// src/micro_wakeword_lib.c
#include "micro_wakeword.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <stdint.h>

// Include micro_features for feature extraction
#include "micro_features.h"

// Constants
#define STRIDE 3
#define SAMPLES_PER_CHUNK 160  // 10ms @ 16kHz
#define BYTES_PER_CHUNK (SAMPLES_PER_CHUNK * 2)  // 16-bit samples
#define BYTES_PER_SAMPLE 2

// TensorFlow Lite C API types
typedef int TfLiteStatus;  // kTfLiteOk == 0
typedef void *TfLiteModel;
typedef void *TfLiteInterpreter;
typedef void *TfLiteTensor;

typedef struct {
	float scale;
	int32_t zero_point;
} TfLiteQuantizationParams;

// TensorFlow Lite C API function pointers
typedef TfLiteModel (*TfLiteModelCreateFromFileFunc)(const char *);
typedef TfLiteInterpreter (*TfLiteInterpreterCreateFunc)(TfLiteModel, void *);
typedef TfLiteStatus (*TfLiteInterpreterAllocateTensorsFunc)(TfLiteInterpreter);
typedef TfLiteStatus (*TfLiteInterpreterInvokeFunc)(TfLiteInterpreter);
typedef TfLiteTensor (*TfLiteInterpreterGetInputTensorFunc)(TfLiteInterpreter, int32_t);
typedef TfLiteTensor (*TfLiteInterpreterGetOutputTensorFunc)(TfLiteInterpreter, int32_t);
typedef size_t (*TfLiteTensorByteSizeFunc)(TfLiteTensor);
typedef TfLiteQuantizationParams (*TfLiteTensorQuantizationParamsFunc)(TfLiteTensor);
typedef TfLiteStatus (*TfLiteTensorCopyFromBufferFunc)(TfLiteTensor, const void *, size_t);
typedef TfLiteStatus (*TfLiteTensorCopyToBufferFunc)(TfLiteTensor, void *, size_t);
typedef void (*TfLiteInterpreterDeleteFunc)(TfLiteInterpreter);
typedef void (*TfLiteModelDeleteFunc)(TfLiteModel);

// Feature buffer entry
typedef struct {
	float *features;
	size_t features_size;
} FeatureBufferEntry;

// Probability window (circular buffer)
typedef struct {
	float *probabilities;
	size_t size;
	size_t count;
	size_t head;
} ProbabilityWindow;

// MicroWakeWord structure
struct MicroWakeWord {
	void *tflite_handle;  // dlopen handle for tensorflowlite_c
	TfLiteModel model;
	TfLiteInterpreter interpreter;
	TfLiteTensor input_tensor;
	TfLiteTensor output_tensor;

	// Quantization parameters
	float input_scale;
	int32_t input_zero_point;
	float output_scale;
	int32_t output_zero_point;

	// Feature buffer (STRIDE entries)
	FeatureBufferEntry feature_buffer[STRIDE];
	size_t feature_buffer_count;

	// Probability sliding window
	ProbabilityWindow prob_window;

	// Configuration
	char *model_path;  // Stored for reset
	float probability_cutoff;
	size_t sliding_window_size;

	// Function pointers
	TfLiteModelCreateFromFileFunc TfLiteModelCreateFromFile;
	TfLiteInterpreterCreateFunc TfLiteInterpreterCreate;
	TfLiteInterpreterAllocateTensorsFunc TfLiteInterpreterAllocateTensors;
	TfLiteInterpreterInvokeFunc TfLiteInterpreterInvoke;
	TfLiteInterpreterGetInputTensorFunc TfLiteInterpreterGetInputTensor;
	TfLiteInterpreterGetOutputTensorFunc TfLiteInterpreterGetOutputTensor;
	TfLiteTensorByteSizeFunc TfLiteTensorByteSize;
	TfLiteTensorQuantizationParamsFunc TfLiteTensorQuantizationParams;
	TfLiteTensorCopyFromBufferFunc TfLiteTensorCopyFromBuffer;
	TfLiteTensorCopyToBufferFunc TfLiteTensorCopyToBuffer;
	TfLiteInterpreterDeleteFunc TfLiteInterpreterDelete;
	TfLiteModelDeleteFunc TfLiteModelDelete;
};

// MicroWakeWordFeatures structure
struct MicroWakeWordFeatures {
	MicroFrontend *frontend;
	uint8_t *audio_buffer;
	size_t audio_buffer_size;
	size_t audio_buffer_capacity;
};

// Helper function to find tensorflowlite_c library
static const char *find_tflite_lib(const char *user_path) {
	if (user_path && user_path[0] != '\0') {
		return user_path;
	}

	// Try relative paths for development builds
	static const char *dev_paths[] = {
		"../lib/linux_amd64/libtensorflowlite_c.so",
		"../lib/linux_arm64/libtensorflowlite_c.so",
		"../lib/linux_armv7/libtensorflowlite_c.so",
		"./libtensorflowlite_c.so",
		NULL
	};

	for (size_t i = 0; dev_paths[i]; ++i) {
		FILE *f = fopen(dev_paths[i], "r");
		if (f) {
			fclose(f);
			return dev_paths[i];
		}
	}

	// Return system library name - dlopen will search standard paths
	// (LD_LIBRARY_PATH, /usr/lib, /lib, etc.)
	return "libtensorflowlite_c.so";
}

// Load TensorFlow Lite C API functions
static int load_tflite_functions(MicroWakeWord *mww, const char *lib_path) {
	const char *lib = find_tflite_lib(lib_path);
	if (!lib) {
		return -1;
	}

	mww->tflite_handle = dlopen(lib, RTLD_LAZY | RTLD_GLOBAL);
	if (!mww->tflite_handle) {
		return -2;
	}

	// Load function pointers
	mww->TfLiteModelCreateFromFile = (TfLiteModelCreateFromFileFunc)
		dlsym(mww->tflite_handle, "TfLiteModelCreateFromFile");
	mww->TfLiteInterpreterCreate = (TfLiteInterpreterCreateFunc)
		dlsym(mww->tflite_handle, "TfLiteInterpreterCreate");
	mww->TfLiteInterpreterAllocateTensors = (TfLiteInterpreterAllocateTensorsFunc)
		dlsym(mww->tflite_handle, "TfLiteInterpreterAllocateTensors");
	mww->TfLiteInterpreterInvoke = (TfLiteInterpreterInvokeFunc)
		dlsym(mww->tflite_handle, "TfLiteInterpreterInvoke");
	mww->TfLiteInterpreterGetInputTensor = (TfLiteInterpreterGetInputTensorFunc)
		dlsym(mww->tflite_handle, "TfLiteInterpreterGetInputTensor");
	mww->TfLiteInterpreterGetOutputTensor = (TfLiteInterpreterGetOutputTensorFunc)
		dlsym(mww->tflite_handle, "TfLiteInterpreterGetOutputTensor");
	mww->TfLiteTensorByteSize = (TfLiteTensorByteSizeFunc)
		dlsym(mww->tflite_handle, "TfLiteTensorByteSize");
	mww->TfLiteTensorQuantizationParams = (TfLiteTensorQuantizationParamsFunc)
		dlsym(mww->tflite_handle, "TfLiteTensorQuantizationParams");
	mww->TfLiteTensorCopyFromBuffer = (TfLiteTensorCopyFromBufferFunc)
		dlsym(mww->tflite_handle, "TfLiteTensorCopyFromBuffer");
	mww->TfLiteTensorCopyToBuffer = (TfLiteTensorCopyToBufferFunc)
		dlsym(mww->tflite_handle, "TfLiteTensorCopyToBuffer");
	mww->TfLiteInterpreterDelete = (TfLiteInterpreterDeleteFunc)
		dlsym(mww->tflite_handle, "TfLiteInterpreterDelete");
	mww->TfLiteModelDelete = (TfLiteModelDeleteFunc)
		dlsym(mww->tflite_handle, "TfLiteModelDelete");

	// Check if all functions loaded
	if (!mww->TfLiteModelCreateFromFile || !mww->TfLiteInterpreterCreate ||
	    !mww->TfLiteInterpreterAllocateTensors || !mww->TfLiteInterpreterInvoke ||
	    !mww->TfLiteInterpreterGetInputTensor || !mww->TfLiteInterpreterGetOutputTensor ||
	    !mww->TfLiteTensorByteSize || !mww->TfLiteTensorQuantizationParams ||
	    !mww->TfLiteTensorCopyFromBuffer || !mww->TfLiteTensorCopyToBuffer ||
	    !mww->TfLiteInterpreterDelete || !mww->TfLiteModelDelete) {
		dlclose(mww->tflite_handle);
		return -3;
	}

	return 0;
}

// Initialize probability window
static int init_probability_window(ProbabilityWindow *window, size_t size) {
	window->probabilities = (float *)malloc(size * sizeof(float));
	if (!window->probabilities) {
		return -1;
	}
	window->size = size;
	window->count = 0;
	window->head = 0;
	return 0;
}

// Add probability to window
static void add_probability(ProbabilityWindow *window, float prob) {
	window->probabilities[window->head] = prob;
	window->head = (window->head + 1) % window->size;
	if (window->count < window->size) {
		window->count++;
	}
}

// Calculate mean of probability window
static float mean_probability(const ProbabilityWindow *window) {
	if (window->count == 0) {
		return 0.0f;
	}

	float sum = 0.0f;
	for (size_t i = 0; i < window->count; ++i) {
		sum += window->probabilities[i];
	}
	return sum / window->count;
}

// Load model
static int load_model(MicroWakeWord *mww, const char *model_path) {
	mww->model = mww->TfLiteModelCreateFromFile(model_path);
	if (!mww->model) {
		return -1;
	}

	mww->interpreter = mww->TfLiteInterpreterCreate(mww->model, NULL);
	if (!mww->interpreter) {
		mww->TfLiteModelDelete(mww->model);
		mww->model = NULL;
		return -2;
	}

	if (mww->TfLiteInterpreterAllocateTensors(mww->interpreter) != 0) {
		mww->TfLiteInterpreterDelete(mww->interpreter);
		mww->TfLiteModelDelete(mww->model);
		mww->interpreter = NULL;
		mww->model = NULL;
		return -3;
	}

	mww->input_tensor = mww->TfLiteInterpreterGetInputTensor(mww->interpreter, 0);
	mww->output_tensor = mww->TfLiteInterpreterGetOutputTensor(mww->interpreter, 0);

	if (!mww->input_tensor || !mww->output_tensor) {
		mww->TfLiteInterpreterDelete(mww->interpreter);
		mww->TfLiteModelDelete(mww->model);
		mww->interpreter = NULL;
		mww->model = NULL;
		return -4;
	}

	// Get quantization parameters
	TfLiteQuantizationParams input_q = mww->TfLiteTensorQuantizationParams(mww->input_tensor);
	TfLiteQuantizationParams output_q = mww->TfLiteTensorQuantizationParams(mww->output_tensor);

	mww->input_scale = input_q.scale;
	mww->input_zero_point = input_q.zero_point;
	mww->output_scale = output_q.scale;
	mww->output_zero_point = output_q.zero_point;

	return 0;
}

MicroWakeWord *micro_wakeword_create(const MicroWakeWordConfig *config) {
	if (!config || !config->model_path) {
		return NULL;
	}

	MicroWakeWord *mww = (MicroWakeWord *)calloc(1, sizeof(MicroWakeWord));
	if (!mww) {
		return NULL;
	}

	// Load TensorFlow Lite library
	int result = load_tflite_functions(mww, config->libtensorflowlite_c);
	if (result != 0) {
		free(mww);
		return NULL;
	}

	// Initialize probability window
	if (init_probability_window(&mww->prob_window, config->sliding_window_size) != 0) {
		dlclose(mww->tflite_handle);
		free(mww);
		return NULL;
	}

	mww->probability_cutoff = config->probability_cutoff;
	mww->sliding_window_size = config->sliding_window_size;
	mww->feature_buffer_count = 0;

	// Store model path for reset
	mww->model_path = strdup(config->model_path);
	if (!mww->model_path) {
		free(mww->prob_window.probabilities);
		dlclose(mww->tflite_handle);
		free(mww);
		return NULL;
	}

	// Load model
	if (load_model(mww, config->model_path) != 0) {
		free(mww->model_path);
		free(mww->prob_window.probabilities);
		dlclose(mww->tflite_handle);
		free(mww);
		return NULL;
	}

	return mww;
}

bool micro_wakeword_process_streaming(MicroWakeWord *mww,
				       const float *features,
				       size_t features_size) {
	if (!mww || !features || !mww->interpreter || !mww->model) {
		return false;
	}

	// Always add current features to buffer first (matching Python: self._features.append(features))
	FeatureBufferEntry *entry = &mww->feature_buffer[mww->feature_buffer_count];
	entry->features = (float *)malloc(features_size * sizeof(float));
	if (!entry->features) {
		return false;
	}
	memcpy(entry->features, features, features_size * sizeof(float));
	entry->features_size = features_size;
	mww->feature_buffer_count++;

	// Check if we have enough features (matching Python: if len(self._features) < STRIDE)
	if (mww->feature_buffer_count < STRIDE) {
		return false;  // Not enough features yet
	}

	// Concatenate features (matching Python: np.concatenate(self._features, axis=1))
	size_t total_features = 0;
	for (size_t i = 0; i < STRIDE; ++i) {
		total_features += mww->feature_buffer[i].features_size;
	}

	// Allocate and concatenate
	float *concatenated = (float *)malloc(total_features * sizeof(float));
	if (!concatenated) {
		return false;
	}

	size_t offset = 0;
	for (size_t i = 0; i < STRIDE; ++i) {
		memcpy(concatenated + offset, mww->feature_buffer[i].features,
		       mww->feature_buffer[i].features_size * sizeof(float));
		offset += mww->feature_buffer[i].features_size;
	}

	// Quantize input
	uint8_t *quant_features = (uint8_t *)malloc(total_features * sizeof(uint8_t));
	if (!quant_features) {
		free(concatenated);
		return false;
	}

	for (size_t i = 0; i < total_features; ++i) {
		// Match Python: np.round(...).astype(np.uint8)
		// uint8 casting wraps negative values (e.g., -128 becomes 128)
		float quant = roundf(concatenated[i] / mww->input_scale + mww->input_zero_point);
		// Cast directly to uint8_t - this will wrap negative values correctly
		// e.g., -128 wraps to 128, -1 wraps to 255
		quant_features[i] = (uint8_t)(int32_t)quant;
	}

	// Copy to input tensor
	if (mww->TfLiteTensorCopyFromBuffer(mww->input_tensor, quant_features,
					     total_features * sizeof(uint8_t)) != 0) {
		free(quant_features);
		free(concatenated);
		return false;
	}

	// Run inference
	if (mww->TfLiteInterpreterInvoke(mww->interpreter) != 0) {
		free(quant_features);
		free(concatenated);
		return false;
	}

	// Read output
	size_t output_bytes = mww->TfLiteTensorByteSize(mww->output_tensor);
	uint8_t *output_data = (uint8_t *)malloc(output_bytes);
	if (!output_data) {
		free(quant_features);
		free(concatenated);
		return false;
	}

	if (mww->TfLiteTensorCopyToBuffer(mww->output_tensor, output_data, output_bytes) != 0) {
		free(output_data);
		free(quant_features);
		free(concatenated);
		return false;
	}

	// Debug: check output tensor size
	// Note: For quantized models, output should be uint8, but tensor might be larger
	// Python reads the entire output_bytes and converts to float32 array

	// Dequantize output
	// Python does: (output_data.astype(np.float32) - zero_point) * scale
	// where output_data is a numpy array. For a single-element output, this becomes:
	// (float32(output_data[0]) - zero_point) * scale
	float result = ((float)output_data[0] - mww->output_zero_point) * mww->output_scale;

	// Add to probability window
	add_probability(&mww->prob_window, result);

	// Clear feature buffer (stride instead of rolling)
	// Note: Python version clears buffer completely, next feature window starts fresh
	for (size_t i = 0; i < STRIDE; ++i) {
		free(mww->feature_buffer[i].features);
		mww->feature_buffer[i].features = NULL;
		mww->feature_buffer[i].features_size = 0;
	}
	mww->feature_buffer_count = 0;

	free(output_data);
	free(quant_features);
	free(concatenated);

	// Check if enough probabilities
	if (mww->prob_window.count < mww->sliding_window_size) {
		return false;
	}

	// Check if mean probability exceeds cutoff
	float mean_prob = mean_probability(&mww->prob_window);
	return mean_prob > mww->probability_cutoff;
}

void micro_wakeword_reset(MicroWakeWord *mww) {
	if (!mww) {
		return;
	}

	// Clear feature buffer
	for (size_t i = 0; i < STRIDE; ++i) {
		free(mww->feature_buffer[i].features);
		mww->feature_buffer[i].features = NULL;
		mww->feature_buffer[i].features_size = 0;
	}
	mww->feature_buffer_count = 0;

	// Clear probability window
	mww->prob_window.count = 0;
	mww->prob_window.head = 0;

	// Reload model to reset internal state
	if (mww->interpreter) {
		mww->TfLiteInterpreterDelete(mww->interpreter);
		mww->interpreter = NULL;
	}
	if (mww->model) {
		mww->TfLiteModelDelete(mww->model);
		mww->model = NULL;
	}

	// Reload model
	if (mww->model_path) {
		load_model(mww, mww->model_path);
	}
}

void micro_wakeword_get_quantization_params(MicroWakeWord *mww,
					    float *input_scale,
					    int32_t *input_zero_point,
					    float *output_scale,
					    int32_t *output_zero_point) {
	if (!mww) {
		return;
	}
	if (input_scale) *input_scale = mww->input_scale;
	if (input_zero_point) *input_zero_point = mww->input_zero_point;
	if (output_scale) *output_scale = mww->output_scale;
	if (output_zero_point) *output_zero_point = mww->output_zero_point;
}

size_t micro_wakeword_get_buffer_size(MicroWakeWord *mww) {
	if (!mww) {
		return 0;
	}
	return mww->feature_buffer_count;
}

size_t micro_wakeword_get_probabilities(MicroWakeWord *mww,
					float *latest_prob,
					float *mean_prob) {
	if (!mww) {
		if (latest_prob) *latest_prob = 0.0f;
		if (mean_prob) *mean_prob = 0.0f;
		return 0;
	}

	if (mww->prob_window.count > 0) {
		// Get latest probability (most recently added)
		size_t latest_idx = (mww->prob_window.head == 0) ?
			mww->prob_window.size - 1 : mww->prob_window.head - 1;
		if (latest_prob) {
			*latest_prob = mww->prob_window.probabilities[latest_idx];
		}
		if (mean_prob) {
			*mean_prob = mean_probability(&mww->prob_window);
		}
	} else {
		if (latest_prob) *latest_prob = 0.0f;
		if (mean_prob) *mean_prob = 0.0f;
	}

	return mww->prob_window.count;
}

void micro_wakeword_destroy(MicroWakeWord *mww) {
	if (!mww) {
		return;
	}

	// Clear feature buffer
	for (size_t i = 0; i < STRIDE; ++i) {
		free(mww->feature_buffer[i].features);
	}

	// Free probability window
	free(mww->prob_window.probabilities);

	// Delete interpreter and model
	if (mww->interpreter) {
		mww->TfLiteInterpreterDelete(mww->interpreter);
	}
	if (mww->model) {
		mww->TfLiteModelDelete(mww->model);
	}

	// Free model path
	free(mww->model_path);

	// Close library
	if (mww->tflite_handle) {
		dlclose(mww->tflite_handle);
	}

	free(mww);
}

MicroWakeWordFeatures *micro_wakeword_features_create(void) {
	MicroWakeWordFeatures *features = (MicroWakeWordFeatures *)calloc(1, sizeof(MicroWakeWordFeatures));
	if (!features) {
		return NULL;
	}

	features->frontend = micro_frontend_create();
	if (!features->frontend) {
		free(features);
		return NULL;
	}

	features->audio_buffer_size = 0;
	features->audio_buffer_capacity = 4096;  // Initial capacity
	features->audio_buffer = (uint8_t *)malloc(features->audio_buffer_capacity);
	if (!features->audio_buffer) {
		micro_frontend_destroy(features->frontend);
		free(features);
		return NULL;
	}

	return features;
}

int micro_wakeword_features_process_streaming(
	MicroWakeWordFeatures *features,
	const uint8_t *audio_bytes,
	size_t audio_size,
	float **features_out,
	size_t *features_size_out) {
	if (!features || !audio_bytes || !features_out || !features_size_out) {
		return -1;
	}

	*features_out = NULL;
	*features_size_out = 0;

	// Append to buffer (audio_bytes is already in the correct format - raw bytes from WAV)
	if (features->audio_buffer_size + audio_size > features->audio_buffer_capacity) {
		size_t new_capacity = features->audio_buffer_capacity * 2;
		while (new_capacity < features->audio_buffer_size + audio_size) {
			new_capacity *= 2;
		}
		uint8_t *new_buffer = (uint8_t *)realloc(features->audio_buffer, new_capacity);
		if (!new_buffer) {
			return -2;
		}
		features->audio_buffer = new_buffer;
		features->audio_buffer_capacity = new_capacity;
	}

	memcpy(features->audio_buffer + features->audio_buffer_size, audio_bytes, audio_size);
	features->audio_buffer_size += audio_size;

	// Process chunks
	if (features->audio_buffer_size < BYTES_PER_CHUNK) {
		return 0;  // Not enough data
	}

	// Estimate max features (one per chunk, 40 features each)
	size_t max_features = (features->audio_buffer_size / BYTES_PER_CHUNK) * 40;
	float *all_features = (float *)malloc(max_features * sizeof(float));
	if (!all_features) {
		return -3;
	}

	size_t total_features = 0;
	size_t buffer_idx = 0;

	while (buffer_idx + BYTES_PER_CHUNK <= features->audio_buffer_size) {
		MicroFrontendOutput output;
		int16_t *chunk_samples = (int16_t *)(features->audio_buffer + buffer_idx);
		// micro_frontend_process_samples expects number of samples, not bytes
		int result = micro_frontend_process_samples(features->frontend, chunk_samples,
							    SAMPLES_PER_CHUNK, &output);

		if (result == 0 && output.features_size > 0) {
			// Resize if needed
			if (total_features + output.features_size > max_features) {
				max_features = (total_features + output.features_size) * 2;
				float *new_features = (float *)realloc(all_features,
									max_features * sizeof(float));
				if (!new_features) {
					free(output.features);
					free(all_features);
					return -4;
				}
				all_features = new_features;
			}

			memcpy(all_features + total_features, output.features,
			       output.features_size * sizeof(float));
			total_features += output.features_size;
		}

		if (output.features) {
			free(output.features);
		}

		buffer_idx += output.samples_read * BYTES_PER_SAMPLE;
	}

	// Remove processed audio
	if (buffer_idx > 0) {
		memmove(features->audio_buffer,
			features->audio_buffer + buffer_idx,
			features->audio_buffer_size - buffer_idx);
		features->audio_buffer_size -= buffer_idx;
	}

	*features_out = all_features;
	*features_size_out = total_features;
	return 0;
}

void micro_wakeword_features_reset(MicroWakeWordFeatures *features) {
	if (!features) {
		return;
	}

	micro_frontend_reset(features->frontend);
	features->audio_buffer_size = 0;
}

void micro_wakeword_features_destroy(MicroWakeWordFeatures *features) {
	if (!features) {
		return;
	}

	micro_frontend_destroy(features->frontend);
	free(features->audio_buffer);
	free(features);
}
