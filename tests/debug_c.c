// Debug version of test to compare with Python
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include "micro_wakeword.h"
#include "wav_reader.h"

#define BYTES_PER_CHUNK (160 * 2)
#define FEATURES_PER_WINDOW 40

// Helper to find model file
static const char *find_model_file(const char *model_name) {
	static char path[512];
	const char *paths[] = {
		"pymicro_wakeword/models/%s.tflite",
		"../pymicro_wakeword/models/%s.tflite",
		"../../pymicro_wakeword/models/%s.tflite",
		NULL
	};

	for (size_t i = 0; paths[i]; ++i) {
		snprintf(path, sizeof(path), paths[i], model_name);
		FILE *f = fopen(path, "r");
		if (f) {
			fclose(f);
			return path;
		}
	}
	return NULL;
}

// Helper to find tensorflowlite_c library
static const char *find_tflite_lib(void) {
	const char *paths[] = {
		"lib/linux_amd64/libtensorflowlite_c.so",
		"lib/linux_arm64/libtensorflowlite_c.so",
		"lib/linux_armv7/libtensorflowlite_c.so",
		"../lib/linux_amd64/libtensorflowlite_c.so",
		"../lib/linux_arm64/libtensorflowlite_c.so",
		"../lib/linux_armv7/libtensorflowlite_c.so",
		NULL
	};

	for (size_t i = 0; paths[i]; ++i) {
		FILE *f = fopen(paths[i], "r");
		if (f) {
			fclose(f);
			return paths[i];
		}
	}
	return NULL;
}

// Helper to find WAV file
static const char *find_wav_file(const char *model_name, int number) {
	static char wav_path[512];
	const char *base_paths[] = {"tests", ".", "../tests", NULL};

	for (size_t i = 0; base_paths[i]; ++i) {
		snprintf(wav_path, sizeof(wav_path), "%s/%s/%d.wav",
			base_paths[i], model_name, number);
		FILE *f = fopen(wav_path, "rb");
		if (f) {
			fclose(f);
			return wav_path;
		}
	}
	return NULL;
}

// Calculate mean of array
static float mean_float(const float *arr, size_t count) {
	if (count == 0) return 0.0f;
	float sum = 0.0f;
	for (size_t i = 0; i < count; ++i) {
		sum += arr[i];
	}
	return sum / count;
}

// Find min/max
static void min_max_float(const float *arr, size_t count, float *min_out, float *max_out) {
	if (count == 0) {
		*min_out = 0.0f;
		*max_out = 0.0f;
		return;
	}
	float min = arr[0];
	float max = arr[0];
	for (size_t i = 1; i < count; ++i) {
		if (arr[i] < min) min = arr[i];
		if (arr[i] > max) max = arr[i];
	}
	*min_out = min;
	*max_out = max;
}

int main(int argc, char *argv[]) {
	const char *model_name = "okay_nabu";
	int wav_number = 1;

	printf("Loading model: %s\n", model_name);
	const char *model_path = find_model_file(model_name);
	if (!model_path) {
		fprintf(stderr, "Failed to find model file\n");
		return 1;
	}

	const char *lib_path = find_tflite_lib();

	MicroWakeWordConfig config = {
		.model_path = model_path,
		.libtensorflowlite_c = lib_path,
		.probability_cutoff = 0.97f,
		.sliding_window_size = 5
	};

	MicroWakeWord *mww = micro_wakeword_create(&config);
	if (!mww) {
		fprintf(stderr, "Failed to create wake word detector\n");
		return 1;
	}

	// Get quantization parameters
	float input_scale, output_scale;
	int32_t input_zero_point, output_zero_point;
	micro_wakeword_get_quantization_params(mww, &input_scale, &input_zero_point,
					       &output_scale, &output_zero_point);

	printf("Model loaded:\n");
	printf("  input_scale: %.17g\n", input_scale);
	printf("  input_zero_point: %d\n", input_zero_point);
	printf("  output_scale: %.8g\n", output_scale);
	printf("  output_zero_point: %d\n", output_zero_point);
	printf("  probability_cutoff: %.2f\n", config.probability_cutoff);
	printf("  sliding_window_size: %zu\n", config.sliding_window_size);

	printf("\nLoading WAV file: %s/%d.wav\n", model_name, wav_number);
	const char *wav_path = find_wav_file(model_name, wav_number);
	if (!wav_path) {
		fprintf(stderr, "Failed to find WAV file\n");
		micro_wakeword_destroy(mww);
		return 1;
	}

	WavFile wav;
	if (wav_file_read(wav_path, &wav) != 0) {
		fprintf(stderr, "Failed to read WAV file\n");
		micro_wakeword_destroy(mww);
		return 1;
	}

	printf("Audio size: %zu bytes (%.1f samples)\n", wav.data_size, (float)wav.data_size / 2.0f);

	MicroWakeWordFeatures *features = micro_wakeword_features_create();
	if (!features) {
		fprintf(stderr, "Failed to create feature generator\n");
		wav_file_free(&wav);
		micro_wakeword_destroy(mww);
		return 1;
	}

	printf("\nProcessing features...\n");

	uint8_t *audio_bytes = (uint8_t *)wav.data;
	size_t audio_size = wav.data_size;

	float *feature_array = NULL;
	size_t feature_count = 0;

	int result = micro_wakeword_features_process_streaming(
		features, audio_bytes, audio_size, &feature_array, &feature_count);

	if (result != 0 || !feature_array || feature_count == 0) {
		fprintf(stderr, "Failed to process features\n");
		wav_file_free(&wav);
		micro_wakeword_features_destroy(features);
		micro_wakeword_destroy(mww);
		return 1;
	}

	size_t feature_window_count = 0;
	bool detected = false;
	float probabilities[1000];  // Store probabilities for summary
	size_t prob_count = 0;

	// Process each feature window
	for (size_t i = 0; i < feature_count; i += FEATURES_PER_WINDOW) {
		if (i + FEATURES_PER_WINDOW <= feature_count) {
			feature_window_count++;
			const float *window_features = &feature_array[i];

			printf("\nFeature window #%zu:\n", feature_window_count);
			printf("  Shape: (1, 1, %d)\n", FEATURES_PER_WINDOW);
			printf("  First 5 values: [%f %f %f %f %f]\n",
				window_features[0], window_features[1], window_features[2],
				window_features[3], window_features[4]);

			float min_val, max_val;
			min_max_float(window_features, FEATURES_PER_WINDOW, &min_val, &max_val);
			float mean_val = mean_float(window_features, FEATURES_PER_WINDOW);
			printf("  Min: %.6f, Max: %.6f, Mean: %.6f\n", min_val, max_val, mean_val);

			// Check buffer state before processing
			size_t buffer_size_before = micro_wakeword_get_buffer_size(mww);
			printf("  Buffer size before: %zu\n", buffer_size_before);

			bool result = micro_wakeword_process_streaming(mww, window_features, FEATURES_PER_WINDOW);

			// Check buffer state after processing
			size_t buffer_size_after = micro_wakeword_get_buffer_size(mww);
			printf("  Buffer size after: %zu\n", buffer_size_after);

			// Get probability information
			float latest_prob, mean_prob;
			size_t prob_window_size = micro_wakeword_get_probabilities(mww, &latest_prob, &mean_prob);

			if (prob_window_size > 0) {
				if (prob_count < sizeof(probabilities) / sizeof(probabilities[0])) {
					probabilities[prob_count++] = latest_prob;
				}
				printf("  Latest probability: %.6f\n", latest_prob);
				printf("  Mean probability: %.6f (window size: %zu)\n", mean_prob, prob_window_size);
			}
			printf("  Detection: %s\n", result ? "True" : "False");

			if (result) {
				printf("\n*** WAKE WORD DETECTED at feature window #%zu ***\n", feature_window_count);
				detected = true;
				break;
			}
		}
	}

	printf("\nSummary:\n");
	printf("  Total feature windows processed: %zu\n", feature_window_count);
	printf("  Total probabilities: %zu\n", prob_count);
	if (prob_count > 0) {
		printf("  First 5 probabilities: ");
		for (size_t i = 0; i < 5 && i < prob_count; ++i) {
			printf("%.6f ", probabilities[i]);
		}
		printf("\n");
		if (prob_count > 5) {
			printf("  Last 5 probabilities: ");
			size_t start = (prob_count > 5) ? prob_count - 5 : 0;
			for (size_t i = start; i < prob_count; ++i) {
				printf("%.6f ", probabilities[i]);
			}
			printf("\n");
		}
		float max_prob = probabilities[0];
		float min_prob = probabilities[0];
		for (size_t i = 1; i < prob_count; ++i) {
			if (probabilities[i] > max_prob) max_prob = probabilities[i];
			if (probabilities[i] < min_prob) min_prob = probabilities[i];
		}
		printf("  Max probability: %.6f\n", max_prob);
		printf("  Min probability: %.6f\n", min_prob);
	}

	free(feature_array);
	wav_file_free(&wav);
	micro_wakeword_features_destroy(features);
	micro_wakeword_destroy(mww);

	return 0;
}
