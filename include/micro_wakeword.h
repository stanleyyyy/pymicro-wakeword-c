#ifndef MICRO_WAKEWORD_H_
#define MICRO_WAKEWORD_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for the wake word detector instance
typedef struct MicroWakeWord MicroWakeWord;

// Opaque handle for the feature generator instance
typedef struct MicroWakeWordFeatures MicroWakeWordFeatures;

// Configuration structure for creating a wake word detector
typedef struct {
	const char *model_path;           // Path to .tflite model file
	const char *libtensorflowlite_c;  // Path to libtensorflowlite_c.so (optional, NULL for default)
	float probability_cutoff;         // Detection threshold (0.0-1.0)
	size_t sliding_window_size;       // Number of probabilities to average
} MicroWakeWordConfig;

// Create a new wake word detector instance
// Returns NULL on error
MicroWakeWord *micro_wakeword_create(const MicroWakeWordConfig *config);

// Process audio features and return true if wake word is detected
// features: pointer to feature array (1D, size should match model input)
// features_size: number of features
// Returns true if wake word detected, false otherwise, NULL on error
// Note: This function maintains internal state (feature buffer, probability window)
bool micro_wakeword_process_streaming(MicroWakeWord *mww,
				       const float *features,
				       size_t features_size);

// Reset the wake word detector state
void micro_wakeword_reset(MicroWakeWord *mww);

// Get quantization parameters (for debugging)
void micro_wakeword_get_quantization_params(MicroWakeWord *mww,
					    float *input_scale,
					    int32_t *input_zero_point,
					    float *output_scale,
					    int32_t *output_zero_point);

// Get buffer size (for debugging)
size_t micro_wakeword_get_buffer_size(MicroWakeWord *mww);

// Get probability information (for debugging)
// Returns the number of probabilities in the window
size_t micro_wakeword_get_probabilities(MicroWakeWord *mww,
					float *latest_prob,
					float *mean_prob);

// Destroy the wake word detector instance and free all resources
void micro_wakeword_destroy(MicroWakeWord *mww);

// Create a new feature generator instance
// Returns NULL on error
MicroWakeWordFeatures *micro_wakeword_features_create(void);

// Process raw audio bytes and generate features
// audio_bytes: pointer to 16-bit PCM audio data (16kHz, mono)
// audio_size: size in bytes
// features_out: output array pointer (caller must free)
// features_size_out: number of features generated
// Returns 0 on success, non-zero on error
// Note: The features_out array must be freed by the caller
int micro_wakeword_features_process_streaming(
	MicroWakeWordFeatures *features,
	const uint8_t *audio_bytes,
	size_t audio_size,
	float **features_out,
	size_t *features_size_out);

// Reset the feature generator state
void micro_wakeword_features_reset(MicroWakeWordFeatures *features);

// Destroy the feature generator instance and free all resources
void micro_wakeword_features_destroy(MicroWakeWordFeatures *features);

#ifdef __cplusplus
}
#endif

#endif  // MICRO_WAKEWORD_H_
