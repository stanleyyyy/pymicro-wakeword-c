// examples/wakeword_example.c
// Example usage of the micro_wakeword library

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "micro_wakeword.h"

int main(int argc, char *argv[]) {
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <model.tflite> [libtensorflowlite_c.so]\n", argv[0]);
		fprintf(stderr, "Example: %s ../pymicro_wakeword/models/okay_nabu.tflite\n", argv[0]);
		return 1;
	}

	// Create feature generator
	MicroWakeWordFeatures *features = micro_wakeword_features_create();
	if (!features) {
		fprintf(stderr, "Failed to create feature generator\n");
		return 1;
	}

	// Configure wake word detector
	MicroWakeWordConfig config = {
		.model_path = argv[1],
		.libtensorflowlite_c = (argc > 2) ? argv[2] : NULL,
		.probability_cutoff = 0.97f,
		.sliding_window_size = 5
	};

	// Create wake word detector
	MicroWakeWord *mww = micro_wakeword_create(&config);
	if (!mww) {
		fprintf(stderr, "Failed to create wake word detector\n");
		micro_wakeword_features_destroy(features);
		return 1;
	}

	printf("Wake word detector created successfully\n");
	printf("Processing audio from stdin (16kHz, 16-bit, mono)...\n");

	// Process audio from stdin
	uint8_t audio_buffer[320];  // 10ms at 16kHz (160 samples * 2 bytes)
	size_t bytes_read;
	bool detected = false;

	while ((bytes_read = fread(audio_buffer, 1, sizeof(audio_buffer), stdin)) > 0) {
		// Generate features
		float *feature_array = NULL;
		size_t feature_count = 0;
		int result = micro_wakeword_features_process_streaming(
			features, audio_buffer, bytes_read, &feature_array, &feature_count);

		if (result != 0) {
			fprintf(stderr, "Failed to process features\n");
			break;
		}

		// Process each feature window
		// The feature array contains features in a flat format
		// For microWakeWord, each window has 40 features
		if (feature_array && feature_count > 0) {
			size_t features_per_window = 40;
			for (size_t i = 0; i < feature_count; i += features_per_window) {
				if (i + features_per_window <= feature_count) {
					if (micro_wakeword_process_streaming(
						mww, &feature_array[i], features_per_window)) {
						printf("Wake word detected!\n");
						detected = true;
						break;
					}
				}
			}
			free(feature_array);
		}

		if (detected) {
			break;
		}
	}

	if (!detected) {
		printf("No wake word detected\n");
	}

	// Clean up
	micro_wakeword_destroy(mww);
	micro_wakeword_features_destroy(features);

	return 0;
}
