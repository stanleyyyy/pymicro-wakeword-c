// tests/test_micro_wakeword.c
// C test program based on Python test_microwakeword.py

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "micro_wakeword.h"
#include "wav_reader.h"

#define BYTES_PER_CHUNK (160 * 2)  // 10ms @ 16kHz (16-bit mono)
#define SAMPLES_PER_CHUNK 160
#define FEATURES_PER_WINDOW 40

// Helper to find model file
static const char *find_model_file(const char *model_name) {
	static char path[512];

	// Try different paths (relative to test binary location)
	const char *paths[] = {
		"./models/%s.tflite",
		"models/%s.tflite",
		"../models/%s.tflite",
		"pymicro_wakeword/models/%s.tflite",
		"../pymicro_wakeword/models/%s.tflite",
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

// Helper to find tensorflowlite_c library (for development only)
// On installed systems, rely on system loader to find it in standard paths
static const char *find_tflite_lib(void) {
	// Only check relative paths for development builds
	// On installed systems, return NULL to let system loader find it
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

	// Return NULL to let system loader find it (dlopen will search LD_LIBRARY_PATH, /usr/lib, etc.)
	return NULL;
}

// Test basic creation and destruction
static int test_create_destroy(void) {
	printf("Running test_create_destroy...\n");

	const char *model_path = find_model_file("okay_nabu");
	if (!model_path) {
		printf("  SKIPPED: Model file not found\n");
		return 0;
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

	micro_wakeword_destroy(mww);

	// Test features
	MicroWakeWordFeatures *features = micro_wakeword_features_create();
	if (!features) {
		fprintf(stderr, "Failed to create feature generator\n");
		return 1;
	}

	micro_wakeword_features_destroy(features);

	printf("  test_create_destroy: PASSED\n");
	return 0;
}

// Test reset functionality
static int test_reset(void) {
	printf("Running test_reset...\n");

	const char *model_path = find_model_file("okay_nabu");
	if (!model_path) {
		printf("  SKIPPED: Model file not found\n");
		return 0;
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

	MicroWakeWordFeatures *features = micro_wakeword_features_create();
	if (!features) {
		micro_wakeword_destroy(mww);
		return 1;
	}

	// Process some dummy audio
	uint8_t dummy_audio[320] = {0};
	float *feature_array = NULL;
	size_t feature_count = 0;

	micro_wakeword_features_process_streaming(features, dummy_audio,
						  sizeof(dummy_audio),
						  &feature_array, &feature_count);
	if (feature_array) {
		free(feature_array);
	}

	// Reset
	micro_wakeword_reset(mww);
	micro_wakeword_features_reset(features);

	micro_wakeword_destroy(mww);
	micro_wakeword_features_destroy(features);

	printf("  test_reset: PASSED\n");
	return 0;
}

// Test processing with WAV file
static int test_process_wav(const char *model_name, const char *wav_path, bool should_detect) {
	WavFile wav;
	if (wav_file_read(wav_path, &wav) != 0) {
		fprintf(stderr, "Failed to read WAV file: %s\n", wav_path);
		return 1;
	}

	const char *model_path = find_model_file(model_name);
	if (!model_path) {
		wav_file_free(&wav);
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
		wav_file_free(&wav);
		return 1;
	}

	MicroWakeWordFeatures *features = micro_wakeword_features_create();
	if (!features) {
		micro_wakeword_destroy(mww);
		wav_file_free(&wav);
		return 1;
	}

	// Process audio (pass all at once, like Python version)
	// Convert int16_t audio data to uint8_t bytes
	uint8_t *audio_bytes = (uint8_t *)wav.data;
	size_t audio_size = wav.data_size;

	bool detected = false;

	// Process all audio at once (matching Python: mww_features.process_streaming(audio_bytes))
	float *feature_array = NULL;
	size_t feature_count = 0;

	int result = micro_wakeword_features_process_streaming(
		features, audio_bytes, audio_size, &feature_array, &feature_count);

	if (result == 0 && feature_array && feature_count > 0) {
		// Process each feature window (matching Python: for features in ...)
		// Each window has FEATURES_PER_WINDOW (40) features
		for (size_t i = 0; i < feature_count; i += FEATURES_PER_WINDOW) {
			if (i + FEATURES_PER_WINDOW <= feature_count) {
				// Process one feature window at a time (matching Python: mww.process_streaming(features))
				if (micro_wakeword_process_streaming(mww,
								  &feature_array[i],
								  FEATURES_PER_WINDOW)) {
					detected = true;
					break;
				}
			}
		}
		free(feature_array);
	}

	micro_wakeword_destroy(mww);
	micro_wakeword_features_destroy(features);
	wav_file_free(&wav);

	if (detected != should_detect) {
		fprintf(stderr, "Expected detection=%d, got %d for %s\n",
			should_detect, detected, wav_path);
		return 1;
	}

	return 0;
}

// Helper to find WAV file (similar to Python's _DIR / model_name / f"{number}.wav")
static const char *find_wav_file(const char *model_name, int number) {
	static char wav_path[512];

	// Try different base paths (relative to test binary)
	const char *base_paths[] = {
		"./tests",           // From project root or installed location
		"tests",             // From project root
		".",                 // From tests directory
		"../tests",          // From build directory
		NULL
	};

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

// Test with WAV files if available
static int test_wav_files(void) {
	printf("Running test_wav_files...\n");

	// Try to find WAV files in test directories (matching Python test)
	const char *models[] = {"okay_nabu", "hey_jarvis", "hey_mycroft", "alexa", NULL};

	bool found_any = false;

	for (size_t m = 0; models[m]; ++m) {
		// Try to find WAV files for this model (1, 2, 3)
		for (int num = 1; num <= 3; ++num) {
			const char *wav_path = find_wav_file(models[m], num);
			if (wav_path) {
				found_any = true;

				// Test positive (same model) - should detect
				if (test_process_wav(models[m], wav_path, true) != 0) {
					fprintf(stderr, "Failed positive test for %s/%d.wav\n",
						models[m], num);
					return 1;
				}

				// Test negative (different model) - should not detect
				for (size_t other = 0; models[other]; ++other) {
					if (other != m) {
						if (test_process_wav(models[other], wav_path, false) != 0) {
							fprintf(stderr, "Failed negative test: %s model detected %s/%d.wav\n",
								models[other], models[m], num);
							return 1;
						}
						break;  // Just test one negative per file
					}
				}
			}
		}
	}

	if (!found_any) {
		printf("  SKIPPED: No WAV test files found\n");
		return 0;
	}

	printf("  test_wav_files: PASSED\n");
	return 0;
}

int main(int argc, char *argv[]) {
	int failures = 0;

	failures += test_create_destroy();
	failures += test_reset();
	failures += test_wav_files();

	if (failures == 0) {
		printf("\nAll tests PASSED\n");
		return 0;
	} else {
		printf("\n%d test(s) FAILED\n", failures);
		return 1;
	}
}
