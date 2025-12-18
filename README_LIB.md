# Micro Wake Word C/C++ Library

This is a pure C/C++ library that provides wake word detection using TensorFlow Lite models. It is a standalone version of the Python library functionality.

## API Overview

The library provides a simple C API for processing 16kHz 16-bit audio and detecting wake words using TensorFlow Lite models.

### Header File

Include `micro_wakeword.h` in your project:

```c
#include "micro_wakeword.h"
```

### Functions

#### `MicroWakeWord *micro_wakeword_create(const MicroWakeWordConfig *config)`

Creates a new wake word detector instance. Returns `NULL` on error.

**Configuration structure:**
```c
typedef struct {
	const char *model_path;           // Path to .tflite model file
	const char *libtensorflowlite_c; // Path to libtensorflowlite_c.so (optional, NULL for default)
	float probability_cutoff;        // Detection threshold (0.0-1.0)
	size_t sliding_window_size;       // Number of probabilities to average
} MicroWakeWordConfig;
```

#### `bool micro_wakeword_process_streaming(MicroWakeWord *mww, const float *features, size_t features_size)`

Processes audio features and returns true if wake word is detected.

**Parameters:**
- `mww`: Wake word detector instance
- `features`: Pointer to feature array (typically 40 features per window)
- `features_size`: Number of features

**Returns:**
- `true` if wake word detected
- `false` if not detected or not enough data yet
- Note: This function maintains internal state (feature buffer, probability window)

#### `void micro_wakeword_reset(MicroWakeWord *mww)`

Resets the wake word detector state to initial conditions.

#### `void micro_wakeword_destroy(MicroWakeWord *mww)`

Destroys the wake word detector instance and frees all resources.

#### `MicroWakeWordFeatures *micro_wakeword_features_create(void)`

Creates a new feature generator instance. Returns `NULL` on error.

#### `int micro_wakeword_features_process_streaming(MicroWakeWordFeatures *features, const uint8_t *audio_bytes, size_t audio_size, float **features_out, size_t *features_size_out)`

Processes raw audio bytes and generates features.

**Parameters:**
- `features`: Feature generator instance
- `audio_bytes`: Pointer to 16-bit PCM audio data (16kHz, mono)
- `audio_size`: Size in bytes
- `features_out`: Output array pointer (caller must free)
- `features_size_out`: Number of features generated

**Returns:**
- `0` on success
- Non-zero on error

**Note:** The `features_out` array must be freed by the caller using `free()`.

#### `void micro_wakeword_features_reset(MicroWakeWordFeatures *features)`

Resets the feature generator state to initial conditions.

#### `void micro_wakeword_features_destroy(MicroWakeWordFeatures *features)`

Destroys the feature generator instance and frees all resources.

## Building

### Prerequisites

- C99 or C++11 compiler
- [micro_features library](../pymicro-features) (must be built first)
- TensorFlow Lite C shared library (`libtensorflowlite_c.so`)

### Using the Makefile

First, build the micro_features library:

```bash
cd ../pymicro-features
make -f Makefile.lib
cd ../pymicro-wakeword
```

Then build this library:

```bash
make -f Makefile.lib
```

This will build:
- `libmicro_wakeword.a` - Static library
- `examples/example_c` - C example
- `examples/example_cpp` - C++ example

### Building Tests

To build the test executable:

```bash
make -f Makefile.lib test
```

This will create `tests/test_micro_wakeword` executable. The test will automatically build the library and its dependencies if needed.

To build everything (library, examples, and tests) at once:

```bash
make -f Makefile.lib all test
```

### Running Tests

After building, run the test executable:

```bash
./tests/test_micro_wakeword
```

The test suite includes:
- Basic creation and destruction tests
- Reset functionality tests
- WAV file processing tests (if test WAV files are available)

**Note:** The test will look for:
- Model files in `pymicro_wakeword/models/` (e.g., `okay_nabu.tflite`)
- WAV test files in `tests/<model_name>/` directories (e.g., `tests/okay_nabu/1.wav`)
- TensorFlow Lite library in `lib/` subdirectories

If model or WAV files are not found, some tests will be skipped with a "SKIPPED" message.

### Manual Build

1. Compile `src/micro_wakeword_lib.c` with appropriate flags
2. Link against:
   - `libmicro_features.a` (from pymicro-features)
   - `libtensorflowlite_c.so` (dynamically loaded via dlopen)
   - `libdl` (for dynamic library loading)
3. Include the `include/` directory and micro_features `include/` directory

## Usage Example (C)

```c
#include "micro_wakeword.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
	// Create feature generator
	MicroWakeWordFeatures *features = micro_wakeword_features_create();
	if (!features) {
		fprintf(stderr, "Failed to create feature generator\n");
		return 1;
	}

	// Configure wake word detector
	MicroWakeWordConfig config = {
		.model_path = "models/okay_nabu.tflite",
		.libtensorflowlite_c = NULL,  // Auto-detect
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

	// Process audio (16kHz, 16-bit, mono)
	uint8_t audio_buffer[320];  // 10ms of audio
	// ... read audio into buffer ...

	float *feature_array = NULL;
	size_t feature_count = 0;
	int result = micro_wakeword_features_process_streaming(
		features, audio_buffer, sizeof(audio_buffer),
		&feature_array, &feature_count);

	if (result == 0 && feature_array && feature_count > 0) {
		// Process each feature window (40 features each)
		for (size_t i = 0; i < feature_count; i += 40) {
			if (i + 40 <= feature_count) {
				if (micro_wakeword_process_streaming(mww,
								      &feature_array[i], 40)) {
					printf("Wake word detected!\n");
					break;
				}
			}
		}
		free(feature_array);
	}

	// Clean up
	micro_wakeword_destroy(mww);
	micro_wakeword_features_destroy(features);
	return 0;
}
```

## Usage Example (C++)

See `examples/example.cpp` for a C++ wrapper class that provides RAII semantics.

## Requirements

- C99 or C++11 compiler
- [micro_features library](../pymicro-features) - for audio feature extraction
- TensorFlow Lite C shared library - dynamically loaded at runtime
- libdl - for dynamic library loading (usually included with glibc)

## Configuration

The wake word detector uses the following parameters (matching the Python library):
- Sample rate: 16kHz
- Audio format: 16-bit PCM, mono
- Feature window size: 40 features
- Stride: 3 feature windows per inference
- Sliding window: Configurable (typically 5 probabilities)

These settings are configured through the `MicroWakeWordConfig` structure when creating a detector instance.

## Model Files

Model files (`.tflite`) and their configuration (`.json`) can be found in `pymicro_wakeword/models/`. The library expects the `.tflite` model file path.

## TensorFlow Lite Library

The library dynamically loads `libtensorflowlite_c.so` at runtime. It will search for the library in:
1. Path specified in `libtensorflowlite_c` config parameter
2. Current directory
3. `../lib/linux_amd64/`, `../lib/linux_arm64/`, `../lib/linux_armv7/`

Pre-built libraries are available in the `lib/` directory of this repository.
