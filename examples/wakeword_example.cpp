// examples/wakeword_example.cpp
// C++ example usage of the micro_wakeword library

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include "micro_wakeword.h"

// C++ wrapper class for convenience
class MicroWakeWordWrapper {
public:
	MicroWakeWordWrapper(const MicroWakeWordConfig &config)
		: mww_(micro_wakeword_create(&config)) {
		if (!mww_) {
			throw std::runtime_error("Failed to create wake word detector");
		}
	}

	~MicroWakeWordWrapper() {
		if (mww_) {
			micro_wakeword_destroy(mww_);
		}
	}

	// Delete copy constructor and assignment
	MicroWakeWordWrapper(const MicroWakeWordWrapper &) = delete;
	MicroWakeWordWrapper &operator=(const MicroWakeWordWrapper &) = delete;

	// Move constructor
	MicroWakeWordWrapper(MicroWakeWordWrapper &&other) noexcept
		: mww_(other.mww_) {
		other.mww_ = nullptr;
	}

	// Move assignment
	MicroWakeWordWrapper &operator=(MicroWakeWordWrapper &&other) noexcept {
		if (this != &other) {
			if (mww_) {
				micro_wakeword_destroy(mww_);
			}
			mww_ = other.mww_;
			other.mww_ = nullptr;
		}
		return *this;
	}

	bool process_streaming(const std::vector<float> &features) {
		return micro_wakeword_process_streaming(mww_, features.data(),
							features.size());
	}

	void reset() {
		micro_wakeword_reset(mww_);
	}

private:
	MicroWakeWord *mww_;
};

// C++ wrapper for feature generator
class MicroWakeWordFeaturesWrapper {
public:
	MicroWakeWordFeaturesWrapper()
		: features_(micro_wakeword_features_create()) {
		if (!features_) {
			throw std::runtime_error("Failed to create feature generator");
		}
	}

	~MicroWakeWordFeaturesWrapper() {
		if (features_) {
			micro_wakeword_features_destroy(features_);
		}
	}

	// Delete copy constructor and assignment
	MicroWakeWordFeaturesWrapper(const MicroWakeWordFeaturesWrapper &) = delete;
	MicroWakeWordFeaturesWrapper &operator=(const MicroWakeWordFeaturesWrapper &) = delete;

	// Move constructor
	MicroWakeWordFeaturesWrapper(MicroWakeWordFeaturesWrapper &&other) noexcept
		: features_(other.features_) {
		other.features_ = nullptr;
	}

	// Move assignment
	MicroWakeWordFeaturesWrapper &operator=(MicroWakeWordFeaturesWrapper &&other) noexcept {
		if (this != &other) {
			if (features_) {
				micro_wakeword_features_destroy(features_);
			}
			features_ = other.features_;
			other.features_ = nullptr;
		}
		return *this;
	}

	std::vector<float> process_streaming(const std::vector<uint8_t> &audio) {
		float *features_out = nullptr;
		size_t features_size = 0;
		int result = micro_wakeword_features_process_streaming(
			features_, audio.data(), audio.size(), &features_out, &features_size);

		if (result != 0) {
			throw std::runtime_error("Failed to process features");
		}

		std::vector<float> features(features_out, features_out + features_size);
		free(features_out);  // Free the C-allocated memory

		return features;
	}

	void reset() {
		micro_wakeword_features_reset(features_);
	}

private:
	MicroWakeWordFeatures *features_;
};

int main(int argc, char *argv[]) {
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0]
			  << " <model.tflite> [libtensorflowlite_c.so]\n";
		std::cerr << "Example: " << argv[0]
			  << " ../pymicro_wakeword/models/okay_nabu.tflite\n";
		return 1;
	}

	try {
		// Create feature generator
		MicroWakeWordFeaturesWrapper features;

		// Configure wake word detector
		MicroWakeWordConfig config = {};
		config.model_path = argv[1];
		config.libtensorflowlite_c = (argc > 2) ? argv[2] : nullptr;
		config.probability_cutoff = 0.97f;
		config.sliding_window_size = 5;

		// Create wake word detector
		MicroWakeWordWrapper mww(config);

		std::cout << "Wake word detector created successfully\n";
		std::cout << "Processing audio from stdin (16kHz, 16-bit, mono)...\n";

		// Process audio from stdin
		std::vector<uint8_t> audio_buffer(320);  // 10ms at 16kHz
		bool detected = false;
		size_t features_per_window = 40;

		while (std::cin.read(reinterpret_cast<char *>(audio_buffer.data()),
				     audio_buffer.size())) {
			// Generate features
			std::vector<float> feature_array = features.process_streaming(audio_buffer);

			// Process each feature window
			for (size_t i = 0; i < feature_array.size(); i += features_per_window) {
				if (i + features_per_window <= feature_array.size()) {
					std::vector<float> window_features(
						feature_array.begin() + i,
						feature_array.begin() + i + features_per_window);
					if (mww.process_streaming(window_features)) {
						std::cout << "Wake word detected!\n";
						detected = true;
						break;
					}
				}
			}

			if (detected) {
				break;
			}
		}

		if (!detected) {
			std::cout << "No wake word detected\n";
		}

	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}

	return 0;
}
