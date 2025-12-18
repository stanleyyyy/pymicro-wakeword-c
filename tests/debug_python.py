#!/usr/bin/env python3
"""Debug script to print intermediate values from Python implementation."""

import sys
from pathlib import Path
import wave
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymicro_wakeword import MicroWakeWord, Model, MicroWakeWordFeatures

def load_wav(model_name: str, number: int) -> bytes:
	_DIR = Path(__file__).parent
	wav_path = _DIR / model_name / f"{number}.wav"
	with wave.open(str(wav_path), "rb") as wav_file:
		assert wav_file.getframerate() == 16000
		assert wav_file.getsampwidth() == 2
		assert wav_file.getnchannels() == 1
		return wav_file.readframes(wav_file.getnframes())

def main():
	model_name = "okay_nabu"
	wav_number = 1

	print(f"Loading model: {model_name}")
	mww = MicroWakeWord.from_builtin(Model.OKAY_NABU)
	mww_features = MicroWakeWordFeatures()

	print(f"Model loaded:")
	print(f"  input_scale: {mww.input_scale}")
	print(f"  input_zero_point: {mww.input_zero_point}")
	print(f"  output_scale: {mww.output_scale}")
	print(f"  output_zero_point: {mww.output_zero_point}")
	print(f"  probability_cutoff: {mww.probability_cutoff}")
	print(f"  sliding_window_size: {mww.sliding_window_size}")

	print(f"\nLoading WAV file: {model_name}/{wav_number}.wav")
	audio_bytes = load_wav(model_name, wav_number)
	print(f"Audio size: {len(audio_bytes)} bytes ({len(audio_bytes) / 2} samples)")

	print(f"\nProcessing features...")
	feature_count = 0
	probabilities = []

	for features in mww_features.process_streaming(audio_bytes):
		feature_count += 1
		print(f"\nFeature window #{feature_count}:")
		print(f"  Shape: {features.shape}")
		# Format to match C output: [0.000000 0.000000 0.000000 0.000000 0.000000]
		first_5 = features.flatten()[:5]
		print(f"  First 5 values: [{first_5[0]:.6f} {first_5[1]:.6f} {first_5[2]:.6f} {first_5[3]:.6f} {first_5[4]:.6f}]")
		print(f"  Min: {features.min():.6f}, Max: {features.max():.6f}, Mean: {features.mean():.6f}")

		# Check buffer state before processing
		buffer_len = len(mww._features)
		print(f"  Buffer size before: {buffer_len}")

		result = mww.process_streaming(features)

		# Check buffer state after processing
		buffer_len_after = len(mww._features)
		print(f"  Buffer size after: {buffer_len_after}")

		# Get probability from sliding window
		if len(mww._probabilities) > 0:
			prob = mww._probabilities[-1]
			probabilities.append(prob)
			mean_prob = sum(mww._probabilities) / len(mww._probabilities)
			print(f"  Latest probability: {prob:.6f}")
			print(f"  Mean probability: {mean_prob:.6f} (window size: {len(mww._probabilities)})")

		# Always print detection for comparison
		print(f"  Detection: {result}")

		if result:
			print(f"\n*** WAKE WORD DETECTED at feature window #{feature_count} ***")
			break

	print(f"\nSummary:")
	print(f"  Total feature windows processed: {feature_count}")
	print(f"  Total probabilities: {len(probabilities)}")
	if probabilities:
		print(f"  First 5 probabilities: {probabilities[:5]}")
		print(f"  Last 5 probabilities: {probabilities[-5:]}")
		print(f"  Max probability: {max(probabilities):.6f}")
		print(f"  Min probability: {min(probabilities):.6f}")

if __name__ == "__main__":
	main()
