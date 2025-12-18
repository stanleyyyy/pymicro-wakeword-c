// tests/wav_reader.c
// Simple WAV file reader for 16kHz, 16-bit, mono PCM files

#include "wav_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// WAV file header structure
#pragma pack(push, 1)
typedef struct {
	char riff[4];           // "RIFF"
	uint32_t file_size;     // File size - 8
	char wave[4];            // "WAVE"
	char fmt[4];             // "fmt "
	uint32_t fmt_size;       // Format chunk size (usually 16)
	uint16_t audio_format;   // 1 = PCM
	uint16_t num_channels;   // Number of channels
	uint32_t sample_rate;    // Sample rate
	uint32_t byte_rate;      // Byte rate
	uint16_t block_align;    // Block align
	uint16_t bits_per_sample; // Bits per sample
	char data[4];            // "data"
	uint32_t data_size;      // Data chunk size
} WavHeader;
#pragma pack(pop)

int wav_file_read(const char *filename, WavFile *wav) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		return -1;
	}

	// Read RIFF header
	char riff[4];
	uint32_t file_size;
	char wave[4];
	if (fread(riff, 4, 1, file) != 1 || fread(&file_size, 4, 1, file) != 1 ||
	    fread(wave, 4, 1, file) != 1) {
		fclose(file);
		return -2;
	}

	// Verify RIFF header
	if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
		fclose(file);
		return -3;
	}

	// Read fmt chunk
	char fmt_id[4];
	uint32_t fmt_size;
	uint16_t audio_format;
	uint16_t num_channels;
	uint32_t sample_rate;
	uint32_t byte_rate;
	uint16_t block_align;
	uint16_t bits_per_sample;

	if (fread(fmt_id, 4, 1, file) != 1 || fread(&fmt_size, 4, 1, file) != 1) {
		fclose(file);
		return -4;
	}

	if (memcmp(fmt_id, "fmt ", 4) != 0) {
		fclose(file);
		return -4;
	}

	if (fread(&audio_format, 2, 1, file) != 1 ||
	    fread(&num_channels, 2, 1, file) != 1 ||
	    fread(&sample_rate, 4, 1, file) != 1 ||
	    fread(&byte_rate, 4, 1, file) != 1 ||
	    fread(&block_align, 2, 1, file) != 1 ||
	    fread(&bits_per_sample, 2, 1, file) != 1) {
		fclose(file);
		return -4;
	}

	// Skip any extra bytes in fmt chunk (if fmt_size > 16)
	if (fmt_size > 16) {
		fseek(file, fmt_size - 16, SEEK_CUR);
	}

	// Verify expected format: 16kHz, 16-bit, mono
	if (sample_rate != 16000 || bits_per_sample != 16 || num_channels != 1) {
		fclose(file);
		return -8;
	}

	// Find data chunk (skip any other chunks like LIST/INFO)
	char chunk_id[4];
	uint32_t chunk_size;
	uint32_t data_size = 0;

	while (fread(chunk_id, 4, 1, file) == 1) {
		if (fread(&chunk_size, 4, 1, file) != 1) {
			fclose(file);
			return -5;
		}

		if (memcmp(chunk_id, "data", 4) == 0) {
			data_size = chunk_size;
			break;
		}

		// Skip this chunk (pad to even byte boundary)
		if (chunk_size % 2 != 0) {
			chunk_size++;
		}
		fseek(file, chunk_size, SEEK_CUR);
	}

	if (data_size == 0) {
		fclose(file);
		return -7;
	}

	// Allocate memory for audio data
	size_t num_samples = data_size / 2; // 16-bit = 2 bytes per sample
	int16_t *data = (int16_t *)malloc(data_size);
	if (!data) {
		fclose(file);
		return -9;
	}

	// Read audio data
	if (fread(data, 2, num_samples, file) != num_samples) {
		free(data);
		fclose(file);
		return -10;
	}

	fclose(file);

	// Fill WavFile structure
	wav->sample_rate = sample_rate;
	wav->bits_per_sample = bits_per_sample;
	wav->num_channels = num_channels;
	wav->data_size = data_size;
	wav->data = data;

	return 0;
}

void wav_file_free(WavFile *wav) {
	if (wav && wav->data) {
		free(wav->data);
		wav->data = NULL;
		wav->data_size = 0;
	}
}
