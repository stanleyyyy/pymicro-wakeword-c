#ifndef WAV_READER_H_
#define WAV_READER_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Simple WAV file structure
typedef struct {
	uint32_t sample_rate;
	uint16_t bits_per_sample;
	uint16_t num_channels;
	size_t data_size;
	int16_t *data;
} WavFile;

// Read a WAV file (16kHz, 16-bit, mono expected)
// Returns 0 on success, non-zero on error
// Caller must free data with wav_file_free()
int wav_file_read(const char *filename, WavFile *wav);

// Free WAV file data
void wav_file_free(WavFile *wav);

#ifdef __cplusplus
}
#endif

#endif  // WAV_READER_H_
