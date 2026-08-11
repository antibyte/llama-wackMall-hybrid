// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

#include "ggml-quants.h"

#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

static const float turbo4_k_centroids[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f,
};

static const int8_t turbo4_k_signs_first[QK_TURBO4_K] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,
};

static const int8_t turbo4_k_signs_second[QK_TURBO4_K] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,
};

static void turbo4_k_transform(float values[QK_TURBO4_K], bool inverse) {
    const int8_t * first  = inverse ? turbo4_k_signs_second : turbo4_k_signs_first;
    const int8_t * second = inverse ? turbo4_k_signs_first : turbo4_k_signs_second;

    for (int i = 0; i < QK_TURBO4_K; ++i) {
        values[i] *= first[i];
    }
    for (int width = 1; width < QK_TURBO4_K; width *= 2) {
        for (int begin = 0; begin < QK_TURBO4_K; begin += 2 * width) {
            for (int i = 0; i < width; ++i) {
                const float a = values[begin + i];
                const float b = values[begin + i + width];
                values[begin + i] = a + b;
                values[begin + i + width] = a - b;
            }
        }
    }
    for (int i = 0; i < QK_TURBO4_K; ++i) {
        values[i] *= 0.08838834764831845f * second[i];
    }
}

static uint8_t turbo4_k_nearest(float value) {
    int best = 0;
    float best_error = fabsf(value - turbo4_k_centroids[0]);
    for (int i = 1; i < 16; ++i) {
        const float error = fabsf(value - turbo4_k_centroids[i]);
        if (error < best_error) {
            best = i;
            best_error = error;
        }
    }
    return (uint8_t) best;
}

void quantize_row_turbo4_k_ref(
        const float * GGML_RESTRICT x,
        block_turbo4_k * GGML_RESTRICT y,
        int64_t k) {
    assert(k % QK_TURBO4_K == 0);
    const int64_t blocks = k / QK_TURBO4_K;
    for (int64_t block = 0; block < blocks; ++block) {
        const float * source = x + block * QK_TURBO4_K;
        float rotated[QK_TURBO4_K];
        memcpy(rotated, source, sizeof(rotated));
        turbo4_k_transform(rotated, false);

        double source_norm_sq = 0.0;
        for (int i = 0; i < QK_TURBO4_K; ++i) {
            source_norm_sq += (double) source[i] * (double) source[i];
        }
        const float source_norm = (float) sqrt(source_norm_sq);
        const float inv_norm = source_norm > 1e-10f ? 1.0f / source_norm : 0.0f;
        double reconstructed_norm_sq = 0.0;
        memset(y[block].qs, 0, sizeof(y[block].qs));
        for (int i = 0; i < QK_TURBO4_K; ++i) {
            const uint8_t index = turbo4_k_nearest(rotated[i] * inv_norm);
            const float centroid = turbo4_k_centroids[index];
            reconstructed_norm_sq += (double) centroid * (double) centroid;
            y[block].qs[i / 2] |= (uint8_t) ((index & 0x0f) << (4 * (i % 2)));
        }
        const float reconstructed_norm = (float) sqrt(reconstructed_norm_sq);
        const float corrected_norm = reconstructed_norm > 1e-10f
            ? source_norm / reconstructed_norm
            : source_norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);
        y[block].reserved = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_turbo4_k(
        const block_turbo4_k * GGML_RESTRICT x,
        float * GGML_RESTRICT y,
        int64_t k) {
    assert(k % QK_TURBO4_K == 0);
    const int64_t blocks = k / QK_TURBO4_K;
    for (int64_t block = 0; block < blocks; ++block) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        float values[QK_TURBO4_K];
        for (int i = 0; i < QK_TURBO4_K; ++i) {
            const uint8_t index = (x[block].qs[i / 2] >> (4 * (i % 2))) & 0x0f;
            values[i] = turbo4_k_centroids[index] * norm;
        }
        turbo4_k_transform(values, true);
        memcpy(y + block * QK_TURBO4_K, values, sizeof(values));
    }
}

size_t quantize_turbo4_k(
        const float * GGML_RESTRICT src,
        void * GGML_RESTRICT dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4_K == 0);
    const size_t row_size = (size_t) (n_per_row / QK_TURBO4_K) * sizeof(block_turbo4_k);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo4_k_ref(
            src + row * n_per_row,
            (block_turbo4_k *) ((char *) dst + row * row_size),
            n_per_row);
    }
    return (size_t) nrows * row_size;
}
