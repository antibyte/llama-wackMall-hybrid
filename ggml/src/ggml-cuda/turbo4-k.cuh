// Copyright (c) 2025 atomicmilkshake and contributors
// Copyright (c) 2026 The llama-wackMall-hybrid contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "common.cuh"

static __constant__ float turbo4_k_centroids_cuda[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f,
};

static __constant__ float turbo4_k_midpoints_cuda[15] = {
    -0.1455605f, -0.1033610f, -0.0791415f, -0.0600090f,
    -0.0434295f, -0.0282930f, -0.0139635f,  0.0f,
     0.0139635f,  0.0282930f,  0.0434295f,  0.0600090f,
     0.0791415f,  0.1033610f,  0.1455605f,
};

// Least-squares signed-int8 approximation of the 16 centroids. It is used
// only by the separately enabled DP4A experiment.
static __constant__ int8_t turbo4_k_centroid_i8_cuda[16] = {
    -127, -86, -65, -50, -37, -26, -15, -5,
       5,  15,  26,  37,  50,  65,  86, 127,
};

static constexpr float turbo4_k_centroid_i8_scale_cuda = 0.0013702924565985558f;

static __device__ __forceinline__ void dequantize_turbo4_k_rotated(
        const void * vx, const int64_t ib, const int iqs, float2 & value) {
    const block_turbo4_k * blocks = (const block_turbo4_k *) vx;
    const float norm = __half2float(blocks[ib].norm);
    const uint8_t packed = blocks[ib].qs[iqs/2];
    value.x = norm*turbo4_k_centroids_cuda[ packed       & 0x0f];
    value.y = norm*turbo4_k_centroids_cuda[(packed >> 4) & 0x0f];
}

static __constant__ int8_t turbo4_k_signs_first_cuda[QK_TURBO4_K] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,
};

static __constant__ int8_t turbo4_k_signs_second_cuda[QK_TURBO4_K] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,
};

static __device__ __forceinline__ uint8_t turbo4_k_nearest_cuda(float value) {
    uint8_t index = 0;
#pragma unroll
    for (int i = 0; i < 15; ++i) {
        index += value > turbo4_k_midpoints_cuda[i];
    }
    return index;
}
