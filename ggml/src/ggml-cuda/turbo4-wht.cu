// Copyright (c) 2025 atomicmilkshake and contributors
// Copyright (c) 2026 The llama-wackMall-hybrid contributors
// SPDX-License-Identifier: MIT

#include "turbo4-k.cuh"
#include "turbo4-wht.cuh"

#include <cstdlib>
#include <cstring>

static bool ggml_cuda_turbo4_wht_shuffle_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("GGML_CUDA_TURBO4_WHT_SHUFFLE");
        if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
            return false;
        }
        if (std::strcmp(value, "1") != 0) {
            GGML_LOG_WARN("invalid GGML_CUDA_TURBO4_WHT_SHUFFLE='%s'; using 0\n", value);
            return false;
        }
        return true;
    }();
    return enabled;
}

static __global__ void k_turbo4_wht_f32(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int64_t n_groups,
        bool inverse) {
    const int64_t group = blockIdx.x;
    if (group >= n_groups) {
        return;
    }

    const int lane = threadIdx.x;
    __shared__ float values[QK_TURBO4_K];
    const int8_t sign_in = inverse ? turbo4_k_signs_second_cuda[lane] : turbo4_k_signs_first_cuda[lane];
    values[lane] = src[group*QK_TURBO4_K + lane] * (float) sign_in;
    __syncthreads();

#define TURBO4_K_WHT_STAGE(width)                                                                                 \
    if (lane % (2*(width)) < (width)) {                                                                           \
        const float a = values[lane];                                                                              \
        const float b = values[lane + (width)];                                                                    \
        values[lane] = a + b;                                                                                      \
        values[lane + (width)] = a - b;                                                                            \
    }                                                                                                              \
    __syncthreads()

    TURBO4_K_WHT_STAGE(1);
    TURBO4_K_WHT_STAGE(2);
    TURBO4_K_WHT_STAGE(4);
    TURBO4_K_WHT_STAGE(8);
    TURBO4_K_WHT_STAGE(16);
    TURBO4_K_WHT_STAGE(32);
    TURBO4_K_WHT_STAGE(64);
#undef TURBO4_K_WHT_STAGE

    const int8_t sign_out = inverse ? turbo4_k_signs_first_cuda[lane] : turbo4_k_signs_second_cuda[lane];
    dst[group*QK_TURBO4_K + lane] = values[lane] * 0.08838834764831845f * (float) sign_out;
}

static __global__ void k_turbo4_wht_f32_shuffle(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int64_t n_groups,
        bool inverse) {
    const int64_t group = blockIdx.x;
    if (group >= n_groups) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int8_t sign_in = inverse ? turbo4_k_signs_second_cuda[tid] : turbo4_k_signs_first_cuda[tid];
    float value = src[group*QK_TURBO4_K + tid] * (float) sign_in;

#pragma unroll
    for (int width = 1; width <= 16; width *= 2) {
        const float other = __shfl_xor_sync(0xffffffff, value, width);
        value = (lane & width) ? other - value : value + other;
    }

    // Widths 32 and 64 cross warp boundaries. Separate source and destination
    // arrays avoid a read/write race between independently scheduled warps.
    __shared__ float stage16[QK_TURBO4_K];
    __shared__ float stage32[QK_TURBO4_K];
    stage16[tid] = value;
    __syncthreads();

    const float other32 = stage16[tid ^ 32];
    value = (tid & 32) ? other32 - value : value + other32;
    stage32[tid] = value;
    __syncthreads();

    const float other64 = stage32[tid ^ 64];
    value = (tid & 64) ? other64 - value : value + other64;

    const int8_t sign_out = inverse ? turbo4_k_signs_first_cuda[tid] : turbo4_k_signs_second_cuda[tid];
    dst[group*QK_TURBO4_K + tid] = value * 0.08838834764831845f * (float) sign_out;
}

void ggml_cuda_turbo4_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src->ne[0] % QK_TURBO4_K == 0);
    const bool inverse = ggml_get_op_params_i32(dst, 0) != 0;

    const int64_t n_groups = ggml_nelements(src) / QK_TURBO4_K;
    if (n_groups > 0) {
        if (ggml_cuda_turbo4_wht_shuffle_enabled()) {
            k_turbo4_wht_f32_shuffle<<<(int) n_groups, QK_TURBO4_K, 0, ctx.stream()>>>(
                (const float *) src->data, (float *) dst->data, n_groups, inverse);
        } else {
            k_turbo4_wht_f32<<<(int) n_groups, QK_TURBO4_K, 0, ctx.stream()>>>(
                (const float *) src->data, (float *) dst->data, n_groups, inverse);
        }
    }
}
