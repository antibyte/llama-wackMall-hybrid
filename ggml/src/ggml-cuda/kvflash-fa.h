#pragma once

// Flash-attention entry for the KVFlash stream path (src/kvflash-stream.cu).
// Plain C types only: libllama includes this without the ggml-cuda internals.

#include <cstddef>

// All pointers are device memory.
// Q: f32 [head_dim, n_q, n_head], byte strides q_nb1/q_nb2.
// K/V: f16 token-major rows, row stride k_nb1/v_nb1 bytes, head stride head_dim.
// mask: f16 [n_kv, n_q], 0 or -inf. n_kv must be a multiple of 256.
// blocks == 1: normalized result in dst [head_dim, n_head, n_q].
// blocks >= 2: unnormalized parts [n_q*n_head][blocks][head_dim] and
// meta [n_q*n_head][blocks] as (KQ max, KQ sum) float pairs.
struct ggml_cuda_kvflash_fa_call {
    const float * q;
    size_t q_nb1;
    size_t q_nb2;
    const void * k;
    const void * v;
    size_t k_nb1;
    size_t v_nb1;
    const void * mask;
    float * dst;
    float * parts;
    float * meta;
    size_t cap_parts;    // floats
    size_t cap_meta;     // pairs
    int allow_single;    // 0: always export parts (needed to merge with other calls)
    int n_q;
    int n_kv;
    int n_head;
    int n_head_kv;
    int head_dim;
    float scale;
    int blocks;          // out
};

// The CUDA stream every ggml_cuda_kvflash_fa call on this device runs on (cudaStream_t).
extern "C" void * ggml_cuda_kvflash_stream(int device);

// 0 when the call ran; nonzero when no FA kernel fits this shape.
extern "C" int ggml_cuda_kvflash_fa(int device, ggml_cuda_kvflash_fa_call * call);
