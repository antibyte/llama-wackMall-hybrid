#pragma once

#include <cstddef>
#include <cstdint>

// Full-context attention over KVFlash chunks.
// Q is contiguous f32 [head_dim, n_q, n_head].
// dst is contiguous f32 [head_dim * n_head, n_q].
// K/V rows are f16, k_row/v_row bytes apart. Causal: query q_pos0+qi sees KV pos <= that.
//
// All device memory comes from one arena sized at graph build time
// (kvflash_stream_arena_bytes). The context is attended in windows of
// LLAMA_KVFLASH_STREAM_WINDOW tokens (default 16384) and query blocks of
// LLAMA_KVFLASH_STREAM_QBLOCK (default 512); windows merge by their softmax
// statistics, so the arena does not grow with the context. Q and the running
// state of LLAMA_KVFLASH_STREAM_QBATCH queries (default 0: the whole graph
// batch) stay on the GPU, so each window is packed once per batch.

// host: 0 device pool, 1 pageable host memory, 2 pinned host memory (read by the GPU in place).
struct kvflash_stream_chunk {
    const void * k;
    const void * v;
    int host;
    int n_tok;
    int64_t pos0;
    size_t k_stride;
    size_t v_stride;
};

struct kvflash_stream_shape {
    int n_head;
    int n_head_kv;
    int head_dim;
    size_t k_row;   // bytes per token, as stored in the chunks
    size_t v_row;
    int n_q_max;    // largest query batch of the graphs
    int n_ctx;      // positions the cache can hold
};

struct kvflash_stream_args {
    const float * q;
    float * dst;
    int n_q;
    int q_pos0;
    float scale;
    kvflash_stream_shape shape;
    void * arena;           // device memory
    size_t arena_bytes;
    const kvflash_stream_chunk * chunks;
    int n_chunks;
};

// Device bytes one layer of this shape needs. The arena is the max over layers.
size_t kvflash_stream_arena_bytes(const kvflash_stream_shape * shape);

// The window that kvflash_stream_arena_bytes plans for.
int kvflash_stream_window(const kvflash_stream_shape * shape);

bool kvflash_stream_attn(const kvflash_stream_args * args);
