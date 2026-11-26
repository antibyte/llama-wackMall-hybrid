#pragma once

#include <cstddef>
#include <cstdint>

// Full-context attention over KVFlash chunks.
// Q is contiguous f32 [head_dim, n_q, n_head].
// dst is contiguous f32 [head_dim * n_head, n_q].
// K/V rows are f16, stride bytes apart. Causal: query q_pos0+qi sees KV pos <= that.

struct kvflash_stream_chunk {
    const void * k;
    const void * v;
    int host;
    int n_tok;
    int64_t pos0;
    size_t k_stride;
    size_t v_stride;
};

struct kvflash_stream_args {
    const float * q;
    float * dst;
    int n_q;
    int n_head;
    int n_head_kv;
    int head_dim;
    float scale;
    int q_pos0;
    const kvflash_stream_chunk * chunks;
    int n_chunks;
};

bool kvflash_stream_attn(const kvflash_stream_args * args);
