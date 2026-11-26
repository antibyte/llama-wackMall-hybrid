#include "kvflash-stream.h"

#include "../ggml/src/ggml-cuda/kvflash-fa.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kWarp = 32;
constexpr int kBQ = 8;              // queries per block in the fallback kernel
constexpr int kPartBlocks = 2;      // FA parallel blocks the parts region holds at a full query block (4 or 8: no gain)
constexpr int kSegsPerLaunch = 128; // gather segments passed as one kernel parameter
constexpr size_t kAlign = 256;

int env_int(const char * name, int def) {
    const char * s = std::getenv(name);
    if (!s || !*s) {
        return def;
    }
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    return end != s && v > 0 && v < (1L << 30) ? (int) v : def;
}

int pad256(int x) {
    return (x + 255) & ~255;
}

size_t align_up(size_t x) {
    return (x + kAlign - 1) / kAlign * kAlign;
}

int g_window_override = 0; // self-test only
int g_qblock_override = 0;

int cfg_window() {
    if (g_window_override > 0) {
        return g_window_override;
    }
    static const int w = pad256(std::max(env_int("LLAMA_KVFLASH_STREAM_WINDOW", 16384), 1024));
    return w;
}

int cfg_qblock() {
    if (g_qblock_override > 0) {
        return g_qblock_override;
    }
    static const int q = std::max(env_int("LLAMA_KVFLASH_STREAM_QBLOCK", 512), 16);
    return q;
}

int cfg_qbatch() {
    static const int q = env_int("LLAMA_KVFLASH_STREAM_QBATCH", 0); // 0: the whole graph batch
    return q;
}

struct Layout {
    int window = 0;
    int q_batch = 0; // queries whose Q and state stay on the GPU across windows
    int q_block = 0; // queries per FA call (mask and parts scale with it)
    size_t k = 0;
    size_t v = 0;
    size_t q = 0;
    size_t acc = 0;
    size_t ml = 0;
    size_t mask = 0;
    size_t parts = 0;
    size_t meta = 0;
    size_t cap_parts = 0;
    size_t cap_meta = 0;
    size_t bytes = 0;
};

Layout plan(const kvflash_stream_shape & s) {
    Layout L;
    L.window = std::min(cfg_window(), pad256(std::max(s.n_ctx, 1)));
    L.q_batch = std::max(1, cfg_qbatch() > 0 ? std::min(cfg_qbatch(), s.n_q_max) : s.n_q_max);
    L.q_block = std::min(cfg_qblock(), L.q_batch);
    const size_t rows_batch = (size_t) L.q_batch * s.n_head;
    const size_t rows = (size_t) L.q_block * s.n_head;
    size_t off = 0;
    auto take = [&](size_t bytes) {
        const size_t at = off;
        off = align_up(off + bytes);
        return at;
    };
    L.k = take((size_t) L.window * s.k_row);
    L.v = take((size_t) L.window * s.v_row);
    L.q = take(rows_batch * s.head_dim * sizeof(float));
    L.acc = take(rows_batch * s.head_dim * sizeof(float));
    L.ml = take(rows_batch * sizeof(float2));
    L.mask = take((size_t) L.q_block * L.window * sizeof(half));
    L.cap_parts = (size_t) kPartBlocks * rows * s.head_dim;
    L.cap_meta = (size_t) kPartBlocks * rows;
    L.parts = take(L.cap_parts * sizeof(float));
    L.meta = take(L.cap_meta * sizeof(float2));
    L.bytes = off;
    return L;
}

bool shape_ok(const kvflash_stream_shape & s) {
    return s.n_head > 0 && s.n_head_kv > 0 && s.n_head % s.n_head_kv == 0 &&
        s.head_dim >= kWarp && s.head_dim <= 256 && s.head_dim % kWarp == 0 &&
        s.k_row >= (size_t) s.n_head_kv * s.head_dim * sizeof(half) && s.k_row % sizeof(half) == 0 &&
        s.v_row >= (size_t) s.n_head_kv * s.head_dim * sizeof(half) && s.v_row % sizeof(half) == 0 &&
        s.n_q_max > 0 && s.n_ctx > 0;
}

struct Stats {
    int calls = 0;
    int64_t host_bytes = 0;
    int64_t device_bytes = 0;
    int64_t passes = 0;
    int64_t fallback_passes = 0;
    double gpu_s = 0;
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;

    void reset() {
        calls = 0;
        host_bytes = 0;
        device_bytes = 0;
        passes = 0;
        fallback_passes = 0;
        gpu_s = 0;
    }

    ~Stats() {
        if (calls > 0) {
            std::fprintf(stderr,
                    "kvflash stream: calls=%d host=%.2f MiB device=%.2f MiB passes=%lld fallback=%lld gpu=%.3fs\n",
                    calls, host_bytes / (1024.0 * 1024.0), device_bytes / (1024.0 * 1024.0),
                    (long long) passes, (long long) fallback_passes, gpu_s);
        }
    }
};

Stats & stats() {
    static Stats s;
    return s;
}

bool check(cudaError_t err, const char * what) {
    if (err == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "kvflash stream: %s: %s\n", what, cudaGetErrorString(err));
    cudaGetLastError(); // ggml checks cudaGetLastError after its own launches
    return false;
}

// Rows [dst_row, dst_row + n_rows) of the window come from k/v (device or pinned host).
struct GatherSeg {
    const uint4 * k;
    const uint4 * v;
    int dst_row;
    int n_rows;
};

struct GatherBatch {
    int n;
    GatherSeg seg[kSegsPerLaunch];
};
static_assert(sizeof(GatherBatch) <= 4000, "kernel parameter space is 4 KiB");

__global__ void gather_rows(const GatherBatch batch, uint4 * __restrict__ kdst, uint4 * __restrict__ vdst,
        int k_u4, int v_u4) {
    const int s = (int) blockIdx.y;
    if (s >= batch.n) {
        return;
    }
    const GatherSeg seg = batch.seg[s];
    const int nk = seg.n_rows * k_u4;
    const int nv = seg.n_rows * v_u4;
    const int n = max(nk, nv);
    uint4 * kd = kdst + (size_t) seg.dst_row * k_u4;
    uint4 * vd = vdst + (size_t) seg.dst_row * v_u4;
    for (int i = (int) (blockIdx.x * blockDim.x + threadIdx.x); i < n; i += (int) (gridDim.x * blockDim.x)) {
        if (i < nk) {
            kd[i] = seg.k[i];
        }
        if (i < nv) {
            vd[i] = seg.v[i];
        }
    }
}

// mask[t][j]: key kv_pos0+j is visible to query q_pos0+t.
__global__ void causal_mask(half * __restrict__ mask, int n_kv, int n_valid, int n_rows, int kv_pos0, int q_pos0) {
    const int j = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int t = (int) blockIdx.y;
    if (j >= n_kv || t >= n_rows) {
        return;
    }
    const bool vis = j < n_valid && kv_pos0 + j <= q_pos0 + t;
    mask[(size_t) t * n_kv + j] = __float2half(vis ? 0.f : -INFINITY);
}

// Folds FA partials into the running (max, sum) and unnormalized accumulator of each row.
// sum == 0 marks an empty state or a part that saw no visible key.
__global__ void merge_parts(const float * __restrict__ parts, const float2 * __restrict__ meta, int blocks,
        float2 * __restrict__ ml, float * __restrict__ acc, int D) {
    const int r = (int) blockIdx.x;
    const float2 * mr = meta + (size_t) r * blocks;
    const float2 st = ml[r];
    float m_new = st.y > 0.f ? st.x : -FLT_MAX;
    for (int l = 0; l < blocks; ++l) {
        if (mr[l].y > 0.f) {
            m_new = fmaxf(m_new, mr[l].x);
        }
    }
    if (m_new == -FLT_MAX) {
        return;
    }
    const float a = st.y > 0.f ? expf(st.x - m_new) : 0.f;
    float * out = acc + (size_t) r * D;
    const float * pr = parts + (size_t) r * blocks * D;
    for (int d = (int) threadIdx.x; d < D; d += (int) blockDim.x) {
        float val = st.y > 0.f ? out[d] * a : 0.f;
        for (int l = 0; l < blocks; ++l) {
            if (mr[l].y > 0.f) {
                val += expf(mr[l].x - m_new) * pr[(size_t) l * D + d];
            }
        }
        out[d] = val;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float lsum = st.y * a;
        for (int l = 0; l < blocks; ++l) {
            if (mr[l].y > 0.f) {
                lsum += expf(mr[l].x - m_new) * mr[l].y;
            }
        }
        ml[r] = make_float2(m_new, lsum);
    }
}

__global__ void finalize_rows(const float2 * __restrict__ ml, float * __restrict__ acc, int D) {
    const int r = (int) blockIdx.x;
    const float l = ml[r].y;
    const float inv = l > 0.f ? 1.f / l : 0.f;
    float * out = acc + (size_t) r * D;
    for (int d = (int) threadIdx.x; d < D; d += (int) blockDim.x) {
        out[d] = l > 0.f ? out[d] * inv : 0.f;
    }
}

// Fallback for head shapes without an FA kernel: one warp per (query, head) over one window,
// accumulating into the same (max, sum, acc) state as merge_parts.
template <int D>
__global__ void attn_window(
        const float * __restrict__ q, int q_head_stride, const half * __restrict__ k, const half * __restrict__ v,
        int k_stride, int v_stride,
        float2 * __restrict__ ml, float * __restrict__ acc,
        int n_rows, int n_head, int n_head_kv, float scale, int q_pos0, int kv_pos0, int n_kv) {
    constexpr int BT = D <= 128 ? 32 : 16;
    constexpr int dims = D / kWarp;

    const int t0 = (int) blockIdx.x * kBQ;
    const int h = (int) blockIdx.y;
    const int warp = (int) threadIdx.x >> 5;
    const int lane = (int) threadIdx.x & 31;
    const int t = t0 + warp;
    const bool active = t < n_rows;
    const int kvh = h / (n_head / n_head_kv);
    const int qp = q_pos0 + t;
    const size_t r = (size_t) t * n_head + h;
    // Keys the last query of this block can see.
    const int n_lim = min(n_kv, q_pos0 + min(t0 + kBQ, n_rows) - kv_pos0);

    __shared__ float sk[BT * D];
    __shared__ float sv[BT * D];

    float qv[dims];
    float a[dims];
    float m = -FLT_MAX;
    float l = 0.f;
    if (active) {
        const float * qq = q + (size_t) D * ((size_t) t + (size_t) q_head_stride * h);
        const float2 st = ml[r];
#pragma unroll
        for (int d = 0; d < dims; ++d) {
            qv[d] = qq[lane + d * kWarp];
            a[d] = st.y > 0.f ? acc[r * D + lane + d * kWarp] : 0.f;
        }
        if (st.y > 0.f) {
            m = st.x;
            l = st.y;
        }
    }

    const int tid = (int) threadIdx.x;
    const int nthreads = kBQ * kWarp;
    for (int base = 0; base < n_lim; base += BT) {
        const int nt = min(BT, n_lim - base);
        for (int e = tid; e < nt * D; e += nthreads) {
            const int tt = e / D;
            const int d = e - tt * D;
            sk[e] = __half2float(k[(size_t) (base + tt) * k_stride + (size_t) kvh * D + d]);
            sv[e] = __half2float(v[(size_t) (base + tt) * v_stride + (size_t) kvh * D + d]);
        }
        __syncthreads();
        if (active) {
            for (int j = 0; j < nt; ++j) {
                if (kv_pos0 + base + j > qp) {
                    break;
                }
                float part = 0.f;
#pragma unroll
                for (int d = 0; d < dims; ++d) {
                    part += qv[d] * sk[j * D + lane + d * kWarp];
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    part += __shfl_xor_sync(0xffffffff, part, off);
                }
                const float score = part * scale;
                const float m_new = fmaxf(m, score);
                const float alpha = expf(m - m_new);
                const float p = expf(score - m_new);
                l = l * alpha + p;
#pragma unroll
                for (int d = 0; d < dims; ++d) {
                    a[d] = a[d] * alpha + p * sv[j * D + lane + d * kWarp];
                }
                m = m_new;
            }
        }
        __syncthreads();
    }

    if (!active || l <= 0.f) {
        return;
    }
#pragma unroll
    for (int d = 0; d < dims; ++d) {
        acc[r * D + lane + d * kWarp] = a[d];
    }
    if (lane == 0) {
        ml[r] = make_float2(m, l);
    }
}

template <int D>
void launch_window(const float * q, int qs, const half * k, const half * v, int k_stride, int v_stride,
        float2 * ml, float * acc, int n_rows, int n_head, int n_head_kv, float scale,
        int q_pos0, int kv_pos0, int n_kv, cudaStream_t stream) {
    const dim3 grid((n_rows + kBQ - 1) / kBQ, n_head);
    attn_window<D><<<grid, kBQ * kWarp, 0, stream>>>(
            q, qs, k, v, k_stride, v_stride, ml, acc, n_rows, n_head, n_head_kv, scale, q_pos0, kv_pos0, n_kv);
}

bool launch_window_switch(int D, const float * q, int qs, const half * k, const half * v, int k_stride, int v_stride,
        float2 * ml, float * acc, int n_rows, int n_head, int n_head_kv, float scale,
        int q_pos0, int kv_pos0, int n_kv, cudaStream_t stream) {
    switch (D) {
        case 32:  launch_window<32>(q, qs, k, v, k_stride, v_stride, ml, acc, n_rows, n_head, n_head_kv, scale, q_pos0, kv_pos0, n_kv, stream); break;
        case 64:  launch_window<64>(q, qs, k, v, k_stride, v_stride, ml, acc, n_rows, n_head, n_head_kv, scale, q_pos0, kv_pos0, n_kv, stream); break;
        case 96:  launch_window<96>(q, qs, k, v, k_stride, v_stride, ml, acc, n_rows, n_head, n_head_kv, scale, q_pos0, kv_pos0, n_kv, stream); break;
        case 128: launch_window<128>(q, qs, k, v, k_stride, v_stride, ml, acc, n_rows, n_head, n_head_kv, scale, q_pos0, kv_pos0, n_kv, stream); break;
        case 160: launch_window<160>(q, qs, k, v, k_stride, v_stride, ml, acc, n_rows, n_head, n_head_kv, scale, q_pos0, kv_pos0, n_kv, stream); break;
        case 192: launch_window<192>(q, qs, k, v, k_stride, v_stride, ml, acc, n_rows, n_head, n_head_kv, scale, q_pos0, kv_pos0, n_kv, stream); break;
        case 224: launch_window<224>(q, qs, k, v, k_stride, v_stride, ml, acc, n_rows, n_head, n_head_kv, scale, q_pos0, kv_pos0, n_kv, stream); break;
        case 256: launch_window<256>(q, qs, k, v, k_stride, v_stride, ml, acc, n_rows, n_head, n_head_kv, scale, q_pos0, kv_pos0, n_kv, stream); break;
        default:  return false;
    }
    return check(cudaGetLastError(), "window kernel");
}

int arena_device(const void * arena) {
    static const void * last = nullptr;
    static int dev = 0;
    if (arena != last) {
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, arena) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        last = arena;
        dev = attr.device;
    }
    return dev;
}

bool run_attn(const kvflash_stream_args * a) {
    if (!a || !a->q || !a->dst || a->n_q <= 0 || a->q_pos0 < 0 || a->n_chunks <= 0 || !a->chunks ||
        !shape_ok(a->shape) || !a->arena) {
        return false;
    }
    const kvflash_stream_shape & s = a->shape;
    const Layout L = plan(s);
    if (L.bytes > a->arena_bytes) {
        std::fprintf(stderr, "kvflash stream: arena %zu bytes, layer needs %zu\n", a->arena_bytes, L.bytes);
        return false;
    }
    const int dev = arena_device(a->arena);
    cudaStream_t stream = dev < 0 ? nullptr : (cudaStream_t) ggml_cuda_kvflash_stream(dev);
    if (!stream) {
        return false;
    }
    Stats & st = stats();
    if (!st.ev0 && (!check(cudaEventCreate(&st.ev0), "event") || !check(cudaEventCreate(&st.ev1), "event"))) {
        return false;
    }

    const int D = s.head_dim;
    const int n_q = a->n_q;
    const int n_pos = a->q_pos0 + n_q;
    int64_t covered = 0;
    for (int c = 0; c < a->n_chunks; ++c) {
        const kvflash_stream_chunk & ch = a->chunks[c];
        if (ch.pos0 != covered || ch.n_tok <= 0 || !ch.k || !ch.v ||
            ch.k_stride != s.k_row || ch.v_stride != s.v_row) {
            std::fprintf(stderr, "kvflash stream: chunk %d does not continue the positions\n", c);
            return false;
        }
        covered += ch.n_tok;
    }
    if (covered < n_pos) {
        std::fprintf(stderr, "kvflash stream: chunks cover %lld of %d positions\n", (long long) covered, n_pos);
        return false;
    }

    uint8_t * base = (uint8_t *) a->arena;
    uint8_t * kbuf = base + L.k;
    uint8_t * vbuf = base + L.v;
    float * qbuf = (float *) (base + L.q);
    float * accbuf = (float *) (base + L.acc);
    float2 * mlbuf = (float2 *) (base + L.ml);
    half * maskbuf = (half *) (base + L.mask);
    float * partsbuf = (float *) (base + L.parts);
    float * metabuf = (float *) (base + L.meta);

    const bool gather_rows_ok = s.k_row % 16 == 0 && s.v_row % 16 == 0;
    const int k_u4 = (int) (s.k_row / 16);
    const int v_u4 = (int) (s.v_row / 16);

    // Equal windows of at most L.window tokens; content does not depend on the query block.
    const int n_win0 = (n_pos + L.window - 1) / L.window;
    const int win = n_win0 == 1 ? pad256(n_pos) : pad256((n_pos + n_win0 - 1) / n_win0);
    const int n_win = (n_pos + win - 1) / win;

    auto pack = [&](int w) -> bool {
        const int w0 = w * win;
        const int w1 = std::min(w0 + win, n_pos);
        GatherBatch batch;
        batch.n = 0;
        int max_n = 0;
        auto flush = [&]() -> bool {
            if (batch.n == 0) {
                return true;
            }
            const int gx = std::min(std::max((max_n + 255) / 256, 1), 128);
            gather_rows<<<dim3(gx, batch.n), 256, 0, stream>>>(batch, (uint4 *) kbuf, (uint4 *) vbuf, k_u4, v_u4);
            batch.n = 0;
            max_n = 0;
            return check(cudaGetLastError(), "gather");
        };
        for (int c = 0; c < a->n_chunks; ++c) {
            const kvflash_stream_chunk & ch = a->chunks[c];
            const int r0 = (int) std::max<int64_t>(ch.pos0, w0);
            const int r1 = (int) std::min<int64_t>(ch.pos0 + ch.n_tok, w1);
            if (r0 >= r1) {
                continue;
            }
            const uint8_t * ks = (const uint8_t *) ch.k + (size_t) (r0 - ch.pos0) * s.k_row;
            const uint8_t * vs = (const uint8_t *) ch.v + (size_t) (r0 - ch.pos0) * s.v_row;
            const int dst_row = r0 - w0;
            const int n = r1 - r0;
            const int64_t bytes = (int64_t) n * (int64_t) (s.k_row + s.v_row);
            if (ch.host) {
                st.host_bytes += bytes;
            } else {
                st.device_bytes += bytes;
            }
            const bool direct = ch.host != 1 && gather_rows_ok &&
                (uintptr_t) ks % 16 == 0 && (uintptr_t) vs % 16 == 0;
            if (!direct) {
                if (!check(cudaMemcpyAsync(kbuf + (size_t) dst_row * s.k_row, ks, (size_t) n * s.k_row,
                                cudaMemcpyDefault, stream), "copy k") ||
                    !check(cudaMemcpyAsync(vbuf + (size_t) dst_row * s.v_row, vs, (size_t) n * s.v_row,
                                cudaMemcpyDefault, stream), "copy v")) {
                    return false;
                }
                continue;
            }
            if (batch.n > 0) {
                GatherSeg & p = batch.seg[batch.n - 1];
                if ((const uint8_t *) p.k + (size_t) p.n_rows * s.k_row == ks &&
                    (const uint8_t *) p.v + (size_t) p.n_rows * s.v_row == vs &&
                    p.dst_row + p.n_rows == dst_row) {
                    p.n_rows += n;
                    max_n = std::max(max_n, p.n_rows * std::max(k_u4, v_u4));
                    continue;
                }
            }
            if (batch.n == kSegsPerLaunch && !flush()) {
                return false;
            }
            batch.seg[batch.n++] = { (const uint4 *) ks, (const uint4 *) vs, dst_row, n };
            max_n = std::max(max_n, n * std::max(k_u4, v_u4));
        }
        if (!flush()) {
            return false;
        }
        // Padding rows must hold finite values: masked keys still enter Q*K.
        const int len = w1 - w0;
        const int len_pad = pad256(len);
        if (len_pad > len &&
            (!check(cudaMemsetAsync(kbuf + (size_t) len * s.k_row, 0, (size_t) (len_pad - len) * s.k_row, stream), "pad k") ||
             !check(cudaMemsetAsync(vbuf + (size_t) len * s.v_row, 0, (size_t) (len_pad - len) * s.v_row, stream), "pad v"))) {
            return false;
        }
        return true;
    };

    // One FA call: queries [b0, b0+n_rows) of the batch over one window.
    // normalized: the result went straight to acc; otherwise it was merged into the state.
    auto pass = [&](int nbatch, int b0, int n_rows, int qp0, int w0, int n_valid, bool allow_single,
            bool & normalized) -> bool {
        normalized = false;
        st.passes += 1;
        const int n_kv = pad256(n_valid);
        causal_mask<<<dim3((n_kv + 255) / 256, n_rows), 256, 0, stream>>>(maskbuf, n_kv, n_valid, n_rows, w0, qp0);
        if (!check(cudaGetLastError(), "mask")) {
            return false;
        }
        const float * qb = qbuf + (size_t) b0 * D;
        float * accb = accbuf + (size_t) b0 * s.n_head * D;
        float2 * mlb = mlbuf + (size_t) b0 * s.n_head;
        ggml_cuda_kvflash_fa_call c = {};
        c.q = qb;
        c.q_nb1 = (size_t) D * sizeof(float);
        c.q_nb2 = (size_t) D * nbatch * sizeof(float);
        c.k = kbuf;
        c.v = vbuf;
        c.k_nb1 = s.k_row;
        c.v_nb1 = s.v_row;
        c.mask = maskbuf;
        c.dst = accb;
        c.parts = partsbuf;
        c.meta = metabuf;
        c.cap_parts = L.cap_parts;
        c.cap_meta = L.cap_meta;
        c.allow_single = allow_single ? 1 : 0;
        c.n_q = n_rows;
        c.n_kv = n_kv;
        c.n_head = s.n_head;
        c.n_head_kv = s.n_head_kv;
        c.head_dim = D;
        c.scale = a->scale;
        const int rc = ggml_cuda_kvflash_fa(dev, &c);
        if (!check(cudaGetLastError(), "fa")) {
            return false;
        }
        const int rows = n_rows * s.n_head;
        if (rc == 0 && c.blocks == 1) {
            normalized = true;
            return true;
        }
        if (rc == 0 && c.blocks >= 2) {
            merge_parts<<<rows, std::min(D, 256), 0, stream>>>(partsbuf, (const float2 *) metabuf, c.blocks, mlb, accb, D);
            return check(cudaGetLastError(), "merge");
        }
        st.fallback_passes += 1;
        return launch_window_switch(D, qb, nbatch, (const half *) kbuf, (const half *) vbuf,
                (int) (s.k_row / sizeof(half)), (int) (s.v_row / sizeof(half)),
                mlb, accb, n_rows, s.n_head, s.n_head_kv, a->scale, qp0, w0, n_valid, stream);
    };

    if (!check(cudaEventRecord(st.ev0, stream), "record")) {
        return false;
    }
    // Q and state of a batch of queries stay on the GPU while the windows stream past,
    // so every window is packed once per batch. Graphs never exceed L.q_batch queries;
    // larger calls still work batch by batch.
    std::vector<char> normalized_blk;
    for (int s0 = 0; s0 < n_q; s0 += L.q_batch) {
        const int nbatch = std::min(L.q_batch, n_q - s0);
        const int qpos_batch = a->q_pos0 + s0;
        const size_t rows_batch = (size_t) nbatch * s.n_head;
        if (!check(cudaMemcpy2DAsync(qbuf, (size_t) nbatch * D * sizeof(float),
                        a->q + (size_t) s0 * D, (size_t) n_q * D * sizeof(float),
                        (size_t) nbatch * D * sizeof(float), (size_t) s.n_head, cudaMemcpyHostToDevice, stream), "q") ||
            !check(cudaMemsetAsync(mlbuf, 0, rows_batch * sizeof(float2), stream), "state")) {
            return false;
        }
        const int n_blk = (nbatch + L.q_block - 1) / L.q_block;
        normalized_blk.assign((size_t) n_blk, 0);
        const int need_batch = qpos_batch + nbatch;
        const int w_end = (need_batch + win - 1) / win;
        for (int w = 0; w < w_end; ++w) {
            if (!pack(w)) {
                return false;
            }
            for (int bi = 0; bi < n_blk; ++bi) {
                const int b0 = bi * L.q_block;
                const int nb = std::min(L.q_block, nbatch - b0);
                const int qp0 = qpos_batch + b0;
                const int need = qp0 + nb; // keys [0, need) are visible to some query of the block
                if (w * win >= need) {
                    continue;
                }
                bool normalized = false;
                if (!pass(nbatch, b0, nb, qp0, w * win, std::min(win, need - w * win), n_win == 1, normalized)) {
                    return false;
                }
                normalized_blk[(size_t) bi] = normalized ? 1 : 0;
            }
        }
        for (int bi = 0; bi < n_blk; ++bi) {
            if (normalized_blk[(size_t) bi]) {
                continue;
            }
            const int b0 = bi * L.q_block;
            const int nb = std::min(L.q_block, nbatch - b0);
            finalize_rows<<<(unsigned) (nb * s.n_head), std::min(D, 256), 0, stream>>>(
                    mlbuf + (size_t) b0 * s.n_head, accbuf + (size_t) b0 * s.n_head * D, D);
            if (!check(cudaGetLastError(), "finalize")) {
                return false;
            }
        }
        if (!check(cudaMemcpyAsync(a->dst + (size_t) s0 * s.n_head * D, accbuf, rows_batch * D * sizeof(float),
                        cudaMemcpyDeviceToHost, stream), "dst")) {
            return false;
        }
    }
    if (!check(cudaEventRecord(st.ev1, stream), "record") ||
        !check(cudaStreamSynchronize(stream), "sync") ||
        !check(cudaGetLastError(), "kernels")) {
        return false;
    }
    float ms = 0.f;
    cudaEventElapsedTime(&ms, st.ev0, st.ev1);
    st.gpu_s += (double) ms / 1e3;
    st.calls += 1;
    return true;
}

uint16_t f2h(float x) {
    const __half_raw r = __float2half(x);
    return r.x;
}

float h2f(uint16_t x) {
    __half_raw r;
    r.x = x;
    return __half2float(__half(r));
}

// CPU reference for the self-test: K/V f16 rows, positions 0..n_kv-1.
void reference(const std::vector<float> & q, const std::vector<uint16_t> & k, const std::vector<uint16_t> & v,
        int n_q, int n_head, int n_head_kv, int D, int q_pos0, int n_kv, float scale, std::vector<float> & out) {
    const size_t row = (size_t) n_head_kv * D;
    out.assign((size_t) n_q * n_head * D, 0.f);
    std::vector<float> sc((size_t) n_kv);
    for (int h = 0; h < n_head; ++h) {
        const int kvh = h / (n_head / n_head_kv);
        for (int t = 0; t < n_q; ++t) {
            const float * qq = q.data() + (size_t) D * ((size_t) t + (size_t) n_q * h);
            const int n_vis = std::min(n_kv, q_pos0 + t + 1);
            float m = -FLT_MAX;
            for (int j = 0; j < n_vis; ++j) {
                float dot = 0.f;
                for (int d = 0; d < D; ++d) {
                    dot += qq[d] * h2f(k[(size_t) j * row + (size_t) kvh * D + d]);
                }
                sc[(size_t) j] = dot * scale;
                m = std::max(m, sc[(size_t) j]);
            }
            float l = 0.f;
            float * o = out.data() + ((size_t) t * n_head + h) * D;
            for (int j = 0; j < n_vis; ++j) {
                const float p = std::exp(sc[(size_t) j] - m);
                l += p;
                for (int d = 0; d < D; ++d) {
                    o[d] += p * h2f(v[(size_t) j * row + (size_t) kvh * D + d]);
                }
            }
            for (int d = 0; d < D; ++d) {
                o[d] = l > 0.f ? o[d] / l : 0.f;
            }
        }
    }
}

// Two windows and two query blocks against the CPU, once with an FA head size and once
// with one that only the fallback kernel handles.
bool self_test_case(void * arena, size_t arena_bytes, int D, int n_q, int n_pos, int n_head, int n_head_kv) {
    const int chunk = 64;
    const size_t row = (size_t) n_head_kv * D;
    std::vector<uint16_t> k((size_t) n_pos * row);
    std::vector<uint16_t> v((size_t) n_pos * row);
    uint32_t seed = 12345u + (uint32_t) D;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return ((seed >> 8) & 0xffff) / 32768.0f - 1.0f;
    };
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = f2h(rnd());
        v[i] = f2h(rnd());
    }
    std::vector<float> q((size_t) D * n_q * n_head);
    for (float & x : q) {
        x = rnd();
    }
    std::vector<kvflash_stream_chunk> chunks;
    for (int p = 0; p < n_pos; p += chunk) {
        kvflash_stream_chunk ch;
        ch.k = k.data() + (size_t) p * row;
        ch.v = v.data() + (size_t) p * row;
        ch.host = 1;
        ch.n_tok = std::min(chunk, n_pos - p);
        ch.pos0 = p;
        ch.k_stride = row * sizeof(uint16_t);
        ch.v_stride = row * sizeof(uint16_t);
        chunks.push_back(ch);
    }
    kvflash_stream_args args = {};
    args.q = q.data();
    std::vector<float> dst((size_t) n_q * n_head * D, -1.f);
    args.dst = dst.data();
    args.n_q = n_q;
    args.q_pos0 = n_pos - n_q;
    args.scale = 1.f / std::sqrt((float) D);
    args.shape = { n_head, n_head_kv, D, row * sizeof(uint16_t), row * sizeof(uint16_t), n_q, n_pos };
    args.arena = arena;
    args.arena_bytes = arena_bytes;
    args.chunks = chunks.data();
    args.n_chunks = (int) chunks.size();
    if (plan(args.shape).bytes > arena_bytes) {
        std::fprintf(stderr, "kvflash stream: self-test D=%d skipped (arena too small)\n", D);
        return true;
    }
    if (!run_attn(&args)) {
        std::fprintf(stderr, "kvflash stream: self-test D=%d did not run\n", D);
        return false;
    }
    std::vector<float> ref;
    reference(q, k, v, n_q, n_head, n_head_kv, D, args.q_pos0, n_pos, args.scale, ref);
    float err = 0.f;
    for (size_t i = 0; i < ref.size(); ++i) {
        err = std::max(err, std::fabs(dst[i] - ref[i]));
    }
    if (!(err < 5e-3f)) {
        std::fprintf(stderr, "kvflash stream: self-test D=%d err %g\n", D, err);
        return false;
    }
    std::fprintf(stderr, "kvflash stream: self-test D=%d ok (err %g)\n", D, err);
    return true;
}

bool self_test(void * arena, size_t arena_bytes) {
    // Window 1024 over 1500 positions: two windows; 40 queries in blocks of 16: three blocks.
    // D 128 runs the FA kernels, D 32 has none and runs the fallback kernel.
    g_window_override = 1024;
    g_qblock_override = 16;
    const bool ok = self_test_case(arena, arena_bytes, 128, 40, 1500, 4, 2) &&
        self_test_case(arena, arena_bytes, 32, 40, 1500, 4, 2) &&
        self_test_case(arena, arena_bytes, 128, 1, 1500, 4, 2);
    g_window_override = 0;
    g_qblock_override = 0;
    return ok;
}

}  // namespace

size_t kvflash_stream_arena_bytes(const kvflash_stream_shape * shape) {
    if (!shape || !shape_ok(*shape)) {
        return 0;
    }
    return plan(*shape).bytes;
}

int kvflash_stream_window(const kvflash_stream_shape * shape) {
    return shape ? plan(*shape).window : 0;
}

bool kvflash_stream_attn(const kvflash_stream_args * args) {
    static int tested = 0;
    if (!tested && args && args->arena) {
        tested = self_test(args->arena, args->arena_bytes) ? 1 : -1;
        stats().reset();
    }
    if (tested < 0) {
        return false;
    }
    return run_attn(args);
}
