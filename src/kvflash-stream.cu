#include "kvflash-stream.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

struct DevSeg {
    const uint16_t * k;
    const uint16_t * v;
    int n_tok;
    int pos0;
    int k_stride;
    int v_stride;
};

extern "C" int ggml_cuda_kvflash_fa(
        const float * q, const DevSeg * segs, int n_segs, float * dst,
        int n_q, int n_head, int n_head_kv, int head_dim,
        float scale, int q_pos0, int pos_span);

namespace {

constexpr int kWarp = 32;
constexpr int kMaxDimsPerLane = 8;
constexpr int kBQ = 8;

struct DeviceBuf {
    float * q = nullptr;
    float * dst = nullptr;
    uint16_t * stage = nullptr;
    DevSeg * segs = nullptr;
    float * partial_m = nullptr;
    float * partial_l = nullptr;
    float * partial_acc = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    size_t q_cap = 0;
    size_t dst_cap = 0;
    size_t stage_cap = 0;
    size_t seg_cap = 0;
    size_t partial_cap = 0;
    size_t acc_cap = 0;
    double copy_s = 0;
    double kern_s = 0;
    int64_t host_bytes = 0;
    int64_t device_bytes = 0;
    int calls = 0;

    ~DeviceBuf() {
        if (calls > 0) {
            std::fprintf(stderr,
                    "kvflash stream: calls=%d host=%.2f MiB device=%.2f MiB copy=%.3fs kern=%.3fs\n",
                    calls,
                    host_bytes / (1024.0 * 1024.0),
                    device_bytes / (1024.0 * 1024.0),
                    copy_s, kern_s);
        }
    }
};

DeviceBuf & buf() {
    static DeviceBuf b;
    return b;
}

bool grow(void ** ptr, size_t & cap, size_t need) {
    if (cap >= need) {
        return true;
    }
    void * next = nullptr;
    if (cudaMalloc(&next, need) != cudaSuccess) {
        return false;
    }
    cudaFree(*ptr);
    *ptr = next;
    cap = need;
    return true;
}

bool check(cudaError_t err, const char * what) {
    if (err == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "kvflash stream: %s: %s\n", what, cudaGetErrorString(err));
    return false;
}

__device__ void softmax_step(float & running_m, float & running_l, float * acc, int dims,
        int lane, float score, const float * sv, int t, int D) {
    const float m_new = fmaxf(running_m, score);
    const float alpha = __expf(running_m - m_new);
    const float p = __expf(score - m_new);
    running_l = running_l * alpha + p;
    for (int d = 0; d < dims; ++d) {
        const int dim = lane + d * kWarp;
        acc[d] = acc[d] * alpha + p * sv[t * D + dim];
    }
    running_m = m_new;
}

// One warp per query. Four KV tokens are dotted together so the fmas stay independent.
// blockIdx.z splits the position range; partial_m == nullptr writes the final output.
template <int D>
__global__ void attn_tiled(
        const float * __restrict__ q,
        const DevSeg * __restrict__ segs,
        float * __restrict__ dst,
        float * __restrict__ partial_m,
        float * __restrict__ partial_l,
        float * __restrict__ partial_acc,
        int n_q, int n_segs, int n_head, int n_head_kv,
        float scale, int q_pos0, int pos_span) {
    constexpr int BT = D <= 128 ? 32 : 16;
    constexpr int dims = D / kWarp;

    const int qi0 = (int) blockIdx.x * kBQ;
    const int hi = (int) blockIdx.y;
    const int split = (int) blockIdx.z;
    const int n_splits = (int) gridDim.z;
    if (qi0 >= n_q || hi >= n_head) {
        return;
    }
    const int pos_lo = (int) (((int64_t) split * pos_span) / n_splits);
    const int pos_hi = (int) (((int64_t) (split + 1) * pos_span) / n_splits);

    const int warp = (int) threadIdx.x >> 5;
    const int lane = (int) threadIdx.x & 31;
    const int qi = qi0 + warp;
    const bool active = qi < n_q;
    const int kvh = hi / (n_head / n_head_kv);
    const int qp = q_pos0 + qi;
    const float * qq = active ? q + (size_t) D * ((size_t) qi + (size_t) n_q * hi) : nullptr;

    __shared__ float sk[BT * D];
    __shared__ float sv[BT * D];

    float acc[dims];
#pragma unroll
    for (int d = 0; d < dims; ++d) {
        acc[d] = 0.f;
    }
    float running_m = -INFINITY;
    float running_l = 0.f;

    const int tid = (int) threadIdx.x;
    const int nthreads = kBQ * kWarp;

    for (int s = 0; s < n_segs; ++s) {
        const DevSeg seg = segs[s];
        if (seg.pos0 + seg.n_tok <= pos_lo || seg.pos0 >= pos_hi) {
            continue;
        }
        for (int base = 0; base < seg.n_tok; base += BT) {
            const int nt = min(BT, seg.n_tok - base);
            const int pos_base = seg.pos0 + base;
            if (pos_base + nt <= pos_lo || pos_base >= pos_hi) {
                continue;
            }
            for (int e = tid; e < nt * D; e += nthreads) {
                const int t = e / D;
                const int d = e - t * D;
                const size_t row = (size_t) (base + t);
                sk[e] = __half2float(__ushort_as_half(
                        seg.k[row * seg.k_stride + (size_t) kvh * D + d]));
                sv[e] = __half2float(__ushort_as_half(
                        seg.v[row * seg.v_stride + (size_t) kvh * D + d]));
            }
            __syncthreads();

            if (active && qp >= pos_lo) {
                int t = 0;
                for (; t + 4 <= nt; t += 4) {
                    const int p0 = pos_base + t;
                    if (p0 >= pos_hi || p0 > qp) {
                        break;
                    }
                    float part[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int d = 0; d < dims; ++d) {
                        const int dim = lane + d * kWarp;
                        const float qv = qq[dim];
                        part[0] += qv * sk[(t + 0) * D + dim];
                        part[1] += qv * sk[(t + 1) * D + dim];
                        part[2] += qv * sk[(t + 2) * D + dim];
                        part[3] += qv * sk[(t + 3) * D + dim];
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
#pragma unroll
                        for (int off = 16; off > 0; off >>= 1) {
                            part[i] += __shfl_xor_sync(0xffffffff, part[i], off);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int kp = p0 + i;
                        if (kp <= qp && kp >= pos_lo && kp < pos_hi) {
                            softmax_step(running_m, running_l, acc, dims, lane, part[i] * scale, sv, t + i, D);
                        }
                    }
                }
                for (; t < nt; ++t) {
                    const int kp = pos_base + t;
                    if (kp > qp || kp >= pos_hi) {
                        break;
                    }
                    if (kp < pos_lo) {
                        continue;
                    }
                    float partial = 0.f;
#pragma unroll
                    for (int d = 0; d < dims; ++d) {
                        const int dim = lane + d * kWarp;
                        partial += qq[dim] * sk[t * D + dim];
                    }
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        partial += __shfl_xor_sync(0xffffffff, partial, off);
                    }
                    softmax_step(running_m, running_l, acc, dims, lane, partial * scale, sv, t, D);
                }
            }
            __syncthreads();
        }
    }

    if (!active) {
        return;
    }
    if (partial_m == nullptr) {
        float * out = dst + ((size_t) qi * n_head + hi) * D;
        if (running_l <= 0.f) {
            for (int d = 0; d < dims; ++d) {
                out[lane + d * kWarp] = 0.f;
            }
            return;
        }
        const float inv = 1.f / running_l;
#pragma unroll
        for (int d = 0; d < dims; ++d) {
            out[lane + d * kWarp] = acc[d] * inv;
        }
        return;
    }
    const int pair = qi * n_head + hi;
    const size_t slot = ((size_t) split * n_q * n_head + pair);
    if (lane == 0) {
        partial_m[slot] = running_m;
        partial_l[slot] = running_l;
    }
    float * out = partial_acc + slot * D;
#pragma unroll
    for (int d = 0; d < dims; ++d) {
        out[lane + d * kWarp] = acc[d];
    }
}

template <int D>
__global__ void attn_combine(
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        const float * __restrict__ partial_acc,
        float * __restrict__ dst,
        int n_q, int n_head, int n_splits) {
    constexpr int dims = D / kWarp;
    const int pair = (int) blockIdx.x;
    const int n_pairs = n_q * n_head;
    if (pair >= n_pairs) {
        return;
    }
    const int lane = (int) threadIdx.x;
    float acc[dims];
#pragma unroll
    for (int d = 0; d < dims; ++d) {
        acc[d] = 0.f;
    }
    float running_m = -INFINITY;
    float running_l = 0.f;
    for (int s = 0; s < n_splits; ++s) {
        const size_t slot = ((size_t) s * n_pairs + pair);
        const float split_l = partial_l[slot];
        if (split_l <= 0.f) {
            continue;
        }
        const float split_m = partial_m[slot];
        const float m_new = fmaxf(running_m, split_m);
        const float alpha = __expf(running_m - m_new);
        const float beta = __expf(split_m - m_new);
        const float * src = partial_acc + slot * D;
        running_l = running_l * alpha + split_l * beta;
#pragma unroll
        for (int d = 0; d < dims; ++d) {
            const int dim = lane + d * kWarp;
            acc[d] = acc[d] * alpha + src[dim] * beta;
        }
        running_m = m_new;
    }
    const int qi = pair / n_head;
    const int hi = pair - qi * n_head;
    float * out = dst + ((size_t) qi * n_head + hi) * D;
    if (running_l <= 0.f) {
        for (int d = 0; d < dims; ++d) {
            out[lane + d * kWarp] = 0.f;
        }
        return;
    }
    const float inv = 1.f / running_l;
#pragma unroll
    for (int d = 0; d < dims; ++d) {
        out[lane + d * kWarp] = acc[d] * inv;
    }
}

template <int D>
bool launch_attn(
        const float * q, const DevSeg * segs, float * dst,
        float * partial_m, float * partial_l, float * partial_acc,
        int n_q, int n_segs, int n_head, int n_head_kv, int n_splits, int pos_span,
        float scale, int q_pos0, cudaStream_t stream) {
    const dim3 grid((n_q + kBQ - 1) / kBQ, n_head, n_splits);
    attn_tiled<D><<<grid, kBQ * kWarp, 0, stream>>>(
            q, segs, dst, partial_m, partial_l, partial_acc,
            n_q, n_segs, n_head, n_head_kv, scale, q_pos0, pos_span);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    if (n_splits > 1) {
        attn_combine<D><<<n_q * n_head, kWarp, 0, stream>>>(
                partial_m, partial_l, partial_acc, dst, n_q, n_head, n_splits);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool launch_switch(
        int head_dim,
        const float * q, const DevSeg * segs, float * dst,
        float * partial_m, float * partial_l, float * partial_acc,
        int n_q, int n_segs, int n_head, int n_head_kv, int n_splits, int pos_span,
        float scale, int q_pos0, cudaStream_t stream) {
    switch (head_dim) {
        case 32:  return launch_attn<32>(q, segs, dst, partial_m, partial_l, partial_acc, n_q, n_segs, n_head, n_head_kv, n_splits, pos_span, scale, q_pos0, stream);
        case 64:  return launch_attn<64>(q, segs, dst, partial_m, partial_l, partial_acc, n_q, n_segs, n_head, n_head_kv, n_splits, pos_span, scale, q_pos0, stream);
        case 96:  return launch_attn<96>(q, segs, dst, partial_m, partial_l, partial_acc, n_q, n_segs, n_head, n_head_kv, n_splits, pos_span, scale, q_pos0, stream);
        case 128: return launch_attn<128>(q, segs, dst, partial_m, partial_l, partial_acc, n_q, n_segs, n_head, n_head_kv, n_splits, pos_span, scale, q_pos0, stream);
        case 160: return launch_attn<160>(q, segs, dst, partial_m, partial_l, partial_acc, n_q, n_segs, n_head, n_head_kv, n_splits, pos_span, scale, q_pos0, stream);
        case 192: return launch_attn<192>(q, segs, dst, partial_m, partial_l, partial_acc, n_q, n_segs, n_head, n_head_kv, n_splits, pos_span, scale, q_pos0, stream);
        case 256: return launch_attn<256>(q, segs, dst, partial_m, partial_l, partial_acc, n_q, n_segs, n_head, n_head_kv, n_splits, pos_span, scale, q_pos0, stream);
        default:  return false;
    }
}

bool run_attn(const kvflash_stream_args * args) {
    if (!args || args->n_q <= 0 || args->n_chunks <= 0 || args->n_head <= 0 ||
        args->n_head_kv <= 0 || args->n_head % args->n_head_kv != 0 ||
        args->head_dim < kWarp || args->head_dim > kWarp * kMaxDimsPerLane ||
        args->head_dim % kWarp != 0 || !args->q || !args->dst) {
        return false;
    }
    const int n_q = args->n_q;
    const int n_head = args->n_head;
    const int head_dim = args->head_dim;
    const size_t q_bytes = (size_t) head_dim * n_q * n_head * sizeof(float);

    const int k_stride0 = (int) (args->chunks[0].k_stride / sizeof(uint16_t));
    const int v_stride0 = (int) (args->chunks[0].v_stride / sizeof(uint16_t));
    if ((size_t) k_stride0 * sizeof(uint16_t) != args->chunks[0].k_stride ||
        (size_t) v_stride0 * sizeof(uint16_t) != args->chunks[0].v_stride ||
        k_stride0 < args->n_head_kv * head_dim || v_stride0 < args->n_head_kv * head_dim) {
        return false;
    }

    DeviceBuf & b = buf();
    if (!b.stream && !check(cudaStreamCreate(&b.stream), "stream")) {
        return false;
    }
    if (!b.ev0 && (!check(cudaEventCreate(&b.ev0), "event") || !check(cudaEventCreate(&b.ev1), "event"))) {
        return false;
    }
    if (!grow((void **) &b.q, b.q_cap, q_bytes) ||
        !grow((void **) &b.dst, b.dst_cap, q_bytes) ||
        !grow((void **) &b.segs, b.seg_cap, (size_t) args->n_chunks * sizeof(DevSeg))) {
        return false;
    }

    size_t host_need = 0;
    for (int c = 0; c < args->n_chunks; ++c) {
        const kvflash_stream_chunk & ch = args->chunks[c];
        if (ch.n_tok <= 0 || !ch.k || !ch.v ||
            ch.k_stride != args->chunks[0].k_stride || ch.v_stride != args->chunks[0].v_stride) {
            return false;
        }
        if (ch.host) {
            host_need += (size_t) ch.n_tok * (ch.k_stride + ch.v_stride);
        }
    }
    if (host_need > 0 && !grow((void **) &b.stage, b.stage_cap, host_need)) {
        return false;
    }

    if (!check(cudaEventRecord(b.ev0, b.stream), "record") ||
        !check(cudaMemcpyAsync(b.q, args->q, q_bytes, cudaMemcpyHostToDevice, b.stream), "q")) {
        return false;
    }

    std::vector<DevSeg> segs((size_t) args->n_chunks);
    size_t stage_off = 0;
    for (int c = 0; c < args->n_chunks; ++c) {
        const kvflash_stream_chunk & ch = args->chunks[c];
        DevSeg & seg = segs[(size_t) c];
        seg.n_tok = ch.n_tok;
        seg.pos0 = (int) ch.pos0;
        seg.k_stride = k_stride0;
        seg.v_stride = v_stride0;
        const size_t kb = (size_t) ch.n_tok * ch.k_stride;
        const size_t vb = (size_t) ch.n_tok * ch.v_stride;
        if (ch.host) {
            uint16_t * dstk = (uint16_t *) ((uint8_t *) b.stage + stage_off);
            if (!check(cudaMemcpyAsync(dstk, ch.k, kb, cudaMemcpyHostToDevice, b.stream), "k") ||
                !check(cudaMemcpyAsync(dstk + kb / sizeof(uint16_t), ch.v, vb, cudaMemcpyHostToDevice, b.stream), "v")) {
                return false;
            }
            seg.k = dstk;
            seg.v = dstk + kb / sizeof(uint16_t);
            stage_off += kb + vb;
            b.host_bytes += (int64_t) (kb + vb);
        } else {
            seg.k = (const uint16_t *) ch.k;
            seg.v = (const uint16_t *) ch.v;
            b.device_bytes += (int64_t) (kb + vb);
        }
    }
    if (!check(cudaMemcpyAsync(b.segs, segs.data(), segs.size() * sizeof(DevSeg), cudaMemcpyHostToDevice, b.stream), "segs") ||
        !check(cudaEventRecord(b.ev1, b.stream), "record") ||
        !check(cudaEventSynchronize(b.ev1), "sync")) {
        return false;
    }
    float copy_ms = 0.f;
    cudaEventElapsedTime(&copy_ms, b.ev0, b.ev1);
    b.copy_s += (double) copy_ms / 1e3;

    int pos_span = 1;
    for (const DevSeg & seg : segs) {
        pos_span = std::max(pos_span, seg.pos0 + seg.n_tok);
    }
    // Decode has one query per head, so split the KV to fill the GPU.
    const int n_splits = n_q <= 4 && pos_span >= 256 ? 32 : 1;
    const size_t n_slots = (size_t) n_splits * n_q * n_head;
    float * pm = nullptr;
    float * pl = nullptr;
    float * pa = nullptr;
    if (n_splits > 1) {
        static size_t partial_acc_cap = 0;
        if (!grow((void **) &b.partial_m, b.partial_cap, n_slots * sizeof(float)) ||
            !grow((void **) &b.partial_l, b.acc_cap, n_slots * sizeof(float)) ||
            !grow((void **) &b.partial_acc, partial_acc_cap, n_slots * (size_t) head_dim * sizeof(float))) {
            return false;
        }
        pm = b.partial_m;
        pl = b.partial_l;
        pa = b.partial_acc;
    }

    if (!check(cudaEventRecord(b.ev0, b.stream), "record")) {
        return false;
    }
    const bool used_fa = ggml_cuda_kvflash_fa(
            b.q, segs.data(), (int) segs.size(), b.dst,
            n_q, n_head, args->n_head_kv, head_dim,
            args->scale, args->q_pos0, pos_span) == 0;
    if (!used_fa && !launch_switch(head_dim, b.q, b.segs, b.dst, pm, pl, pa,
                n_q, args->n_chunks, n_head, args->n_head_kv, n_splits, pos_span,
                args->scale, args->q_pos0, b.stream)) {
        std::fprintf(stderr, "kvflash stream: launch failed head_dim=%d\n", head_dim);
        return false;
    }
    if (!check(cudaMemcpyAsync(args->dst, b.dst, q_bytes, cudaMemcpyDeviceToHost, b.stream), "dst") ||
        !check(cudaEventRecord(b.ev1, b.stream), "record") ||
        !check(cudaEventSynchronize(b.ev1), "sync")) {
        return false;
    }
    float kern_ms = 0.f;
    cudaEventElapsedTime(&kern_ms, b.ev0, b.ev1);
    b.kern_s += (double) kern_ms / 1e3;
    b.calls += 1;
    return true;
}

void f16_row(uint16_t * dst, float value, int n) {
    const uint16_t h = value == 0.f ? 0 : value == 1.f ? 0x3c00 : value == 2.f ? 0x4000 : value == 3.f ? 0x4200 : 0x3c00;
    for (int i = 0; i < n; ++i) {
        dst[i] = h;
    }
}

bool self_test() {
    constexpr int head = 32;
    constexpr int n_tok = 2;
    std::vector<float> q((size_t) head, 1.f);
    std::vector<uint16_t> k((size_t) n_tok * head);
    std::vector<uint16_t> v((size_t) n_tok * head);
    f16_row(k.data(), 1.f, head);
    f16_row(k.data() + head, 0.f, head);
    f16_row(v.data(), 1.f, head);
    f16_row(v.data() + head, 2.f, head);
    kvflash_stream_chunk ch;
    ch.k = k.data();
    ch.v = v.data();
    ch.host = 1;
    ch.n_tok = n_tok;
    ch.pos0 = 0;
    ch.k_stride = head * sizeof(uint16_t);
    ch.v_stride = head * sizeof(uint16_t);
    std::vector<float> dst((size_t) head, -1.f);
    kvflash_stream_args args;
    args.q = q.data();
    args.dst = dst.data();
    args.n_q = 1;
    args.n_head = 1;
    args.n_head_kv = 1;
    args.head_dim = head;
    args.scale = 1.f;
    args.q_pos0 = 1;
    args.chunks = &ch;
    args.n_chunks = 1;
    if (!run_attn(&args)) {
        std::fprintf(stderr, "kvflash stream: self-test launch failed\n");
        return false;
    }
    float err = 0.f;
    for (float x : dst) {
        err = std::max(err, std::fabs(x - 1.f));
    }

    // Two queries, equal K. Query 0 sees only V=1. Query 1 sees V=1 and V=3, mean 2.
    std::vector<float> q2((size_t) head * 2, 1.f);
    f16_row(k.data(), 1.f, head * 2);
    f16_row(v.data(), 1.f, head);
    f16_row(v.data() + head, 3.f, head);
    std::vector<float> dst2((size_t) head * 2, -1.f);
    args.q = q2.data();
    args.dst = dst2.data();
    args.n_q = 2;
    args.q_pos0 = 0;
    if (!run_attn(&args)) {
        std::fprintf(stderr, "kvflash stream: self-test multi-query launch failed\n");
        return false;
    }
    for (int i = 0; i < head; ++i) {
        err = std::max(err, std::fabs(dst2[(size_t) i] - 1.f));
        err = std::max(err, std::fabs(dst2[(size_t) head + i] - 2.f));
    }
    if (!(err < 1e-3f)) {
        std::fprintf(stderr, "kvflash stream: self-test err %g\n", err);
        return false;
    }
    std::fprintf(stderr, "kvflash stream: self-test ok (err %g)\n", err);
    return true;
}

}  // namespace

bool kvflash_stream_attn(const kvflash_stream_args * args) {
    static int tested = 0;
    if (!tested) {
        tested = 1;
        if (!self_test()) {
            tested = -1;
        }
    }
    if (tested < 0) {
        return false;
    }
    return run_attn(args);
}
