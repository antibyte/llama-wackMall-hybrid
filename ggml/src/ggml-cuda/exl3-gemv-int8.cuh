#pragma once

// EXL3 int8-activation decode gemv (4 bpw, m <= 4), after exllamav3's exl3_gemv_int8 (MIT).
//
// The mul1 codebook value is affine in the byte sum of (window * 0x83DCD12D):
//   v = k_inv * (1024 + bytesum) + bias
// so with activations quantized to int8, dp4a(window * M, splat(a), acc) accumulates a * bytesum
// exactly in int32 and
//   y[n] = q * (k_inv * acc[n] + (1024 * k_inv + bias) * sum(a))
// In residual mode the int8 rounding error (|r| <= q/2) is quantized with scale q2 = q / 254 and
// accumulated by a second dp4a sharing the decoded windows (~15-16 bit activation precision).
//
// Quantization is per k-slice: every block quantizes its own k range of the Hadamard-transformed
// activations (F16 xh from the fp16 path's input kernel) while staging the byte splats, so the
// scale tracks the local magnitude (this is what keeps plain int8 at fp16 parity).  Each block
// writes its float partial y_inner for its 256 columns; the last k-split block per column group
// sums the partials in slice order (bitwise deterministic), applies the output Hadamard and svh.
//
// gemv_int8_kernel grid (n/256, ksplit), 256 threads: warp = 2 adjacent n tiles, lane = uint2 of
// its tile word pair, two-row register prefetch, splats read from smem as uint4.

#include "exl3-dq.cuh"
#include "exl3-had.cuh"
#include "exl3-gemv.cuh"

namespace exl3_int8 {

constexpr int THREADS = 256;
constexpr int COLS    = 256;   // columns per block: 8 warps x 2 tiles
constexpr int MAX_M   = 8;     // covers speculative verify batches (draft-max 3 default, up to 7)

__device__ __forceinline__ void cp_async16(void * smem, const void * glob) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= 800
    const unsigned s = unsigned(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(glob));
#else
    // Volta/Turing use synchronous copies into the same warp-private ring.
    // The consumer's __syncwarp() orders these stores before cross-lane reads.
    *static_cast<uint4 *>(smem) = *static_cast<const uint4 *>(glob);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// pair-row staging depth (rows in flight per warp) for the smem unit (K = 5..8)
constexpr int STAGE_D = 4;
// The SM86 multi-row head favors two warp-private stages; the scalar decoder
// retains its independently tuned four-stage pipeline.
constexpr int HEAD_STAGE_D = 2;
__host__ __device__ constexpr bool stage_smem(int bits) { return bits >= 5; }
__host__ __device__ constexpr int stage_bytes(int bits) { return stage_smem(bits) ? 8 * STAGE_D * 16 * bits * 4 : 0; }

// Shared by the device specialization and host allocation: compact storage
// must never be selected for a row count that still uses the scalar executor.
__host__ __device__ constexpr bool sm86_matrix_shape(int bits, int cb, int m, bool residual, bool grouped) {
    return cb == 2 && !grouped && (
            (bits >= 2 && bits <= 4 &&
             (m == 3 || m == 4 || m == 8 || (bits == 4 && (m == 5 || m == 6 || m == 7 || (m == 13 && !residual)))) &&
             !(bits == 4 && m == 3 && !residual)) ||
            (bits == 6 && residual && m >= 3 && m <= 8));
}

__host__ __device__ constexpr int head_stage_bytes(int bits) {
    return bits == 6 ? 8 * HEAD_STAGE_D * 16 * bits * 4 : 0;
}

// Reuse the same transform/quantization implementation when a large head can
// amortize one preparation launch across its output columns. No decoded weights
// are retained, and each original K slice keeps its scales and sum order.
enum class input_mode { inline_quantize, prepare, consume };

__device__ __forceinline__ float dot2(half2 w, half2 x) {
    const float2 wf = __half22float2(w), xf = __half22float2(x);
    return wf.x * xf.x + wf.y * xf.y;
}

__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c) {
#if defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))
    return __builtin_amdgcn_sudot4(false, int(a), true, int(b), c, false);
#elif !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= 610
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#else
    // Match unsigned weight bytes, signed activation bytes and wrapping sum.
    uint32_t sum = uint32_t(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += uint32_t(int((a >> (8*i)) & 255u) * int(int8_t(b >> (8*i))));
    }
    return int32_t(sum);
#endif
}

// 4 bpw: 8 windows for run t0..t0+7 from words (a = previous, b = this)
__device__ __forceinline__ void extract8_4bits(uint32_t a, uint32_t b, uint32_t & w0, uint32_t & w1,
        uint32_t & w2, uint32_t & w3, uint32_t & w4, uint32_t & w5, uint32_t & w6, uint32_t & w7) {
    uint32_t s;
    EXL3_FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    EXL3_BFE16_IMM(w6, b, 4);
    EXL3_BFE16_IMM(w5, b, 8);
    EXL3_BFE16_IMM(w4, b, 12);
    EXL3_BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    EXL3_BFE16_IMM(w1, s, 4);
    EXL3_BFE16_IMM(w0, s, 8);
}

// Generic K: 8 windows for run t0..t0+7 straight from the tile words (pointer extraction, the
// index math of exl3_dq.cuh's dq8 paths).  Windows wrap around the 256*bits-bit tile stream.
template <int bits>
__device__ __forceinline__ int wrap_idx(int i) {
    constexpr int words = bits * 8;
    return i >= words ? i - words : i;
}

template <int bits>
__device__ __forceinline__ void ext4w(const uint32_t * ptr, int t0, uint32_t & w0, uint32_t & w1, uint32_t & w2, uint32_t & w3) {
    const int b0 = (t0 + 257) * bits - 16;
    const int b2 = b0 + 3 * bits + 16;
    const int i0 = b0 / 32;
    const int i2 = (b2 - 1) / 32;
    const int s2 = (i2 + 1) * 32 - b2;
    const uint32_t a = ptr[wrap_idx<bits>(i0)];
    const uint32_t b = ptr[wrap_idx<bits>(i2)];
    w3 = exl3::fshift(b, a, s2) & 0xffff;
    w2 = exl3::fshift(b, a, s2 + bits) & 0xffff;
    w1 = exl3::fshift(b, a, s2 + bits * 2) & 0xffff;
    w0 = exl3::fshift(b, a, s2 + bits * 3) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext2w(const uint32_t * ptr, int t0, uint32_t & w0, uint32_t & w1) {
    const int b0 = (t0 + 257) * bits - 16;
    const int b2 = b0 + bits + 16;
    const int i0 = b0 / 32;
    const int i2 = (b2 - 1) / 32;
    const int s2 = (i2 + 1) * 32 - b2;
    const uint32_t a = ptr[wrap_idx<bits>(i0)];
    const uint32_t b = ptr[wrap_idx<bits>(i2)];
    w1 = exl3::fshift(b, a, s2) & 0xffff;
    w0 = exl3::fshift(b, a, s2 + bits) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext8w(const uint32_t * ptr, int t0, uint32_t & w0, uint32_t & w1, uint32_t & w2, uint32_t & w3,
        uint32_t & w4, uint32_t & w5, uint32_t & w6, uint32_t & w7) {
    if constexpr (bits == 1) {
        const uint32_t i1 = t0 >> 5;
        const uint32_t i0 = (i1 + 7) & 7;
        const uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = exl3::fshift(b, a, ((~t0) & 24));
        w7 = b & 0xffff;
        EXL3_BFE16_IMM(w6, b, 1); EXL3_BFE16_IMM(w5, b, 2); EXL3_BFE16_IMM(w4, b, 3); EXL3_BFE16_IMM(w3, b, 4);
        EXL3_BFE16_IMM(w2, b, 5); EXL3_BFE16_IMM(w1, b, 6); EXL3_BFE16_IMM(w0, b, 7);
    } else if constexpr (bits == 2) {
        const uint32_t i1 = t0 >> 4;
        const uint32_t i0 = (i1 + 15) & 15;
        const uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = exl3::fshift(b, a, ((~t0) & 8) << 1);
        w7 = b & 0xffff;
        EXL3_BFE16_IMM(w6, b, 2); EXL3_BFE16_IMM(w5, b, 4); EXL3_BFE16_IMM(w4, b, 6); EXL3_BFE16_IMM(w3, b, 8);
        EXL3_BFE16_IMM(w2, b, 10); EXL3_BFE16_IMM(w1, b, 12); EXL3_BFE16_IMM(w0, b, 14);
    } else if constexpr (bits == 3) {
        const int b1 = (t0 + 257) * bits;
        const int b0 = b1 - 16;
        const int b2 = b1 + bits * 7;
        const int i0 = b0 / 32;
        const int i2 = (b2 - 1) / 32;
        const int s2 = (i2 + 1) * 32 - b2;
        const uint32_t a = ptr[wrap_idx<bits>(i0)];
        const uint32_t b = ptr[wrap_idx<bits>(i2)];
        w7 = exl3::fshift(b, a, s2);
        w6 = w7 >> bits; w5 = w6 >> bits; w4 = w5 >> bits;
        w3 = exl3::fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits; w1 = w2 >> bits; w0 = w1 >> bits;
        w7 &= 0xffff; w6 &= 0xffff; w5 &= 0xffff; w4 &= 0xffff;
        w3 &= 0xffff; w2 &= 0xffff; w1 &= 0xffff; w0 &= 0xffff;
    } else if constexpr (bits == 4) {
        const uint32_t i1 = t0 >> 3;
        const uint32_t i0 = (i1 + 31) & 31;
        extract8_4bits(ptr[i0], ptr[i1], w0, w1, w2, w3, w4, w5, w6, w7);
    } else if constexpr (bits == 7) {
        ext2w<bits>(ptr, t0,     w0, w1);
        ext2w<bits>(ptr, t0 + 2, w2, w3);
        ext2w<bits>(ptr, t0 + 4, w4, w5);
        ext2w<bits>(ptr, t0 + 6, w6, w7);
    } else {  // 5, 6, 8
        ext4w<bits>(ptr, t0,     w0, w1, w2, w3);
        ext4w<bits>(ptr, t0 + 4, w4, w5, w6, w7);
    }
}

// Register forms (exl3-gemv.cuh's dq8_regs_2bits / dq8_regs_3bits without the decode): the eight
// windows of run t0..t0+7 from the two lane-selected words.
__device__ __forceinline__ void regs2_windows(uint32_t a, uint32_t b, int t_offset, uint32_t * w) {
    b = exl3::fshift(b, a, ((~t_offset) & 8) << 1);
    w[7] = b & 0xffff;
    EXL3_BFE16_IMM(w[6], b, 2); EXL3_BFE16_IMM(w[5], b, 4); EXL3_BFE16_IMM(w[4], b, 6); EXL3_BFE16_IMM(w[3], b, 8);
    EXL3_BFE16_IMM(w[2], b, 10); EXL3_BFE16_IMM(w[1], b, 12); EXL3_BFE16_IMM(w[0], b, 14);
}

__device__ __forceinline__ void regs3_windows(uint32_t a, uint32_t b, int s2, uint32_t * w) {
    w[7] = exl3::fshift(b, a, s2);
    w[6] = w[7] >> 3; w[5] = w[6] >> 3; w[4] = w[5] >> 3;
    w[3] = exl3::fshift(b, a, s2 + 12);
    w[2] = w[3] >> 3; w[1] = w[2] >> 3; w[0] = w[1] >> 3;
#pragma unroll
    for (int j = 0; j < 8; ++j) w[j] &= 0xffff;
}

// K = 5, 6, 8: ext4w with the two words fetched by lane shuffle instead of pointer (word q of the
// tile lives in lane q/2, component q%2 of the uint2 each lane loaded).  K = 7 uses ext2w groups.
template <int bits>
__device__ __forceinline__ uint32_t shfl_word(uint2 r, int q) {
    const uint32_t x = __shfl_sync(0xffffffffu, r.x, q >> 1);
    const uint32_t y = __shfl_sync(0xffffffffu, r.y, q >> 1);
    return (q & 1) ? y : x;
}

template <int bits>
__device__ __forceinline__ void regsw4(uint2 r, int t0, uint32_t & w0, uint32_t & w1, uint32_t & w2, uint32_t & w3) {
    const int b0 = (t0 + 257) * bits - 16;
    const int b2 = b0 + 3 * bits + 16;
    const int i0 = b0 / 32;
    const int i2 = (b2 - 1) / 32;
    const int s2 = (i2 + 1) * 32 - b2;
    const uint32_t a = shfl_word<bits>(r, wrap_idx<bits>(i0));
    const uint32_t b = shfl_word<bits>(r, wrap_idx<bits>(i2));
    w3 = exl3::fshift(b, a, s2) & 0xffff;
    w2 = exl3::fshift(b, a, s2 + bits) & 0xffff;
    w1 = exl3::fshift(b, a, s2 + bits * 2) & 0xffff;
    w0 = exl3::fshift(b, a, s2 + bits * 3) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void regsw2(uint2 r, int t0, uint32_t & w0, uint32_t & w1) {
    const int b0 = (t0 + 257) * bits - 16;
    const int b2 = b0 + bits + 16;
    const int i0 = b0 / 32;
    const int i2 = (b2 - 1) / 32;
    const int s2 = (i2 + 1) * 32 - b2;
    const uint32_t a = shfl_word<bits>(r, wrap_idx<bits>(i0));
    const uint32_t b = shfl_word<bits>(r, wrap_idx<bits>(i2));
    w1 = exl3::fshift(b, a, s2) & 0xffff;
    w0 = exl3::fshift(b, a, s2 + bits) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void regsw_windows(uint2 r, int t0, uint32_t * w) {
    if constexpr (bits == 7) {
        regsw2<bits>(r, t0,     w[0], w[1]);
        regsw2<bits>(r, t0 + 2, w[2], w[3]);
        regsw2<bits>(r, t0 + 4, w[4], w[5]);
        regsw2<bits>(r, t0 + 6, w[6], w[7]);
    } else {
        regsw4<bits>(r, t0,     w[0], w[1], w[2], w[3]);
        regsw4<bits>(r, t0 + 4, w[4], w[5], w[6], w[7]);
    }
}

// ---- gemv ------------------------------------------------------------------------------------

// B: n-tile-major tile stream; x: [M][k] F32 activations, suh: [k] F16 input signs; y: [M][n] F32;
// partials: [ksplit][M][n] F32 (fully overwritten); counters: n/256 ints, zero at rest.
// Routing descriptor for the grouped (MoE) launch: block z = (token, expert slot) pair.
struct grouped_args {
    const int32_t * ids;        // [n_expert_used, n_tokens], row stride ids_nb1 (elements)
    int    ids_nb1;
    int    n_expert_used;
    int    ne11;                // activation rows per token (1 = broadcast, or n_expert_used)
    size_t x_nb1, x_nb2;        // activation strides (elements)
    size_t expert_stride;       // bytes between expert weight blocks
};

// cb == 2 (mul1): int8 activations, dp4a; other codebooks: F16 activations, decoded weights, fp32 FMA.
// Independent second projection for a paired dense launch. Each projection keeps
// its own input signs, scales, K-slices and ordered partial reduction.
struct bundle_args {
    const uint8_t * weights;
    const half * suh;
    const half * svh;
    float * output;
    float * partials;
    int * counters;
    int n, nrows, ksplit, first_blocks, second_blocks;
};

template <int bits, int cb, int M, bool RESID, bool GROUPED, bool BUNDLE = false, input_mode INPUT = input_mode::inline_quantize>
__global__ void __launch_bounds__(THREADS) gemv_int8_kernel(const uint8_t * __restrict__ B,
        const float * __restrict__ x, const half * __restrict__ suh, const half * __restrict__ svh, float * __restrict__ y,
        float * __restrict__ partials, int * __restrict__ counters, int k, int n, int nrows_max, grouped_args ga, bundle_args bundle = {}, uint8_t * prepared = nullptr) {
    int column_block = blockIdx.x, k_block = blockIdx.y, k_blocks = gridDim.y;
    if constexpr (BUNDLE) {
        static_assert(!GROUPED);
        // Alternate independent projection blocks; append any unmatched tail.
        const int paired = min(bundle.first_blocks, bundle.second_blocks);
        const int flat = blockIdx.x;
        const bool second = flat < 2 * paired ? (flat & 1) : bundle.second_blocks > bundle.first_blocks;
        const int index = flat < 2 * paired ? flat / 2 : flat - paired;
        if (second) {
            B = bundle.weights; suh = bundle.suh; svh = bundle.svh;
            y = bundle.output; partials = bundle.partials; counters = bundle.counters;
            n = bundle.n; nrows_max = bundle.nrows; k_blocks = bundle.ksplit;
        } else {
            k_blocks = bundle.first_blocks / ((n + COLS - 1) / COLS);
        }
        const int columns = (n + COLS - 1) / COLS;
        column_block = index % columns;
        k_block = index / columns;
    }
    // Visit adjacent K slices first for dense single-row SM86 decode. Keep the
    // same slice boundaries/scales and ordered reduction; only CTA order changes.
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == 860
    if constexpr (M == 1 && !GROUPED && !BUNDLE) {
        const int linear_block = blockIdx.y * gridDim.x + blockIdx.x;
        column_block = linear_block / gridDim.y;
        k_block = linear_block % gridDim.y;
    }
#endif
    constexpr int TWORDS = 8 * bits;
    constexpr bool WIDE = bits == 4;   // uint2-per-lane block pair; other K use pointer extraction
    constexpr bool INT8 = cb == 2;
    static_assert(INT8 || !RESID, "residual pass is an int8-mode feature");
    constexpr int NACC = (RESID ? 2 : 1) * M;
#if defined(GGML_USE_HIP) && defined(RDNA4)
    // Measured dense mul1 crossover: retain vector dots for one/two rows.
    // Other row counts/precisions and grouped MoE retain their existing executor.
    constexpr bool WMMA = INT8 && !GROUPED && bits >= 2 && bits <= 4 && (M == 3 || M == 4 || M == 8);
#elif !defined(GGML_USE_HIP) && __CUDA_ARCH__ == 750
    // Turing uses four smaller integer MMAs. Keep vector dots for the
    // unmeasured formats/rows and for small plain-int8 verification batches.
    constexpr bool WMMA = INT8 && !GROUPED && bits == 4 && (M == 8 || (RESID && M == 4));
#elif !defined(GGML_USE_HIP) && __CUDA_ARCH__ == 860
    // SM86: matrix cores help dense verify batches; plain K4 M3 favors vector dots.
    // K6 residual batches cover the vocabulary head without dropping its correction.
    constexpr bool WMMA = sm86_matrix_shape(bits, cb, M, RESID, GROUPED);
#else
    constexpr bool WMMA = false;
#endif
    if constexpr (GROUPED) {
        // this block's (token, expert slot) pair selects the expert's weights, scales and rows
        const int pair = blockIdx.z;
        const int t = pair / ga.n_expert_used, e = pair - t * ga.n_expert_used;
        const int expert = ga.ids[size_t(t) * ga.ids_nb1 + e];
        // Expert windows use -1 for pairs owned by another device. The whole
        // block skips them; the window dispatcher zeroes their output rows.
        if (expert < 0) return;
        B   += size_t(expert) * ga.expert_stride;
        svh += size_t(expert) * n;
        suh += size_t(expert) * k;
        x   += size_t(t) * ga.x_nb2 + (ga.ne11 == 1 ? 0 : size_t(e) * ga.x_nb1);
        y   += size_t(pair) * n;
        partials += size_t(pair) * k_blocks * M * n;
        counters += size_t(pair) * gridDim.x;
    }
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == 860
    constexpr bool COMPACT = WMMA && M >= 3 && M <= 8;
#else
    constexpr bool COMPACT = false;
#endif
    extern __shared__ uint32_t sh_as[];
    half * sh_xh = reinterpret_cast<half *>(reinterpret_cast<uint8_t *>(sh_as) + size_t(NACC) * nrows_max * 16 * (COMPACT ? 1 : 4));
    uint32_t * sh_stage = reinterpret_cast<uint32_t *>(sh_xh +
            (INPUT == input_mode::consume ? 0 : size_t(M) * nrows_max * 16));
    __shared__ float sh_y[M][COLS];
    __shared__ float sh_redf[THREADS / 32][M];
    __shared__ int   sh_redi[THREADS / 32][NACC];
    __shared__ float sh_q[NACC];
    __shared__ int   sh_s[NACC];
    __shared__ int   sh_last;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, lq = lane & 15;
    const int kslices = k / 16;
    const int kb0   = k_block * nrows_max;
    const int nrows = min(nrows_max, kslices - kb0);
    const int kn    = nrows * 16;
    constexpr int HEADER_BYTES = 2 * NACC * sizeof(uint32_t);
    if constexpr (INPUT != input_mode::inline_quantize) {
        static_assert(M >= 3 && M <= 8 && bits == 6 && RESID && !GROUPED && !BUNDLE);
        prepared += size_t(k_block) * (HEADER_BYTES + NACC * nrows_max * 16);
    }

    // input transform of this block's own k range (128-aligned: nrows % 8 == 0): xh = had128(x * suh) / sqrt(128),
    // F16 in smem, with the per-slice max |xh| per row
    if constexpr (INPUT != input_mode::consume) {
        float amax[M];
#pragma unroll
        for (int r = 0; r < M; ++r) amax[r] = 0.0f;
        for (int b = warp; b < (kn / 128) * M; b += THREADS / 32) {
            const int r   = b / (kn / 128);
            const int col = kb0 * 16 + (b - r * (kn / 128)) * 128 + lane * 4;
            const float4 xv = *reinterpret_cast<const float4 *>(x + size_t(r) * k + col);
            const half2 s01 = *reinterpret_cast<const half2 *>(suh + col);
            const half2 s23 = *reinterpret_cast<const half2 *>(suh + col + 2);
            float v0 = xv.x * __low2float(s01);
            float v1 = xv.y * __high2float(s01);
            float v2 = xv.z * __low2float(s23);
            float v3 = xv.w * __high2float(s23);
            exl3_had::had128(v0, v1, v2, v3, lane);
            const half2 h01 = __floats2half2_rn(v0 * exl3_had::SCALE, v1 * exl3_had::SCALE);
            const half2 h23 = __floats2half2_rn(v2 * exl3_had::SCALE, v3 * exl3_had::SCALE);
            half * dst = sh_xh + size_t(r) * nrows_max * 16 + (col - kb0 * 16);
            *reinterpret_cast<half2 *>(dst)     = h01;
            *reinterpret_cast<half2 *>(dst + 2) = h23;
            amax[r] = fmaxf(amax[r], fmaxf(fmaxf(fabsf(__low2float(h01)), fabsf(__high2float(h01))),
                                           fmaxf(fabsf(__low2float(h23)), fabsf(__high2float(h23)))));
        }
#pragma unroll
        for (int r = 0; r < M; ++r) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) amax[r] = fmaxf(amax[r], __shfl_xor_sync(0xffffffffu, amax[r], o));
            if (lane == 0) sh_redf[warp][r] = amax[r];
        }
        __syncthreads();
        if (threadIdx.x < NACC) {
            const int r = RESID ? threadIdx.x >> 1 : threadIdx.x;
            float mx = 0.0f;
            for (int w = 0; w < THREADS / 32; ++w) mx = fmaxf(mx, sh_redf[w][r]);
            const float q = fmaxf(mx, 1e-30f) / 127.0f;
            sh_q[threadIdx.x] = (RESID && (threadIdx.x & 1)) ? q / 254.0f : q;
        }
        __syncthreads();
    }
    auto splat_index = [&](int p, int i) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == 860
        // Spread MMA activation planes across banks without padding the allocation.
        if constexpr (WMMA) i ^= (p & 7) * 4;
#endif
        return p * nrows_max * 16 + i;
    };
    auto store_splat = [&](int p, int i, int v) {
        if constexpr (COMPACT) reinterpret_cast<uint8_t *>(sh_as)[splat_index(p, i)] = uint8_t(int8_t(v));
        else sh_as[splat_index(p, i)] = uint32_t(uint8_t(int8_t(v))) * 0x01010101u;
    };
    auto load_splat = [&](int p, int i) {
        if constexpr (COMPACT) return uint32_t(reinterpret_cast<uint8_t *>(sh_as)[splat_index(p, i)]) * 0x01010101u;
        else return sh_as[splat_index(p, i)];
    };
    // quantize inline while staging the splats; exact int sums per plane
    if constexpr (INT8 && INPUT != input_mode::consume) {
        int sum[NACC];
#pragma unroll
        for (int p = 0; p < NACC; ++p) sum[p] = 0;
        for (int i = threadIdx.x; i < kn; i += THREADS) {
#pragma unroll
            for (int r = 0; r < M; ++r) {
                const int p0 = RESID ? 2 * r : r;
                const float a  = __half2float(sh_xh[size_t(r) * nrows_max * 16 + i]);
                const float q  = sh_q[p0];
                int v = __float2int_rn(a / q);
                v = max(-127, min(127, v));
                store_splat(p0, i, v);
                sum[p0] += v;
                if constexpr (RESID) {
                    const float rr = a - q * float(v);
                    int v2 = __float2int_rn(rr / sh_q[p0 + 1]);
                    v2 = max(-127, min(127, v2));
                    store_splat(p0 + 1, i, v2);
                    sum[p0 + 1] += v2;
                }
            }
        }
#pragma unroll
        for (int p = 0; p < NACC; ++p) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum[p] += __shfl_xor_sync(0xffffffffu, sum[p], o);
            if (lane == 0) sh_redi[warp][p] = sum[p];
        }
        __syncthreads();
        if (threadIdx.x < NACC) {
            int t = 0;
            for (int w = 0; w < THREADS / 32; ++w) t += sh_redi[w][threadIdx.x];
            sh_s[threadIdx.x] = t;
        }
        __syncthreads();
    }

    if constexpr (INPUT == input_mode::prepare) {
        for (int i = threadIdx.x; i < NACC * nrows_max * 4; i += THREADS) {
            reinterpret_cast<uint32_t *>(prepared + HEADER_BYTES)[i] = i % (nrows_max * 4) < kn / 4 ? sh_as[i] : 0;
        }
        if (threadIdx.x < NACC) {
            reinterpret_cast<uint32_t *>(prepared)[threadIdx.x] = __float_as_uint(sh_q[threadIdx.x]);
            reinterpret_cast<uint32_t *>(prepared)[NACC + threadIdx.x] = uint32_t(sh_s[threadIdx.x]);
        }
        return;
    }
    if constexpr (INPUT == input_mode::consume) {
        for (int i = threadIdx.x; i < NACC * nrows_max * 4; i += THREADS) {
            sh_as[i] = reinterpret_cast<const uint32_t *>(prepared + HEADER_BYTES)[i];
        }
        if (threadIdx.x < NACC) {
            sh_q[threadIdx.x] = __uint_as_float(reinterpret_cast<uint32_t *>(prepared)[threadIdx.x]);
            sh_s[threadIdx.x] = int(reinterpret_cast<uint32_t *>(prepared)[NACC + threadIdx.x]);
        }
        __syncthreads();
    }

    const float k_inv = __half2float(__ushort_as_half(0x1eee));
    const float bias  = __half2float(__ushort_as_half(0xc931));
    const float cbias = 1024.0f * k_inv + bias;
    const uint32_t * B32 = reinterpret_cast<const uint32_t *>(B);

    const int n_tiles = n / 16;
    if constexpr (WMMA) {
        static_assert(bits >= 2 && bits <= 6 && NACC <= 16);
        const int nt = column_block * 16 + warp * 2;
#if defined(GGML_USE_HIP) && defined(RDNA4)
        using i2 = int __attribute__((ext_vector_type(2)));
        using i8 = int __attribute__((ext_vector_type(8)));
        i8 acc[2] = {};
#elif !defined(GGML_USE_HIP) && (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 750)
        constexpr int PLANES = (NACC + 7) / 8;
        int acc[2][PLANES][4] = {};
#endif
        // Each lane loads one packed word per tile (two for K6). Prefetch four K tiles, then
        // shuffle word pairs and extract nearby windows together for the matrix core.
        constexpr int RING = 4;
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == 860
        constexpr bool WSTAGE = bits == 6 && M >= 3 && M <= 8;
#else
        constexpr bool WSTAGE = false;
#endif
        constexpr int PAIRW = 2 * TWORDS;
        uint32_t * warp_stage = sh_stage + warp * HEAD_STAGE_D * PAIRW;
        auto stage_weights = [&](int kb) {
            if constexpr (WSTAGE) {
                const int tile = lane / (TWORDS / 4);
                if (lane < PAIRW / 4 && kb < nrows && nt + tile < n_tiles) {
                    const auto * src = B32 + (size_t(nt + tile) * kslices + kb0 + kb) * TWORDS + (lane % (TWORDS / 4)) * 4;
                    cp_async16(warp_stage + (kb % HEAD_STAGE_D) * PAIRW + lane * 4, src);
                }
                cp_async_commit();
            }
        };
        if constexpr (WSTAGE) {
#pragma unroll
            for (int i = 0; i < HEAD_STAGE_D - 1; ++i) stage_weights(i);
        }
        uint32_t words[2][RING] = {};
        uint32_t words_hi[2][RING] = {};
        auto load_word = [&](int tile, int kb, int extra = 0) {
            return nt + tile < n_tiles && lane + extra < TWORDS && kb < nrows
                ? exl3::load_streaming(B32 + (size_t(nt + tile) * kslices + kb0 + kb) * TWORDS + lane + extra) : 0u;
        };
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
            for (int d = 0; d < RING; ++d) {
                if constexpr (!WSTAGE) {
                    words[tile][d] = load_word(tile, d);
                    if constexpr (bits > 4) words_hi[tile][d] = load_word(tile, d, 32);
                }
            }
        }
        for (int base = 0; base < nrows; base += RING) {
#pragma unroll
            for (int drow = 0; drow < RING; ++drow) {
                const int kb = base + drow;
                if (kb >= nrows) break;
                if constexpr (WSTAGE) {
                    cp_async_wait<HEAD_STAGE_D - 2>();
                    __syncwarp();
                    stage_weights(kb + HEAD_STAGE_D - 1);
                }
#if defined(GGML_USE_HIP) && defined(RDNA4)
                i2 decoded[2][4];
#pragma unroll
                for (int tile = 0; tile < 2; ++tile) {
                    const uint32_t current = words[tile][drow];
                    words[tile][drow] = load_word(tile, kb + RING);
#pragma unroll
                    for (int g = 0; g < 2; ++g) {
                        // RDNA4 fragment: four consecutive trellis windows per group.
                        const int t = (lq % 8) * 32 + (lane / 16) * 8 + (lq / 8) * 4 + g * 16;
                        const int end = (t + 257 + 3) * bits;
                        const int first = (end - 3 * bits - 16) / 32, last = (end - 1) / 32;
                        const uint32_t lo = __shfl_sync(0xffffffffu, current, last % TWORDS);
                        const uint32_t hi = __shfl_sync(0xffffffffu, current, first % TWORDS);
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            const uint32_t w = exl3::fshift(lo, hi, (last + 1) * 32 - end + (3 - s) * bits) & 65535u;
                            decoded[tile][g + 2 * (s / 2)][s & 1] = int(w * 0x83DCD12Du);
                        }
                    }
                }
#pragma unroll
                for (int j0 = 0; j0 < 16; j0 += 4) {
                    i2 b;
#pragma unroll
                    for (int l = 0; l < 2; ++l) {
                        const int j = j0 + 2 * (lane / 16) + l;
                        b[l] = lq < NACC ? int(sh_as[lq * nrows_max * 16 + kb * 16 + j]) : 0;
                    }
#pragma unroll
                    for (int tile = 0; tile < 2; ++tile) {
                        const i2 a = decoded[tile][j0 / 4];
                        // Unsigned product bytes times signed activation splats:
                        // the same integer dot as dp4a_us, including residual planes.
                        acc[tile] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, true, b, acc[tile], false);
                    }
                }
#elif !defined(GGML_USE_HIP) && (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 750)
                uint32_t current[2];
                uint32_t current_hi[2] = {};
                uint32_t operands[2][PLANES][2];
#pragma unroll
                for (int half = 0; half < 2; ++half) {
#pragma unroll
                    for (int plane = 0; plane < PLANES; ++plane) {
                        const int p = plane * 8 + lane / 4;
                        const int j = half * 8 + lane % 4;
                        if constexpr (M <= 8) {
                            operands[half][plane][0] = p < NACC ? load_splat(p, kb * 16 + j) : 0;
                            operands[half][plane][1] = p < NACC ? load_splat(p, kb * 16 + j + 4) : 0;
                        }
                    }
                }
#pragma unroll
                for (int tile = 0; tile < 2; ++tile) {
                    if constexpr (!WSTAGE) {
                        current[tile] = words[tile][drow];
                        words[tile][drow] = load_word(tile, kb + RING);
                        if constexpr (bits > 4) {
                            current_hi[tile] = words_hi[tile][drow];
                            words_hi[tile][drow] = load_word(tile, kb + RING, 32);
                        }
                    }
                }
#pragma unroll
                for (int tile = 0; tile < 2; ++tile) {
                    uint32_t decoded[2][4];
#pragma unroll
                    for (int g = 0; g < 2; ++g) {
                        // Ampere fragment: four stride-two windows per group.
                        const int t = (lane / 4) * 32 + ((lane % 4) / 2) * 8 + (lane & 1) + g * 16;
                        if constexpr (bits > 4) {
                            // Four stride-two K6 windows can span three packed words.
                            // Decode pairs so each funnel shift covers adjacent words.
#pragma unroll
                            for (int pair = 0; pair < 2; ++pair) {
                                const int end = (t + 257 + 4 * pair + 2) * bits;
                                const int first = (end - 2 * bits - 16) / 32, last = (end - 1) / 32;
                                const int ilo = last % TWORDS, ihi = first % TWORDS;
                                uint32_t lo, hi;
                                if constexpr (WSTAGE) {
                                    const uint32_t * row = warp_stage + (kb % HEAD_STAGE_D) * PAIRW + tile * TWORDS;
                                    lo = nt + tile < n_tiles ? row[ilo] : 0;
                                    hi = nt + tile < n_tiles ? row[ihi] : 0;
                                } else {
                                    const uint32_t lo0 = __shfl_sync(0xffffffffu, current[tile], ilo & 31);
                                    const uint32_t hi0 = __shfl_sync(0xffffffffu, current[tile], ihi & 31);
                                    const uint32_t lo1 = __shfl_sync(0xffffffffu, current_hi[tile], ilo & 31);
                                    const uint32_t hi1 = __shfl_sync(0xffffffffu, current_hi[tile], ihi & 31);
                                    lo = ilo < 32 ? lo0 : lo1; hi = ihi < 32 ? hi0 : hi1;
                                }
#pragma unroll
                                for (int s = 0; s < 2; ++s) {
                                    const uint32_t w = exl3::fshift(lo, hi, (last + 1) * 32 - end + (2 - 2 * s) * bits) & 65535u;
                                    decoded[s][2 * g + pair] = w * 0x83DCD12Du;
                                }
                            }
                        } else {
                            const int end = (t + 257 + 6) * bits;
                            const int first = (end - 6 * bits - 16) / 32, last = (end - 1) / 32;
                            const uint32_t lo = __shfl_sync(0xffffffffu, current[tile], last % TWORDS);
                            const uint32_t hi = __shfl_sync(0xffffffffu, current[tile], first % TWORDS);
#pragma unroll
                            for (int s = 0; s < 4; ++s) {
                                const uint32_t w = exl3::fshift(lo, hi, (last + 1) * 32 - end + (6 - 2 * s) * bits) & 65535u;
                                decoded[s & 1][2 * g + s / 2] = w * 0x83DCD12Du;
                            }
                        }
                    }
#pragma unroll
                    for (int j0 = 0; j0 < 16; j0 += 8) {
                        const uint32_t * a = decoded[j0 / 8];
#pragma unroll
                        for (int plane = 0; plane < PLANES; ++plane) {
                            const int p = plane * 8 + lane / 4;
                            const int j = j0 + lane % 4;
                            const uint32_t b0 = M <= 8 ? operands[j0 / 8][plane][0] : (p < NACC ? load_splat(p, kb * 16 + j) : 0);
                            const uint32_t b1 = M <= 8 ? operands[j0 / 8][plane][1] : (p < NACC ? load_splat(p, kb * 16 + j + 4) : 0);
                            int * d = acc[tile][plane];
#if __CUDA_ARCH__ == 750
                            // Same integer dot and accumulator layout as m16n8k32;
                            // Turing's native instruction covers eight rows / K16.
                            asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(b0));
                            asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                : "+r"(d[2]), "+r"(d[3]) : "r"(a[1]), "r"(b0));
                            asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                : "+r"(d[0]), "+r"(d[1]) : "r"(a[2]), "r"(b1));
                            asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                : "+r"(d[2]), "+r"(d[3]) : "r"(a[3]), "r"(b1));
#else
                            asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
                        }
                    }
                }
#endif
            }
        }
        if constexpr (WSTAGE) {
            cp_async_wait<0>();
            __syncwarp();
        }
#if defined(GGML_USE_HIP) && defined(RDNA4)
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                // RDNA4 accumulator layout: lane%16 is the activation plane;
                // 8*(lane/16)+l is the output column within this 16-column tile.
                int second = 0;
                if constexpr (RESID) second = __shfl_down_sync(0xffffffffu, acc[tile][l], 1);
                if (nt + tile < n_tiles && lq < NACC && (!RESID || !(lq & 1))) {
                    // Move the integer residual before folding; shuffling rounded
                    // residual floats here would change the original rounding order.
                    float v = sh_q[lq] * (k_inv * float(acc[tile][l]) + cbias * float(sh_s[lq]));
                    if constexpr (RESID) v += sh_q[lq + 1] * (k_inv * float(second) + cbias * float(sh_s[lq + 1]));
                    const int r = RESID ? lq / 2 : lq;
                    partials[(size_t(k_block) * M + r) * n + (nt + tile) * 16 + 8 * (lane / 16) + l] = v;
                }
            }
        }
#elif !defined(GGML_USE_HIP) && (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 750)
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
            for (int plane = 0; plane < PLANES; ++plane) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    // MMA C layout: adjacent registers are adjacent activation planes.
                    // In residual mode even registers own the primary/residual pair.
                    const int p = plane * 8 + (lane % 4) * 2 + l % 2;
                    const int c = lane / 4 + (l / 2) * 8;
                    if (nt + tile < n_tiles && p < NACC && (!RESID || !(l & 1))) {
                        float v = sh_q[p] * (k_inv * float(acc[tile][plane][l]) + cbias * float(sh_s[p]));
                        if constexpr (RESID) v += sh_q[p + 1] * (k_inv * float(acc[tile][plane][l + 1]) + cbias * float(sh_s[p + 1]));
                        const int r = RESID ? p / 2 : p;
                        partials[(size_t(k_block) * M + r) * n + (nt + tile) * 16 + c] = v;
                    }
                }
            }
        }
#endif
    } else if constexpr (WIDE) {
        const int nt = column_block * 16 + warp * 2 + (lane >> 4);
        const bool active = nt < n_tiles;   // partial last block: warps beyond n idle (whole warp)
        const uint32_t * bp = B32 + (size_t(nt) * kslices + kb0) * TWORDS + 2 * lq;
        const int c2 = (lane & 1) ? 4 : 0;
        const int shfl_src = (lane & 16) | ((lane + 15) & 15);

        int acc0[NACC], acc1[NACC];
        float facc0[M], facc1[M];
#pragma unroll
        for (int p = 0; p < NACC; ++p) { acc0[p] = 0; acc1[p] = 0; }
#pragma unroll
        for (int r = 0; r < M; ++r) { facc0[r] = 0.0f; facc1[r] = 0.0f; }

        uint2 r0 = (active && nrows > 0) ? exl3::load_streaming(reinterpret_cast<const uint2 *>(bp)) : make_uint2(0, 0);
        uint2 r1 = (active && nrows > 1) ? exl3::load_streaming(reinterpret_cast<const uint2 *>(bp + TWORDS)) : make_uint2(0, 0);
        for (int kb = 0; kb < (active ? nrows : 0); ++kb) {
            uint2 r2 = make_uint2(0, 0);
            if (kb + 2 < nrows) r2 = exl3::load_streaming(reinterpret_cast<const uint2 *>(bp + size_t(kb + 2) * TWORDS));
            const uint32_t prev = __shfl_sync(0xffffffffu, r0.y, shfl_src);
            uint32_t w0, w1, w2, w3, w4, w5, w6, w7, v0, v1, v2, v3, v4, v5, v6, v7;
            extract8_4bits(prev, r0.x, w0, w1, w2, w3, w4, w5, w6, w7);   // run t = 8*(2m)
            extract8_4bits(r0.x, r0.y, v0, v1, v2, v3, v4, v5, v6, v7);   // run t = 8*(2m+1)
            if constexpr (!INT8) {
                // rows c2+{0,1,8,9} for the w run, c2+{2,3,10,11} for the v run
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    const half * xr = sh_xh + size_t(r) * nrows_max * 16 + (kb << 4) + c2;
                    const half2 x01 = *reinterpret_cast<const half2 *>(xr);
                    const half2 x89 = *reinterpret_cast<const half2 *>(xr + 8);
                    const half2 x23 = *reinterpret_cast<const half2 *>(xr + 2);
                    const half2 xab = *reinterpret_cast<const half2 *>(xr + 10);
                    facc0[r] += dot2(exl3_gemv::exl3_decode_pair<cb>(w0, w1), x01) + dot2(exl3_gemv::exl3_decode_pair<cb>(w2, w3), x89)
                              + dot2(exl3_gemv::exl3_decode_pair<cb>(v0, v1), x23) + dot2(exl3_gemv::exl3_decode_pair<cb>(v2, v3), xab);
                    facc1[r] += dot2(exl3_gemv::exl3_decode_pair<cb>(w4, w5), x01) + dot2(exl3_gemv::exl3_decode_pair<cb>(w6, w7), x89)
                              + dot2(exl3_gemv::exl3_decode_pair<cb>(v4, v5), x23) + dot2(exl3_gemv::exl3_decode_pair<cb>(v6, v7), xab);
                }
                r0 = r1;
                r1 = r2;
                continue;
            }
            w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
            w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
            v0 *= 0x83DCD12Du; v1 *= 0x83DCD12Du; v2 *= 0x83DCD12Du; v3 *= 0x83DCD12Du;
            v4 *= 0x83DCD12Du; v5 *= 0x83DCD12Du; v6 *= 0x83DCD12Du; v7 *= 0x83DCD12Du;
#pragma unroll
            for (int p = 0; p < NACC; ++p) {
                const uint32_t * as = sh_as + p * nrows_max * 16 + (kb << 4);
                const uint4 as0 = *reinterpret_cast<const uint4 *>(as + c2);
                const uint4 as8 = *reinterpret_cast<const uint4 *>(as + c2 + 8);
                int i0 = acc0[p], i1 = acc1[p];
                i0 = dp4a_us(w0, as0.x, i0); i0 = dp4a_us(w1, as0.y, i0); i0 = dp4a_us(w2, as8.x, i0); i0 = dp4a_us(w3, as8.y, i0);
                i1 = dp4a_us(w4, as0.x, i1); i1 = dp4a_us(w5, as0.y, i1); i1 = dp4a_us(w6, as8.x, i1); i1 = dp4a_us(w7, as8.y, i1);
                i0 = dp4a_us(v0, as0.z, i0); i0 = dp4a_us(v1, as0.w, i0); i0 = dp4a_us(v2, as8.z, i0); i0 = dp4a_us(v3, as8.w, i0);
                i1 = dp4a_us(v4, as0.z, i1); i1 = dp4a_us(v5, as0.w, i1); i1 = dp4a_us(v6, as8.z, i1); i1 = dp4a_us(v7, as8.w, i1);
                acc0[p] = i0; acc1[p] = i1;
            }
            r0 = r1;
            r1 = r2;
        }

        // lanes l, l^1 hold the same two columns; fold the affine codebook terms with this slice's scales
#pragma unroll
        for (int p = 0; p < NACC; ++p) {
            acc0[p] += __shfl_xor_sync(0xffffffffu, acc0[p], 1);
            acc1[p] += __shfl_xor_sync(0xffffffffu, acc1[p], 1);
        }
#pragma unroll
        for (int r = 0; r < M; ++r) {
            facc0[r] += __shfl_xor_sync(0xffffffffu, facc0[r], 1);
            facc1[r] += __shfl_xor_sync(0xffffffffu, facc1[r], 1);
        }
        if (active && !(lane & 1)) {
            const int n0 = nt * 16 + (lq >> 1);
#pragma unroll
            for (int r = 0; r < M; ++r) {
                const int p0 = RESID ? 2 * r : r;
                float o0, o1;
                if constexpr (INT8) {
                    o0 = sh_q[p0] * (k_inv * float(acc0[p0]) + cbias * float(sh_s[p0]));
                    o1 = sh_q[p0] * (k_inv * float(acc1[p0]) + cbias * float(sh_s[p0]));
                    if constexpr (RESID) {
                        o0 += sh_q[p0 + 1] * (k_inv * float(acc0[p0 + 1]) + cbias * float(sh_s[p0 + 1]));
                        o1 += sh_q[p0 + 1] * (k_inv * float(acc1[p0 + 1]) + cbias * float(sh_s[p0 + 1]));
                    }
                } else {
                    o0 = facc0[r];
                    o1 = facc1[r];
                }
                float * part = partials + (size_t(k_block) * M + r) * n;
                part[n0]     = o0;
                part[n0 + 8] = o1;
            }
        }
    } else {
        // pair unit: lane = standard lane t0 = 8*lane for both tiles of the pair; four lanes share
        // each column.  K = 2/3 stream one word per lane per tile through a two-row register ring
        // and resolve the windows with lane shuffles; other K read the windows by pointer (L1).
        constexpr bool REG   = bits == 2 || bits == 3;
        constexpr bool STAGE = stage_smem(bits);         // cp.async pair rows into warp-private smem
        constexpr bool REGW  = bits >= 5 && !STAGE;      // two words per lane per tile
        const int ntA = column_block * 16 + warp * 2;
        const bool active = ntA < n_tiles;   // partial last block: idle warps skip loads and stores
        const uint32_t * bpA = B32 + (size_t(ntA) * kslices + kb0) * TWORDS;
        const uint32_t * bpB = B32 + (size_t(ntA + 1) * kslices + kb0) * TWORDS;
        const int c2 = 2 * (lane & 3);
        const int t0 = lane << 3;
        [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
        if constexpr (bits == 2) {
            const int i1 = lane >> 1;
            x_src_b = i1;
            x_src_a = (i1 + 15) & 15;
        }
        if constexpr (bits == 3) {
            const int b1 = (t0 + 257) * 3;
            const int b2 = b1 + 21;
            const int i0 = (b1 - 16) / 32;
            const int i2 = (b2 - 1) / 32;
            x_s2 = (i2 + 1) * 32 - b2;
            x_src_a = i0 % 24;
            x_src_b = i2 % 24;
        }
        // K = 2: one word per lane covers both tiles (lanes 0..15 tile A, 16..31 tile B);
        // K = 3: lanes 0..23 load word `lane` of tile A and of tile B
        auto load_row = [&](int kb, uint32_t & wa, uint32_t & wb) {
            if constexpr (bits == 2) {
                wa = exl3::load_streaming((lane < 16 ? bpA : bpB) + size_t(kb) * TWORDS + (lane & 15));
                wb = 0;
            } else if constexpr (bits == 3) {
                wa = lane < 24 ? exl3::load_streaming(bpA + size_t(kb) * TWORDS + lane) : 0u;
                wb = lane < 24 ? exl3::load_streaming(bpB + size_t(kb) * TWORDS + lane) : 0u;
            } else {
                wa = 0; wb = 0;
            }
        };
        constexpr int RING = 4;   // rows in flight per lane for the K = 2/3 unit
        [[maybe_unused]] uint32_t pa[RING], pb[RING];
        if constexpr (REG) {
#pragma unroll
            for (int d = 0; d < RING; ++d) { pa[d] = 0; pb[d] = 0; if (active && d < nrows) load_row(d, pa[d], pb[d]); }
        }
        auto load_row2 = [&](int kb, uint2 & wa, uint2 & wb) {
            if (lane < TWORDS / 2) {
                wa = exl3::load_streaming(reinterpret_cast<const uint2 *>(bpA + size_t(kb) * TWORDS + 2 * lane));
                wb = exl3::load_streaming(reinterpret_cast<const uint2 *>(bpB + size_t(kb) * TWORDS + 2 * lane));
            } else {
                wa = make_uint2(0, 0); wb = make_uint2(0, 0);
            }
        };
        [[maybe_unused]] uint2 qa0 = make_uint2(0, 0), qb0 = qa0, qa1 = qa0, qb1 = qa0;
        if constexpr (REGW) {
            if (active && nrows > 0) load_row2(0, qa0, qb0);
            if (active && nrows > 1) load_row2(1, qa1, qb1);
        }
        // smem unit: pair row = [TWORDS words tile A][TWORDS words tile B], 16-byte cp.async chunks
        constexpr int PAIRW = 2 * TWORDS;
        [[maybe_unused]] uint32_t * sb = sh_stage + warp * (STAGE_D * PAIRW);
        auto stage_row = [&](int kb) {
            if constexpr (STAGE) {
                constexpr int CHUNKS = PAIRW / 4;
                if (active && kb < nrows && lane < CHUNKS) {
                    const uint32_t * src = lane < TWORDS / 4 ? bpA + size_t(kb) * TWORDS + 4 * lane
                                                             : bpB + size_t(kb) * TWORDS + 4 * (lane - TWORDS / 4);
                    cp_async16(sb + (kb % STAGE_D) * PAIRW + 4 * lane, src);
                }
                cp_async_commit();
            }
        };
        if constexpr (STAGE) {
#pragma unroll
            for (int r = 0; r < STAGE_D - 1; ++r) stage_row(r);
        }

        int ia0[NACC], ia1[NACC], ib0[NACC], ib1[NACC];
        float fa0[M], fa1[M], fb0[M], fb1[M];
#pragma unroll
        for (int p = 0; p < NACC; ++p) { ia0[p] = 0; ia1[p] = 0; ib0[p] = 0; ib1[p] = 0; }
#pragma unroll
        for (int r = 0; r < M; ++r) { fa0[r] = 0.0f; fa1[r] = 0.0f; fb0[r] = 0.0f; fb1[r] = 0.0f; }
        for (int kb0i = 0; kb0i < (active ? nrows : 0); kb0i += (REG ? RING : 1)) {
#pragma unroll
        for (int d = 0; d < (REG ? RING : 1); ++d) {
            const int kb = kb0i + d;
            if (kb >= nrows) break;
            uint32_t wA[8], wB[8];
            if constexpr (REG) {
                const uint32_t ca = pa[d], word_b = pb[d];
                if (kb + RING < nrows) load_row(kb + RING, pa[d], pb[d]);
                if constexpr (bits == 2) {
                    regs2_windows(__shfl_sync(0xffffffffu, ca, x_src_a), __shfl_sync(0xffffffffu, ca, x_src_b), t0, wA);
                    regs2_windows(__shfl_sync(0xffffffffu, ca, 16 + x_src_a), __shfl_sync(0xffffffffu, ca, 16 + x_src_b), t0, wB);
                } else {
                    regs3_windows(__shfl_sync(0xffffffffu, ca, x_src_a), __shfl_sync(0xffffffffu, ca, x_src_b), x_s2, wA);
                    regs3_windows(__shfl_sync(0xffffffffu, word_b, x_src_a), __shfl_sync(0xffffffffu, word_b, x_src_b), x_s2, wB);
                }
            } else if constexpr (STAGE) {
                cp_async_wait<STAGE_D - 2>();
                __syncwarp();   // also orders last iteration's smem reads before the overwrite
                stage_row(kb + STAGE_D - 1);
                const uint32_t * rowp = sb + (kb % STAGE_D) * PAIRW;
                ext8w<bits>(rowp, t0, wA[0], wA[1], wA[2], wA[3], wA[4], wA[5], wA[6], wA[7]);
                ext8w<bits>(rowp + TWORDS, t0, wB[0], wB[1], wB[2], wB[3], wB[4], wB[5], wB[6], wB[7]);
            } else if constexpr (REGW) {
                const uint2 ca = qa0, word_b = qb0;
                qa0 = qa1; qb0 = qb1;
                if (kb + 2 < nrows) load_row2(kb + 2, qa1, qb1);
                regsw_windows<bits>(ca, t0, wA);
                regsw_windows<bits>(word_b, t0, wB);
            } else {
                ext8w<bits>(bpA + size_t(kb) * TWORDS, t0, wA[0], wA[1], wA[2], wA[3], wA[4], wA[5], wA[6], wA[7]);
                ext8w<bits>(bpB + size_t(kb) * TWORDS, t0, wB[0], wB[1], wB[2], wB[3], wB[4], wB[5], wB[6], wB[7]);
            }
            if constexpr (!INT8) {
                // rows c2+{0,1,8,9}: the same windows for tile A (cols cA, cA+8) and tile B
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    const half * xr = sh_xh + size_t(r) * nrows_max * 16 + (kb << 4) + c2;
                    const half2 x01 = *reinterpret_cast<const half2 *>(xr);
                    const half2 x89 = *reinterpret_cast<const half2 *>(xr + 8);
                    fa0[r] += dot2(exl3_gemv::exl3_decode_pair<cb>(wA[0], wA[1]), x01) + dot2(exl3_gemv::exl3_decode_pair<cb>(wA[2], wA[3]), x89);
                    fa1[r] += dot2(exl3_gemv::exl3_decode_pair<cb>(wA[4], wA[5]), x01) + dot2(exl3_gemv::exl3_decode_pair<cb>(wA[6], wA[7]), x89);
                    fb0[r] += dot2(exl3_gemv::exl3_decode_pair<cb>(wB[0], wB[1]), x01) + dot2(exl3_gemv::exl3_decode_pair<cb>(wB[2], wB[3]), x89);
                    fb1[r] += dot2(exl3_gemv::exl3_decode_pair<cb>(wB[4], wB[5]), x01) + dot2(exl3_gemv::exl3_decode_pair<cb>(wB[6], wB[7]), x89);
                }
                continue;
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) { wA[j] *= 0x83DCD12Du; wB[j] *= 0x83DCD12Du; }
            const uint32_t * as_kb = sh_as + (kb << 4);
#pragma unroll
            for (int p = 0; p < NACC; ++p) {
                const uint32_t * as = as_kb + p * nrows_max * 16;
                const uint2 as01 = *reinterpret_cast<const uint2 *>(as + c2);
                const uint2 as89 = *reinterpret_cast<const uint2 *>(as + c2 + 8);
                int a0 = ia0[p], a1 = ia1[p], b0 = ib0[p], b1 = ib1[p];
                a0 = dp4a_us(wA[0], as01.x, a0); a0 = dp4a_us(wA[1], as01.y, a0); a0 = dp4a_us(wA[2], as89.x, a0); a0 = dp4a_us(wA[3], as89.y, a0);
                a1 = dp4a_us(wA[4], as01.x, a1); a1 = dp4a_us(wA[5], as01.y, a1); a1 = dp4a_us(wA[6], as89.x, a1); a1 = dp4a_us(wA[7], as89.y, a1);
                b0 = dp4a_us(wB[0], as01.x, b0); b0 = dp4a_us(wB[1], as01.y, b0); b0 = dp4a_us(wB[2], as89.x, b0); b0 = dp4a_us(wB[3], as89.y, b0);
                b1 = dp4a_us(wB[4], as01.x, b1); b1 = dp4a_us(wB[5], as01.y, b1); b1 = dp4a_us(wB[6], as89.x, b1); b1 = dp4a_us(wB[7], as89.y, b1);
                ia0[p] = a0; ia1[p] = a1; ib0[p] = b0; ib1[p] = b1;
            }
        }
        }
        // lanes with equal lane/4 share the same columns (col lane/4 and +8 of each tile)
#pragma unroll
        for (int p = 0; p < NACC; ++p) {
            ia0[p] += __shfl_xor_sync(0xffffffffu, ia0[p], 1); ia0[p] += __shfl_xor_sync(0xffffffffu, ia0[p], 2);
            ia1[p] += __shfl_xor_sync(0xffffffffu, ia1[p], 1); ia1[p] += __shfl_xor_sync(0xffffffffu, ia1[p], 2);
            ib0[p] += __shfl_xor_sync(0xffffffffu, ib0[p], 1); ib0[p] += __shfl_xor_sync(0xffffffffu, ib0[p], 2);
            ib1[p] += __shfl_xor_sync(0xffffffffu, ib1[p], 1); ib1[p] += __shfl_xor_sync(0xffffffffu, ib1[p], 2);
        }
#pragma unroll
        for (int r = 0; r < M; ++r) {
            fa0[r] += __shfl_xor_sync(0xffffffffu, fa0[r], 1); fa0[r] += __shfl_xor_sync(0xffffffffu, fa0[r], 2);
            fa1[r] += __shfl_xor_sync(0xffffffffu, fa1[r], 1); fa1[r] += __shfl_xor_sync(0xffffffffu, fa1[r], 2);
            fb0[r] += __shfl_xor_sync(0xffffffffu, fb0[r], 1); fb0[r] += __shfl_xor_sync(0xffffffffu, fb0[r], 2);
            fb1[r] += __shfl_xor_sync(0xffffffffu, fb1[r], 1); fb1[r] += __shfl_xor_sync(0xffffffffu, fb1[r], 2);
        }
        if (active && !(lane & 3)) {
            const int cA = ntA * 16 + (lane >> 2);
            const int cB = cA + 16;
#pragma unroll
            for (int r = 0; r < M; ++r) {
                const int p0 = RESID ? 2 * r : r;
                auto fold = [&](int a, int p) { return sh_q[p] * (k_inv * float(a) + cbias * float(sh_s[p])); };
                float oa0, oa1, ob0, ob1;
                if constexpr (INT8) {
                    oa0 = fold(ia0[p0], p0); oa1 = fold(ia1[p0], p0); ob0 = fold(ib0[p0], p0); ob1 = fold(ib1[p0], p0);
                    if constexpr (RESID) {
                        oa0 += fold(ia0[p0 + 1], p0 + 1); oa1 += fold(ia1[p0 + 1], p0 + 1);
                        ob0 += fold(ib0[p0 + 1], p0 + 1); ob1 += fold(ib1[p0 + 1], p0 + 1);
                    }
                } else {
                    oa0 = fa0[r]; oa1 = fa1[r]; ob0 = fb0[r]; ob1 = fb1[r];
                }
                float * part = partials + (size_t(k_block) * M + r) * n;
                part[cA] = oa0; part[cA + 8] = oa1;
                part[cB] = ob0; part[cB + 8] = ob1;
            }
        }
    }

    // last k-split block for these 256 columns reduces the partials in slice order
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        sh_last = atomicAdd(counters + column_block, 1) == int(k_blocks) - 1;
    }
    __syncthreads();
    if (!sh_last) return;
    __threadfence();

    const int col = column_block * COLS + threadIdx.x;
#pragma unroll
    for (int r = 0; r < M; ++r) {
        float v = 0.0f;
        if (col < n) {
            for (int sl = 0; sl < int(k_blocks); ++sl) {
#if defined(GGML_USE_HIP)
                // Read other blocks' published partials through an agent-scope load.
                v += __hip_atomic_load(partials + (size_t(sl) * M + r) * n + col,
                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
                v += __ldcg(partials + (size_t(sl) * M + r) * n + col);
#endif
            }
        }
        sh_y[r][threadIdx.x] = v;
    }
    if (threadIdx.x == 0) counters[column_block] = 0;
    __syncthreads();
    // output Hadamard: 2 x 128-blocks per row, one warp each
    for (int b = warp; b < 2 * M; b += THREADS / 32) {
        const int r = b >> 1;
        const int c = (b & 1) * 128 + lane * 4;
        if (column_block * COLS + (b & 1) * 128 >= n) continue;   // partial block: second 128-half absent
        float v0 = sh_y[r][c], v1 = sh_y[r][c + 1], v2 = sh_y[r][c + 2], v3 = sh_y[r][c + 3];
        exl3_had::had128(v0, v1, v2, v3, lane);
        const int gc = column_block * COLS + c;
        const half2 s01 = *reinterpret_cast<const half2 *>(svh + gc);
        const half2 s23 = *reinterpret_cast<const half2 *>(svh + gc + 2);
        float4 o;
        o.x = v0 * exl3_had::SCALE * __low2float(s01);
        o.y = v1 * exl3_had::SCALE * __high2float(s01);
        o.z = v2 * exl3_had::SCALE * __low2float(s23);
        o.w = v3 * exl3_had::SCALE * __high2float(s23);
        *reinterpret_cast<float4 *>(y + size_t(r) * n + gc) = o;
    }
}

} // namespace exl3_int8
