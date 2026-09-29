#pragma once

// SM86 M8/M16 executor: original INT8 quantization, expressed as exact integer
// products on F16/F32 MMA. K groups and their floating reduction order remain
// identical to the scalar path, including the vocabulary head's residual.
namespace exl3_int8 {

struct warpk_pair {
    const uint8_t * weights;
    const half * suh;
    uint8_t * prepared;
    float * output;
    int n, rows, splits;
};

// One warp owns one token row. Its max and integer sum need no cross-warp
// reduction. Prepared uint2 fragments give the consumer coalesced MMA-B loads.
template <bool RESID, bool PAIR, int M = 8>
__global__ __launch_bounds__(M * 32) void prepare_warpk(const float * x, const half * suh,
        uint8_t * prepared, int k, int nrows_max, int m, warpk_pair pair) {
    constexpr int NACC = M * (RESID ? 2 : 1), PLANES = NACC / 8;
    const int row = threadIdx.x / 32, lane = threadIdx.x % 32;
    if constexpr (PAIR) {
        if (blockIdx.z) { suh = pair.suh; prepared = pair.prepared; nrows_max = pair.rows; }
    }
    const int kb0 = blockIdx.y * nrows_max;
    if (kb0 >= k / 16) return;
    const int nrows = min(nrows_max, k / 16 - kb0), kn = nrows * 16;
    prepared += size_t(blockIdx.y) * (8 * NACC + NACC * nrows_max * 32);
    extern __shared__ uint32_t scratch[];
    auto * quant = reinterpret_cast<int8_t *>(scratch);
    auto * xh = reinterpret_cast<half *>(quant + NACC * nrows_max * 16);
    float maximum = 0;
    for (int i = lane * 4; i < kn; i += 128) {
        const int col = kb0 * 16 + i;
        const float4 xv = row < m ? *reinterpret_cast<const float4 *>(x + size_t(row) * k + col) : make_float4(0, 0, 0, 0);
        const half2 s01 = *reinterpret_cast<const half2 *>(suh + col);
        const half2 s23 = *reinterpret_cast<const half2 *>(suh + col + 2);
        float v0 = xv.x * __low2float(s01), v1 = xv.y * __high2float(s01);
        float v2 = xv.z * __low2float(s23), v3 = xv.w * __high2float(s23);
        exl3_had::had128(v0, v1, v2, v3, lane);
        const half2 h01 = __floats2half2_rn(v0 * exl3_had::SCALE, v1 * exl3_had::SCALE);
        const half2 h23 = __floats2half2_rn(v2 * exl3_had::SCALE, v3 * exl3_had::SCALE);
        half * dst = xh + size_t(row) * nrows_max * 16 + i;
        *reinterpret_cast<half2 *>(dst) = h01;
        *reinterpret_cast<half2 *>(dst + 2) = h23;
        maximum = fmaxf(maximum, fmaxf(fmaxf(fabsf(__low2float(h01)), fabsf(__high2float(h01))),
                                      fmaxf(fabsf(__low2float(h23)), fabsf(__high2float(h23)))));
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, d));
    const float q = fmaxf(maximum, 1e-30f) / 127.f, q2 = q / 254.f;
    const int p0 = RESID ? 2 * row : row;
    int sum = 0, sum2 = 0;
    __syncwarp();
    auto offset = [&](int p, int i) { return p * nrows_max * 16 + (i ^ ((p & 7) * 4)); };
    // The XOR swizzle preserves each four-byte group. Quantize the same
    // scalars with the same divisions, then issue aligned word stores.
    for (int i = lane * 4; i < kn; i += 128) {
        union { uint2 words; uint16_t halves[4]; } values;
        values.words = *reinterpret_cast<const uint2 *>(xh + size_t(row) * nrows_max * 16 + i);
        uint32_t packed = 0, packed2 = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float a = __half2float(__ushort_as_half(values.halves[t]));
            const int v = max(-127, min(127, __float2int_rn(a / q)));
            packed |= uint32_t(uint8_t(v)) << (8*t);
            sum += v;
            if constexpr (RESID) {
                const float rr = a - q * float(v);
                const int v2 = max(-127, min(127, __float2int_rn(rr / q2)));
                packed2 |= uint32_t(uint8_t(v2)) << (8*t);
                sum2 += v2;
            }
        }
        *reinterpret_cast<uint32_t *>(quant + offset(p0, i)) = packed;
        if constexpr (RESID) *reinterpret_cast<uint32_t *>(quant + offset(p0+1, i)) = packed2;
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, d);
        if constexpr (RESID) sum2 += __shfl_xor_sync(0xffffffffu, sum2, d);
    }
    if (!lane) {
        reinterpret_cast<float *>(prepared)[p0] = q;
        reinterpret_cast<int *>(prepared)[NACC + p0] = sum;
        if constexpr (RESID) {
            reinterpret_cast<float *>(prepared)[p0 + 1] = q2;
            reinterpret_cast<int *>(prepared)[NACC + p0 + 1] = sum2;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nrows_max * 32 * PLANES; i += M * 32) {
        const int kb = i / (32 * PLANES), p = ((i / 32) % PLANES) * 8 + (i % 32) / 4, j = (i % 4) * 2;
        auto value = [&](int t) { return __int2half_rn(kb < nrows ? int(quant[offset(p, kb * 16 + j + t)]) : 0); };
        reinterpret_cast<half2 *>(prepared + 8 * NACC)[2 * i] = __halves2half2(value(0), value(1));
        reinterpret_cast<half2 *>(prepared + 8 * NACC)[2 * i + 1] = __halves2half2(value(8), value(9));
    }
}

template <int WK, int BITS, bool RESID, bool PAIR, int M = 8, bool COMPACT = false>
__device__ __forceinline__ void gemv_warpk_impl(const uint8_t * weights,
        const uint8_t * prepared, float * output, int k, int n, int nrows_max, int ksplit, int m, warpk_pair pair) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == 860
    constexpr int COLS = 32, RING = 4, NACC = M * (RESID ? 2 : 1), PLANES = NACC / 8, TWORDS = BITS * 8;
    constexpr bool HOIST_FOLD = BITS == 4 && M == 8 && WK == 16;
    static_assert(M == 8 || M == 16);
    static_assert(BITS == 4 || BITS == 6);
    if constexpr (PAIR) {
        if (blockIdx.z) {
            weights = pair.weights; prepared = pair.prepared; output = pair.output;
            n = pair.n; nrows_max = pair.rows; ksplit = pair.splits;
        }
    }
    if (blockIdx.x >= n / 32) return;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nt = blockIdx.x * 2, kslices = k / 16;
    extern __shared__ float partial[];
    __shared__ __align__(16) uint32_t stage[BITS == 6 ? WK : 1][BITS == 6 ? 4 * TWORDS : 1];
    const uint32_t * b = reinterpret_cast<const uint32_t *>(weights);
    const float inv = __half2float(__ushort_as_half(0x1eee));
    const float bias = 1024.f * inv + __half2float(__ushort_as_half(0xc931));
    auto centered_pair = [](uint32_t w0, uint32_t w1) {
        const uint32_t s0 = exl3::byte_sum(w0 * 0x83DCD12Du, 0x6400u);
        const uint32_t s1 = exl3::byte_sum(w1 * 0x83DCD12Du, 0x6400u);
        union { half2 h; uint32_t u; } v;
        v.h = __hsub2(__halves2half2(__ushort_as_half(uint16_t(s0)), __ushort_as_half(uint16_t(s1))), __float2half2_rn(1536.f));
        return v.u;
    };
    for (int group = warp; group < ksplit; group += WK) {
        const int kb0 = group * nrows_max, nr = min(nrows_max, kslices - kb0);
        const uint8_t * prep = prepared + size_t(group) * (8 * NACC + NACC * nrows_max * 32);
        const float * scales = reinterpret_cast<const float *>(prep);
        const int * sums = reinterpret_cast<const int *>(prep) + NACC;
        const uint2 * x = reinterpret_cast<const uint2 *>(prep + 8 * NACC);
        float accum[2][PLANES][4] = {};
        int carry[2][PLANES][4] = {};
        uint32_t ring[2][RING];
        auto load = [&](int tile, int kb) {
            return kb < nr ? exl3::load_streaming(b + (size_t(nt + tile) * kslices + kb0 + kb) * TWORDS + lane) : 0u;
        };
        auto prefetch = [&](int kb) {
            if constexpr (BITS == 6) {
                if (kb < nr && lane < 2 * TWORDS / 4) {
                    const int t = lane / (TWORDS / 4), w = (lane % (TWORDS / 4)) * 4;
                    void * dst = stage[warp] + (kb % 2) * 2 * TWORDS + lane * 4;
                    const void * src = b + (size_t(nt + t) * kslices + kb0 + kb) * TWORDS + w;
                    if constexpr (M == 8 && WK == 3) {
                        // Large-vocabulary head: fetch neighboring packed weights
                        // into L2 while copying this lane's unchanged 16 bytes.
                        const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
                        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                                     :: "r"(smem), "l"(src));
                    } else {
                        cp_async16(dst, src);
                    }
                }
                cp_async_commit();
            }
        };
        if constexpr (BITS == 4) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
#pragma unroll
                for (int j = 0; j < RING; ++j) ring[t][j] = load(t, j);
            }
        } else prefetch(0);
        for (int base = 0; base < nr; base += RING) {
#pragma unroll
            for (int j = 0; j < RING; ++j) {
                const int kb = base + j;
                if (kb >= nr) break;
                if constexpr (BITS == 6) {
                    cp_async_wait<0>(); __syncwarp(); prefetch(kb + 1);
                }
                uint2 bv[PLANES];
#pragma unroll
                for (int p = 0; p < PLANES; ++p) bv[p] = __ldg(x + (kb * PLANES + p) * 32 + lane);
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
                    if constexpr (BITS == 4) {
                        const uint32_t word = ring[t][j];
                        ring[t][j] = load(t, kb + RING);
                        const uint32_t previous = __shfl_sync(0xffffffffu, word, (lane + 31) & 31);
                        extract8_4bits(previous, word, w0, w1, w2, w3, w4, w5, w6, w7);
                    } else {
                        ext8w<6>(stage[warp] + (kb % 2) * 2 * TWORDS + t * TWORDS, lane * 8, w0, w1, w2, w3, w4, w5, w6, w7);
                    }
                    const uint32_t a0 = centered_pair(w0, w1), a1 = centered_pair(w4, w5);
                    const uint32_t a2 = centered_pair(w2, w3), a3 = centered_pair(w6, w7);
#pragma unroll
                    for (int p = 0; p < PLANES; ++p) {
                        float * a = accum[t][p];
                        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+f"(a[0]), "+f"(a[1]), "+f"(a[2]), "+f"(a[3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bv[p].x), "r"(bv[p].y));
                        // Centered weights are [-512,508], activations [-127,127].
                        // 256 products stay exact: 512*127*256 < 2^24. Fold before
                        // that bound is exceeded; the remaining sum is integer.
                        if (!HOIST_FOLD && ((kb & 15) == 15 || kb + 1 == nr)) {
#pragma unroll
                            for (int l = 0; l < 4; ++l) { carry[t][p][l] += int(a[l]); a[l] = 0; }
                        }
                    }
                }
                // Hoist folding across independent tiles for this many-warp
                // geometry. Wide projections prefer the interleaved schedule.
                // Each accumulator still folds at the identical K positions.
                if (HOIST_FOLD && ((kb & 15) == 15 || kb + 1 == nr)) {
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
#pragma unroll
                        for (int p = 0; p < PLANES; ++p) {
#pragma unroll
                            for (int l = 0; l < 4; ++l) {
                                carry[t][p][l] += int(accum[t][p][l]);
                                accum[t][p][l] = 0;
                            }
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int plane = 0; plane < PLANES; ++plane) {
#pragma unroll
                for (int l = 0; l < 4; l += (RESID ? 2 : 1)) {
                    const int p = plane * 8 + (lane % 4) * 2 + l % 2, c = t * 16 + lane / 4 + (l / 2) * 8;
                    const int total = carry[t][plane][l] + 512 * sums[p];
                    float v = scales[p] * (inv * float(total) + bias * float(sums[p]));
                    if constexpr (RESID) {
                        const int residual = carry[t][plane][l + 1] + 512 * sums[p + 1];
                        v += scales[p + 1] * (inv * float(residual) + bias * float(sums[p + 1]));
                    }
                    const int row = RESID ? p / 2 : p;
                    if (!COMPACT || row < m) {
                        partial[(group * (COMPACT ? m : M) + row) * COLS + c] = v;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < m * COLS; i += blockDim.x) {
        float v = 0;
        for (int g = 0; g < ksplit; ++g) v = __fadd_rn(v, partial[g * (COMPACT ? m : M) * COLS + i]);
        output[size_t(i / COLS) * n + blockIdx.x * COLS + i % COLS] = v;
    }
#else
    NO_DEVICE_CODE;
#endif
}

// Keep the original one-argument launch bound: an explicit minimum of one
// changes register allocation for unrelated specializations on SM86.
template <int WK, int BITS, bool RESID, bool PAIR, int M = 8>
__global__ __launch_bounds__(WK * 32) void gemv_warpk(const uint8_t * weights,
        const uint8_t * prepared, float * output, int k, int n, int nrows_max, int ksplit, int m, warpk_pair pair) {
    gemv_warpk_impl<WK, BITS, RESID, PAIR, M>(weights, prepared, output, k, n, nrows_max, ksplit, m, pair);
}

template <bool PAIR>
__global__ __launch_bounds__(512, 2) void gemv_warpk_m16(const uint8_t * weights,
        const uint8_t * prepared, float * output, int k, int n, int nrows_max, int ksplit, int m, warpk_pair pair) {
    gemv_warpk_impl<16, 4, false, PAIR, 16>(weights, prepared, output, k, n, nrows_max, ksplit, m, pair);
}

template <bool PAIR>
__global__ __launch_bounds__(512, 2) void gemv_warpk_m16_compact(const uint8_t * weights,
        const uint8_t * prepared, float * output, int k, int n, int nrows_max, int ksplit, int m, warpk_pair pair) {
    gemv_warpk_impl<16, 4, false, PAIR, 16, true>(weights, prepared, output, k, n, nrows_max, ksplit, m, pair);
}

} // namespace exl3_int8
