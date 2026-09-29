#pragma once

#include "exl3-dq.cuh"

namespace exl3_head {

// Six-bit mul1 vocabulary projection. Sixteen fixed K partitions and an
// ordered F32 reduction give each row the same arithmetic at M=1..13.
// Two eight-row planes reuse decoded weights during wider verification.
template <int PLANES>
__global__ void __launch_bounds__(512, 2) project(const half * __restrict__ A,
        const uint8_t * __restrict__ B, float * __restrict__ C, int m, int k, int n) {
#if __CUDA_ARCH__ >= 800
    constexpr int WK = 16, WNT = 2, PF = 4;
    constexpr int COLS = 32, ROWS = 8 * PLANES, TWORDS = 48;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int kslices = k / 16;
    const int chunk = (kslices + WK - 1) / WK;
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const auto * B32 = reinterpret_cast<const uint32_t *>(B);
    const auto * A2 = reinterpret_cast<const half2 *>(A);
    const half2 zero = __half2half2(__ushort_as_half(0));
    const int r0 = lane >> 2;
    __shared__ float red[WK][ROWS][COLS];
    __shared__ uint32_t stage[WK][2 * WNT * TWORDS];

    for (int group = blockIdx.x; group < n / COLS; group += gridDim.x) {
        const uint32_t * bg = B32 + size_t(group * WNT) * kslices * TWORDS;
        auto tile_ptr = [&](int t, int kt) { return bg + (size_t(t) * kslices + kt) * TWORDS; };
        auto stage_weight = [&](int i) {
            if (i < myn) {
                for (int l = lane; l < WNT * TWORDS / 4; l += 32) {
                    const int t = l / (TWORDS / 4), word = l % (TWORDS / 4) * 4;
                    const auto * src = tile_ptr(t, ks0 + i) + word;
                    auto * dst = stage[warp] + (i % 2) * WNT * TWORDS + l * 4;
                    const unsigned smem = unsigned(__cvta_generic_to_shared(dst));
                    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(smem), "l"(src));
                }
            }
            asm volatile("cp.async.commit_group;");
        };
        stage_weight(0);
        float acc[WNT][PLANES][4] = {};
        for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
            for (int d = 0; d < PF; ++d) {
                const int i = ib + d;
                if (i >= myn) break;
                const int kt = ks0 + i;
                asm volatile("cp.async.wait_group 0;");
                __syncwarp();
                stage_weight(i + 1);

                const size_t a_col = size_t(kt) * 8 + (lane & 3);
                half2 a0[PLANES], a1[PLANES];
#pragma unroll
                for (int p = 0; p < PLANES; ++p) {
                    const int row = r0 + 8 * p;
                    const size_t offset = size_t(row) * (k / 2) + a_col;
                    a0[p] = row < m ? A2[offset] : zero;
                    a1[p] = row < m ? A2[offset + 4] : zero;
                }
#pragma unroll
                for (int t = 0; t < WNT; ++t) {
                    exl3::FragB f0, f1;
                    exl3::dq_dispatch<6, 2>(&stage[warp][(i % 2) * WNT * TWORDS + t * TWORDS], lane * 8, f0, f1);
                    const auto * w0 = reinterpret_cast<const uint32_t *>(&f0);
                    const auto * w1 = reinterpret_cast<const uint32_t *>(&f1);
#pragma unroll
                    for (int p = 0; p < PLANES; ++p) {
                        const uint32_t b0 = *reinterpret_cast<const uint32_t *>(&a0[p]);
                        const uint32_t b1 = *reinterpret_cast<const uint32_t *>(&a1[p]);
                        float * sum = acc[t][p];
                        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+f"(sum[0]), "+f"(sum[1]), "+f"(sum[2]), "+f"(sum[3])
                            : "r"(w0[0]), "r"(w1[0]), "r"(w0[1]), "r"(w1[1]), "r"(b0), "r"(b1));
                    }
                }
            }
        }
        // MMA output is transposed: adjacent registers hold adjacent tokens.
#pragma unroll
        for (int t = 0; t < WNT; ++t) {
#pragma unroll
            for (int p = 0; p < PLANES; ++p) {
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    const int row = 8 * p + 2 * (lane % 4) + f % 2;
                    const int col = t * 16 + lane / 4 + (f / 2) * 8;
                    red[warp][row][col] = acc[t][p][f];
                }
            }
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < COLS * min(m, ROWS); idx += WK * 32) {
            const int row = idx / COLS, col = idx % COLS;
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < WK; ++j) sum += red[j][row][col];
            C[size_t(row) * n + group * COLS + col] = sum;
        }
        __syncthreads();
    }
#else
    // The host dispatcher only admits SM86.
    (void) A; (void) B; (void) C; (void) m; (void) k; (void) n;
    __trap();
#endif
}

} // namespace exl3_head
