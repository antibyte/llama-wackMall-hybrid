#pragma once
#include <mma.h>

// Compressed-weight GEMM for bounded dense prefill tiles. Decode into shared
// memory, reusing each weight across BM tokens without a full F16 reconstruction.
// The caller owns both Hadamard transforms. F16 inputs/weights and F32 accumulation
// match the cuBLAS path's precision, but not its floating-point reduction order.
// SM86 measurements favor BM=64 for short prompts, BM=128 for longer prompts.
template <int bits, int cb, int BM, int min_blocks>
__global__ void __launch_bounds__(256, min_blocks)
exl3_gemm_kernel(const uint8_t *weights, const half *xh, float *y, int k, int n, int m) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BN = 64, BK = BM <= 64 ? 128 : 64, STRIDE = BK + 8;
    static_assert(BM == 32 || BM == 64 || BM == 128);
    constexpr int MV = BM / 32;
    __shared__ __align__(32) half a[BN * STRIDE];
    __shared__ __align__(32) half b[BM * STRIDE];
    // The final K-loop barrier retires all readers of a before the epilogue.
    // Reuse that tile for warp-private output staging, saving 8 KiB per CTA.
    static_assert(sizeof(a) >= 8 * 256 * sizeof(float));
    float *c = reinterpret_cast<float *>(a);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[MV];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MV];
#pragma unroll
    for (int v = 0; v < MV; ++v) {
        wmma::fill_fragment(acc[v], 0.f);
    }
    for (int k0 = 0; k0 < k; k0 += BK) {
        for (int tile = warp; tile < (BN / 16) * (BK / 16); tile += 8) {
            const int nt = tile / (BK / 16), kt = tile % (BK / 16);
            const auto *tp = reinterpret_cast<const uint32_t *>(
                weights + (size_t(n0 / 16 + nt) * (k / 16) + (k0 / 16 + kt)) * 32 * bits);
            uint32_t w[8];
            exl3_int8::ext8w<bits>(tp, lane * 8, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
            const int row = (lane & 3) * 2, col = lane / 4;
            half *dst = a + (nt * 16 + col) * STRIDE + kt * 16 + row;
            *reinterpret_cast<half2 *>(dst) = exl3_gemv::exl3_decode_pair<cb>(w[0], w[1]);
            *reinterpret_cast<half2 *>(dst + 8) = exl3_gemv::exl3_decode_pair<cb>(w[2], w[3]);
            *reinterpret_cast<half2 *>(dst + 8 * STRIDE) = exl3_gemv::exl3_decode_pair<cb>(w[4], w[5]);
            *reinterpret_cast<half2 *>(dst + 8 * STRIDE + 8) = exl3_gemv::exl3_decode_pair<cb>(w[6], w[7]);
        }
        for (int i = threadIdx.x; i < BM * BK / 8; i += 256) {
            const int row = i / (BK / 8), col = i % (BK / 8) * 8;
            *reinterpret_cast<uint4 *>(b + row * STRIDE + col) =
                m0 + row < m ? *reinterpret_cast<const uint4 *>(xh + size_t(m0 + row) * k + k0 + col)
                             : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::load_matrix_sync(fa, a + (warp / 2) * 16 * STRIDE + kk, STRIDE);
#pragma unroll
            for (int v = 0; v < MV; ++v) {
                wmma::load_matrix_sync(fb[v], b + ((warp % 2) * MV + v) * 16 * STRIDE + kk, STRIDE);
            }
#pragma unroll
            for (int v = 0; v < MV; ++v) {
                wmma::mma_sync(acc[v], fa, fb[v], acc[v]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < MV; ++v) {
        wmma::store_matrix_sync(c + warp * 256, acc[v], 16, wmma::mem_col_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            const int row = n0 + (warp / 2) * 16 + i % 16, token = m0 + ((warp % 2) * MV + v) * 16 + i / 16;
            if (row < n && token < m) {
                y[size_t(token) * n + row] = c[warp * 256 + i];
            }
        }
        __syncwarp();
    }
#endif
}
