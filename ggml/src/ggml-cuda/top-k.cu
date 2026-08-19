#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

// Ordered warp top-k for small power-of-two rows. Older CUB versions do not
// provide DeviceTopK and otherwise fall back to sorting every element.
template <int ncols>
static __global__ void top_k_f32_i32_cuda(const float * src, int * dst, const int nrows, const int k) {
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= nrows) {
        return;
    }

    constexpr int values_per_thread = ncols > WARP_SIZE ? ncols / WARP_SIZE : 1;
    float values[values_per_thread];
    bool  active[values_per_thread];

#pragma unroll
    for (int i = 0; i < values_per_thread; ++i) {
        const int col = lane + i * WARP_SIZE;
        active[i] = col < ncols;
        values[i] = active[i] ? src[row * ncols + col] : -INFINITY;
        if (__isnanf(values[i])) {
            values[i] = -FLT_MAX;
        }
    }

    for (int rank = 0; rank < k; ++rank) {
        float max_val = -INFINITY;
        int   max_col = INT_MAX;

#pragma unroll
        for (int i = 0; i < values_per_thread; ++i) {
            const int col = lane + i * WARP_SIZE;
            if (active[i] && (max_col == INT_MAX || values[i] > max_val ||
                    (values[i] == max_val && col < max_col))) {
                max_val = values[i];
                max_col = col;
            }
        }

#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   other_col = __shfl_xor_sync(0xFFFFFFFF, max_col, mask, WARP_SIZE);
            if (other_col != INT_MAX && (max_col == INT_MAX || other_val > max_val ||
                    (other_val == max_val && other_col < max_col))) {
                max_val = other_val;
                max_col = other_col;
            }
        }

        if (lane == 0) {
            dst[row * k + rank] = max_col;
        }
        if (max_col % WARP_SIZE == lane) {
            active[max_col / WARP_SIZE] = false;
        }
    }
}

static bool top_k_f32_i32_cuda_ordered(
        const float * src, int * dst, const int ncols, const int nrows, const int k, cudaStream_t stream) {
    if (k > WARP_SIZE || k > ncols) {
        return false;
    }

    constexpr int rows_per_block = 4;
    const dim3 block_dims(WARP_SIZE, rows_per_block, 1);
    const dim3 grid_dims((nrows + rows_per_block - 1) / rows_per_block, 1, 1);

#define GGML_CUDA_TOP_K_CASE(N) \
    case N: \
        top_k_f32_i32_cuda<N><<<grid_dims, block_dims, 0, stream>>>(src, dst, nrows, k); \
        return true

    switch (ncols) {
        GGML_CUDA_TOP_K_CASE(1);
        GGML_CUDA_TOP_K_CASE(2);
        GGML_CUDA_TOP_K_CASE(4);
        GGML_CUDA_TOP_K_CASE(8);
        GGML_CUDA_TOP_K_CASE(16);
        GGML_CUDA_TOP_K_CASE(32);
        GGML_CUDA_TOP_K_CASE(64);
        GGML_CUDA_TOP_K_CASE(128);
        GGML_CUDA_TOP_K_CASE(256);
        GGML_CUDA_TOP_K_CASE(512);
        default: return false;
    }

#undef GGML_CUDA_TOP_K_CASE
}

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#else  // CUB_TOP_K_AVAILABLE
    if (top_k_f32_i32_cuda_ordered(src0_d, dst_d, ncols, nrows, k, stream)) {
        return;
    }
#if defined(GGML_CUDA_USE_CUB)
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
#endif  // CUB_TOP_K_AVAILABLE
}
