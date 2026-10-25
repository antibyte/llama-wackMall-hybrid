#include "convert.cuh"
#include "dequantize.cuh"
#include "turbo4-k.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
            y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

#ifdef GGML_CUDA_TURBO4_F16_PREFILL
static int ggml_cuda_turbo4_fast_f16_convert_mode() {
    static const int mode = []() {
        const char * value = std::getenv("GGML_CUDA_TURBO4_FAST_F16_CONVERT");
        if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
            return 0;
        }
        if (std::strcmp(value, "1") == 0) {
            return 1;
        }
        if (std::strcmp(value, "2") == 0) {
            return 2;
        }
        {
            GGML_LOG_WARN(
                "invalid GGML_CUDA_TURBO4_FAST_F16_CONVERT='%s'; using 0\n", value);
            return 0;
        }
    }();
    return mode;
}

template <bool need_check>
static __global__ void dequantize_block_turbo4_f16_fast(
        const block_turbo4_k * __restrict__ x,
        half * __restrict__ y,
        const int64_t k) {
    constexpr int warps_per_block = CUDA_DEQUANTIZE_BLOCK_SIZE / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t ib = int64_t(blockIdx.x)*warps_per_block + warp;
    const int64_t i0 = ib*QK_TURBO4_K;

    if (need_check && i0 >= k) {
        return;
    }

    const float norm = __shfl_sync(0xffffffff, lane == 0 ? __half2float(x[ib].norm) : 0.0f, 0);
    const uint16_t packed = *reinterpret_cast<const uint16_t *>(x[ib].qs + 2*lane);
    const uint8_t packed0 = packed & 0xff;
    const uint8_t packed1 = packed >> 8;

    half2 * y2 = reinterpret_cast<half2 *>(y + i0);
    y2[2*lane + 0] = __float22half2_rn(make_float2(
        norm*turbo4_k_centroids_cuda[ packed0       & 0x0f],
        norm*turbo4_k_centroids_cuda[(packed0 >> 4) & 0x0f]));
    y2[2*lane + 1] = __float22half2_rn(make_float2(
        norm*turbo4_k_centroids_cuda[ packed1       & 0x0f],
        norm*turbo4_k_centroids_cuda[(packed1 >> 4) & 0x0f]));
}

static void dequantize_block_turbo4_f16_fast_cuda(
        const void * __restrict__ vx,
        half * __restrict__ y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_TURBO4_K == 0);
    constexpr int warps_per_block = CUDA_DEQUANTIZE_BLOCK_SIZE / WARP_SIZE;
    const int64_t n_quant_blocks = k / QK_TURBO4_K;
    const int num_blocks = (n_quant_blocks + warps_per_block - 1) / warps_per_block;

    if (n_quant_blocks % warps_per_block == 0) {
        dequantize_block_turbo4_f16_fast<false><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(
            static_cast<const block_turbo4_k *>(vx), y, k);
    } else {
        dequantize_block_turbo4_f16_fast<true><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(
            static_cast<const block_turbo4_k *>(vx), y, k);
    }
}

static __global__ void dequantize_block_turbo4_f16_fast_nc(
        const block_turbo4_k * __restrict__ x,
        half * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03) {
    constexpr int warps_per_block = 2;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t blocks_per_row = ne00 / QK_TURBO4_K;
    const int64_t ib_row = int64_t(blockIdx.x)*warps_per_block + warp;

    if (ib_row >= blocks_per_row) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t) i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;
            const int64_t ib = i03*s03 + i02*s02 + i01*s01 + ib_row;
            const int64_t i0 = ((i0203*ne01 + i01)*ne00) + ib_row*QK_TURBO4_K;

            const float norm = __shfl_sync(0xffffffff, lane == 0 ? __half2float(x[ib].norm) : 0.0f, 0);
            const uint16_t packed = *reinterpret_cast<const uint16_t *>(x[ib].qs + 2*lane);
            const uint8_t packed0 = packed & 0xff;
            const uint8_t packed1 = packed >> 8;
            half2 * y2 = reinterpret_cast<half2 *>(y + i0);

            y2[2*lane + 0] = __float22half2_rn(make_float2(
                norm*turbo4_k_centroids_cuda[ packed0       & 0x0f],
                norm*turbo4_k_centroids_cuda[(packed0 >> 4) & 0x0f]));
            y2[2*lane + 1] = __float22half2_rn(make_float2(
                norm*turbo4_k_centroids_cuda[ packed1       & 0x0f],
                norm*turbo4_k_centroids_cuda[(packed1 >> 4) & 0x0f]));
        }
    }
}

static void dequantize_block_turbo4_f16_fast_nc_cuda(
        const void * x,
        half * y,
        int64_t ne00,
        int64_t ne01,
        int64_t ne02,
        int64_t ne03,
        int64_t s01,
        int64_t s02,
        int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TURBO4_K == 0);
    constexpr int warps_per_block = 2;
    const int64_t blocks_per_row = ne00 / QK_TURBO4_K;
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks(
        (blocks_per_row + warps_per_block - 1) / warps_per_block,
        (int) std::min(ne01, (int64_t) 65535),
        (int) std::min(ne0203, (int64_t) 65535));
    dequantize_block_turbo4_f16_fast_nc<<<num_blocks, warps_per_block*WARP_SIZE, 0, stream>>>(
        static_cast<const block_turbo4_k *>(x), y,
        ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <bool need_check>
static __global__ void dequantize_block_turbo4_f16_2warp(
        const block_turbo4_k * __restrict__ x,
        half * __restrict__ y,
        const int64_t k) {
    constexpr int threads_per_quant_block = 2*WARP_SIZE;
    constexpr int quant_blocks_per_cuda_block = CUDA_DEQUANTIZE_BLOCK_SIZE / threads_per_quant_block;
    const int local_quant_block = threadIdx.x / threads_per_quant_block;
    const int byte = threadIdx.x % threads_per_quant_block;
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t ib = int64_t(blockIdx.x)*quant_blocks_per_cuda_block + local_quant_block;
    const int64_t i0 = ib*QK_TURBO4_K;

    if (need_check && i0 >= k) {
        return;
    }

    const float norm = __shfl_sync(0xffffffff, lane == 0 ? __half2float(x[ib].norm) : 0.0f, 0);
    const uint8_t packed = x[ib].qs[byte];
    reinterpret_cast<half2 *>(y + i0)[byte] = __float22half2_rn(make_float2(
        norm*turbo4_k_centroids_cuda[ packed       & 0x0f],
        norm*turbo4_k_centroids_cuda[(packed >> 4) & 0x0f]));
}

static void dequantize_block_turbo4_f16_2warp_cuda(
        const void * __restrict__ vx,
        half * __restrict__ y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_TURBO4_K == 0);
    constexpr int quant_blocks_per_cuda_block = CUDA_DEQUANTIZE_BLOCK_SIZE / (2*WARP_SIZE);
    const int64_t n_quant_blocks = k / QK_TURBO4_K;
    const int num_blocks = (n_quant_blocks + quant_blocks_per_cuda_block - 1) / quant_blocks_per_cuda_block;

    if (n_quant_blocks % quant_blocks_per_cuda_block == 0) {
        dequantize_block_turbo4_f16_2warp<false><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(
            static_cast<const block_turbo4_k *>(vx), y, k);
    } else {
        dequantize_block_turbo4_f16_2warp<true><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(
            static_cast<const block_turbo4_k *>(vx), y, k);
    }
}

static __global__ void dequantize_block_turbo4_f16_2warp_nc(
        const block_turbo4_k * __restrict__ x,
        half * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03) {
    constexpr int threads_per_quant_block = 2*WARP_SIZE;
    constexpr int quant_blocks_per_cuda_block = 2;
    const int local_quant_block = threadIdx.x / threads_per_quant_block;
    const int byte = threadIdx.x % threads_per_quant_block;
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t blocks_per_row = ne00 / QK_TURBO4_K;
    const int64_t ib_row = int64_t(blockIdx.x)*quant_blocks_per_cuda_block + local_quant_block;

    if (ib_row >= blocks_per_row) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t) i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;
            const int64_t ib = i03*s03 + i02*s02 + i01*s01 + ib_row;
            const int64_t i0 = ((i0203*ne01 + i01)*ne00) + ib_row*QK_TURBO4_K;

            const float norm = __shfl_sync(0xffffffff, lane == 0 ? __half2float(x[ib].norm) : 0.0f, 0);
            const uint8_t packed = x[ib].qs[byte];
            reinterpret_cast<half2 *>(y + i0)[byte] = __float22half2_rn(make_float2(
                norm*turbo4_k_centroids_cuda[ packed       & 0x0f],
                norm*turbo4_k_centroids_cuda[(packed >> 4) & 0x0f]));
        }
    }
}

static void dequantize_block_turbo4_f16_2warp_nc_cuda(
        const void * x,
        half * y,
        int64_t ne00,
        int64_t ne01,
        int64_t ne02,
        int64_t ne03,
        int64_t s01,
        int64_t s02,
        int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TURBO4_K == 0);
    constexpr int quant_blocks_per_cuda_block = 2;
    const int64_t blocks_per_row = ne00 / QK_TURBO4_K;
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks(
        (blocks_per_row + quant_blocks_per_cuda_block - 1) / quant_blocks_per_cuda_block,
        (int) std::min(ne01, (int64_t) 65535),
        (int) std::min(ne0203, (int64_t) 65535));
    dequantize_block_turbo4_f16_2warp_nc<<<num_blocks, quant_blocks_per_cuda_block*2*WARP_SIZE, 0, stream>>>(
        static_cast<const block_turbo4_k *>(x), y,
        ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}
#endif

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d * (q[l] & 0xF) + dm);
        y[l+16] = ggml_cuda_cast<dst_t>(d * (q[l] >>  4) + dm);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d.x * (q[l] & 0xF) + d.y);
        y[l+16] = ggml_cuda_cast<dst_t>(d.x * (q[l] >>  4) + d.y);
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q2_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q3_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q4_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q5_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q6_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_m(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_nl(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_mxfp4(vx, i, yy + i*QK_K, threadIdx.x);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

// One thread owns 8 values and writes them with one 16-byte store (two for f32).
// Same arithmetic as dequantize_q4_K / q5_K / q6_K / iq4_xs. Measured on sm_75
// at roughly the card bandwidth. GGML_CUDA_DEQUANT_WIDE=0 keeps the per-block
// kernels. Misaligned pointers or a partial super-block fall back to those.
__device__ __forceinline__ uint64_t dequant_ld8(const uint8_t * p) {
    const uint16_t * q = reinterpret_cast<const uint16_t *>(p);
    return (uint64_t) q[0] | ((uint64_t) q[1] << 16) | ((uint64_t) q[2] << 32) | ((uint64_t) q[3] << 48);
}

template<int TYPE, typename dst_t>
__device__ __forceinline__ void dequant_wide8(const void * vx, int group, int piece, dst_t * v) {
    const int l0 = 8 * piece;
    const int ib = group >> 3;
    const int gi = group & 7;
    if constexpr (TYPE == GGML_TYPE_Q4_K) {
        const block_q4_K * b = ((const block_q4_K *) vx) + ib;
        const int j64 = gi >> 1;
        const int hi  = gi & 1;
        const float d    = __low2half(b->dm);
        const float dmin = __high2half(b->dm);
        uint8_t sc, m;
        get_scale_min_k4(gi, b->scales, sc, m);
        const float d1 = d * (float) sc;
        const float m1 = dmin * (float) m;
        const uint64_t q = dequant_ld8(b->qs + 32 * j64 + l0);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int lb = (int) (q >> (8 * t)) & 0xFF;
            v[t] = ggml_cuda_cast<dst_t>(d1 * (float) (hi ? (lb >> 4) : (lb & 0xF)) - m1);
        }
    } else if constexpr (TYPE == GGML_TYPE_Q5_K) {
        const block_q5_K * b = ((const block_q5_K *) vx) + ib;
        const int j64 = gi >> 1;
        const int hi  = gi & 1;
        const float d    = __low2half(b->dm);
        const float dmin = __high2half(b->dm);
        uint8_t sc, m;
        get_scale_min_k4(gi, b->scales, sc, m);
        const float d1 = d * (float) sc;
        const float m1 = dmin * (float) m;
        const uint64_t qh = dequant_ld8(b->qh + l0);
        const uint64_t ql = dequant_ld8(b->qs + 32 * j64 + l0);
        const int ub = 2 * j64 + hi;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int lb = (int) (ql >> (8 * t)) & 0xFF;
            const int hb = (int) (qh >> (8 * t)) & 0xFF;
            const int nib = hi ? (lb >> 4) : (lb & 0xF);
            v[t] = ggml_cuda_cast<dst_t>(d1 * (float) (nib + (((hb >> ub) & 1) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == GGML_TYPE_Q6_K) {
        const block_q6_K * b = ((const block_q6_K *) vx) + ib;
        const int n  = gi >> 2;
        const int qu = gi & 3;
        const uint64_t ql = dequant_ld8(b->ql + 64 * n + ((qu & 1) ? 32 : 0) + l0);
        const uint64_t qh = dequant_ld8(b->qh + 32 * n + l0);
        const int8_t * sc = b->scales + 8 * n;
        const float ds = (float) b->d * (float) sc[l0 / 16 + 2 * qu];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int lb = (int) (ql >> (8 * t)) & 0xFF;
            const int hb = (int) (qh >> (8 * t)) & 0xFF;
            const int q = ((qu >= 2) ? (lb >> 4) : (lb & 0xF)) | (((hb >> (2 * qu)) & 3) << 4);
            v[t] = ggml_cuda_cast<dst_t>(ds * (float) (q - 32));
        }
    } else if constexpr (TYPE == GGML_TYPE_IQ4_XS) {
        const block_iq4_xs * b = ((const block_iq4_xs *) vx) + ib;
        const uint16_t scales_h = b->scales_h;
        const int ls = ((b->scales_l[gi >> 1] >> (4 * (gi & 1))) & 0xf) | (((scales_h >> (2 * gi)) & 3) << 4);
        const float dl = (float) b->d * (float) (ls - 32);
        const uint64_t qs = dequant_ld8(b->qs + 16 * gi + 8 * (piece & 1));
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int lb = (int) (qs >> (8 * t)) & 0xFF;
            v[t] = ggml_cuda_cast<dst_t>(dl * (float) kvalues_iq4nl[piece >= 2 ? (lb >> 4) : (lb & 0xf)]);
        }
    }
}

template<int TYPE, typename dst_t>
__global__ void dequant_wide_kernel(const void * __restrict__ vx, dst_t * __restrict__ y, uint32_t groups) {
    static_assert(sizeof(dst_t) == 2 || sizeof(dst_t) == 4, "wide dequant store width");
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t g = i >> 2;
    if (g >= groups) {
        return;
    }
    const int piece = (int) (i & 3);
    __align__(16) dst_t v[8];
    dequant_wide8<TYPE>(vx, (int) g, piece, v);
    uint4 * o = reinterpret_cast<uint4 *>(y + (int64_t) g * 32 + 8 * piece);
    const uint4 * s = reinterpret_cast<const uint4 *>(v);
#pragma unroll
    for (int t = 0; t < (int) (sizeof(v) / sizeof(uint4)); ++t) {
        o[t] = s[t];
    }
}

static bool ggml_cuda_dequant_wide_enabled() {
    static const int enabled = []() {
        const char * v = std::getenv("GGML_CUDA_DEQUANT_WIDE");
        if (v == nullptr || v[0] == '\0') {
            return 1;
        }
        return std::atoi(v) != 0 ? 1 : 0;
    }();
    return enabled != 0;
}

template<typename dst_t>
static bool dequant_wide_launch(ggml_type type, const void * vx, dst_t * y, int64_t k, cudaStream_t stream) {
    if (!ggml_cuda_dequant_wide_enabled()) {
        return false;
    }
    if (k <= 0 || (k % QK_K) != 0) {
        return false;
    }
    const uint64_t groups64 = (uint64_t) k / 32;
    if (groups64 == 0 || groups64 > (uint64_t) UINT32_MAX || groups64 * 4 >= (1ull << 31)) {
        return false;
    }
    if ((((uintptr_t) vx) & 1u) != 0 || (((uintptr_t) y) & 15u) != 0) {
        return false;
    }
    const uint32_t groups = (uint32_t) groups64;
    const unsigned threads = 256;
    const unsigned blocks = (unsigned) ((groups64 * 4 + threads - 1) / threads);
    switch (type) {
        case GGML_TYPE_Q4_K:
            dequant_wide_kernel<GGML_TYPE_Q4_K, dst_t><<<blocks, threads, 0, stream>>>(vx, y, groups);
            break;
        case GGML_TYPE_Q5_K:
            dequant_wide_kernel<GGML_TYPE_Q5_K, dst_t><<<blocks, threads, 0, stream>>>(vx, y, groups);
            break;
        case GGML_TYPE_Q6_K:
            dequant_wide_kernel<GGML_TYPE_Q6_K, dst_t><<<blocks, threads, 0, stream>>>(vx, y, groups);
            break;
        case GGML_TYPE_IQ4_XS:
            dequant_wide_kernel<GGML_TYPE_IQ4_XS, dst_t><<<blocks, threads, 0, stream>>>(vx, y, groups);
            break;
        default:
            return false;
    }
    return true;
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    if (dequant_wide_launch(GGML_TYPE_Q4_K, vx, y, k, stream)) {
        return;
    }
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    if (dequant_wide_launch(GGML_TYPE_Q5_K, vx, y, k, stream)) {
        return;
    }
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    if (dequant_wide_launch(GGML_TYPE_Q6_K, vx, y, k, stream)) {
        return;
    }
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    if (dequant_wide_launch(GGML_TYPE_IQ4_XS, vx, y, k, stream)) {
        return;
    }
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
#ifdef GGML_CUDA_TURBO4_F16_PREFILL
        case GGML_TYPE_TURBO4_K:
            switch (ggml_cuda_turbo4_fast_f16_convert_mode()) {
                case 1: return dequantize_block_turbo4_f16_fast_cuda;
                case 2: return dequantize_block_turbo4_f16_2warp_cuda;
                default: return dequantize_block_cont_cuda<QK_TURBO4_K, 1, dequantize_turbo4_k_rotated, half>;
            }
#endif
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
#ifdef GGML_CUDA_TURBO4_F16_PREFILL
        case GGML_TYPE_TURBO4_K:
            switch (ggml_cuda_turbo4_fast_f16_convert_mode()) {
                case 1: return dequantize_block_turbo4_f16_fast_nc_cuda;
                case 2: return dequantize_block_turbo4_f16_2warp_nc_cuda;
                default: return dequantize_block_cuda<QK_TURBO4_K, 1, dequantize_turbo4_k_rotated, half>;
            }
#endif
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
