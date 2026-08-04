#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

static __device__ __forceinline__ float fp16_to_fp32(ggml_half value) {
    return __half2float(value);
}

static __device__ __forceinline__ void unpack_q4_k(
        const block_q4_K & block, uint8_t scales[8], uint8_t mins[8]) {
    uint32_t words[4] = {};
    const uint8_t * source = block.scales;
    uint8_t * bytes = reinterpret_cast<uint8_t *>(words);
#pragma unroll
    for (int i = 0; i < 12; ++i) bytes[i] = source[i];
    words[3] = ((words[2] >> 4) & 0x0f0f0f0fU) | (((words[1] >> 6) & 0x03030303U) << 4);
    const uint32_t auxiliary = words[1] & 0x3f3f3f3fU;
    words[1] = (words[2] & 0x0f0f0f0fU) | (((words[0] >> 6) & 0x03030303U) << 4);
    words[2] = auxiliary;
    words[0] &= 0x3f3f3f3fU;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        scales[i] = bytes[i];
        mins[i] = bytes[8 + i];
    }
}

static __device__ __forceinline__ int nearest_int(float value) {
    const float adjusted = __fadd_rn(value, 12582912.0f);
    return (__float_as_int(adjusted) & 0x007fffff) - 0x00400000;
}

static __global__ void quantize_q8_k_kernel(
        const float * input, block_q8_K * output, int blocks) {
    const int block_index = blockIdx.x*blockDim.x + threadIdx.x;
    if (block_index >= blocks) return;
    input += (size_t) block_index*QK_K;
    output += block_index;
    float maximum = 0.0f;
    float absolute_maximum = 0.0f;
    for (int i = 0; i < QK_K; ++i) {
        const float absolute = fabsf(input[i]);
        if (absolute > absolute_maximum) {
            absolute_maximum = absolute;
            maximum = input[i];
        }
    }
    if (absolute_maximum == 0.0f) {
        output->d = 0.0f;
        for (int i = 0; i < QK_K; ++i) output->qs[i] = 0;
        for (int i = 0; i < QK_K/16; ++i) output->bsums[i] = 0;
        return;
    }
    const float inverse_scale = __fdiv_rn(-127.0f, maximum);
    for (int i = 0; i < QK_K; ++i) {
        const int value = nearest_int(__fmul_rn(inverse_scale, input[i]));
        output->qs[i] = (int8_t) min(127, value);
    }
    for (int group = 0; group < QK_K/16; ++group) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) sum += output->qs[16*group + i];
        output->bsums[group] = (int16_t) sum;
    }
    output->d = __fdiv_rn(1.0f, inverse_scale);
}

static __global__ void q4_k_q8_k_avx2_kernel(
        const block_q4_K * weights, const block_q8_K * input,
        int blocks_per_row, int rows, float * output) {
    const int global_thread = blockIdx.x*blockDim.x + threadIdx.x;
    const int row = global_thread/8;
    const int lane = global_thread%8;
    const bool active = row < rows;
    float acc = 0.0f;
    float acc_min = 0.0f;
    if (active) {
        const block_q4_K * row_weights = weights + (size_t) row*blocks_per_row;
        for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
            const block_q4_K & weight = row_weights[block_index];
            const block_q8_K & activation = input[block_index];
            uint8_t scales[8];
            uint8_t mins[8];
            unpack_q4_k(weight, scales, mins);
            const float d = __fmul_rn(activation.d, fp16_to_fp32(weight.data.d));
            const float dmin = __fmul_rn(-activation.d, fp16_to_fp32(weight.data.dmin));
            if (lane < 4) {
                const int base = 4*lane;
                const int first = activation.bsums[base] + activation.bsums[base + 1];
                const int second = activation.bsums[base + 2] + activation.bsums[base + 3];
                const int product = (int) mins[2*lane]*first + (int) mins[2*lane + 1]*second;
                acc_min = __fmaf_rn(dmin, (float) product, acc_min);
            }
            int sum = 0;
#pragma unroll
            for (int group = 0; group < 4; ++group) {
                const int index = 32*group + 4*lane;
#if __CUDA_ARCH__ >= 610
                const int packed = *(const int *) (weight.qs + index);
                const int low = __dp4a(packed & 0x0f0f0f0f,
                        *(const int *) (activation.qs + 64*group + 4*lane), 0);
                const int high = __dp4a((packed >> 4) & 0x0f0f0f0f,
                        *(const int *) (activation.qs + 64*group + 32 + 4*lane), 0);
#else
                int low = 0;
                int high = 0;
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    const uint8_t packed = weight.qs[index + item];
                    low += (int) (packed & 0x0f)*activation.qs[64*group + 4*lane + item];
                    high += (int) (packed >> 4)*activation.qs[64*group + 32 + 4*lane + item];
                }
#endif
                sum += (int) scales[2*group]*low + (int) scales[2*group + 1]*high;
            }
            acc = __fmaf_rn(d, (float) sum, acc);
        }
    }
    const unsigned int mask = 0xffffffffU;
    float reduced = __fadd_rn(acc, __shfl_sync(mask, acc, (lane + 4)%8, 8));
    const float reduced_2 = __shfl_sync(mask, reduced, 2, 8);
    const float reduced_3 = __shfl_sync(mask, reduced, 3, 8);
    if (lane == 0) reduced = __fadd_rn(reduced, reduced_2);
    if (lane == 1) reduced = __fadd_rn(reduced, reduced_3);
    const float reduced_1 = __shfl_sync(mask, reduced, 1, 8);
    if (lane == 0) reduced = __fadd_rn(reduced, reduced_1);

    const float min_2 = __shfl_sync(mask, acc_min, 2, 8);
    const float min_3 = __shfl_sync(mask, acc_min, 3, 8);
    if (lane == 0) acc_min = __fadd_rn(acc_min, min_2);
    if (lane == 1) acc_min = __fadd_rn(acc_min, min_3);
    const float min_1 = __shfl_sync(mask, acc_min, 1, 8);
    if (active && lane == 0) output[row] = __fadd_rn(reduced, __fadd_rn(acc_min, min_1));
}

extern "C" bool expert_compute_q4_k_avx2(
        const void * weights, const void * input, int n_embd, int rows,
        int iterations, float * output, double * kernel_ms) {
    if (!weights || !input || !output || !kernel_ms || n_embd <= 0 || rows <= 0 ||
            iterations <= 0 || n_embd % QK_K != 0) return false;
    const int blocks_per_row = n_embd/QK_K;
    const size_t weight_bytes = (size_t) rows*blocks_per_row*sizeof(block_q4_K);
    const size_t input_bytes = (size_t) blocks_per_row*sizeof(block_q8_K);
    block_q4_K * device_weights = nullptr;
    block_q8_K * device_input = nullptr;
    float * device_output = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    if (cudaMalloc(&device_weights, weight_bytes) != cudaSuccess ||
            cudaMalloc(&device_input, input_bytes) != cudaSuccess ||
            cudaMalloc(&device_output, (size_t) rows*sizeof(float)) != cudaSuccess ||
            cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) {
        cudaFree(device_weights);
        cudaFree(device_input);
        cudaFree(device_output);
        return false;
    }
    bool ok = cudaMemcpy(device_weights, weights, weight_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(device_input, input, input_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        cudaEventRecord(begin);
        for (int i = 0; i < iterations; ++i) {
            q4_k_q8_k_avx2_kernel<<<(8*rows + 127)/128, 128>>>(
                    device_weights, device_input, blocks_per_row, rows, device_output);
        }
        cudaEventRecord(end);
        ok = cudaGetLastError() == cudaSuccess && cudaEventSynchronize(end) == cudaSuccess;
        float elapsed_ms = 0.0f;
        ok = ok && cudaEventElapsedTime(&elapsed_ms, begin, end) == cudaSuccess;
        *kernel_ms = elapsed_ms/iterations;
        ok = ok &&
            cudaMemcpy(output, device_output, (size_t) rows*sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_weights);
    return ok;
}

extern "C" bool expert_compute_q4_k_avx2_f32(
        const void * weights, const float * input, int n_embd, int rows,
        int iterations, float * output, void * quantized, double * kernel_ms) {
    if (!weights || !input || !output || !quantized || !kernel_ms || n_embd <= 0 || rows <= 0 ||
            iterations <= 0 || n_embd % QK_K != 0) return false;
    const int blocks_per_row = n_embd/QK_K;
    const size_t weight_bytes = (size_t) rows*blocks_per_row*sizeof(block_q4_K);
    const size_t input_bytes = (size_t) n_embd*sizeof(float);
    const size_t quantized_bytes = (size_t) blocks_per_row*sizeof(block_q8_K);
    block_q4_K * device_weights = nullptr;
    float * device_input = nullptr;
    block_q8_K * device_quantized = nullptr;
    float * device_output = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    if (cudaMalloc(&device_weights, weight_bytes) != cudaSuccess ||
            cudaMalloc(&device_input, input_bytes) != cudaSuccess ||
            cudaMalloc(&device_quantized, quantized_bytes) != cudaSuccess ||
            cudaMalloc(&device_output, (size_t) rows*sizeof(float)) != cudaSuccess ||
            cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) {
        cudaFree(device_weights);
        cudaFree(device_input);
        cudaFree(device_quantized);
        cudaFree(device_output);
        return false;
    }
    bool ok = cudaMemcpy(device_weights, weights, weight_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(device_input, input, input_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        cudaEventRecord(begin);
        for (int i = 0; i < iterations; ++i) {
            quantize_q8_k_kernel<<<(blocks_per_row + 127)/128, 128>>>(
                    device_input, device_quantized, blocks_per_row);
            q4_k_q8_k_avx2_kernel<<<(8*rows + 127)/128, 128>>>(
                    device_weights, device_quantized, blocks_per_row, rows, device_output);
        }
        cudaEventRecord(end);
        ok = cudaGetLastError() == cudaSuccess && cudaEventSynchronize(end) == cudaSuccess;
        float elapsed_ms = 0.0f;
        ok = ok && cudaEventElapsedTime(&elapsed_ms, begin, end) == cudaSuccess;
        *kernel_ms = elapsed_ms/iterations;
        ok = ok &&
            cudaMemcpy(output, device_output, (size_t) rows*sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(quantized, device_quantized, quantized_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    cudaFree(device_output);
    cudaFree(device_quantized);
    cudaFree(device_input);
    cudaFree(device_weights);
    return ok;
}
