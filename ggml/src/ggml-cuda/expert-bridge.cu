#include "expert-bridge.cuh"

static __device__ __forceinline__ int nearest_int(float value) {
    const float adjusted = __fadd_rn(value, 12582912.0f);
    return (__float_as_int(adjusted) & 0x007fffff) - 0x00400000;
}

static __global__ void quantize_q8_k_kernel(
        const float * input, block_q8_K * output, int blocks) {
    const int block_index = blockIdx.x;
    if (block_index >= blocks) return;
    const int lane = threadIdx.x;
    input += (size_t) block_index*QK_K;
    output += block_index;
    float maximum = 0.0f;
    float absolute_maximum = 0.0f;
    int maximum_index = QK_K;
    for (int i = lane; i < QK_K; i += 32) {
        const float absolute = fabsf(input[i]);
        if (absolute > absolute_maximum) {
            absolute_maximum = absolute;
            maximum = input[i];
            maximum_index = i;
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other_absolute = __shfl_down_sync(0xffffffffU, absolute_maximum, offset);
        const float other_maximum = __shfl_down_sync(0xffffffffU, maximum, offset);
        const int other_index = __shfl_down_sync(0xffffffffU, maximum_index, offset);
        if (other_absolute > absolute_maximum ||
                (other_absolute == absolute_maximum && other_index < maximum_index)) {
            absolute_maximum = other_absolute;
            maximum = other_maximum;
            maximum_index = other_index;
        }
    }
    absolute_maximum = __shfl_sync(0xffffffffU, absolute_maximum, 0);
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    if (absolute_maximum == 0.0f) {
        if (lane == 0) output->d = 0.0f;
        for (int i = lane; i < QK_K; i += 32) output->qs[i] = 0;
        if (lane < QK_K/16) output->bsums[lane] = 0;
        return;
    }
    const float inverse_scale = __fdiv_rn(-127.0f, maximum);
    for (int i = lane; i < QK_K; i += 32) {
        const int value = nearest_int(__fmul_rn(inverse_scale, input[i]));
        output->qs[i] = (int8_t) min(127, value);
    }
    __syncwarp();
    if (lane < QK_K/16) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) sum += output->qs[16*lane + i];
        output->bsums[lane] = (int16_t) sum;
    }
    if (lane == 0) output->d = __fdiv_rn(1.0f, inverse_scale);
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

static __global__ void q4_k_q8_k_avx2_kernel(
        const block_q4_K * weights, const block_q8_K * input,
        int blocks_per_row, int rows, int n_ff,
        int weight_0, int weight_1, int weight_2,
        int output_0, int output_1, int output_2, float * output) {
    const int global_thread = blockIdx.x*blockDim.x + threadIdx.x;
    const int row = global_thread/8;
    const int lane = global_thread%8;
    const bool active = row < rows;
    int weight_row = row;
    int output_row = row;
    if (active && n_ff > 0) {
        const int logical_candidate = row/(2*n_ff);
        const int weight_candidate = logical_candidate == 0 ? weight_0 :
                (logical_candidate == 1 ? weight_1 : weight_2);
        const int output_candidate = logical_candidate == 0 ? output_0 :
                (logical_candidate == 1 ? output_1 : output_2);
        weight_row = weight_candidate*2*n_ff + row%(2*n_ff);
        output_row = output_candidate*2*n_ff + row%(2*n_ff);
    }
    float acc = 0.0f;
    float acc_min = 0.0f;
    if (active) {
        const block_q4_K * row_weights = weights + (size_t) weight_row*blocks_per_row;
        for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
            const block_q4_K & weight = row_weights[block_index];
            const block_q8_K & activation = input[block_index];
            uint8_t scales[8];
            uint8_t mins[8];
            unpack_q4_k(weight, scales, mins);
            const float d = __fmul_rn(activation.d, __half2float(weight.data.d));
            const float dmin = __fmul_rn(-activation.d, __half2float(weight.data.dmin));
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
    if (active && lane == 0) output[output_row] = __fadd_rn(reduced, __fadd_rn(acc_min, min_1));
}

void ggml_cuda_expert_bridge_q4_k_q8_k(
        const void * weights, const void * input, int n_embd, int rows,
        float * output, cudaStream_t stream) {
    GGML_ASSERT(n_embd > 0 && n_embd % QK_K == 0 && rows > 0);
    q4_k_q8_k_avx2_kernel<<<(8*rows + 127)/128, 128, 0, stream>>>(
            (const block_q4_K *) weights, (const block_q8_K *) input,
            n_embd/QK_K, rows, 0, 0, 0, 0, 0, 0, 0, output);
}

void ggml_cuda_expert_bridge_q4_k_q8_k_indexed(
        const void * weights, const void * input, int n_embd, int n_ff,
        const int * weight_candidates, const int * output_candidates,
        int candidate_count, float * output, cudaStream_t stream) {
    GGML_ASSERT(n_embd > 0 && n_embd % QK_K == 0 && n_ff > 0);
    GGML_ASSERT(weight_candidates && output_candidates && candidate_count > 0 && candidate_count <= 3);
    const int weight_0 = weight_candidates[0];
    const int weight_1 = candidate_count > 1 ? weight_candidates[1] : 0;
    const int weight_2 = candidate_count > 2 ? weight_candidates[2] : 0;
    const int output_0 = output_candidates[0];
    const int output_1 = candidate_count > 1 ? output_candidates[1] : 0;
    const int output_2 = candidate_count > 2 ? output_candidates[2] : 0;
    const int rows = candidate_count*2*n_ff;
    q4_k_q8_k_avx2_kernel<<<(8*rows + 127)/128, 128, 0, stream>>>(
            (const block_q4_K *) weights, (const block_q8_K *) input,
            n_embd/QK_K, rows, n_ff,
            weight_0, weight_1, weight_2, output_0, output_1, output_2, output);
}

void ggml_cuda_expert_bridge_quantize_q8_k(
        const float * input, void * output, int n_embd, int rows, cudaStream_t stream) {
    GGML_ASSERT(input && output && n_embd > 0 && n_embd % QK_K == 0 && rows > 0);
    const int blocks = rows*n_embd/QK_K;
    quantize_q8_k_kernel<<<blocks, 32, 0, stream>>>(
            input, (block_q8_K *) output, blocks);
}
