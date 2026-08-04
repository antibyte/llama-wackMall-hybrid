#pragma once

#include "common.cuh"

void ggml_cuda_expert_bridge_q4_k_q8_k(
        const void * weights, const void * input, int n_embd, int rows,
        float * output, cudaStream_t stream);
