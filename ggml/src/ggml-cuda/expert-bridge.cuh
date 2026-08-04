#pragma once

#include "common.cuh"

void ggml_cuda_expert_bridge_q4_k_q8_k(
        const void * weights, const void * input, int n_embd, int rows,
        float * output, cudaStream_t stream);

void ggml_cuda_expert_bridge_q4_k_q8_k_indexed(
        const void * weights, const void * input, int n_embd, int n_ff,
        const int * weight_candidates, const int * output_candidates,
        int candidate_count, float * output, cudaStream_t stream);

void ggml_cuda_expert_bridge_quantize_q8_k(
        const float * input, void * output, int n_embd, int rows, cudaStream_t stream);
