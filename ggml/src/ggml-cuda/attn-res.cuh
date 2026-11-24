#pragma once

#include "common.cuh"

// Fusions for depth attention residuals and low-rank loop gates (Inkling 2); both match the unfused subgraph and
// return the number of nodes to skip after node i, 0 if the subgraph does not match.
//
// GGML_CUDA_ATTN_RES_FUSION: rms_norm(cur) -> concat(blocks, .) -> mul_mat(., queries) -> soft_max ->
// mul_mat(cont(transpose(.)), .): one kernel per token computes the weighted sum of the normalized states.
//
// GGML_CUDA_LOOP_GATE_FUSION: concat(cur, msg) -> rms_norm -> mul_mat(down) -> tanh(mul_mat(forget)),
// tanh(mul_mat(gain)) -> (1 - forget)*cur + gain*msg: two kernels for up to GGML_CUDA_LOOP_GATE_MAX_TOKENS tokens.
int ggml_cuda_try_attn_res_fusion(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
int ggml_cuda_try_loop_gate_fusion(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
