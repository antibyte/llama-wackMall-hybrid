#include "common.cuh"

// SenseNova-Loop residual stream routers (ggml_mhc_route, ggml_mhc_merge)
void ggml_cuda_op_mhc_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mhc_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
