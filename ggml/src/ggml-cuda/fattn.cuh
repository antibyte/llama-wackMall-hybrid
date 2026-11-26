#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

struct ggml_cuda_fattn_parts;

// Runs dst with the parallel-block partials exported to parts (never the stream-K MMA kernel).
// False when no kernel fits or the call cannot export (parts->blocks == 0).
bool ggml_cuda_flash_attn_ext_parts(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_fattn_parts * parts);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
