#include "common.cuh"

// Decode attention for MLA (DKQ 576, DV 512, V a view of K) without tensor cores, see fattn-mla-dec.cu.
// Off unless GGML_CUDA_FA_MLA_DEC=1.
bool ggml_cuda_flash_attn_ext_mla_dec_applies(const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_mla_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
