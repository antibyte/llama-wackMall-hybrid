#pragma once

#include "ggml.h"
#include "llama.h"

struct llama_model;

// expert hot-tiering: pins the most-used routed experts of each MoE layer in
// GPU memory; the remaining (cold) experts are computed on the CPU.
//
// enabled via env:
//   LLAMA_EXPERT_HOT   - path to heat csv with header "layer,expert,count"
//                        (seed; optional if LLAMA_EXPERT_ADAPT=1)
//   LLAMA_EXPERT_S     - hot slots per layer (default 16)
//   LLAMA_EXPERT_TMAX  - max n_tokens for the hot/cold path (default 16)
//   LLAMA_EXPERT_STATS - dump cold-hit stats at exit ("1" = stderr, else path)
//   LLAMA_EXPERT_ADAPT - 1: online repin of hot slots (decay + hysteresis)
//   LLAMA_EXPERT_USAGE - dump learned hot set at exit (heat csv format)

namespace llama_expert_tier {

// call once after the model tensors are loaded; no-op unless LLAMA_EXPERT_HOT is set
void init(const llama_model & model);

// drop-in replacement for ggml_mul_mat_id on MoE expert weights:
// hot part on the GPU via the pinned store, cold part on the CPU via
// ggml_mul_mat_id_cold; falls back to plain ggml_mul_mat_id when the weight
// has no store or the batch is larger than LLAMA_EXPERT_TMAX
ggml_tensor * build_mul_mat_id(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids);

// consume accumulated expert selection counts, update scores and
// (LLAMA_EXPERT_ADAPT) repin hot slots; call after each ubatch compute
void update();

// total bytes of routed-expert weight tensors (for dense-fit estimates)
LLAMA_API size_t expert_weight_bytes(const llama_model & model);

// fused cold-expert path for one MoE layer, two calls per layer:
// begin (before the expert matmuls): validates the layer and makes
//   build_mul_mat_id return hot-only results.
//   eligible must be false unless the layer uses separate gate/up tensors,
//   plain silu swiglu, and no expert biases/scales.
// end (after the down matmul, late in node order so the CPU cold op overlaps
//   GPU hot work): returns the cold contribution [embd, n_used, n_tokens]
//   (add it to the down result) or nullptr. x must be the layer input
//   captured before the matmuls.
bool begin_moe_cold(bool eligible,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * ids);

ggml_tensor * end_moe_cold(ggml_context * ctx,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * x, ggml_tensor * ids);

// count-only path for batches larger than LLAMA_EXPERT_TMAX (prompt
// harvesting): returns a scalar f32 tensor (add it to the layer output to
// keep the op in the graph) or nullptr
ggml_tensor * build_moe_count(ggml_context * ctx, ggml_tensor * down_w, ggml_tensor * ids);

}
