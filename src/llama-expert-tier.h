#pragma once

#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <vector>

struct llama_model;

// expert hot-tiering: pins the most-used routed experts of each MoE layer in
// GPU memory; the remaining (cold) experts are computed on the CPU.
//
// enabled via env:
//   LLAMA_EXPERT_HOT   - path to heat csv with header "layer,expert,count"
//                        (seed; optional if LLAMA_EXPERT_ADAPT=1)
//   LLAMA_EXPERT_S     - hot slots per layer (default 16)
//   LLAMA_EXPERT_PLACEMENT - optional validated per-layer static-slot manifest;
//                            requires W=0 and adaptation off
//   LLAMA_EXPERT_TMAX  - max n_tokens for the hot/cold path (default 16);
//                        warm H2D admits only on graphs at or below this size
//   LLAMA_EXPERT_STATS - dump cold-hit stats at exit ("1" = stderr, "0" = off, else path)
//   LLAMA_EXPERT_ADAPT - 1: online repin of hot slots (decay + hysteresis)
//   LLAMA_EXPERT_USAGE - dump learned hot set at exit (heat csv path, "0" = off)
//   LLAMA_EXPERT_USAGE_MODE - cumulative (seed + observations, default) or
//                             session (only observations from this process)
//   LLAMA_EXPERT_USAGE_CHECKPOINT - request (default) or exit; request writes
//                                   the usage profile atomically on completion
//   LLAMA_EXPERT_ADAPT_INTERVAL - request (default) or graph. MTP always uses
//                                 request-boundary publication.
//   LLAMA_EXPERT_ADAPT_CUDA_GRAPHS - 0 (default) disables CUDA graph caching
//                                    while repinning; 1 is unsafe/diagnostic
//   LLAMA_EXPERT_STATS_JSON - dump machine-readable stats (path, "0" = off)
//   LLAMA_EXPERT_TIMING - 1: collect optional per-layer CPU cold-node wall time
//   LLAMA_EXPERT_CPU_CHUNK - singleton cold-expert row chunk, power of two 16..256
//   LLAMA_EXPERT_CPU_ACT_PARALLEL - 1: split activation/quantization across
//                                   quant blocks when cold columns are sparse
//   LLAMA_EXPERT_CPU_ASYNC - 1: experimental CPU-cold/GPU-hot layer overlap
//   LLAMA_EXPERT_CPU_DOWN_PREFETCH - down-weight row prefetch distance, 0..8
//   LLAMA_EXPERT_CPU_REUSE_ROWS - 1: reuse weight rows across repeated MTP experts
//   LLAMA_EXPERT_CPU_MULTI_ROW - 1: AVX2 Q4_K/Q5_K dots share decoded weight
//                                blocks across repeated MTP expert selections
//   LLAMA_EXPERT_CPU_FUSED_GATE_UP - 1: AVX2 Q4_K Gate/Up dots share Q8_K
//                                    activation loads (default 0)
//   LLAMA_EXPERT_SHARED_HOT_IDS - 1: reuse one per-layer hot-slot ID mapping
//   LLAMA_EXPERT_SKIP_SENTINEL - 1: CUDA MMVQ early-exits cold expert slots that
//                                map to the zeroed sentinel (default 0 until
//                                measured on the target GPU)
//   LLAMA_EXPERT_BW_PROFILE - JSON from bench_expert_bw.py; cpu-heavy keeps W
//                            only when 0 < q* < 1 (capped copies); MTP still W=0
//   LLAMA_EXPERT_WARM_SLOTS - extra bounded warm slots/layer (default 0)
//   LLAMA_EXPERT_WARM_AUTO_MAX - cap for W=auto (default 4; tune on larger GPUs)
//   LLAMA_EXPERT_WARM_POLICY - warm replacement policy (currently lru)
//   LLAMA_EXPERT_WARM_RESET - request or persistent LRU aging
//   LLAMA_EXPERT_WARM_ADMISSION - immediate, second-hit, or frequency
//                                 (default immediate)
//   LLAMA_EXPERT_WARM_ADMISSION_WINDOW - graph window/half-life (default 8)
//   LLAMA_EXPERT_WARM_REPLACE_RATIO - frequency hysteresis vs occupant
//                                 (default 1.0; candidate must exceed
//                                 ratio * occupant, e.g. 1.5)
//   LLAMA_EXPERT_WARM_PREFETCH - 0: synchronous fill; 1: asynchronous H2D fill
//   LLAMA_EXPERT_PREFETCH_STREAMS - async streams (currently exactly 1)
//   LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT - global in-flight copy limit (default 2)
//   LLAMA_EXPERT_VRAM_RESERVE_MIB - conservative post-load reserve (default 512)
//   LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL - 1 bypasses the default MTP warm guard
//   LLAMA_EXPERT_STATIC_NO_SYNC - 1 skips the tier-update barrier only when
//                                 adaptation, warm slots, and stats are disabled

namespace llama_expert_tier {

// Mutable tier tensors and count buffers are process-wide and shared by target
// and MTP draft contexts. Hold this guard from graph construction through the
// post-graph synchronization/update so another context cannot publish a new
// LUT or overwrite shared counts concurrently.
class graph_guard {
public:
    graph_guard();
    ~graph_guard();

    graph_guard(const graph_guard &) = delete;
    graph_guard & operator=(const graph_guard &) = delete;

private:
    bool locked_ = false;
};

// Enable the tier for contexts created through common.cpp. Direct libllama
// callers remain stock unless they explicitly set an LLAMA_EXPERT_* control.
void configure_enabled(bool enabled);

// Server processes call this before model/context initialization so internal
// model warm-up graphs are not persisted as real request usage.  CLI/library
// callers retain the legacy context-lifetime collection behavior.
void configure_request_scoped(bool enabled);

// call once after the model tensors are loaded; no-op unless configured by
// common.cpp or an explicit LLAMA_EXPERT_* environment control is present
void init(const llama_model & model);

// drop-in replacement for ggml_mul_mat_id on MoE expert weights:
// hot part on the GPU via the pinned store, cold part on the CPU via
// ggml_mul_mat_id_cold; falls back to plain ggml_mul_mat_id when the weight
// has no store or the batch is larger than LLAMA_EXPERT_TMAX
ggml_tensor * build_hot_ids(ggml_context * ctx, ggml_tensor * w, ggml_tensor * ids);

ggml_tensor * build_mul_mat_id(
        ggml_context * ctx,
        ggml_tensor * w,
        ggml_tensor * x,
        ggml_tensor * ids,
        ggml_tensor * hot_ids = nullptr);

// consume accumulated expert selection counts, update scores and
// (LLAMA_EXPERT_ADAPT) repin hot slots; call after each ubatch compute
void update();

// Whether the asynchronous graph must be synchronized before update(). The
// explicit static no-sync mode returns false only after strict validation.
bool requires_post_graph_sync();

// Declare the process-wide MTP draft width before target-context creation.
// Warm slots are guarded off for MTP unless the explicit experimental opt-in
// is set. A later MTP context declaration also clears a still-empty/runtime
// warm tier at a context-creation boundary.
void configure_mtp(int mtp_n);

// Mark a server request boundary. Request-granular fixed adaptation publishes
// at most one replacement per layer here, while no graph can observe a partial
// transaction. Warm request-local ages are also reset here when requested.
LLAMA_API void request_begin();

// Mark completion/cancellation of a server request. When request checkpoints
// are enabled, atomically persist learned usage without waiting for process
// exit.
LLAMA_API void request_end();

// Snapshot fixed-tier residency for observational routing traces. The result
// never includes warm slots or pending asynchronous copies.
std::vector<uint8_t> fixed_expert_mask(int il, int n_expert);

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

// True only for the explicit asynchronous cold-split experiment. Graph
// builders use it to place the cold branch before the hot branch in node order.
bool cpu_async_enabled();

ggml_tensor * end_moe_cold(ggml_context * ctx,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * x, ggml_tensor * ids);

// count-only path for batches larger than LLAMA_EXPERT_TMAX (prompt
// harvesting): returns a scalar f32 tensor (add it to the layer output to
// keep the op in the graph) or nullptr
ggml_tensor * build_moe_count(ggml_context * ctx, ggml_tensor * down_w, ggml_tensor * ids);

}
