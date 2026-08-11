# Router Lookahead Analysis

Date: 2026-08-03

This document records Phase 0 of the routing-exact cross-layer lookahead
investigation. It describes the current implementation and the boundaries that
must remain intact. It does not claim benchmark results and does not authorize
expert transfers.

## Protected baseline

- Repository inspected: `/home/andi/src/llama-wackMall-hybrid`
- Requested future-machine path: `/root/llama-wackMall-hybrid`
- Baseline commit: `f337cf7d9e52f6986814c7923ea96b82223f376c`
- Working branch: `codex/router-lookahead-prefetch`
- The original working tree was clean before the branch was created.
- The future GTX 1080 model path under `/root/atomic-nextn-good` is not present
  on this machine. No model, profile, prompt, or benchmark input was modified.

The branch starts from the current complete hybrid baseline rather than the
older `codex/hybrid-profile-placement` branch. The current baseline includes
later MTP, adaptation, quality, and documentation fixes that the lookahead work
must preserve.

## Exact Qwen35MoE data flow

The base-layer graph in `src/models/qwen35moe.cpp` has this relevant order for
layer `L`:

```text
h_out_(L-1)
  -> attention norm
  -> attention or linear attention
  -> add attention residual
  = h_att_L
  -> target layer post-attention RMS norm
  = true_router_input_L
  -> build_layer_ffn
  -> true router and MoE
  -> add FFN residual
  = h_out_L
```

The exact router is built by `llm_graph_context::build_moe_ffn()` in
`src/llama-graph.cpp`. For Qwen35MoE it computes:

```text
router_logits_L = ffn_gate_inp_L * true_router_input_L
router_probs_L  = softmax(router_logits_L)
actual_ids_L    = argsort_top_k(router_probs_L, n_expert_used)
actual_gate_L   = get_rows(router_probs_L, actual_ids_L)
actual_gate_L   = normalized and optionally scaled
```

Consequently, the exact input to router `L` is
`norm_L(h_att_L)`. Neither `h_att_(L-1)` nor `h_out_(L-1)` is an exact router
input for layer `L`.

## Candidate prediction points

For target layer `L` and distance one, the two requested predictors map to the
existing graph as follows:

### Early point: post-attn

```text
Router_L(norm_L(h_att_(L-1)))
```

The source tensor is available after the attention residual of layer `L-1` and
before its MoE. It offers approximately the remaining MoE work of layer `L-1`
plus attention work of layer `L` as a possible transfer window. It is expected
to be less accurate because it omits the source-layer MoE residual.

### Late point: post-moe

```text
Router_L(norm_L(h_out_(L-1)))
```

`h_out_(L-1)` is the input to layer `L`. It offers attention `L` as the main
possible transfer window. It should generally be more accurate, but this is a
measurement question rather than an assumption.

For distance `n`, the source is `L-n`. Target norm uses
`attn_post_norm_L`; source norm uses `attn_post_norm_(L-n)`. Source norm is a
diagnostic variant only. The true router still uses target-layer parameters and
the true target-layer input.

Both norm and router operations are pure graph operations with respect to model
state: they read immutable weights and create new tensors. They can be expanded
at the source point without changing KV, SSM, routing, or sampling state. They
do, however, add GPU work and can affect scheduler splits, memory planning, and
CUDA graph capture. Observational equivalence must therefore be verified with
token and output hashes.

## Current expert ID and cold-compute path

The true selected IDs have two consumers:

1. The fixed-tier GPU lookup maps actual expert IDs through the per-layer LUT
   to a fixed hot slot or the sentinel slot.
2. `GGML_OP_MOE_COLD` consumes the true IDs, true gate weights, cold mask, and
   host weights to compute only the remaining CPU contribution.

The hot and cold contributions are added once in the existing graph. The
sentinel remains the only GPU mapping for an expert without a ready GPU slot.

When a CPU backend consumes IDs produced on CUDA, the ggml scheduler inserts a
cross-backend copy. The scheduler can enqueue an asynchronous tensor readback
when the backend and destination support it, but the CPU consumer cannot run
until its required copy is complete. Adding a new CPU statistics consumer at
the predictor point would therefore create an early host dependency and reduce
the transfer window that is being measured. Phase 1 must not do that.

## Synchronization points relevant to lookahead

The relevant synchronization boundaries are:

- Scheduler split synchronization for cross-backend dependencies in
  `ggml/src/ggml-backend.cpp`.
- The post-graph tier synchronization in `llama_context::process_ubatch()` when
  `llama_expert_tier::requires_post_graph_sync()` is true.
- Output and layer-input readbacks issued with
  `ggml_backend_tensor_get_async()`, which complete at a later backend or
  scheduler synchronization.
- Request begin/end synchronization used to protect tier mutations and drain
  asynchronous warm-cache jobs.
- Warm-cache event completion before a slot is published to the LUT.

The static no-sync mode is deliberately strict. Lookahead tracing cannot claim
to be complete before its asynchronous readbacks are known to be finished.
Phase 1 therefore rejects the combination of trace collection and a skipped
post-graph synchronization instead of adding an otherwise hidden barrier.

## Existing asynchronous infrastructure that can be reused later

`src/llama-expert-tier.cpp` already contains the important lifecycle pieces for
safe H2D work:

- a dedicated backend and CUDA stream,
- pinned host staging buffers,
- backend events,
- a bounded in-flight job set,
- copy-pending state,
- slot ownership and eviction protection,
- event polling,
- publish-after-completion,
- request and shutdown draining.

These mechanisms are useful references for Phase 3. The current warm-cache
residency policy is not the lookahead design: lookahead scratch is transient
scheduling state, must not participate in LRU or long-term admission, and must
never displace the fixed tier.

## Phase 1 readback design boundary

The least invasive trace can expose predictor IDs, actual IDs, and actual gate
weights as graph outputs. After graph execution, their small payloads can be
copied asynchronously into pinned host memory. Accuracy accounting then runs
after the already required tier synchronization.

This has three important properties:

- no new global scheduler barrier is introduced for statistics,
- no CPU node is inserted at the predictor point,
- the true routing and MoE graph remains unchanged.

It also has a limitation: a readback issued after the whole graph cannot by
itself measure the real mid-graph D2H completion time or prove that an H2D copy
would finish before target layer `L`. Phase 1 must label unavailable timing as
unavailable. It must not substitute an assumed PCIe bandwidth. Backend event
instrumentation or a controlled microbenchmark is required before the timing
gate can be closed.

The first runtime implementation exposed another allocator constraint. `ggml_argsort_top_k()` returns a view into the full argsort tensor. Marking only that view as an output did not preserve the backing bytes until the deferred readback. Later layers reused the allocation and produced valid-looking but incorrect trace IDs. The final trace creates dedicated snapshot tensors. I32 snapshots use `GGML_OP_CONT` because the CUDA backend rejects I32 `GGML_OP_DUP`; F32 weight snapshots use `GGML_OP_DUP`. A zero-copy attempt that marked the argsort backing tensors as outputs kept valid IDs but changed the deterministic response hash by changing graph allocation and in-place decisions, so it was rejected.

On the GTX 1660 Ti all three final snapshot tensors are assigned to `CUDA0`, and the deferred destination uses the CUDA host buffer type. The trace reports this as pinned host readback. No predictor ID is consumed by the CPU inside the model graph.

The tested model has per-expert routed-weight slices of 1,900,544 or 2,039,808 bytes depending on the layer. Phase 1 records these model-derived values and reports candidate transfer bytes. It does not convert bytes into milliseconds without a measured transfer model.

## Scratch and CUDA graph lifetime analysis

A per-layer scratch slot is simple to reason about but scales VRAM with layer
count. The observed model would make one slot in every layer expensive, and the
implementation must not assume a particular layer count or expert size.

A small global rotating arena is more portable across 6 GiB and 8 GiB cards,
but it is only safe if all of these hold:

- each allocation has owner layer, owner expert, and generation,
- pending copies are never published as residency,
- a buffer is not reused while any graph or stream can still reference it,
- event completion is checked before publication,
- stale event completion cannot publish a newer generation,
- graph capture sees stable addresses or is invalidated when addresses change.

CUDA graph reuse in `ggml/src/ggml-cuda/ggml-cuda.cu` keys captured work by the
graph and tracks tensor/source pointers, dimensions, and strides. A scratch
address referenced by a captured graph must remain valid and stable for that
graph instance. On sm_61, CUDA graph execution is disabled by the current CUDA
backend capability guard, but the design must also remain correct on sm_75 and
newer GPUs where capture is active.

No scratch choice is approved in Phase 0 or Phase 1.

## MTP boundaries

MTP uses separate target and draft contexts and may verify multiple tokens in a
single graph. The next-token-prediction layer is not a base MoE layer for profile
or lookahead purposes. A scratch mapping referenced by a verify graph would have
to remain immutable for all tokens in that graph and through rollback.

Phase 1 initially supports no-MTP decode only. The trace feature and every
future transfer feature are default-off under MTP. No target KV, draft KV, SSM,
rollback, sampling, or speculative acceptance code is changed by Phase 1.

## Generic and Qwen-coupled parts

Generic mechanisms:

- request-local trace storage and JSON serialization,
- set, recall, Jaccard, and gate-weighted metric computation,
- pinned asynchronous readback,
- fixed-tier residency snapshots,
- transfer-time microbenchmarks and Oracle simulation,
- bounded event/generation ownership for a future scratch arena.

Qwen35MoE-coupled mechanisms:

- locating `h_att_L` and `h_out_L` in this model graph,
- using `attn_post_norm` as the candidate normalization,
- using `ffn_gate_inp` as the target router matrix,
- interpreting the current top-k and gate normalization sequence,
- excluding the model-specific MTP layer.

The trace core should be model-agnostic, while graph insertion is explicitly
enabled only for models whose builder exposes the required tensors and router
parameters.

## Phase 0 risks

Correctness risks:

- accidentally feeding predicted IDs into the real MoE,
- changing the true gate normalization or ordering,
- adding a graph dependency that changes CPU/GPU overlap,
- counting a duplicated prediction more than once,
- mixing MTP draft layers into base-layer statistics.

Performance risks:

- predictor matmul and top-k cost exceeds avoided CPU work,
- predicted IDs require a mid-graph host barrier,
- false-positive copies consume more bandwidth than useful copies,
- graph output retention or pinned buffers increase memory pressure,
- a global scratch arena serializes otherwise independent layers.

Concurrency risks for future phases:

- a late event publishes an obsolete arena generation,
- a slot is overwritten while a graph still references it,
- request cancellation or shutdown leaves an in-flight copy alive,
- an expert contribution is computed on both CPU and GPU or on neither.

Phase 1 is deliberately restricted to observation so these transfer and
residency risks are not introduced yet.
