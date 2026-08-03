# Router Lookahead Design

Date: 2026-08-03

This document defines the Phase 1 trace design and the gates for later phases.
No productive expert prefetch is part of Phase 1.

## Non-negotiable routing invariant

For target layer `L`:

```text
predicted_ids = Router_L(candidate_earlier_hidden)
actual_ids    = Router_L(true_router_input_L)
actual_gate   = weights selected by the true router
```

Only `actual_ids` and `actual_gate` may determine model computation. Predictor
outputs are trace data in Phase 1. In a future prefetch phase they may only
prepare bytes that the actual router can choose to use or ignore.

When lookahead is disabled, no predictor nodes, output tensors, host buffers,
JSON writes, or additional synchronization may be created.

## Phase 1 configuration

All options are environment variables and all behavior is default-off:

```text
LLAMA_EXPERT_LOOKAHEAD_TRACE=0
LLAMA_EXPERT_LOOKAHEAD_TRACE_JSON=
LLAMA_EXPERT_LOOKAHEAD_DISTANCE=1
LLAMA_EXPERT_LOOKAHEAD_TOP_M=16
LLAMA_EXPERT_LOOKAHEAD_POINT=post-attn
LLAMA_EXPERT_LOOKAHEAD_NORM=target
```

Valid point values are `post-attn` and `post-moe`. Valid norm values are
`target` and `source`. Distance must be positive. Top-M is clamped only after
the model expert count is known, and the effective value is reported. No code
assumes 40 layers, 256 experts, top-8 routing, S=70, or a GPU architecture.

Phase 1 initially requires:

- no MTP,
- one sequence and one active request,
- fixed placement during a measured request,
- warm cache disabled,
- productive lookahead disabled.

Unsupported combinations fail closed with a clear message when trace is
requested. They do not silently collect misleading data.

## Graph insertion

The Qwen35MoE graph builder records the source tensors at explicit semantic
points:

- `post-attn`: the attention residual before source-layer post-attention norm,
- `post-moe`: the output after source-layer FFN residual.

For each target layer with a valid source:

1. select source tensor `target - distance`,
2. apply target or source post-attention RMS norm as configured,
3. multiply by the target router matrix,
4. apply the same router probability transformation used by the true path,
5. select the configured Top-M predictor IDs,
6. expand this predictor branch at the source point,
7. retain predictor IDs as a trace output,
8. retain true IDs and true selected gate weights from the unchanged router.

The predictor output is not connected to the true FFN branch.

Retained IDs are explicit CUDA-capable I32 `CONT` snapshots. Retained true weights are F32 `DUP` snapshots. The extra copies are trace-only and prevent allocator reuse from corrupting deferred observations. They make trace throughput unsuitable as a production performance estimate.

## Deferred asynchronous observation

After graph compute is enqueued, the context submits small asynchronous D2H
copies for predictor IDs, actual IDs, and actual gate weights into pinned host
buffers. The copies complete at the existing post-graph synchronization that is
already required by the expert tier. Metrics are evaluated only after that
point.

The trace path must not call `ggml_backend_sched_synchronize()` on its own.
Trace is rejected when the strict static-no-sync path would remove the only
completion boundary.

This deferred scheme measures prediction quality without inserting a CPU
consumer at the source layer. It does not claim to measure a real prefetch
schedule. Phase 2 will combine the routing trace with separately measured event
and transfer timings.

## Request lifecycle

Trace state is reset at request begin and finalized at request end. A monotonic
request ID is included in every output. Multiple requests in one process must
not share counters or raw records.

The JSON target supports two forms:

```text
/path/trace.json
/path/trace-%r.json
```

`%r` is replaced with the request ID. Without `%r`, the latest request is
written atomically to the configured path. The input path is never used as a
model or profile path and is never modified until request finalization.

## Metrics

All set metrics deduplicate IDs and validate them against the model expert
count. Let `A` be the actual top-k set, `P_m` the first `m` unique predicted
IDs, `C` the actual experts outside the fixed tier, and `w(a)` the true,
normalized gate weight.

```text
top1_hit(m) = first predicted ID equals first actual ID
recall(m) = |A intersect P_m| / |A|
jaccard(m) = |A intersect P_m| / |A union P_m|
weighted_coverage(m) = sum(w(a), a in A intersect P_m) / sum(w(a), a in A)
cold_recall(m) = |C intersect P_m| / |C|, or null when C is empty
```

The trace reports prefix metrics for Top-8, Top-12, and Top-16 when the
configured predictor output contains those prefixes. It also reports the model
top-k and configured effective Top-M without assuming top-8.

Counts include:

- predicted IDs already in the fixed tier,
- useful predicted cold IDs,
- false-positive cold IDs,
- actual cold experts missed by the prediction,
- totals per layer and for the complete request.

The model-derived expert bytes are reported per layer. Each prefix aggregate includes estimated candidate H2D bytes and estimated useful H2D bytes. These are byte counts only. Estimated H2D milliseconds and timely fractions remain `null` until Phase 2 supplies measured transfer and layer timing data.

Raw per-token records are retained in trace mode so Phase 2 can replay Top-M
and distance policies without model execution. Records contain actual IDs,
actual weights, predicted IDs, and the fixed-residency snapshot. They contain
no prompt text.

## JSON shape

The versioned JSON has this logical structure:

```json
{
  "schema": "llama-wackmall-router-lookahead-v1",
  "request_id": 1,
  "model": "",
  "config": {},
  "status": {},
  "overall": {},
  "layers": [],
  "records": []
}
```

Unavailable measurements are JSON `null` with a reason in `status`. Predictor
GPU time, predictor-ID D2H latency, attention time, and actual source-to-target
lead time must not be guessed. Separately measured values will be added in
Phase 2.

## Portability requirements

- Shapes and limits come from model metadata and graph tensors.
- Pinned memory and asynchronous copies are backend capability checks, not
  NVIDIA-name checks.
- sm_61 and sm_75 builds use their native architecture values.
- No MMQ forcing is introduced.
- Feature flags work with different layer counts, expert counts, expert sizes,
  fixed-tier sizes, and VRAM budgets.
- A future scratch budget is configured in bytes or slots and is safe to clamp.

## Phase gates

Phase 1 is complete only when:

- feature-off follows the previous graph path,
- baseline and trace token/output hashes match in deterministic no-MTP runs,
- JSON validates and is request-local,
- multiple prompts and at least 512 output tokens have been measured,
- recall and weighted cold coverage are available per layer,
- trace overhead is reported separately from model throughput.

Phase 1 routing-quality collection can complete while the scheduling gate remains open. No productive prefetch is authorized until predictor cost, source-to-target lead time, pinned H2D time, and the Oracle schedule are measured.

Phase 2 may simulate transfers only after real pinned H2D measurements exist.
It must calculate an Oracle with the same scratch and bandwidth constraints.

Phase 3 may start only if both the predictor schedule and the Oracle predict a
material net saving after predictor cost, D2H, H2D, false positives, and late
copies. A positive recall number alone is insufficient.

## Future routing-exact split

If Phase 3 is approved, the actual set will be partitioned into disjoint sets:

```text
fixed_hits + ready_prefetch_hits + cold_misses = actual_ids
```

False positives are outside `actual_ids` and are never computed. Pending or
late copies remain cold misses. True gate weights are applied once. No fixed
slot is mutated. This partition belongs to a future change and is not present
in Phase 1.
