# Hybrid design

Date: 2026-08-02
Status: Phases 0-3 are implemented behind default-off feature flags. Async
prefetch is validated for no-MTP execution. Warmcache plus MTP is guarded off
by default after deterministic controls failed for both synchronous and async
warm residency. Phase 4 has a strictly guarded static no-sync experiment;
deferred adaptation and double-buffered counts are not implemented.

## Goals and non-goals

The hybrid extends llama-wackMall's generic expert tier. The existing llama.cpp CUDA graph, attention kernels, model graph builders, target/draft contexts, speculative verification, and MTP rollback stay authoritative.

In scope:

- strict import of a LuceBox Spark hotness table;
- persistent expert usage compatible with the existing wackMall CSV;
- a small bounded warm LRU after the fixed hot range;
- synchronous promotion and a bounded asynchronous H2D implementation;
- telemetry and hard validation;
- no-MTP and MTP-1/2/3 correctness and performance measurements.

Out of scope:

- LuceBox server, qwen35moe dense/attention graphs, KVFlash, custom sampling, or speculative implementation;
- forced MMQ or non-native PTX builds;
- speculative expert prediction in the first implementation;
- multi-request cache mutation before explicit ownership and locking exist.

## Phase 1: profile conversion

`tools/convert_luce_spark_profile.py` has three independent parsers:

1. Spark dense CSV parser.
2. GGUF metadata reader for scalar/string metadata only.
3. wackMall sparse CSV writer and validator.

The Spark parser requires the dimension header and exactly `n_layer` data rows with exactly `n_expert` unsigned integer counts each. It rejects extra rows, short rows, negative values, floating point values, empty cells, trailing fields, and values beyond signed 64-bit range (the native wackMall parser's limit). It does not normalize counts or infer row order: row index is the layer ID and column index is the expert ID because the source header explicitly declares that format.

The GGUF reader validates magic/version, metadata types, `general.architecture`, `<arch>.block_count`, `<arch>.nextn_predict_layers`, `<arch>.expert_count`, and `<arch>.expert_used_count`. It reads only metadata and skips tensor payloads. The profile layer count must equal `block_count - nextn_predict_layers`; this explicitly excludes appended MTP heads from a base-model hotness profile. The output contains one `layer,expert,count` row for every dense source cell, including zero counts by default, so ordering and expert coverage remain explicit. An optional `--drop-zero` may be used only when requested.

The output path must differ from the input path and an existing output is
always rejected. The file is written to a sibling temporary path, flushed, and
published with an exclusive hard link, so a concurrent writer cannot be
overwritten. Conversion statistics report dimensions, total selections,
non-zero entries, zero entries, per-layer totals, count range, and model
metadata. The report states that Spark CSV has no source-model fingerprint.

Raw conversion preserves every source count exactly. The explicit
`--placement-slots S` mode is a separate, non-silent policy: it keeps each
layer's top-S order but emits bounded scores S..1 and zeros elsewhere. This
prevents large lifetime Spark counts from dominating wackMall's online
adaptation score for an entire request.

## Phase 2: synchronous warmcache

### Configuration

```
LLAMA_EXPERT_WARM_SLOTS=0
LLAMA_EXPERT_WARM_POLICY=lru
LLAMA_EXPERT_WARM_RESET=request
LLAMA_EXPERT_WARM_PREFETCH=0
LLAMA_EXPERT_WARM_ADMISSION=immediate
LLAMA_EXPERT_WARM_ADMISSION_WINDOW=8
```

`W=0` takes the existing allocation and update path without new copies or
policy work. Only `lru` is accepted. Invalid warm settings log a clear error
and disable the warm tier. In server mode, `request` rebases LRU ages after
request validation and before sampler/graph launch; it does not alter slot
ownership. `persistent` retains ages for the context lifetime. Other
executables retain context-lifetime ages unless they call `request_begin()`.

The synchronous MVP supports integer W and `auto`. Auto consumes only the
remaining slot budget after the fixed tier and sentinel, and is deliberately
capped at four slots. A forced integer is clamped down safely and the effective
value is logged. Admission can be immediate, second-hit, or decayed frequency;
the latter two use the explicit graph window. Defaults preserve immediate LRU
behavior.

### Layout

With warmcache enabled, each routed expert weight store allocates:

```
fixed:    slots [0, S)
warm:     slots [S, S + W)
sentinel: slot  S + W
```

All matrices in one layer share the same logical slot ownership. Slot mutation is a layer transaction across gate/up/down or gate_up/down stores.

Host state per layer:

```
fixed_expert[S]
fixed_dwell[S]
warm_expert[W]
warm_last_use[W]
lut_host[N]
score[N]
cumulative[N]
clock
```

The existing CPU mask remains derived state: zero for fixed or ready warm experts, one for sentinel-mapped experts. The GPU LUT is published from `lut_host` after all weight slices and host reverse maps are ready.

### Synchronous update transaction

The completed graph writes selection counts. After the existing scheduler barrier, each layer update executes in this order:

1. Snapshot counts and classify every selection against the pre-update LUT.
2. Accumulate fixed hits, warm hits, cold hits, total selections, and long-term score.
3. Touch resident warm entries in deterministic selection order available from the counter source. If only aggregate counts are available, touch each selected expert once in ascending expert ID and record the limitation; exact routing order requires a later trace buffer.
4. Run at most one fixed-tier adaptation decision on a copy of the ownership
   state. If its source was warm, the old warm slot becomes empty in that copy.
5. Sort cold candidates by count descending, then expert ID ascending.
6. Admit at most W candidates into empty or least-recently-used slots; a slot
   already written in this transaction is protected from a second eviction.
7. Copy every weight slice synchronously before publishing the transaction's
   LUT and mask.
8. Commit the ownership state and upload the derived LUT/mask once.
9. Validate host invariants and zero consumed counters.

`ggml_backend_tensor_set()` has no recoverable status result. The old published
host mapping remains valid until all synchronous writes return, but a backend
or process failure aborts the run; it cannot be converted into a cold fallback
inside this API.

### LRU semantics

- Fixed hits never change fixed placement or warm age.
- Warm hits increment the layer clock and update one warm timestamp.
- A cold miss may fill one warm slot after its current CPU computation is complete.
- Empty warm slots precede eviction.
- Eviction selects the smallest timestamp; slot index breaks ties.
- Repeated selection of one expert in a batch causes one residency operation and contributes its full count to statistics.
- Fixed adaptation and warm LRU never share dwell/timestamp state.

### MTP safety

The current synchronous update remains after `process_ubatch()` completion and
the existing scheduler synchronization. Therefore one target verify graph sees
an immutable LUT and immutable slot contents for all of its tokens. Counts from
the verify graph may update the cache only for a later graph. The MTP draft
graph uses the next-token layer, which is not in the base-layer tier store.

Warmcache remains opt-in. Deterministic 64-token MTP-2 controls failed for both
synchronous and asynchronous warm residency even though no-MTP hashes matched.
The guard is therefore automatic: when `draft-mtp` is parsed, warm slots are
disabled before target-context creation while the fixed tier remains active.
`LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL=1` bypasses this only for controlled
diagnostics. The benchmark evidence and first divergent token are recorded in
`HYBRID_EXPERIMENTS.md`.

## VRAM auto-fit

The existing tier initialization runs after model, KV, and compute allocation. It measures physical free VRAM at that point and subtracts the existing 512 MiB safety reserve.

The allocation decision is:

```
available = max(0, free_vram - safety_reserve)
fixed_effective = min(requested_fixed, available / bytes_per_all_layer_slot)
available -= fixed_effective * bytes_per_all_layer_slot
warm_effective = min(requested_warm, available / bytes_per_all_layer_slot)
```

This preserves the requested priority: dense model, KV, compute, reserve, fixed hot slots, then warm slots. Both requested and effective S/W plus total GiB are logged. `W=1`, `W=2`, and `W=4` are the only initial benchmark sizes.

## Phase 3: asynchronous prefetch

The asynchronous state machine adds per warm slot:

```
EMPTY
READY(expert, generation)
COPYING(expert, generation, event)
```

Rules:

- A `COPYING` slot is never present as a valid expert LUT entry.
- The evicted mapping is removed only when it is safe to replace the slot. Until then, a new miss may be dropped instead of blocking.
- Host expert storage must be pinned or registered for true asynchronous H2D.
- One dedicated prefetch stream is used initially on sm_75.
- All slices are enqueued on that stream and one completion event closes the transaction.
- A worker waits for the copy event without holding tier metadata. The main
  update path polls completed jobs only after the current graph barrier and
  publishes the LUT for a later graph. It never makes the compute stream wait
  for an incomplete copy; the sentinel/CPU path wins instead.
- A maximum of two fills are in flight by default.
- Shutdown drains events before freeing host or device buffers.

Configuration is:

```
LLAMA_EXPERT_WARM_PREFETCH=1
LLAMA_EXPERT_PREFETCH_STREAMS=1
LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT=2
```

Only one stream is accepted on the GTX 1660 Ti. Pinned staging buffers are
allocated through the CUDA backend host-buffer type. One event closes all
gate/up/down slice copies for a layer. In-flight slots are removed from the
LUT, protected from eviction, and represented separately from READY ownership.
Layer admission priority rotates so a global in-flight cap cannot starve later
layers.

The first algorithm is reactive: a cold selection is computed on CPU for the current graph and scheduled for a later graph. No predictor is used.

Graph lifetime must be explicit. A slot cannot be evicted while a submitted target graph may reference its generation. The safe initial implementation keeps the existing post-ubatch barrier even with asynchronous H2D, using async copies only to overlap the next graph's unrelated work. Removing the barrier is a distinct Phase 4 change.

## Phase 4: overlap and synchronization

Instrumentation precedes synchronization changes. Timers are runtime-disabled by default and cover:

- router GPU compute and host readback;
- hot GPU expert path;
- cold CPU expert path;
- H2D and D2H bytes/time;
- scheduler global synchronization;
- event wait;
- hot/cold combine;
- sampling;
- MTP draft and verification.

Candidate changes, in order:

1. Skip the tier-specific unconditional barrier when tier mutation and tier stats are both disabled, relying on the normal output dependency.
2. Update adaptation less often, with double-buffered counts, so most tokens do not mutate weights.
3. Keep repinning at request or larger intervals, outside MTP verify batches.
4. Use pinned cold-output buffers and enqueue combine upload after CPU completion.
5. Replace global tier waits with backend/slot events only after graph ownership is tracked.

No synchronization change is accepted based on GPU utilization alone. It must preserve deterministic controls and improve median 2,000-token sustained decode.

### Static no-sync upper-bound experiment

`LLAMA_EXPERT_STATIC_NO_SYNC=1` removes only the expert-tier post-graph
barrier. It becomes active only when all of the following hold:

```
LLAMA_EXPERT_ADAPT=0
LLAMA_EXPERT_WARM_SLOTS=0
LLAMA_EXPERT_STATS=0
LLAMA_EXPERT_USAGE=0 or unset
LLAMA_EXPERT_STATS_JSON=0 or unset
```

Invalid flag values or any active consumer reject the mode with a diagnostic
and preserve the synchronized path. In the accepted mode the fused cold op is
given no count tensor, prompt-only count nodes are omitted, `update()` is a
defensive no-op, and `process_ubatch()` does not execute the tier-specific
`ggml_backend_sched_synchronize()`. Normal output, sampling, and MTP dependency
synchronization remains authoritative.

The isolated 3x2,000-token comparison improved the static median by only 0.612
percent. This is an upper-bound result for the removed tier barrier, so it does
not justify double-buffered counts or lock-free adaptation on this hardware.

## Statistics

Existing text stats and `LLAMA_EXPERT_USAGE` remain supported. Optional JSON at
`LLAMA_EXPERT_STATS_JSON` is written at orderly shutdown and contains the model
description, per-layer counts, S/W, hit categories, repins, warm
promotions/evictions, transfer counters, timing buckets, and decode fields.

MTP draft width is wired from parsed common parameters and appears in `mtp_n`.
Context, decode timing, CPU/GPU expert time, and synchronization time are not
yet wired into the expert-tier module and therefore appear as literal zero.
The benchmark client and server log provide the available request-level
values. Runtime counters use uint64 and duration totals use milliseconds.
For all three output variables, the literal value `0` now means disabled and
does not create a file named `0`.

## Validation and abort diagnostics

Validation checks every layer after initialization and every mutation:

- every occupied fixed/warm slot owns one in-range expert;
- fixed and warm reverse maps are disjoint;
- each expert has at most one slot;
- `lut_host` agrees with reverse maps;
- all other experts map to the final sentinel;
- CPU masks equal the derived residency state;
- the host sentinel slot is unowned;
- fixed slots are never selected by LRU.
- every pending expert is unique, cold/sentinel-mapped, and paired with one
  empty warm slot;
- a pending slot cannot be selected as an eviction target.

On failure the diagnostic includes layer, state, request ID, and MTP step
(`-1` until that context is plumbed), plus expert/slot detail emitted by the
state validator where relevant, then aborts. Cheap host assertions run after
initialization, mutation, request-age reset, and stats emission. Device LUT and
sentinel-byte readback is not implemented in this MVP.

## Benchmark gate

The benchmark script runs a warm-up process and a fresh measured
server process for each repetition. It fixes model/prompt/context/KV/batch/
ubatch/temperature/output length and records raw logs plus a summary. The
process split avoids contaminating measurements with the independently found
MTP second-request failure; it is a workaround, not a fix. A serious candidate
gets at least three repetitions and reports the median. The primary metric is
sustained decode over 2,000 output tokens.

Stop a direction on output corruption, rollback failure, hang, Xid, OOM, a
greater than 3 percent regression, or higher GPU utilization with lower
throughput. Frequency admission removed most synchronous copy churn. Async W=2
then matched the no-MTP control hashes and recovered the short-run regression,
but its first 2,000-token no-MTP run was 3.20% below the three-run W0 median and
diverged at token 86, so the sustained async direction also hit a stop gate.
Warm+MTP changed deterministic output for both sync and async modes and is
guarded off. The earlier nondeterministic fault resolves to
`dequantize_row_q4_0`; three later 2,000-token release repetitions completed,
but do not prove that independent fault fixed. The guarded Phase 4 upper-bound
experiment is complete, while deferred adaptation remains unimplemented. See
`HYBRID_EXPERIMENTS.md` for exact evidence and limitations.
