# wackMall/LuceBox hybrid analysis

Date: 2026-08-01

This document records the Phase 0 source analysis for an experimental hybrid. The base remains llama-wackMall. LuceBox is a read-only architecture reference; its dense graphs, attention implementation, server, and Turing workarounds are not imported.

## 1. Protected repository inventory

The source repositories and data paths were verified before any hybrid work:

| Item | Verified state |
|---|---|
| `src/llama-wackMall` | commit `1c801502f53a54ac77c1733c1dc2c9d7f7053ff6`, branch `main` |
| `src/lucebox` | commit `bc881af248de9c03336bb9afa54735d1af5f273f`, branch `main` |
| Hybrid worktree | `src/llama-wackMall-hybrid`, branch `codex/hybrid-warmcache`, based on wackMall commit above |
| MTP model | `models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 22,663,387,424 bytes |
| LuceBox model | `models/qwen3.6-35b-a3b-luce/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 22,134,528,992 bytes |
| Spark profile | `models/qwen3.6-35b-a3b-luce/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.spark.csv`, 28,610 bytes |

The original wackMall tree had one tracked local change in `common/arg.cpp`: `LLAMA_CMOE_BATCH` and `LLAMA_CMOE_UBATCH` override the cmoe defaults after argument parsing and reject invalid values. That exact diff was reproduced in the hybrid worktree. Existing untracked benchmark logs, backup files, `BUILD_COMMIT.txt`, and `bench-cmoe-batch.sh` were left untouched in the original tree. The original LuceBox tree has untracked `BUILD_COMMIT.txt` and an old CUDA source backup; neither was modified.

The existing `build` and `build-mmq` directories in the original wackMall tree are protected. The hybrid uses a new `build-hybrid` directory and native `CMAKE_CUDA_ARCHITECTURES=75`; it must not set `GGML_CUDA_FORCE_MMQ`.

## 2. Expert storage layouts

### 2.1 wackMall

The relevant implementation is `src/llama-expert-tier.cpp`, called generically from `llm_graph_context::build_lora_mm_id()` and `build_moe_ffn()` in `src/llama-graph.cpp`.

For every host-resident routed expert weight tensor, wackMall creates:

```
original weight on host: [ne0, ne1, N]
GPU .hot tensor:         [ne0, ne1, S + 1]
GPU LUT:                 int32[N]
CPU cold mask:           int32[N]
CPU counts:              int32[N + 1]
```

Slots `[0, S)` are fixed/adaptive hot slots. The last allocated and initialized slot is the sentinel. `lut[e]` contains a hot slot or the sentinel, while `mask[e]` is zero for GPU-resident experts and one for CPU experts. Cold selections are remapped to the zero-filled sentinel for the normal CUDA `mul_mat_id`; the CPU cold operation computes the complementary contribution. A final `ggml_add` combines the two exactly once.

For models eligible for `MOE_COLD`, the graph builds only hot gate/up/down work through the regular CUDA path and creates one late fused CPU cold node. Qwen3.6 MoE is eligible because it uses the shared `build_moe_ffn()` path, separate or fused gate/up tensors, SiLU, and no excluded Step35 semantics. Larger graphs bypass tiered compute and attach `MOE_COUNT` only, preserving stock prompt performance.

The current allocation is uniform across layers. The auto-fit estimates the bytes for one expert across all host expert tensors, subtracts a flat 512 MiB safety reserve from physically free VRAM, and chooses one `S` for every layer. The allocated GPU buffer is tagged `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`, which is required for scheduler placement.

### 2.2 LuceBox

The common layout is in `server/src/common/moe_hybrid_storage.{h,cpp}`. Model adapters construct `MoeLayerDesc`; the qwen35moe backend builds a completely separate execution pipeline around it.

For every layer LuceBox materializes compact hot and cold stacks and host-side maps:

```
hot stack:  [tensor dimensions, hot_active + cache_slots]
cold stack: [tensor dimensions, cold_count]
global -> hot-local map
global -> cold-local map
VRAM residency bitset
spare_global[cache_slots]
spare_lru[cache_slots]
```

The calibrated hot set occupies `[0, hot_active)`. Bounded cache entries occupy `[hot_active, hot_active + cache_slots)`. There is no sentinel slot in this layout. Callers partition selected global IDs on the host and either remap them to compact local IDs or mask their router weights to zero for a GPU graph. Cold work runs through a compact CPU graph or a custom fused expert runtime.

The qwen35moe `PipelinedDecodeState` also owns cached pre-FFN graphs, routed-FFN graphs, persistent host routing buffers, CPU expert output buffers, and a GPU residual/hot/cold combine graph. This is materially more invasive than wackMall's generic graph rewrite.

### 2.3 Layout decision for the hybrid

The hybrid keeps wackMall's sentinel design and original host tensors:

```
[fixed hot slots S][warm slots W][one zero sentinel]
```

It does not create LuceBox-style dense, attention, cold-stack, or server graphs. A single LUT remains the graph-visible source of truth. Slot ranges give every expert one state:

```
lut[e] in [0, S)       -> HOT_FIXED
lut[e] in [S, S + W)   -> WARM_CACHE
lut[e] == S + W        -> COLD_CPU
```

This preserves the CUDA `.hot` path, generic MoE integration, CPU cold fallback, and sentinel algebra already tested by wackMall.

## 3. Hotness and cache algorithms

### 3.1 wackMall today

- `LLAMA_EXPERT_HOT` reads sparse rows in `layer,expert,count` form.
- Per-layer scores are seeded from the profile.
- CPU `MOE_COLD` or `MOE_COUNT` accumulates expert selection counts.
- `maybe_update()` applies `score = score * decay + count` after a completed ubatch.
- With online adaptation enabled, at most one cold expert replaces the coldest hot slot per layer and update. The incumbent must dwell for 32 updates and the challenger must exceed 1.5 times its score.
- `LLAMA_EXPERT_USAGE` writes cumulative counts in the same sparse format, so learned usage is already persistent across runs if the output is reused manually.
- The original parser was permissive: it skipped invalid layers, did not validate the header, did not reject duplicate expert rows, and indexed seed expert IDs without an explicit range check. The hybrid hardens this parser in addition to validating profiles in the converter.

### 3.2 LuceBox today

- `MoeHybridRoutingStats` accumulates a dense `uint64[n_layer * n_expert]` table from routed IDs.
- Spark CSV begins with `# hotness table: n_layer=... n_expert=... n_expert_used=...`, followed by exactly one dense row per layer and one column per expert.
- `MoeHybridPlacement::build_from_stats_with_layer_bytes()` greedily allocates the next expert with the best activation-count-per-byte gain, subject to a byte budget and per-layer floor.
- The live accumulator is seeded from the loaded profile and saved after requests, creating a true persistent traffic profile.
- Each layer has a bounded spare-slot LRU. A resident cache hit updates its timestamp. A miss chooses an empty slot or the least-recently-used slot, copies the expert's two or three slices, and rewrites host maps.
- The current `moe_hybrid_cache_swap_in()` uses asynchronous tensor writes but publishes residency immediately. It has no explicit per-slot in-flight state or completion event. That behavior is a useful performance reference, not a safe implementation to copy into wackMall.

### 3.3 Hybrid policy

Long-term and short-term state remain separate:

- Fixed slots use imported/persisted scores plus existing hysteretic adaptation.
- Warm slots use request-local or persistent LRU timestamps only.
- Counts update fixed-tier scores regardless of current warm residency.
- A fixed-tier promotion may take an expert from a warm slot, but it must clear that warm slot and publish exactly one LUT mapping.
- The synchronous MVP changes residency only after the current graph has completed. It may use the just-completed graph's selection counts to fill warm slots for later graphs; the completed graph still receives the CPU cold contribution.
- Asynchronous prefetch will publish a warm LUT entry only after its copy event is complete. A not-ready expert remains mapped to the sentinel and is computed on CPU.

The verified Spark file is dense 40 x 256 with declared top-8 routing. Conversion can therefore be exact if the target GGUF reports the same architecture dimensions. The Spark format carries no source model fingerprint, quantization identity, or tensor hashes; the converter must state that limitation instead of claiming model identity.

## 4. Synchronization points in wackMall

### 4.1 Explicit application-level synchronization

1. `llama_context::process_ubatch()`, `src/llama-context.cpp`: every ubatch normally calls `ggml_backend_sched_synchronize()` after asynchronous graph submission, then calls `llama_expert_tier::update()`. This is the expert-tier global barrier. The experimental static no-sync guard skips this pair only when adaptation, warm slots, and every count/stat consumer are disabled; normal output dependencies are unchanged. Mutable modes retain the barrier for count ownership. A process-wide graph guard serializes target/draft construction through post-graph harvesting, while fixed repins publish only from `request_begin()`.
2. `llama_context::process_ubatch()`, lines 1335-1340: graph reuse synchronizes before input mutation when pipeline parallelism is enabled.
3. `llama_context::synchronize()`, lines 698-720: public output/performance synchronization waits all scheduler backends. Logit, embedding, sampling, and MTP hidden-state consumers ultimately use this barrier when host data is needed.
4. `llama_context::output_reserve()` synchronizes before replacing a host output buffer. This is allocation-only, not a per-token steady-state barrier.
5. Scheduler reserve/reallocation calls synchronize before moving graph buffers. These are graph-shape/allocation events rather than normal steady-state decode.

### 4.2 Scheduler-internal synchronization relevant to MoE

`ggml_backend_sched_compute_splits()` in `ggml/src/ggml-backend.cpp` submits backend splits in graph order and uses events where supported. Important waits are:

- Lines 1571-1583: wait before overwriting a split input copy.
- Lines 1597 and 1615-1617: synchronize the input/ID backend when the stock host-weight expert-copy optimization must read selected IDs.
- Lines 1671-1680: synchronize source/destination around a fallback synchronous inter-backend copy when `cpy_tensor_async` is unavailable.
- Lines 1686-1689: normal split graphs are submitted asynchronously.
- Lines 1714-1716: an evaluation callback forces a backend synchronization for requested intermediate tensors.
- `ggml_backend_sched_synchronize()`, lines 1913-1923, synchronizes every backend, not just the one that owns mutable expert state.

The tiered `.hot` tensor is already GPU-resident, and the CPU cold tensor remains host-resident, so it avoids the stock per-token full-expert H2D path. The late `MOE_COLD` node is intentionally built after the hot down projection. The scheduler can have GPU hot work in flight while it executes the CPU split, but dependencies and cross-backend copies still impose waits at split boundaries. Measurement is required before changing this ordering.

## 5. Synchronization points in LuceBox

The most relevant qwen35moe path is `pipelined_decode_one_token()` in `server/src/qwen35moe/qwen35moe_pipelined_decode.cpp`:

1. Lines 448-452: pre-FFN/router graph launches asynchronously, then the entire GPU backend is synchronized so host code can read selected IDs and weights.
2. Lines 458-470: `tensor_get` reads routing data and optionally `ffn_post`; these are host readback dependencies even after the explicit barrier.
3. Lines 527-574: routed hot FFN launches asynchronously, CPU cold compute runs on the calling thread, and cold output is uploaded before the queued combine graph. Same-stream ordering avoids an additional explicit hot wait here.
4. Lines 608-610: entering the attention/fallback split path synchronizes pending work.
5. Lines 654 and 690-711: several attention/dynamic pre-FFN paths use synchronous `graph_compute` rather than async submission.
6. Lines 726-730: selected IDs and weights are copied asynchronously to host and followed by one backend synchronization.
7. Lines 799-841: the alternate routed FFN path again overlaps async hot GPU work with synchronous CPU expert compute and queues combine afterward.
8. Lines 896-905: `ffn_post` is read before launching hot work, intentionally moving the unavoidable readback ahead of the overlap window.
9. Lines 911-1023: split hot work is async, CPU cold work runs concurrently, and hot/cold outputs are queued into the combine graph without an explicit success-path hot barrier.
10. Lines 1047-1051: one final backend synchronization makes the persistent activation available to the caller.

The common `eval_moe_hybrid_ffn_single()` path in `server/src/common/moe_hybrid_ffn_eval.cpp` has the same pattern: async hot launch around lines 1447-1451, synchronous CPU cold graph around 1511, and a hot synchronization immediately before reading/combining around 1537-1543. Error paths synchronize before returning if hot work is in flight. `eval_moe_hybrid_ffn_gpu_resident()` reads `ffn_post` before hot launch, synchronizes hot work only at the combine dependency, and uses synchronous combine at the end.

LuceBox therefore demonstrates real CPU/GPU overlap, but it still pays one router readback barrier per layer and a final barrier. Its measured `Sync Stall` is consistent with this structure. Copying its full pipeline would replace wackMall's faster general CUDA graph, so only its dependency ordering and telemetry categories should influence the hybrid.

## 6. Generic mechanisms versus qwen35moe coupling

### Model-independent mechanisms

- Strict conversion between dense `(layer, expert)` counts and wackMall sparse counts.
- Persistent cumulative expert usage.
- Fixed-slot versus bounded-warm-slot state separation.
- LRU timestamps, empty-slot preference, bounded eviction, and per-slot in-flight state.
- Per-layer LUT uniqueness and sentinel validation.
- Copy-completion events and publish-after-completion semantics.
- Request-local stat deltas, reset policy, and machine-readable telemetry.
- Hysteresis for fixed-tier repinning.
- Byte-based auto-fit using actual expert slice sizes.

These remain generic because wackMall hooks the shared MoE graph builder and obtains all tensor shapes from loaded tensors.

### LuceBox parts too coupled to qwen35moe or its custom server

- `PipelinedDecodeState`, cached DeltaNet pre-FFN graphs, padded attention windows, custom KVFlash handling, and persistent `act_cur`.
- The qwen35moe backend's manual per-layer loop and direct router readback.
- `TargetWeights`, `TargetCache`, qwen-specific SSM snapshots, speculative decoder, and rollback implementation.
- LuceBox's custom fused expert runtime and compact cold-stack descriptors.
- Its server request lifecycle, daemon IPC, pre-gate trace collector, and request-boundary persistence hooks.
- MMQ reduced-stack workarounds and Turing sub-batching. wackMall's sentinel `.hot` path already has its own CUDA fixes and must remain native sm_75 without forced MMQ.
- LuceBox dense, attention, projection, sampling, and speculative graphs.

## 7. MTP interaction

The target and MTP contexts share one loaded model but build different graph types. The Qwen3.6 target graph uses base layers `[0, n_layer)`. The MTP graph in `src/models/qwen35moe.cpp` uses the next-token layer at index `n_layer` and has its own attention KV context. `llama_expert_tier::init()` currently creates stores only for base layers, so the MTP head's routed FFN remains on the stock path. The hot/warm tier affects target prefill/decode/verification, not the MTP draft head itself.

The MTP driver in `common/speculative.cpp`:

- catches the draft context up from target `h_nextn` rows,
- drafts one token per loop iteration for Qwen's single head,
- verifies draft tokens in a target batch,
- keeps the accepted hidden row,
- rolls the draft/target memories back through existing memory APIs.

The hybrid must not change those KV, SSM, hidden-row, sampling, or rollback operations. Warmcache risk comes from target verify graphs containing 1-3 tokens while sharing the same mutable `.hot` tensors and LUT. The safe rule is:

- LUT and slot contents are immutable from target graph submission through target graph completion.
- A synchronous update occurs only after the verify graph is complete.
- An asynchronous fill is not published while any graph can reference that slot.
- Fixed repinning and warm eviction are forbidden inside a running verify graph.
- One verify graph sees one stable LUT for all of its tokens.
- CUDA graph capture/reuse is disabled while fixed expert weights are mutable;
  the CUDA graph path failed on the request after an otherwise valid repin.

No-MTP, MTP-1, MTP-2, and MTP-3 exercise the same target cache. The completed
implementation preserves graph-lifetime rules for both synchronous and async
fills, but deterministic MTP-2 controls still diverged for both modes while
no-MTP controls matched. This shows graph ownership is necessary but not
sufficient: changing a verify expert between CPU and GPU arithmetic can change
speculative decisions. The implemented guard detects parsed `draft-mtp` before
target initialization and disables warm slots by default while retaining fixed
hot slots. An explicit experimental environment variable is required to
bypass it. Exact measurements are in `HYBRID_EXPERIMENTS.md`.

## 8. Work that can be tested before MTP

The following can be validated without speculative decoding:

1. Spark CSV parser and converter unit tests.
2. GGUF dimension and architecture validation.
3. Sparse wackMall profile validation and deterministic conversion.
4. Fixed/warm/sentinel LUT state transitions in a backend-independent state-machine test.
5. LRU hit, empty fill, eviction, fixed promotion, and duplicate prevention.
6. Warm slots disabled (`W=0`) producing the exact previous layout and behavior.
7. Synchronous warm fill after a completed single-token graph.
8. Sentinel-zero and CPU/GPU exactly-once controls.
9. Greedy token/output hashes with adaptation disabled.
10. VRAM auto-fit and forced `W=1,2,4` clamping.

Only after these pass should target verify batches and MTP rollback be tested.

## 9. Planned files and functions

Minimal Phase 1 changes:

| File | Planned change |
|---|---|
| `tools/convert_luce_spark_profile.py` | Strict Spark/wackMall/GGUF validation and dense-to-sparse conversion |
| `tests/test-convert-luce-spark-profile.py` | Parser, dimensions, duplicate/range, malformed metadata, and no-overwrite tests |
| `src/llama-expert-tier.cpp` | Harden native sparse profile parsing; later add persistent JSON stats |
| `src/llama-expert-tier.h` | Small lifecycle/testing interfaces only if required |
| `README.md` | Document new opt-in environment variables |

Synchronous warmcache changes:

| File/function | Planned change |
|---|---|
| `src/llama-expert-tier.cpp:store` | Keep tensor pointers; classify slot range and expose counts source |
| `src/llama-expert-tier.cpp:layer_tier` | Add `n_fixed`, `n_warm`, warm owner/LRU arrays, counters, and invariant state |
| `src/llama-expert-tier.cpp:init()` | Parse flags, allocate `S + W + 1`, auto-fit fixed first and warm second, initialize sentinel at the final slot |
| `src/llama-expert-tier.cpp:maybe_update()` | Attribute hot/warm/cold hits from the pre-update LUT, update LRU, synchronously fill cold misses, then perform separated fixed adaptation |
| `src/llama-expert-tier.cpp:dump_stats()` | Extended invariant checks and optional JSON stats |
| `src/llama-context.cpp:process_ubatch()` | Keep the existing safe barrier for MVP; later make it conditional/measure it |
| `tests/test-expert-warm-cache.cpp` | Pure state-machine tests for slots, LUT, eviction, promotion, sentinel, and MTP-style stable batches |
| `CMakeLists.txt` or `tests/CMakeLists.txt` | Register the focused test target |

Asynchronous work, only after synchronous correctness:

| Area | Planned change |
|---|---|
| CUDA/backend helper | One prefetch stream and completion event per warm slot, without forcing MMQ |
| Tier state | `EMPTY`, `COPYING`, `READY`, generation/owner, no-evict-in-flight checks |
| Publish protocol | Copy all expert slices, record event, publish LUT only after completion |
| Shutdown/cancel | Drain or cancel safely before freeing buffers; never expose partially replaced weights |
| Instrumentation | hot GPU, cold CPU, H2D, D2H/readback, scheduler sync, event wait, combine, router, sampling, MTP draft/verify |

Implementation addendum (2026-08-02): the profile converter, JSON hit/copy
statistics, bounded LRU state, immediate/second-hit/frequency admission,
synchronous fill, and one-stream asynchronous prefetch are implemented. Async
uses pinned staging, completion events, a bounded global job pool, rotating
layer priority, sentinel fallback, and publish-after-completion. Full timing
instrumentation and scheduler-barrier removal remain pending. Warm+MTP remains
guarded after the deterministic failure described above.

## 10. Risk analysis

| Risk | Failure mode | Mitigation/gate |
|---|---|---|
| Incorrect sentinel index | A cold ID reads stale/non-zero GPU weights | Sentinel is always final slot `S+W`; zero its slices at allocation; assert every cold host LUT entry equals it; add device readback before claiming full validation |
| Duplicate residency | Expert is valid in fixed and warm slots or two warm slots | Single host LUT source of truth; reverse-map uniqueness check after every mutation in validation mode |
| Exactly-once error | Hot/warm expert also computed by CPU, or cold contribution omitted | Derive CPU mask from published LUT only; publish after all slice copies; count combine paths in tests |
| In-flight overwrite | GPU graph reads a warm slot while it is evicted or copied | Synchronous MVP mutates only after global completion; async phase adds slot events, graph generations, and no-evict-in-flight |
| Partial expert copy | gate/up/down belong to different experts | Treat all slices as one transaction and publish LUT last; synchronous backend/process failures abort because `tensor_set` has no recoverable status |
| Repin/warm race | Long-term adaptation and LRU move the same expert concurrently | Serialize metadata under one tier update owner; clear warm ownership before fixed publication |
| MTP verify instability | Tokens in one verify batch see different LUT/weights, or CPU/GPU arithmetic changes verify decisions | No tier mutation during a graph; explicit batch-stability test; default-off warm guard after deterministic MTP mismatch |
| MTP rollback corruption | Cache lifecycle accidentally changes KV/SSM state | Do not touch speculative/KV APIs; compare MTP acceptance, accepted length, output/token hash, and rollback controls |
| VRAM OOM | Warm allocation consumes graph-capture/KV reserve | Fixed allocation first; warm slots use only remaining measured budget; clamp manual request; conservative W=1/2/4 tests |
| Hidden memory regression | One `W` means one slot in every MoE layer and every routed tensor | Report bytes per all-layer slot and actual total before allocation; `W=0` default |
| Deadlock | Waiting for a copy event on a stream that depends on compute holding metadata | No blocking while holding metadata lock; one-way event dependencies; bounded in-flight queue; shutdown drain test |
| Data race in global state | Multiple contexts/parallel requests mutate process-global tier maps | Shared graph guard covers target/draft execution and publication; request-boundary transaction plus validation abort includes request/MTP data |
| Counter race/overflow | CPU counts are read/reset while a graph writes them, or int32 wraps | MVP retains scheduler barrier; use uint64 cumulative totals; future readback uses double-buffered counters/events |
| Performance regression | Warm copies and global update cost exceed saved CPU work | Feature off by default; benchmark 2,000-token sustained decode; stop if median drops more than 3 percent |
| Turing kernel fault | Reduced-stack/MMQ behavior differs on sm_75 | Preserve wackMall `.hot` CUDA fixes; native arch 75; do not force MMQ; stop on hang, Xid, or incoherent output |
| Profile mismatch | Correct dimensions but unrelated model/traffic | Validate all available metadata, print source-profile identity limitation, never infer normalization or ordering, allow user review before use |
| Persistence corruption | Crash truncates learned profile | Converter never overwrites input; future runtime persistence writes a temporary sibling then atomically renames, with dimensions in header |

## 11. Phase 0 conclusion

The low-risk integration seam is `llama-expert-tier`, not the qwen35moe graph or speculative decoder. Phase 1 can be completed without altering compute semantics: validate/convert the dense Spark profile to wackMall's sparse profile and harden profile ingestion. The synchronous warmcache can then reuse the existing `.hot` tensor, LUT, mask, CPU fallback, and completed-ubatch mutation boundary.

Asynchronous prefetch was subsequently implemented with explicit slot lifetime
and completion state; LuceBox's immediate residency publication was not copied.
Removal of the unconditional post-ubatch global barrier remains a separate
experiment. CPU/GPU overlap should be improved only after telemetry shows
whether the dominant remaining wait is router readback, CPU cold compute,
inter-backend copy, or the final adaptation barrier.
