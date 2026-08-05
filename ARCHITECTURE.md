# Architecture: llama-wackMall expert-granular MoE tiering engine

First public disclosure: 2026-07-26. Updated 2026-07-29 for v3 (RAM pool,
pre-gated prefetch, per-layer budget fix, learned router heads roadmap).
This document is published to disclose the methods and systems described
herein as of those dates. Code is Apache 2.0 licensed (see LICENSE);
the authors grant no patent rights over the disclosed concepts.

Design rule: all model-specific knowledge lives in data (seed files), never
in code. Hooks exist only at the shared MoE graph-builder level
(build_moe_ffn / build_lora_mm_id), so every mechanism below applies to any
llama.cpp MoE architecture with the standard ffn_{gate,up,down}_exps tensor
layout. No per-model branches.

---

## 1. System overview

A sparse MoE layer routes each token to k of N experts. Stock llama.cpp
offloads whole layers; on small GPUs most experts end up on CPU and every
token streams its k active experts from RAM (~25 MB/token for a 122B-class
model), which sets the speed floor.

This engine tiers experts individually: a small set of hot experts lives in
VRAM, the rest stay RAM-resident and are computed on CPU only when routed.
Per-token traffic becomes (cold-selected unique experts) x (bytes/slot)
instead of (all active experts) x (bytes/slot).

```
GPU hot tier: S slots/layer x 3 matrices x ~1 MB/slot
    |
    v
Demand RAM pool: pool fill/evict with hysteresis, ptrs indirection
    |
    v
Speculative pool (prefetch): worker-fed, probe-served, learned heads target
    |
    v
mmap fallback: SSD model file, OS page cache
```

## 2. Hot store layout and the sentinel trick

Per MoE layer and per weight matrix (gate, up, down):

- `.hot` tensor [ne0, ne1, S+1] on GPU: S hot expert slices plus one
  zero-filled sentinel slot at index S.
- `lut` int32[N] on GPU: lut[e] = slot index if expert e is hot, else S.
- Graph rewrite for small graphs: ids_hot = get_rows(lut, ids), then the
  stock mul_mat_id runs on the .hot tensor with ids_hot. Cold selections
  map to the zeroed sentinel, contributing an exact zero:
  SWIGLU: silu(x*0) * (x*0) = 0; GELU variant: gelu(x*0) * (x*0) = 0.
  No masking, lossless by construction.

Because selection affects only which slot is read, placement changes speed,
never logits.

## 3. Cold execution paths (two kernels + one counter)

a) MUL_MAT_ID_COLD (generic, per-matrix). CPU op with the same I/O layout
as stock mul_mat_id. Computes only experts that are both selected and cold
(dedup inside the op), writes zeros for hot rows; a ggml_add merges with the
GPU result. Uses ptrs indirection for weight source resolution (pool | mmap).

b) MOE_COLD (fused 3-phase, gated). For architectures with fused gate_up
blocks, GELU-family activation, and no per-expert biases. Uses ptrs
indirection for gate/up/down weight addresses. Calls the predict hook at
ith==0 (pre-gate prediction for spec pool feed).

c) MOE_COUNT (count-only). The tiered kernels engage only for graphs of at
most TMAX tokens (default 16, the decode regime; larger prompt graphs keep
the stock path). MOE_COUNT harvests router decisions from large batches into
the same per-layer count buffer.

All three are registered ggml CPU ops with explicit get_n_tasks cases.

## 4. Hardware-aware auto-fit

Three-stage init at context creation:

1. Dense (non-expert) weights placed by the fit mechanism.
2. KV cache and compute buffers allocated.
3. Tier init measures the physically free VRAM and sets

   S = clamp( floor( (V_free - 512 MB) / bytes_per_slot ), 0, N_experts )

   with bytes_per_slot derived from the actual tensor quantization. S is
   uniform across layers; manual override via LLAMA_EXPERT_S. The 512 MB
   flat reserve covers runtime allocations (CUDA graph capture buffers are
   allocated after measurement).

`-cmoe` additionally auto-configures batch/ubatch 256, flash attention, KV
offload, np=1, and threads.

## 5. Online adaptation (the cache policy)

Graphs are built once and reused, so the learning hook is not in the graph:
it runs after each graph_compute at ubatch granularity. The hook runs only
after a full scheduler synchronize (graph_compute is async — swapping
buffers that an in-flight graph reads causes corruption).

- Counting: the cold kernels increment counts[expert] (plus a total); the
  hook consumes and zeroes them.
- Score: score[e] = score[e]*decay + count[e]. decay defaults to 1.0
  (pure cumulative); values < 1 recency-weight.
- Re-pin: per layer, hottest cold expert ec replaces coldest hot slot si
  iff score[ec] > 1.5 * score[slot_expert[si]] AND the incumbent has dwelt
  >= 32 updates. Empty slots fill first. Promotion H2D source uses ptrs
  (pool copy when resident) then frees the pool slot. DONTNEED on the mmap
  source via madvise (gated by weights_discardable).
- Warm start: optional heat CSV via LLAMA_EXPERT_HOT; LLAMA_EXPERT_USAGE
  dumps counts at exit, reusable as the next session's seed.

## 6. Demand RAM pool (LLAMA_EXPERT_RAMPOOL)

Intermediate cache tier between VRAM and mmap. Per-layer contiguous blocks,
one slot = one expert x all its tiered tensors (64B-aligned slices).

- ptrs table (i64[n_expert]): each entry is the full memory address of that
  expert's weight slice — pool slot if resident, else mmap fallback address.
  Cold ops read weights through ptrs transparently.
- Fill: hottest non-pooled cold expert by score, memcpy from mmap into a free
  pool slot, then DONTNEED the mmap range (pool = sole resident copy).
  Fill budget: 16 MiB per layer per update() call (v3 fix: per-layer, not
  global — v2's global budget starved layers 6+).
- Eviction: same hysteresis as hot tier (dwell >= 32, score > 1.5x coldest
  pooled). After eviction, ptrs[e] restored to mmap fallback.
- Promotion: when a pooled expert becomes hot, the H2D copy reads from the
  pool slot and the slot is freed. If a worker is filling the slot (FILLING
  state), the window recycles it next step.
- Seeding: when LLAMA_EXPERT_HOT is present, hottest cold experts by seed
  score are pre-filled into the pool at init.
- Page hints (LLAMA_EXPERT_MADVISE, default 1): madvise(DONTNEED) on mmap
  pages after pool fill; madvise(WILLNEED) on demotion back to cold.
  Gated by weights_discardable() — only mmap-backed, non-mlocked weights.
  Frees ~5.4 GiB VmRSS on 35B models.

## 7. Pre-gated predictive prefetch (LLAMA_EXPERT_PREDICT, LLAMA_EXPERT_PREFETCH_GB)

Pre-gate predictor: runs layer L+1's exact router on layer L's hidden state
(one step ahead of demand). No learned parameters — the model's own gating
function applied early. Norm reconstruction is exact: norm_{L+1} * (x / norm_L),
eps-guarded for zeroed REAP weights.

- Oracle: top-1 ~99%, top-8 covers 81.5% of actual selections (122B).
- ~100 MiB CPU mirror of router gate + norm weights per model.
- Predictions push (layer+1, expert) pairs into a work queue.

Speculative pool (LLAMA_EXPERT_PREFETCH_GB): separate from the demand pool.
Worker threads drain the prediction queue mid-step: claim FREE spec slots
via CAS (FREE->FILLING), memcpy from mmap, release-store READY. Cold ops
probe READY slots via moe_cold_addr — hit costs one atomic load, miss falls
back to ptrs/mmap transparently. byte-identical correctness.

- State machine: 0 FREE / 1 FILLING / 2 READY / 3 RESIDENT.
- Workers own state 1; the update() window owns 0/2/3.
- Window publishes READY->RESIDENT and evicts oldest-timestamped to
  keep ~25% slots free.
- Env: LLAMA_EXPERT_PREFETCH_THREADS (default 2), LLAMA_EXPERT_PREFETCH_MB
  (default 64 MiB in-flight cap).

## 8. Learned router heads (NEXT — DESIGN.md sec 13)

Per-layer MLP heads trained on frozen hidden states to predict P(expert
fires). Replaces the stateless pre-gate exact-router lookahead with learned
scores. Persisted per-model to disk. Training uses LLAMA_EXPERT_TRACEX dumps
— offline, single pass, consumer-GPU feasible. No backprop through the base
model.

## 9. Integration invariants

- mm_ids_helper assumed at most one use of an expert per token; sentinel
  duplicates violate that. Fixed to count+rank semantics.
- The CUDA scheduler anchors pass-1 weights only on buffers with usage
  WEIGHTS; hot buffers must be explicitly tagged or they get swept to CPU.
- mmq mul_mat_id ncols_max is relaxed only for .hot tensors.
- Dispatch follows the src0 buffer: hot tensors in CUDA buffers execute on
  GPU; cold host-pinned tensors stay on CPU.
- graph_compute is async: any mutation of w_hot/lut/mask must happen only
  after ggml_backend_sched_synchronize (the ADAPT=1 corruption bug).
- Cold op ptrs tables are read-only during compute; modified only inside the
  post-sync update() window.
- Predict + probe hooks cross ggml via function pointers; no new op codes.

## 10. Verified benchmarks (v3, 2026-07-29)

RTX 3070 8 GB VRAM, 31 GB RAM, SSD model. PPL = 1.6088 (all configs).

| 122B IQ2_M config | n | tok/s | pool hit |
|---|---|---|---|
| RAMPOOL=10 | 256 | 1.31 | 57.1% |
| RAMPOOL=10 | 1024 | 2.22 | 73.2% |
| RAMPOOL=10 + PREFETCH_GB=2 | 256 | 1.41 | 57.1% demand + 21.9% spec probe |

### Prior benchmarks (v2/original)

| Model | Quant | Stock | wackMall | Win |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | IQ2_M (11 GB) | 27.74 tok/s | 63.54 tok/s (S=112) | +129% |
| Qwen3.6-35B-A3B | Q4_K_M (20 GB) | 26.89 tok/s | 49.93 tok/s (S=64) | +86% |
| gemma-4-26B-A4B | Q5_K_S (17 GB) | 19.50 tok/s | 56 tok/s (S=39) | +187% |
| Qwen3.5-122B-A10B | IQ2_M (28 GB) | ~8.0 tok/s | 10.60 tok/s (S=28) | +33% |

Apache 2.0 License. See LICENSE.
