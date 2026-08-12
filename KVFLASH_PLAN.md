# KVFlash port plan (llama-wackMall-hybrid)

Based on [Lucebox KVFlash](https://github.com/Luce-Org/lucebox/tree/main/optimizations/kvflash)
(FlashMemory-style decode-time KV paging, arXiv 2606.09079).

**Goal:** hard O(pool) VRAM for full-attention KV on Qwen3.6 hybrid, with bit-exact
host recall of cold 64-token chunks. GDN/SSM recurrent state is never paged.

**Non-goal (P0–P2):** tree-verify + pool, multi-seq concurrent pool, pflash BSA.

---

## PR stack

| PR | Name | Deliverable | Gate |
|----|------|-------------|------|
| **PR1** | Pager core | `common/kvflash_pager.h` (+ scorer iface), unit tests, no model | `test-kvflash` green |
| **PR2** | Pool-sized hybrid KV | Allocate attn tensors at `pool_tokens` when enabled; logical `n_ctx` unchanged; recurrent full size | Load Qwen3.6-35B hybrid, pool=2048, generate short text without crash |
| **PR3** | Slot map + mask | Wire `slot_for` into `set_rows` idxs / FA span; slot-validity mask (or zero free slots) | Teacher-forced argmax flip ≤1% vs full cache on 2k tokens (small model or forced) |
| **PR4** | Live LRU paging | Eviction under pool≪prompt+gen; host D2H/H2D; decode continues | Page stats >0, coherent completion, VRAM peak ~pool |
| **PR5** | Chunked prefill | Prompt > pool: chunked prefill with live eviction | 32k prompt fits 4k pool; linear time |
| **PR6** | DFlash chain on pool | Spec verify slot-mapped; rejected draft slots validity-excluded | Accept rate ±1% abs vs pool-off |
| **PR7** | Scorers | Drafter / target-QK reselect τ; CLI `--kvflash` | NIAH light: drafter ≥ LRU on mid-depth needles |

**Default off** until PR4 gates pass. Env: `LLAMA_KVFLASH` (tokens|auto|0).

---

## Architecture mapping

| Lucebox | wackMall hybrid |
|---------|-----------------|
| `KvFlashPager` | `common/kvflash_pager.h` → `common_kvflash_pager` |
| `create_target_cache(pool)` | `llama_kv_cache` ctor / hybrid: pass `n_kv = pool` for attn layers only |
| `kv_write_rows` physical slot | `llama_kv_cache::*_set_rows` idxs from pager |
| `positions` logical | existing `ubatch.pos` (RoPE) — unchanged |
| Slot mask | tighten in `llm_graph_input_attn_kv::set_input` (same pattern as tree visibility) |
| Recurrent | `llama_memory_recurrent` — no change |
| Drafter scorer | later: DFlash draft ctx as indexer (optional) |

### Why relocation is legal
- RoPE is baked into K at write from logical `pos`.
- Attention sees only resident pool slots via mask (or zeroed free slots).
- Quantized / turbo4 rows copy as opaque bytes.

### Hybrid MoE notes (1660 Ti)
- Pool shrinks attn KV VRAM → more headroom for expert S / larger logical ctx.
- Prefer **masked** path first (exact); maskless+zero is the qwen35moe Spark approximation.
- CUDA graphs: FA span clamps to pool once filled → fewer rebuilds after warmup.
- PCIe: P0 uses sync `ggml_backend_tensor_get/set`; async DMA is a follow-up.

---

## CLI / env (target surface)

```
--kvflash N|auto|0          # resident pool tokens (multiple of 256)
--kvflash-tau N             # reselect interval floor (default 64)
--kvflash-policy lru|drafter|qk
LLAMA_KVFLASH=...
LLAMA_KVFLASH_TAU=...
LLAMA_KVFLASH_POLICY=...
LLAMA_KVFLASH_MAX_POOL=16384
```

`auto`: half free VRAM after weights (minus reserve), density-converted, capped by speed knee and `n_ctx`.

---

## File touch list (by PR)

### PR1 (this slice)
- `common/kvflash_pager.h` — ported pager (sync I/O, optional empty tensors)
- `common/kvflash_scorer.h` — scorer interface
- `tests/test-kvflash.cpp` — mapping, LRU eviction, reselect, mask, identity
- `tests/CMakeLists.txt`, `common/CMakeLists.txt` if needed

### PR2–PR3
- `src/llama-kv-cache.{h,cpp}` — optional pool capacity
- `src/llama-memory-hybrid.{h,cpp}` — wire pager lifetime to ctx
- `src/llama-context.{h,cpp}` — cparams + public set/get
- `src/llama-graph.cpp` — mask tighten when pager active
- `include/llama.h` — API if needed
- `common/arg.cpp` / server — flags

### PR4–PR7
- prefill paths in server/context
- speculative / DFlash verify idxs
- scorer implementations + benches

---

## Test matrix

| ID | Test | When |
|----|------|------|
| A | Unit: slot map + LRU eviction | PR1 |
| B | Unit: reselect score_hook | PR1 |
| C | Unit: mask / fill_slot_pos | PR1 |
| D | Relocation teacher-force (shuffled blocks) | PR3 |
| E | Live paging smoke n_predict=256 pool=1024 | PR4 |
| F | Prefill 16k–32k / pool 2k–4k | PR5 |
| G | DFlash accept parity pool on/off | PR6 |
| H | NIAH 8k/32k residency 9–25% | PR7 |

---

## Risks

1. **Cell allocator mismatch** — llama.cpp cells ≠ Lucebox dense pool rows; may need a “dense pool mode” that disables defrag assumptions.
2. **Checkpoints** — refuse or serialize page table once `page_outs > 0`.
3. **Tree-verify** — requires identity prefix; refuse tree path when non-identity.
4. **Multi-seq** — P0–P6 single sequence only (`n_parallel=1`).
5. **Quality** — LRU alone fails mid-context NIAH; document until scorer lands.

---

## Success criteria (product)

On GTX 1660 Ti (6 GiB), Qwen3.6-35B-A3B hybrid + DFlash:

- Logical context ≥ 32k–64k with resident pool 2k–4k.
- Decode tok/s within ~15% of full-cache short-ctx baseline at same pool span.
- Peak VRAM drops enough to raise expert S or context vs today (~24k @ ~5.5 GiB).
- Chain DFlash still works (PR6).

---

## Status

- [x] Feasibility review
- [x] PR plan (this doc)
- [x] PR1 pager + unit tests (`common/kvflash_*.h`, `tests/test-kvflash.cpp`)
- [x] PR2 wiring:
  - `cparams.kvflash_pool` from `LLAMA_KVFLASH`
  - hybrid `attn_kv_size` = pool when enabled
  - `llama_kv_cache::init_kvflash` + `find_slot` physical map
  - cell clear on block eviction; FA `n_kv` = used_max (full pool only after page_outs)
  - smoke (2026-08-12): base/kvf × ±dflash all HTTP 200, **identical sha** short gen
- [x] Fixes after first smoke:
  - same-pos re-apply skips purge (double apply no longer wipes sequence)
  - permanent prepare restored for KVFlash (pager can't dry-run page_out)
  - vocab `token_to_piece` guards NULL/OOB (DFlash crash on cache.at(-1))
  - packed-row page copy for llama KV layout; default LRU reselect
  - τ-reselect after decode (`LLAMA_KVFLASH_TAU`, default 64)
- [x] Live eviction smoke: pool=320, n_predict=700, ignore_eos → HTTP 200 ~21 tok/s (must page)
- [ ] PR7 scorer (drafter / target-QK) + NIAH suite
- [ ] Async DMA page stream; pooled snapshots
