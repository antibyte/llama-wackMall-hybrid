# KVFlash implementation status (llama-wackMall-hybrid)

The implementation is based on the design in
[Lucebox KVFlash](https://github.com/Luce-Org/lucebox/tree/main/optimizations/kvflash),
adapted to llama.cpp's hybrid attention/recurrent memory split.

## Goal

Keep the full-attention K/V cache at a fixed device-resident size while the
logical context grows. Cold 64-token chunks are copied bit-exactly to host
memory. Recurrent GDN/SSM state remains managed by the normal hybrid cache and
is never paged.

## Implemented architecture

- Logical token positions map to fixed-size physical blocks in the resident
  attention cache.
- K/V data is copied as opaque packed rows, including quantized cache types.
- Paging uses the context's real backends. All layer copies for a page batch are
  queued before one synchronization, preserving graph/copy ordering.
- Host snapshots use pooled slabs (up to 64 pages or 64 MiB per allocation)
  instead of one pinned allocation per page.
- llama.cpp cell metadata is snapshotted together with each cold chunk and
  restored before the chunk becomes visible to the graph.
- Each micro-batch is mapped immediately before its graph runs. Requested
  chunks are pinned as one transaction while victims are selected.
- Pure LRU eviction is demand-driven. With no scorer installed, the periodic
  reselect path is a no-op and causes no additional page traffic.
- Victim selection, resident masks, and identity checks scan the resident pool,
  not the complete logical context. The normal decode hot path is O(pool).
- Host-page position bounds use incremental ordered indices, so the batch
  allocator does not rescan all cold metadata on each decode.
- Sink and recent-tail chunks are structurally protected.

## Supported scope and guardrails

KVFlash is enabled only for a causal hybrid target context with all of the
following properties:

- Flash Attention remains enabled after backend probing.
- Exactly one sequence and one stream are used.
- Positions use either the normal 1D layout or llama.cpp's 4-channel M-RoPE
  layout. The primary position must remain the unique sequential token index;
  the additional cell metadata is paged with the K/V rows.
- The model does not use SWA.
- The resident pool is smaller than the logical context.

Unsupported configurations are disabled with a warning. Position shifts,
position division, cross-sequence cache copies, and attention-cache state
serialization fail explicitly instead of silently corrupting the page table.
Partial hybrid state operations that serialize only recurrent state remain
available.

## Configuration

```text
LLAMA_KVFLASH=N|auto|0
LLAMA_KVFLASH_MAX_POOL=N
LLAMA_KVFLASH_TAU=N
LLAMA_KVFLASH_POLICY=lru
LLAMA_KVFLASH_STATS=1
```

- `N` is rounded to a cache-safe alignment and clamped to the logical context.
- `auto` currently uses a conservative context-derived fallback because the
  context does not yet provide a post-weight free-VRAM budget to the pager.
- `LLAMA_KVFLASH_POLICY` currently supports only `lru`; other values warn and
  fall back to LRU.
- `LLAMA_KVFLASH_TAU` becomes active only when a relevance scorer is attached.
- `LLAMA_KVFLASH_STATS=1` reports page-ins, page-outs, used and allocated host
  bytes, transferred bytes, resident blocks, and host-slab count when paging
  activity changes.

## Verification completed

- CPU pager tests: mapping, exact/partial masks, LRU, callback ordering,
  transactional attach, invalid inputs, and K/V byte roundtrip.
- CUDA pager byte roundtrip on an NVIDIA GTX 1660 Ti.
- AddressSanitizer test run (LeakSanitizer is unavailable under the runner's
  ptrace supervision).
- A 1M-token map-only LRU test with a 512-token resident pool; the complete unit
  suite runs in about 0.06 seconds on the test host.
- Qwen3.6-35B-A3B M-RoPE hybrid integration with a 1024-token logical context,
  a 512-token resident pool, and a 651-token prompt. Runtime statistics showed
  three page-outs, 1.05 MiB transferred, 1.05 MiB of used snapshots in one
  5.62 MiB host slab, and all eight blocks resident. Prompt processing was about
  23.7 tokens/s on the GTX 1660 Ti test system. A same-command smoke run with
  KVFlash disabled also measured about 23.7 tokens/s; this is a parity smoke
  check, not a statistically rigorous benchmark.

## Remaining work

1. Add and evaluate a real relevance scorer (drafter or target-QK). LRU alone
   can lose important middle-context chunks and is not a quality replacement
   for relevance-based selection.
2. Feed an actual post-load device-memory budget into `auto` sizing.
3. Define a page-table-aware state serialization format.
4. Run long-context quality suites (including NIAH), 32K+ prefill stress tests,
   and DFlash acceptance-parity tests.
5. Evaluate double-buffered page-in/page-out overlap. The current implementation
   batches asynchronous copies but synchronizes at each required residency
   transition for correctness.
