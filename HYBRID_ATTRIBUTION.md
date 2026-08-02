# Hybrid attribution

Date: 2026-08-01

This experimental branch is based on llama-wackMall at commit `1c801502f53a54ac77c1733c1dc2c9d7f7053ff6`. llama-wackMall is Apache-2.0 licensed and includes MIT-licensed llama.cpp work as described by the repository `NOTICE` file.

LuceBox at commit `bc881af248de9c03336bb9afa54735d1af5f273f` was used as a read-only architecture and behavior reference. LuceBox is Apache-2.0 licensed. No file in the LuceBox repository was modified.

## LuceBox files examined

The analysis directly examined these implementation and documentation files:

- `optimizations/spark/README.md`
- `optimizations/spark/RESULTS.md`
- `optimizations/spark/spark/calibrate.py`
- `server/src/common/moe_hybrid_types.h`
- `server/src/common/moe_hybrid_storage.h`
- `server/src/common/moe_hybrid_storage.cpp`
- `server/src/common/moe_hybrid_placement.h`
- `server/src/common/moe_hybrid_placement.cpp`
- `server/src/common/moe_hybrid_routing_stats.h`
- `server/src/common/moe_hybrid_routing_stats.cpp`
- `server/src/common/moe_hybrid_swap_manager.h`
- `server/src/common/moe_hybrid_swap_manager.cpp`
- `server/src/common/moe_hybrid_ffn_eval.h`
- `server/src/common/moe_hybrid_ffn_eval.cpp`
- `server/src/qwen35moe/qwen35moe_backend.cpp`
- `server/src/qwen35moe/qwen35moe_pipelined_decode.h`
- `server/src/qwen35moe/qwen35moe_pipelined_decode.cpp`

Additional filenames under `server/src/common`, `server/src/qwen35moe`, and `optimizations/spark` were inventoried to determine scope and coupling, but were not treated as implementation sources unless listed above.

## Concepts used as reference

The following LuceBox concepts informed the hybrid design:

- dense per-layer expert activation counts learned from real traffic;
- persistence and reuse of learned counts across process restarts;
- byte-budgeted placement based on expert frequency;
- a bounded spare-slot expert cache separated from calibrated fixed placement;
- LRU replacement for short-lived request-local working sets;
- hot GPU and cold CPU execution overlap;
- moving the required host readback before the hot launch;
- delaying the hot/cold join until the real dependency;
- telemetry for router readback, hot/cold compute, synchronization, and combine;
- keeping MTP/speculative verification cache residency stable for a batch.

These are architecture ideas. The hybrid expresses them through wackMall's existing generic `.hot` tensor, sentinel LUT, CPU cold operation, scheduler, and MTP implementation.

## Directly copied code

None. No LuceBox function, class, source block, comment, or test has been copied into the hybrid worktree.

The profile converter implements the published Spark CSV grammar independently from the format emitted by `MoeHybridRoutingStats::save_csv()`. Format compatibility requires using the same header and row meaning, but the parser and validation code are new.

The warmcache state machine and slot transaction code are a new implementation for wackMall. In particular, it does not copy LuceBox's immediate map publication after asynchronous tensor writes; the hybrid requires explicit completion before publication.

## Modified versus new files

- Existing wackMall/llama.cpp files retain their original license and copyright
  notices. The hybrid modifies `NOTICE`, `common/arg.cpp`, `common/common.cpp`,
  `ggml/include/ggml-cpu.h`, `ggml/src/ggml-cpu/ggml-cpu.c`,
  `ggml/src/ggml-cpu/ggml-cpu.cpp`,
  `src/CMakeLists.txt`, `src/llama-context.cpp`, `src/llama-expert-tier.cpp`,
  `src/llama-expert-tier.h`, `src/llama-graph.cpp`, `tests/CMakeLists.txt`, and
  `tools/server/server-context.cpp`.
- New implementation/test/tool files are `src/llama-expert-cache.cpp`,
  `src/llama-expert-cache.h`, `tests/test-expert-warm-cache.cpp`,
  `src/llama-expert-placement.cpp`, `src/llama-expert-placement.h`,
  `tests/test-expert-placement.cpp`, `tests/test-aggregate-expert-profiles.py`,
  `tests/test-optimize-expert-placement.py`,
  `tests/test-convert-luce-spark-profile.py`,
  `tools/convert_luce_spark_profile.py`, `tools/bench_hybrid_client.py`,
  `tools/aggregate_expert_profiles.py`, `tools/optimize_expert_placement.py`,
  `tools/debug_sigsegv_backtrace.cpp`, `scripts/collect_expert_profiles.sh`,
  `scripts/hybrid_profile_corpus.jsonl`, and `scripts/bench_hybrid.sh`.
- New project records are `HYBRID_ANALYSIS.md`, `HYBRID_DESIGN.md`, `HYBRID_ATTRIBUTION.md`, and `HYBRID_EXPERIMENTS.md`. Raw benchmark artifacts remain local and untracked below `benchmark-results/`.
- New hybrid files are contributed under the repository's Apache-2.0 license unless a file states otherwise.
- Files adapted from upstream llama.cpp retain the MIT attribution already present in `NOTICE`.

The CPU phase timers and block-parallel activation scheduling are original
implementations for the existing wackMall fused cold operation. Their design
was driven by local measurements and does not copy LuceBox runtime code.

The experimental CPU-split coordinator, graph reordering, Down-row prefetch,
MTP row-reuse loop, and associated telemetry are also new wackMall
implementations. LuceBox's high-level readback-before-hot and delayed-join
concept motivated the ordering; no LuceBox scheduler, worker, graph, or
synchronization code was copied.

If later work copies or closely adapts LuceBox code, this file must be updated with the exact source path, function/range, copyright notice, nature of modifications, and corresponding `NOTICE` entry before distribution.
