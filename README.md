# llama-wackMall

> Expert-granular MoE tiering for llama.cpp: hot experts in VRAM, cold experts in RAM, adaptive online cache, zero-config auto-fit.

`llama-wackMall` is a research fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
that runs large sparse Mixture-of-Experts (MoE) models - Qwen3.5-122B-A10B,
Qwen3.6-35B-A3B, gemma-4-26B-A4B - at high token-generation throughput on
consumer GPUs with as little as 8 GB VRAM.

First public disclosure: 2026-07-26. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the full technical specification. Released under the Apache 2.0 License (see [LICENSE](LICENSE) for upstream MIT;
[NOTICE](NOTICE) for additions).

---

## The bottleneck

Stock llama.cpp offloads at layer granularity (`-ngl N`): an MoE layer with
128-256 expert tensors is an all-or-nothing block. On small GPUs, fitting a
28 GB MoE model means most layers fall back to CPU, and every generated token
streams its active experts through system RAM - the memory-bandwidth wall.

## What this engine does

Replaces layer-granular offload with **expert-granular offload**:

1. **Hot/cold expert split.** Per MoE layer, the S currently-hottest experts
   are pinned in VRAM; the rest stay in RAM and are computed on CPU only when
   actually routed. Per-token traffic drops from "all active experts" to
   "only the cold-selected ones".
2. **Hardware-aware auto-fit.** At startup the engine places dense weights,
   KV cache and compute buffers first, measures the remaining free VRAM, and
   computes the exact number of hot expert slots S that fits. No manual
   `-ngl` tuning.
3. **Adaptive online cache.** Router decisions are counted per token; a
   cumulative usage score with hysteresis (1.5x score ratio, 32-token
   minimum dwell) swaps hot/cold assignments online (optional exponential
   decay for recency weighting). No offline profiling required; optional
   warm-start seeds are supported.
4. **Zero-config entry point.** One flag, `-cmoe`, enables the whole stack.

The design is model-agnostic: all hooks live at the shared MoE graph-builder
level (no per-model branches), covering any llama.cpp MoE architecture whose
expert tensors use the standard `ffn_{gate,up,down}_exps` layout.

## Verified results

Hardware: RTX 3070 8 GB VRAM, 31 GB RAM, `--temp 0`, `-n 256 --ignore-eos`,
single run per config, same binary/session per A/B pair.

| Model | Quant (size) | Stock | wackMall | Speedup |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | IQ2_M (11 GB) | 42.70 tok/s | **74 tok/s** (server, n=1024) | +73% |
| Qwen3.6-35B-A3B | Q4_K_M (20 GB) | 26.89 tok/s | **49.93 tok/s** (S=64) | +86% |
| gemma-4-26B-A4B | Q5_K_S (17 GB) | 19.50 tok/s | **56 tok/s** (S=39) | +187% |
| Qwen3.5-122B-A10B | IQ2_M (28 GB) | ~8.0 tok/s (best layer-split config) | **10.60 tok/s** (S=28) | +33% |
| Long context (67k prompt) | - | CUDA OOM | **410.38 tok/s** (prompt eval) | runs cleanly |
| Qwen3.5-122B, 16 GB RAM cap | IQ3_XS (34 GB) | 2.40 tok/s | **2.84 tok/s** | +18% |

Latest rebased tree (upstream `0e4a036`, 2026-07-27): 35B IQ2_M server at
**63.25 tok/s** (n=1024, adaptation on, ~4250 online re-pins in one run),
with fresh-server determinism verified byte-identical across restarts.
Vulkan tiering verified: 35B Q4_K_M at **34.58 tok/s** on an RX 570 - use
K-quants on Vulkan, the IQ-quant MUL_MAT_ID shaders are not well optimized
there yet.

All numbers measured manually by the authors; single run per config,
same-flags stock baselines on the identical upstream base. Correctness:
perplexity within rounding noise of stock (9-chunk protocol); bit-identical
to stock when forced through identical compute paths; adaptive re-pin
bookkeeping machine-checked by a permanent invariant guard. Greedy output
can tie-flip vs stock (different-but-valid rounding, same class as changing
batch size); output quality is equivalent. Details in ARCHITECTURE.md
section 5.

## Quick start

Requirements: CMake 3.18+, GCC/Clang with OpenMP, CUDA toolkit for NVIDIA
builds.

```bash
mkdir build && cd build
cmake -DGGML_CUDA=ON ..
cmake --build . -j --target llama-server
```

Run (tiering on by default):

```bash
./bin/llama-server -m /path/to/moe-model.gguf -c 4096 --port 8080
```

Use `-no-cmoe` for stock llama.cpp behavior. `-t 10` for manual threads
(10 is the automatic default on 12-core CPUs; auto-threads picks 80% of
hardware when dense weights fit VRAM). KV cache defaults to f16; on K-quant
models at long context, add `-ctk q8_0 -ctv q8_0`.

Tiering also auto-configures auto-fit offloading, batch/ubatch 256, flash
attention and KV offload. Interactive use: `llama-cli` with the same flags
(add `--jinja` for architectures with custom chat templates, e.g. gemma4).
Vulkan: configure with `-DGGML_VULKAN=ON` and prefer K-quant models.

### Optional environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_EXPERT_S` | auto | Hot slots per layer (0 = all CPU, N = force N) |
| `LLAMA_EXPERT_HOT` | - | CSV heat seed for warm start |
| `LLAMA_EXPERT_ADAPT` | 1 | Online adaptation on/off |
| `LLAMA_EXPERT_DECAY` | 1.0 | Score decay per update (1.0 = cumulative) |
| `LLAMA_EXPERT_TMAX` | 16 | Max tokens/graph that takes the tiered path |
| `LLAMA_EXPERT_STATS` | - | 1 or path: dump cache statistics at exit |
| `LLAMA_EXPERT_USAGE` | - | Path: dump counts (reusable as next seed) |

## Status

Research preview. Verified on `qwen35moe` and `gemma4` architectures. On the
roadmap: disk as third tier for models exceeding RAM (mmap-based, see
ARCHITECTURE.md section 8), predictive expert prefetching (semantic seeding,
Markov correlation, learned router heads - also section 8), multi-GPU tier
priority, per-layer slot skew.

## Prior art notice

This repository and ARCHITECTURE.md are published to disclose the described
methods and systems as of the first publication date, with the intent that
this disclosure serve as prior art. The original code is MIT licensed; colibri-tier
additions are Apache 2.0 (see [NOTICE](NOTICE)). The authors grant no patent
rights and intend none to be asserted over the disclosed concepts.
