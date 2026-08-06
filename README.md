IMPORTANT: Current autofit is bugged to use max context. Set -c manually to see real benefit. Outside of that no other argument is needed



I am throwing ideas one by one in an attempt to recreate colibri tier performance using llama.cpp base and model compatibility

This version is behind master llama as my focus is on getting both my merge
request sorted and the still not ready to publish WIP additions. This version
is about 1 week behind my local as of writing this, as I have not yet
confirmed that rebasing it onto master will not break it.

This project is in active development. Feel free to contact me if you wish to
help me test or contribute to the latest version at miltiadiskd@gmail.com

# llama-wackMall

> Expert-granular MoE tiering for llama.cpp: hot experts in VRAM, cold experts in RAM, adaptive online cache, RAM pool, pre-gated predictive prefetch, zero-config auto-fit.

`llama-wackMall` is a research fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
that runs large sparse Mixture-of-Experts (MoE) models - Qwen3.5-122B-A10B,
Qwen3.6-35B-A3B, gemma-4-26B-A4B - at high token-generation throughput on
consumer GPUs with as little as 8 GB VRAM.

First public disclosure: 2026-07-26. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the full technical specification. Apache 2.0 License (see [LICENSE](LICENSE) for upstream MIT;
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
3. **Demand RAM pool.** Intermediate cache tier one level below VRAM.
   Fills/evicts follow score + hysteresis in the update() window. ptrs tables
   let cold ops read from pool or mmap transparently. Env: LLAMA_EXPERT_RAMPOOL.
4. **Pre-gated predictive prefetch.** Runs the next layer's router one step
   early on the current hidden state. Predicted experts are prefetched into
   a speculative pool by worker threads during compute. Env: LLAMA_EXPERT_PREDICT,
   LLAMA_EXPERT_PREFETCH_GB.
5. **Adaptive online cache.** Router decisions are counted per token; an
   exponential-decay score with hysteresis (1.5x score ratio, 32-token
   minimum dwell) swaps hot/cold assignments online. No offline profiling
   required; optional warm-start seeds are supported.
6. **Page hints.** madvise(DONTNEED) on mmap weights after pool fill frees
   VmRSS (~5.4 GiB on 35B). Env: LLAMA_EXPERT_MADVISE.
7. **Zero-config entry point.** Tiering on by default; `-no-cmoe` for stock.
8. **Warm-start sidecar.** On exit the converged expert heat set is persisted
   to `<model>.tier`; on the next run it is loaded at init, so a restart
   skips the convergence burst and starts with a high seed coverage. Delete
   the `.tier` file to force a cold start (no offline profiling required).

The design is model-agnostic: all hooks live at the shared MoE graph-builder
level (no per-model branches), covering any llama.cpp MoE architecture whose
expert tensors use the standard `ffn_{gate,up,down}_exps` layout.

## Verified results

Hardware: RTX 3070 8 GB VRAM, 31 GB RAM, SSD model, `--temp 0`, single
run per config. PPL = 1.6088 all configs (exact match to reference).

| Model | Quant (size) | Stock | wackMall | Speedup |
|---|---|---|---|---|
| Qwen3.6-35B-A3B* | IQ2_M (11 GB) | 42.70 tok/s | **74 tok/s** (server, n=1024) | +73% |
| Qwen3.6-35B-A3B* | Q4_K_M (20 GB) | 26.89 tok/s | **49.93 tok/s** (S=64) | +86% |
| gemma-4-26B-A4B* | Q5_K_S (17 GB) | 19.50 tok/s | **56 tok/s** (S=39) | +187% |
| Qwen3.5-122B-A10B* | IQ2_M (28 GB) | ~8.0 tok/s (best layer-split config) | **10.60 tok/s** (S=28) | +33% |
| Long context (67k prompt) | - | CUDA OOM | **410.38 tok/s** (prompt eval) | runs cleanly |
| Qwen3.5-122B, 16 GB RAM cap | IQ3_XS (34 GB) | 2.40 tok/s | **6.56 tok/s** | +~130% |
Models marked with * have not been updated to include ram-pool performance. Its PROBABLY better

Latest rebased tree (upstream `0e4a036`, 2026-07-27): 35B IQ2_M server at
**63.25 tok/s** (n=1024, adaptation on, ~4250 online re-pins in one run),
with fresh-server determinism verified byte-identical across restarts.
Vulkan tiering verified: 35B Q4_K_M at **34.58 tok/s** on an RX 570 - use
K-quants on Vulkan, the IQ-quant MUL_MAT_ID shaders are not well optimized
there yet.

### 122B IQ2_M (28 GB), RAM-eater pressure (~16 GB free)

| Config | n | tok/s | pool hit |
|---|---|---|---|
| RAMPOOL=10 | 256 | 1.31 | 57.1% |
| RAMPOOL=10 | 1024 | 2.22 | 73.2% |
| RAMPOOL=10 + PREFETCH_GB=2 | 256 | 1.41 | 57.1% + 21.9% spec probe |

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
cd llama-wackMall_v3 && mkdir build && cd build
cmake -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j --target llama-server llama-completion
```

Run server (tiering on by default):

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

### CLI testing (llama-completion)

Non-interactive single-shot generation, useful for coherence and speed tests
(no server, no chat loop):

```bash
./bin/llama-completion \
  -m /path/to/moe-model.gguf \
  -p "Write a comprehensive technical guide to setting up a home Linux server" \
  -n 1024 -no-cnv -st </dev/null
```

Flags: `-n N` tokens to generate, `-no-cnv` non-conversation mode, `-st`
single-turn (exit cleanly after generation instead of looping on the prompt).
Pipe stdin from `/dev/null` so it never blocks on input. `timeout 60` as a
hard safety net if a run hangs.

### Cold vs warm start (sidecar)

The engine persists the converged expert heat set to `<model>.tier` on exit
and reloads it on init (log: `load_sidecar: loaded sidecar ...`). A sidecar
present = warm start (higher seed coverage, no re-convergence burst). To force
a cold start, delete or move the `.tier` file aside before the run.

### Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_EXPERT_S` | auto | Hot slots per layer (auto-fit when unset) |
| `LLAMA_EXPERT_HOT` | - | CSV seed for warm start |
| `LLAMA_EXPERT_ADAPT` | 1 | Online adaptation |
| `LLAMA_EXPERT_DECAY` | 0.999 | Score decay per step (1.0 = cumulative, no aging) |
| `LLAMA_EXPERT_TMAX` | 16 | Max tokens for tiered hot path |
| `LLAMA_EXPERT_RAMPOOL` | 0 | Demand RAM pool (GiB). 0 = off |
| `LLAMA_EXPERT_MADVISE` | 1 | Page hints on mmap'd weights |
| `LLAMA_EXPERT_PREAD` | 1 | pread staging ring for cold weights (0 = plain mmap demand) |
| `LLAMA_EXPERT_PREDICT` | 0 | Pre-gated prefetch hook (set 1 to enable) |
| `LLAMA_EXPERT_PREFETCH_GB` | 0 | Speculative pool (GiB). 0 = off |
| `LLAMA_EXPERT_PREFETCH_THREADS` | 2 | Prefetch worker threads (1..8) |
| `LLAMA_EXPERT_PREFETCH_MB` | 64 | In-flight prefetch bytes cap (MiB) |
| `LLAMA_EXPERT_PREDICT_LOG` | - | Path: dump prediction trace (debug) |
| `LLAMA_EXPERT_STATS` | - | 1 or path: dump stats at exit |
| `LLAMA_EXPERT_USAGE` | - | Path: dump counts (reusable as seed) |

## Status

v3 ACTIVE (2026-08-05). Decay default 0.999 (scores age out over ~1000
tokens so the hot set tracks drift instead of freezing). Repin hysteresis
1.5x score ratio, 32-token minimum dwell. Prefetch worker + speculative
pool re-integrated. CUDA-graph and prefetch machinery verified. MSVC fix:
`__builtin_prefetch` guarded for clang/GCC only in the fused cold kernel.
Verified on `qwen35moe` (35B-A3B IQ2_M, ~37-38 tok/s at S=18 on RTX 3070
8 GB) and `gemma4` architectures.

On the roadmap: disk as third tier for models exceeding RAM (mmap-based, see
ARCHITECTURE.md section 8), learned router heads for persistent per-model
prediction (semantic seeding, Markov correlation - also section 8), multi-GPU
tier priority, per-layer slot skew.

## Prior art notice

This repository and ARCHITECTURE.md are published to disclose the described
methods and systems as of the first publication date, with the intent that
this disclosure serve as prior art. The original code is MIT licensed; the
llama-wackMall additions are Apache 2.0 (see [NOTICE](NOTICE)). The authors
grant no patent rights and intend none to be asserted over the disclosed
concepts.
