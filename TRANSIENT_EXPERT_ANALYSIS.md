# Transient Expert Feasibility Analysis

Date: 2026-08-03

This analysis evaluates routing-exact transient GPU execution before any
productive lookahead transfer is added. The stable fixed tier, sentinel path,
CPU fallback, MTP behavior, and Phase 1 trace are unchanged.

## Protected starting point

```text
source branch: codex/router-lookahead-prefetch
source commit: 4d4f852be998d57b3968718e1d44c8e477e231f2
work branch: codex/transient-expert-feasibility
initial worktree: clean
```

The current host is `/home/andi` with an SM 75 GTX 1660 Ti. The future GTX
1080 paths under `/root` are not present on this machine. No model, profile,
prompt source, or earlier benchmark artifact was modified.

## Assessment of the revised recommendation

The revised order is correct:

1. measure actual transport and resident GPU compute,
2. prove that a routing-exact transient path can beat CPU cold compute,
3. simulate bandwidth, residency, and prediction under those measurements,
4. add productive lookahead only if the gate is positive.

This separates transfer feasibility from predictor recall. SpecPrefetch uses
predictions only for transfer while the native router remains authoritative,
and its scheduler considers bandwidth and completion windows. HybriMoE also
motivates intra-layer CPU/GPU scheduling. These are useful design references,
but neither paper proves that one-shot PCIe transfers win on these desktop
GPUs and CPUs.

Primary references:

- https://arxiv.org/abs/2607.24787
- https://arxiv.org/abs/2504.05897
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html

## Existing runtime pieces

The warm-cache code already has a dedicated backend stream, pinned staging,
events, bounded jobs, pending ownership, and publish-after-completion. Those
mechanisms are relevant to a future transient arena. They do not prove that
the work is profitable.

The existing copy path copies each expert tensor slice from mmap-backed CPU
weights into pinned staging and then uploads it. A full expert is contiguous
within each gate, up, and down tensor but not across all three tensors. The
current model has two base-layer layouts:

```text
layers 0..37:
  gate:       589,824 bytes
  up:         589,824 bytes
  down:       720,896 bytes
  full:     1,900,544 bytes

layers 38..39:
  gate:       589,824 bytes
  up:         589,824 bytes
  down:       860,160 bytes
  full:     2,039,808 bytes
```

These values are read from GGUF tensor metadata. No model dimensions are
hard-coded in the benchmark runner.

## Important corrections to the proposed architecture

### One-shot versus retained residency

A global arena of two to four slots costs little VRAM, but distance-one layer
lookahead cycles through almost every target layer. It provides in-flight
storage, not meaningful next-token retention. Increasing it from one to four
slots did not change the global-arena replay result for the sequential
distance-one policy.

A per-layer slot retains experts across tokens and can amortize a copy, but it
costs about 72.9 MiB per full-model slot for this GGUF. That is the same
fundamental tradeoff as the existing warm cache, not a free global scratch
optimization.

### Matrix-granular streaming

Gate plus up is 1,179,648 bytes, or 62.1 percent of the small full expert.
Down is 720,896 bytes, or 37.9 percent. Therefore early Gate+Up transfer does
not reduce false-positive traffic to an assumed generic two thirds on every
model. The exact tensor sizes must drive the policy.

### Predictability-aware placement

Placement should not move a frequent expert out of the fixed tier merely
because it is predictable. The residual penalty is valid only after the target
hardware shows a positive transient net benefit:

```text
residual_penalty = usage * max(cpu_ms - transient_gpu_ms, 0)
                 * (1 - ready_probability)
```

The transient GPU time must include resident GPU compute and exposed transfer,
not only predictor recall. The existing placement optimizer is intentionally
unchanged until this target-specific input exists.

## Current feasibility conclusion

On the GTX 1660 Ti, routing-exact one-shot streaming fails the necessary cost
gate. In the phase-matched no-MTP control, pinned H2D alone costs roughly four
to eight times the matching CPU phase, depending on layer and matrix phase.
Resident GPU compute removes at most a small part of CPU time and cannot repay
the transfer.

Mapped host memory also fails the lower-bound screen. A sequential mapped read
of one full expert takes about the same time as an explicit H2D copy, before a
quantized expert matmul is performed. NVIDIA documents that mapped access uses
the CPU-GPU interconnect and has higher latency and lower bandwidth than device
memory; it should be measured, not assumed to be a faster general memory tier.

This conclusion is specific to the measured GTX 1660 Ti plus Ryzen 7 4800H.
The same tools must be run on the GTX 1080 plus old quad-core i7. A slower CPU
can change the compute crossover, and a different PCIe topology can change the
transport cost.

## Remaining timing gap

The synthetic overlap test proves that the copy engine can overlap a pinned
copy with a small independent CUDA workload. It does not prove that the copy
can overlap the real attention and fixed-expert kernels without memory or
scheduler contention.

Real source-to-target lead time remains unavailable. A production spike is not
authorized until a one-layer event bridge measures:

- predictor completion,
- ID availability,
- copy start and completion,
- actual-router completion,
- exposed event wait,
- interference with the normal CUDA graph.
