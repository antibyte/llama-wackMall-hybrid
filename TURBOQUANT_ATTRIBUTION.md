# TurboQuant attribution

Date: 2026-08-07

The phase-1 codec was informed by `atomicmilkshake/llama-cpp-turboquant` at
commit `f67e13573ab344d98090ea7612056a0119fcc5ef`:

https://github.com/atomicmilkshake/llama-cpp-turboquant

That repository is MIT licensed:

```text
Copyright (c) 2023-2026 The ggml authors
```

## Files examined

- `ggml/src/ggml-common.h`
- `ggml/src/ggml-turbo-quant.c`
- `ggml/src/ggml-cuda/turbo-quant.cuh`
- `ggml/src/ggml-cuda/set-rows.cu`
- `ggml/src/ggml-cuda/turbo-wht.cu`
- `ggml/src/ggml-cuda/turbo-wht.cuh`
- `ggml/src/ggml-cuda/turbo-innerq.cu`
- `ggml/src/ggml-cuda/turbo-innerq.cuh`
- `src/llama-graph.cpp`
- `src/llama-kv-cache.cpp`
- `src/llama-triattention.cpp`
- `src/llama-triattention.h`

## Directly adapted material

The fixed WHT sign arrays, Turbo3/Turbo4 centroid constants, byte-level index
packing, and corrected-norm formula in `tools/turboquant-ref/turboquant-ref.cpp`
are adapted from the Atomic implementation. The file retains the ggml MIT
copyright and SPDX identifier.

## New implementation

The following parts were written for this fork:

- isolated codec API without `ggml_type` registration;
- explicit 64/128/256 row padding and validation;
- inverse WHT reconstruction path matching the CUDA transform;
- file validation and non-overwrite behavior;
- JSON quality metrics;
- normalized query/key dot-product metric;
- deterministic post-RoPE Q/K capture using the existing llama evaluation
  callback;
- capture-only materialization of a contiguous pre-attention V diagnostic
  tensor, plus independent attention-output error metrics;
- replay-only attention-output delta export, guarded capture-tool injection,
  immutable raw-logit dumps, and logit comparison metrics;
- manifest and raw-file validation;
- GQA-aware causal attention-logit and softmax error analysis;
- offline non-mutating token-retention simulation, including an independently
  implemented form of the documented TriAttention equations and a causal
  past-attention baseline;
- deterministic C++ tests and quality thresholds;
- integration plan for wackMall MTP and separate SM75/SM61 backends.

No Atomic CUDA kernel, KV runtime, TriAttention runtime, or server code has been
copied in phase 1. The original Atomic repository was not modified.

`tools/turboquant-capture/turboquant-capture.cpp` and
`tools/turboquant-ref/analyze-capture.py` are original implementations for this
fork and contain no code copied from the Atomic repository. The same applies
to `tools/turboquant-ref/simulate-triattention.py` and
`tools/turboquant-ref/compare-logits.py`; their equation-level behavior
was checked against Atomic's `docs/TRIATTENTION.md` and
`src/llama-triattention.cpp` at commit
`f67e13573ab344d98090ea7612056a0119fcc5ef`.

## Phase 2 CUDA adaptation

The Phase 2 files `ggml/src/ggml-cuda/turbo4-k.cuh`,
`ggml/src/ggml-cuda/turbo4-wht.cu`, and the Turbo4 sections of
`set-rows.cu` and `fattn-common.cuh` are clean adaptations of the algorithm and
fixed constants in the examined Atomic CUDA files. They were rewritten for
this fork's type ID, K-only semantics, GGML launch helpers, and golden codec;
InnerQ, Turbo2/3, Turbo V, and tail-padding code were deliberately omitted.
The adapted files retain MIT SPDX and Atomic attribution headers where a
standalone file exists. No Atomic server or TriAttention code is included.

The later DP4A and FP16-prefill experiments are new integration work for this
fork. DP4A reuses llama.cpp's existing MIT-licensed
`get_int_from_table_16()` CUDA helper; it does not copy another TurboQuant
kernel. The rotated Turbo4-to-FP16 conversion and batch-dependent
FlashAttention dispatch were implemented locally from the Phase 1 format
definition. No additional Atomic source was copied for either experiment.

`scripts/bench_turboquant_quality.sh` is an original local benchmark wrapper.
It invokes the existing llama.cpp perplexity/KL interface and contains no code
copied from either TurboQuant reference repository.

## Experimental Turbo4 V adaptation

The target-V investigation also examined the MIT-licensed
`TheTom/llama-cpp-turboquant` repository at commit
`284ffc73181f019d7a4f919ba5b9dc8e1c85c388`:

https://github.com/TheTom/llama-cpp-turboquant

Files examined:

- `src/llama-graph.cpp`
- `ggml/src/ggml-common.h`
- `ggml/src/ggml-cuda/fattn-common.cuh`
- `ggml/src/ggml-cuda/fattn-vec.cuh`
- its Turbo4 FlashAttention template-instance files

The linear rotated-domain V accumulation followed by one inverse WHT was
reimplemented for this fork's graph API and existing WHT operation. The
Turbo4 V lane geometry, scaled-centroid reuse, and specialized nibble loop in
`fattn-vec.cuh` are directly adapted from that repository and modified for
`GGML_TYPE_TURBO4_K`, the local 68-byte block with reserved half, the current
llama.cpp kernel structure, and mixed Q8/Turbo4 instances. The source file and
this record mark that adaptation. Parser guards, inverse-op direction,
temporary FP16 dispatch, model tests, and benchmark integration are local
integration work.

No model files, server implementation, InnerQ runtime, TriAttention runtime,
Metal kernels, or HIP kernels were copied. The reference checkout under
`/tmp` was read-only and is not part of this repository.

## ik_llama.cpp experiments

The low-perplexity Q4_0 scale and MTP-only output-head experiments were
informed by the MIT-licensed `ikawrakow/ik_llama.cpp` repository at commit
`40dffce6857b4fe051f096379dc464764c718458`:

https://github.com/ikawrakow/ik_llama.cpp

Files examined for these two implementations:

- `ggml/src/ggml-cuda/cpy-utils.cuh`
- `ggml/src/ggml-quants.c`
- `src/llama.cpp`
- `src/llama-model.h`

The weighted least-squares Q4_0 block-scale formula is adapted from those
quantization implementations. It was integrated through a new explicit
`SET_ROWS` flag so the unflagged llama.cpp operation remains unchanged. CPU
and CUDA paths and the K/V-selective runtime policy were written for this
fork.

The MTP-only output-head feature is a new implementation of the documented
runtime-requantization concept. Its loader integration, owned GGML context and
backend buffer, generic row-parallel conversion, Qwen3.6 graph selection,
environment interface, tests, and benchmark fields were written for this
fork. No ik_llama IQK quantizer, tensor wrapper, graph builder, MTP decoder, or
reload code was copied.

The IQK CPU files were inspected and compiled in a temporary checkout only;
no IQK source is included. Exact benchmark results and the rejection decision
are recorded in `IK_LLAMA_EXPERIMENTS.md`.

The later fused Gate/Up follow-up inspired an independently written AVX2
Q4_K dual-dot experiment. No IQK kernel code was copied: the local function
uses this fork's existing ggml Q4_K arithmetic and only shares reads of the
already quantized Q8_K activation between Gate and Up. The cross-tree
benchmark, exactness test, and stop decision are recorded in
`IK_LLAMA_EXPERIMENTS.md`.

The TriAttention replay/keep-set exporter, guarded delta replay, layer-local
diagnostic live mask, multi-row logit dump, and raw-logit comparator are
original implementations for this repository. No third-party TriAttention
runtime or cache-management source was copied.
