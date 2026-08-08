# TurboQuant analysis

Date: 2026-08-07

## Scope

This analysis uses `atomicmilkshake/llama-cpp-turboquant` commit
`f67e13573ab344d98090ea7612056a0119fcc5ef` as an implementation reference.
The stable wackMall MTP, expert-tier, sampling, KV, and server paths remain the
base. A wholesale merge is not suitable.

Phase 1 was intentionally limited to a CPU reference codec, a dedicated
post-RoPE Q/K capture executable, and offline quality tools. Phase 2 now adds a
default-off `GGML_TYPE_TURBO4_K` runtime spike. The capture callback remains
measurement-only and is never enabled in `llama-server`.

## Atomic implementation findings

The Atomic branch contains real end-to-end work for KV writes, randomized WHT
query rotation, CUDA FlashAttention, mixed K/V cache types, state I/O, and an
experimental TriAttention evictor. It is substantially more complete than a
format-only TurboQuant demonstration.

The current storage definitions differ from several comments and from literal
paper names:

| Format | Bytes per 128 values | Effective bits/value | Current residual |
| --- | ---: | ---: | --- |
| Turbo2 | 34 | 2.125 | none |
| Turbo3 | 50 | 3.125 | none |
| Turbo4 | 68 | 4.25 | none in the default 4-bit mode |

Turbo3 stores one FP16 corrected norm, 2-bit low indices, and one additional
index bit. Turbo4 stores an FP16 corrected norm, one reserved FP16 value, and
4-bit PolarQuant indices. Phase 1 implements only Turbo3 and Turbo4.

The Atomic CPU Turbo4 reference uses a dense generated orthogonal matrix while
the CUDA `SET_ROWS` path uses the same fixed randomized Walsh-Hadamard transform
as Turbo3. The phase-1 codec follows the intended CUDA representation so it can
become a golden reference for the future SM75 port.

An isolated SM75 build compiled all TurboQuant and TriAttention CUDA objects,
including `turbo3 x q8`, `turbo4 x q8`, and the TriAttention scorer. Final
linking failed because `llama-kv-cache.cpp` declares two InnerQ functions with C
linkage while `turbo-innerq.cu` defines C++ symbols. This is a source integration
defect, not an SM75 compilation failure.

## Conflicts with wackMall

- Atomic assigns numeric type IDs 41 through 43 to Turbo formats. This fork
  already assigns 41 and 42 to `Q1_0` and `Q2_0`.
- Atomic does not provide the working wackMall MTP decode and rollback path.
- Target and draft KV types require separate validation in MTP-1, MTP-2, and
  MTP-3.
- Atomic's CUDA path is documented for Turing SM75 and newer. GTX 1080 SM61
  requires a separate backend decision and must retain a Q8/Q4 fallback.
- Several layer-adaptive policies are hard-coded to boundary layer counts and
  must not be copied into a model-independent implementation.

## Qwen3.6 implications

The local 35B-A3B model has recurrent layers plus periodic full-attention
layers. TurboQuant applies only to the actual attention KV cache. It does not
compress recurrent state or expert weights.

The first and only current runtime candidate is Turbo4 K plus Q8 V. Key
vectors affect attention inner products and match the rotated-query
formulation. Keeping V in Q8 avoids introducing a second lossy transform while
the known Q4/Q4 reasoning regression is under investigation.

The initial runtime matrix will be:

1. Q8 K / Q8 V quality reference.
2. Q8 K / Q4 V current quality-speed reference.
3. Turbo4 K / Q8 V.
4. Q4 K / Q4 V only as the known quality-sensitive comparison.

Turbo2 is excluded until Turbo3 has passed model quality and MTP tests.

## Real Qwen3.6 activation findings

The phase-1 capture selects only F32, contiguous `Qcur-L` and `Kcur-L` tensors
whose producing operation is `GGML_OP_ROPE`. This excludes an earlier Qwen
pre-norm projection that reuses the `Kcur-L` name. On the local model the
capture found ten full-attention layers: 3, 7, 11, 15, 19, 23, 27, 31, 35,
and 39. No layer count or interval is hard-coded in the tool.

An 8-token decode control and capture produced identical token IDs and the
same FNV-1a hash, `3f7b30abc1aea598`. A longer prompt processed in 22 physical
128-token prompt batches plus one generated-token graph also remained
identical, hash `d9cdcdeb936dd0f7`. The long capture retained 256 tokens per
attention layer and yielded 5,263,360 causal query/key comparisons per format.

The causal result is more informative than same-token row pairs. It quantizes
only keys, rotates queries with the matching orthogonal transform, compares
each query with every earlier visible key, and reports scaled attention-logit
and softmax errors. Aggregate long-capture results were:

| Format | K cosine | K relative L2 | Mean scaled logit error | Mean softmax KL | Mean maximum probability error | Maximum probability error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Turbo3 | 0.983363 | 0.182296 | 0.293008 | 0.043618 | 0.057609 | 0.354673 |
| Turbo4 | 0.991343 | 0.131348 | 0.203397 | 0.020174 | 0.038949 | 0.238998 |

Turbo4 is the only reasonable first runtime candidate. Turbo3 saves more KV
memory but its attention-distribution error is large enough that it must not be
treated as a quality-safe replacement for Q8. Layer 23 was the worst measured
Turbo4 layer, with mean softmax KL 0.029096 and maximum 0.153296. A future
runtime should retain a per-layer Q8 fallback so layer-adaptive policies can be
tested without assuming all attention layers tolerate the same format.

These metrics are an offline codec gate, not a model-quality pass. The Phase 2
runtime type is therefore explicit opt-in, target-K-only, requires Q8 V and
FlashAttention, and rejects speculative decoding by default. A later guarded
target-only exception permits draft-mtp only when
`LLAMA_TURBO4_MTP_EXPERIMENTAL=1`; the draft cache remains Q8/Q4. It is not a
production default.

## Phase 2 runtime finding

The local runtime uses conflict-free type ID 43, after this fork's Q1/Q2 IDs.
The 128-value serialized block is bit-for-bit identical to the Phase 1 golden
codec. CUDA `SET_ROWS`, the forward query WHT, and only the mixed
Turbo4-K/Q8-V vector FlashAttention instance are implemented. Turbo3, Turbo V,
InnerQ, TriAttention, and generic matrix multiplication remain absent.

An initial CUDA bug treated Turbo4 K like an integer-quantized K type and fed
the float centroid dot product an incompatible query layout. A 128-token model
control exposed severe repetition. Treating Turbo4 as a float-query K type
fixed the layout; the corrected 128-token output is bit-identical to Q8/Q8 and
two multidimensional CUDA-vs-CPU `SET_ROWS` cases pass.

This is a correctness milestone, not a performance win. At 1K context, three
alternating 128-token runs had median decode rates of 28.479 token/s for Q8/Q8
and 28.348 token/s for Turbo4-K/Q8-V (-0.46%). With a 2,747-token prompt, the
single Phase 2 diagnostic was 1.33% slower in prefill and 4.90% slower in the
following 32-token decode. The first token divergence was output index 12.
Long-context medians and quality evaluation are still required.

The first lower-cost K/Q experiment used a signed-int8 centroid approximation
and DP4A. Even after replacing divergent centroid loads with the existing
CUDA byte-permutation lookup, it did not improve phase-matched prefill and
introduced another numerical approximation. It is not an SM75 winner.

Profiling identified kernel selection, rather than the vector dot itself, as
the more useful target. The initial Turbo4 guard forced all query batch sizes
onto vector FlashAttention. A default-off FP16-prefill experiment now keeps
small batches below a configurable threshold on that vector path but
dequantizes rotated Turbo4 K into the existing FP16 scratch for larger tile/MMA
prompt kernels. A threshold of 32 restored the established short-prompt hash.
On the 2,747-token control its median was 35.191 Prompt TPS, 0.40% above the
Q8/Q8 median and 1.56% above the earlier single Turbo4-vector result. A 32K
allocation smoke test completed with about 485 MiB free on the GTX 1660 Ti.
This is a measured prefill improvement, not yet a model-quality or sustained
long-context acceptance.

The first model-level quality chunk produced mixed evidence. Turbo4/Q8 PPL was
11.0549 versus 11.0151 for Q8/Q8 and 11.1209 for Q4/Q4. However, its mean KL
divergence from Q8/Q8 was 0.02023 and the top token agreed 93.933% of the time,
worse than Q4/Q4 at 0.01508 and 94.129%. Turbo4 improves prediction loss over
Q4 K on this chunk but perturbs the full logit distribution more. It therefore
does not pass the model-quality gate yet.

Per-layer fallback is structurally feasible: KV storage tensors already carry
their own type, graph attention dispatch inspects the actual layer K tensor,
and state serialization records and validates each layer type independently.
The current constructor still allocates one requested K type for every layer,
and `type_k()` reports only the first storage layer. A mixed implementation
must fix logging/introspection and validate sharing/reuse before it is safe.
Offline attention error ranks layer 23 worst, followed by 35 and 27, but these
IDs are model observations and must never become hard-coded policy.

## TriAttention decision

TriAttention is a lossy KV token eviction policy, not a faster exact attention
kernel. It is useful only after the realized context becomes large enough for
the reduced attention length to offset scoring and pruning.

The Atomic implementation is not ready for this fork because it has incomplete
calibration tooling, global rather than request-local state, allocation and
synchronization in the pruning path, and no demonstrated MTP rollback safety.
Eviction also does not automatically return already allocated fixed-context KV
memory to expert placement.

TriAttention therefore remains out of the runtime integration. The offline
oracle now includes a capture-only V tensor and attention-output error for
full-attention base layers. A 1,536-token recursive history policy reached
3.65% aggregate normalized RMSE over five prune events, but the final-window
error rose to 6.85%. A second prompt reached 2.27% and 3.56%, respectively.
Keeping 1,792 of the final 2,304 positions reduced aggregate error to 2.86%
and 1.77%, but one final window still reached 4.13%. This is not a production
quality gate. A third, unrelated C++ sampling-source prompt reached 1.66%
aggregate and 1.61% final-window error at the same 1,792-token budget; its
step oracle reached 1.26% and 1.15%. The cross-prompt ranking now passes, but
rare large vector errors remain. A later phase must add downstream-logit
error. No physical eviction should be implemented until that check and the
project's long-reasoning prompts pass.

Per-layer recursive accounting found only layer 39 consistently below 1.2%
RMSE at the 1,792 budget. Restricting eviction to that layer reduces aggregate
error to 0.44%-1.02%, but returns only about 2.2% of the ten full-attention
layers' KV capacity. A milder 2,048 budget lowers that error to 0.10%-0.69%
while returning about 1.1%. This is too little memory to bypass the logit and
quality gates.

The replay-only logit gate is now implemented in the capture tool. A zero
delta produced byte-identical 248,320-value logits and the same generated
token. Three independent 2,304-token prefixes then replayed the causal
History-128 delta only at layer 39 with budget 2,048 of 2,176 old positions.
All three retained the same top token and complete Top-10 set. Relative logit
L2 error was 0.618%-0.870%, maximum absolute error 0.076-0.229, and softmax KL
was 0.0000004-0.000396. An intentionally mismatched 128/128 replay aborted
before prompt evaluation.

The gate was then expanded to all 128 evaluation positions without splitting
the final 256-token batch. A zero-delta replay was byte-identical across all
31,784,960 logits. The History-128 replay retained Top-1 on 381/384 positions
(99.22%) and at least 9/10 Top-10 entries everywhere. Aggregate relative logit
L2 was 0.648%-1.497%; mean row L2 was 0.647%-1.062%, while one Hybrid-analysis
row reached 10.06%. Three changed Top-1 positions occurred in the TurboQuant
document, but each alternative remained in the same Top-10.

A default-off layer-local live-mask diagnostic then duplicated only layer 39's
causal F16 mask and added `-inf` for the 128 exported evictions. On the sampling
prefix it retained all 128 Top-1 decisions with 0.819% relative logit L2 versus
baseline. It differed from the offline-delta replay by 0.769% L2, consistent
with the offline simulator using pre-cache F32 K/V while the live path consumes
Q8/Q8 KV. The feature-off build reproduced the earlier 121-MiB logit matrix
byte for byte. An all-keep live mask (2,176/2,176 old positions retained) was
also byte-identical across all 31,784,960 logits, isolating the measured change
to the 128 actual mask entries. Reapplying the same fixed mask for 64 later
single-token decode graphs also produced the exact baseline token sequence and
hash. These remain sensitivity results, not a runtime eviction or memory win;
layer-39-only eviction returns about 1.1% of full-attention KV.
