# ik_llama.cpp concept audit and experiments

Date: 2026-08-07

Reference: `ikawrakow/ik_llama.cpp` commit
`40dffce6857b4fe051f096379dc464764c718458`, inspected read-only apart from
temporary benchmark-only changes below `/tmp`.

The project is a valuable source of focused inference ideas, but it is based
on a substantially older and heavily modified llama.cpp graph/runtime. A
wholesale merge would replace the fast wackMall CUDA, expert-tier, and MTP
paths and is out of scope. Every experiment below is default-off.

## Candidate ranking

| Concept | Relevance to this fork | Decision |
| --- | --- | --- |
| Low-perplexity Q4_0 KV scale | Directly targets the known Q4 KV quality issue | Implemented and measured as an optional K/V policy |
| Hadamard K/V cache rotation | Reduces quantization outliers | Already present in the Turbo4 path; Q4 ablation confirms it should remain enabled |
| Requantized MTP-only output head | May reduce the very wide draft LM-head cost | Implemented for Q4_K/Q5_K and measured; default-off |
| IQK AVX2 Q4_K GEMM/GEMV | Could accelerate CPU Cold Experts | Rejected after phase-matched synthetic kernel tests |
| Fused MoE Gate/Up | Avoid duplicate activation reads and intermediate graph work | Single-token CUDA fusion already exists; native CPU dual-dot tested below and remains default-off |
| Trellis/IQK model quants | Could improve quality per model byte | Requires a new GGUF/model conversion and complete CPU/CUDA support; separate future model project |
| Row-interleaved repacking | Useful for fully CPU-resident weights | Rejected for the current hybrid: ik_llama itself warns against repacking hybrid K-quant MoE tensors without matching CUDA kernels |
| Auto-fit and per-GPU margin | Important for 6/8 GiB cards | Already provided by wackMall's byte-based expert/KV fit; retain and extend rather than import |
| Fused delta-net/GDN | Relevant to Qwen3.6 recurrent layers | Current wackMall model path already has its own maintained recurrent implementation; no old graph transplant |
| Self-speculative/ngram/suffix | Can be chained ahead of MTP without a second model | Existing N-gram+MTP chain tested; no local winner, remains default-off |
| DFlash draft architecture | Model-specific speculative decoder with target-feature tensors | Not applicable to the local GGUF; requires a separately trained/converted DFlash package |
| TriAttention token eviction | Lossy long-context KV reduction from Atomic TurboQuant, not DFlash | Atomic score rejected for Qwen3.6; causal heavy-hitter selection advances to the next offline gate |
| On-demand tensor reload | Load-time/RAM feature | Does not address sustained decode on the current resident model |

## Low-perplexity Q4_0 KV scale

`LLAMA_KV_Q4_SCALE` accepts `legacy`, `weighted`, `weighted-k`, and
`weighted-v`. The weighted path preserves the ordinary Q4 codes and
recomputes only each block scale by weighted least squares. The legacy path is
the exact default.

On a 1,024-token chunk of `ROUTER_LOOKAHEAD_ANALYSIS.md`, relative to Q8/Q8:

| Q4 mode | PPL | Mean KLD | RMS probability delta | Same top token |
| --- | ---: | ---: | ---: | ---: |
| legacy K/V | 11.1209 | 0.015082 | 2.845% | 94.129% |
| weighted K/V | 11.2732 | 0.016006 | 2.902% | 94.325% |
| weighted K only | 11.0672 | 0.014675 | 2.599% | 94.716% |
| weighted V only | 11.1263 | 0.016536 | 2.845% | 96.086% |

Weighted K is the only promising form on this corpus. A second 1,024-token
chunk was mixed: PPL improved from 14.2782 to 14.1355 and maximum KLD from
0.349 to 0.164, while mean KLD rose slightly from 0.015594 to 0.015909.

Three deterministic 2,000-token MTP-2 runs then measured 42.815 token/s for
weighted K versus 43.177 token/s for legacy (-0.84%). The cache policies
produced different token streams and speculative acceptance, so this is a
system comparison, not a pure kernel result. Weighted K remains a quality
option, never a default or an unqualified quality claim.

Disabling the existing Q4 attention rotation increased mean KLD from 0.015082
to 0.022232 (+47%) and maximum KLD from 0.1792 to 0.5601. The rotation remains
enabled.

## MTP-only output-head requantization

`LLAMA_MTP_REQUANTIZE_OUTPUT=none|q4_K|q5_K` creates an additional derived
output tensor at model load and uses it only in the MTP graph. The target
graph continues to use the original Q6_K output. On the local model the shared
head is `2048 x 248320`, Q6_K, 397.85 MiB. Runtime Q4_K is 272.81 MiB and
takes about 4.8 seconds to construct with eight host threads.

At identical fixed `S=29`, three 512-token runs gave:

| MTP head | Median TPS | Acceptance | Mean accepted | Peak VRAM | Hash relation |
| --- | ---: | ---: | ---: | ---: | --- |
| original Q6_K | 43.574 | 66.14% | 2.32 | 5332 MiB | same |
| derived Q4_K | 43.672 | 65.91% | 2.32 | 5606 MiB | same |

The isolated gain is only 0.23%. A separate three-run comparison on the later
learned profile and `S=28` measured Q5_K at 41.421 token/s versus Q6_K at
41.086 token/s (+0.82%) with identical hashes and MTP statistics. Q5_K needs
about 334 MiB additional VRAM and forced an attempted `S=29` down to `S=28`.

With the later profile, Q4_K could retain `S=29` but changed the target token
stream through the different CPU/GPU expert split and draft rounding. Its
median was 39.290 token/s with 62.33% acceptance. This cannot be interpreted
as a pure Q4 kernel comparison, but it demonstrates that an extra fixed slot
does not guarantee a system win.

Decision for the 6-GiB GTX 1660 Ti: keep the implementation default-off; the
extra tensor displaces too much expert residency for a sub-one-percent
isolated gain. Retest Q4_K and Q5_K on the 8-GiB GTX 1080, where the derived
head may fit without reducing fixed slots. Native SM61 kernel validation is
mandatory.

## IQK AVX2 Q4_K kernel gate

The reference was built separately with GCC 13, `-O3`, `-march=native`, eight
threads, and its optimized IQK matrix multiplications. The same deterministic
benchmark source, two alternating Q4_K matrices, shape `2048 x 512`, and
thread count were then compiled against this fork. This is representative of
one Qwen3.6 gate/up expert, with `n=1/2/4` covering decode and small MTP verify
widths. Fifty iterations were used; both averages include the first cold
iteration, so they are conservative and directly comparable.

| Input columns | ik_llama IQK | wackMall current | wackMall delta |
| ---: | ---: | ---: | ---: |
| 1 | 97.18 GFLOPS | 117.07 GFLOPS | +20.5% |
| 2 | 135.96 GFLOPS | 159.82 GFLOPS | +17.5% |
| 4 | 211.88 GFLOPS | 220.24 GFLOPS | +3.9% |

The current fork wins every MTP-relevant width. Do not import the IQK
subsystem for this model/CPU. The result does not judge IQK's large prompt
GEMMs, new IQ/Trellis model formats, AVX-512 path, or other CPUs. Any future
revisit must start with a standalone benchmark on the target CPU and must not
replace `MOE_COLD` until it wins end-to-end.

### Fused Q4_K Gate/Up follow-up

The generic IQK matrix test does not isolate IQK's fused Gate/Up operation.
A reproducible cross-tree benchmark in
`tools/bench-iqk-fused-up-gate.cpp` therefore measured two `2048 x 512`
Q4_K projections plus SiLU multiplication at one, two, and four input
columns. Across five 500-iteration runs, the IQK fused median was 17.7%, 9.1%,
and 5.2% faster, respectively, than two ordinary wackMall graph matmuls plus
activation. This is an isolated upper-bound signal, not an end-to-end result:
wackMall's `MOE_COLD` already quantizes input once and processes Gate/Up in
one expert loop.

The benchmark is intentionally not a normal CMake target because it is built
twice against incompatible GGML ABIs. Use `-DBENCH_IK_LLAMA` for the ik build;
without it, the source selects the ordinary wackMall graph. Both builds must
use their own headers, shared library, and runtime rpath.

A smaller native experiment was consequently implemented behind
`LLAMA_EXPERT_CPU_FUSED_GATE_UP=1`. Its AVX2 Q4_K dual-dot evaluates separate
Gate and Up weight rows against one Q8_K activation row while loading the
activation block once. Non-x86 and non-Q4_K cases retain the exact old path.
The unit test proves both returned floats are bit-identical to two ordinary
dots.

One phase-timed 256-token MTP-2 pair measured:

| Measurement | Existing | Dual-dot | Difference |
| --- | ---: | ---: | ---: |
| CPU Gate + Up | 1112.610 ms | 1095.173 ms | -1.57% |
| CPU Cold total | 2017.066 ms | 1999.938 ms | -0.85% |
| Decode | 40.964 tok/s | 41.130 tok/s | +0.41% |

The subsequent no-timing static No-Sync test used the established Q4/Q4
MTP-2 kernel controls, S=33, and three fresh 512-token servers per side:

| Dual-dot | Runs | Median decode | Acceptance | Mean accepted | Hashes |
| --- | ---: | ---: | ---: | ---: | --- |
| off | 3 | 42.994 tok/s | 62.97% | 2.26 | identical |
| on | 3 | 43.084 tok/s | 62.97% | 2.26 | identical |

The +0.21% median is below the promotion gate, so no 2,000-token run was
performed and the feature remains default-off. It is still a useful GTX 1080
screen: that host's older quad-core CPU may place more Cold work on the
critical path. Artifacts are under `/tmp/ik-fused-gate-up-*20260807`.

The CUDA side needs no equivalent transplant for the one-token path. With
shared hot IDs, this fork already recognizes the two `MUL_MAT_ID` nodes plus
GLU and launches the existing fused MMVQ operation. The separate
`GGML_CUDA_MOE_MULTI_FUSION` control extends that mechanism to two through
four target tokens on Turing; this was measured earlier and remains guarded
on Pascal pending native SM61 tests.

## N-gram plus MTP chain

The ik_llama audit highlighted a two-stage self-speculative chain. No port was
needed: this wackMall base already prioritizes N-gram implementations and
falls back to `draft-mtp` when they produce no draft. The benchmark runner now
accepts `SPEC_TYPES_OVERRIDE` plus bounded N-gram-mod and N-gram-simple sizes.
Defaults remain unchanged.

A Q8/Q8, static-S screen used MTP-2, a fresh server per run, and 256 generated
tokens. These are stop-rule screens, not medians:

| Workload | Speculation | Decode | Draft acceptance | Mean accepted | Hash relation |
| --- | --- | ---: | ---: | ---: | --- |
| General agent prompt | MTP-2 | 44.825 tok/s | 70.62% | 2.41 | baseline |
| General agent prompt | N-gram-mod 24/2/8 + MTP-2 | 45.031 tok/s | 70.62% | 2.41 | same |
| General agent prompt | N-gram-mod 8/2/8 + MTP-2 | 42.049 tok/s | 68.66% | 2.41 | same |
| Exact code-copy prompt | MTP-2 | 41.791 tok/s | 85.64% | 2.71 | baseline |
| Exact code-copy prompt | N-gram-simple 4/8/1 + MTP-2 | 33.207 tok/s | 74.44% | 2.87 | different |

Match length 8 also triggers the upstream poor-quality warning. The
conservative match-24 result is within single-run noise and appears to fall
through to MTP; it is not a gain claim. Aggressive N-gram-mod regressed 6.2%,
and the code-copy N-gram-simple chain regressed 20.5% while changing the token
stream. The >3% stop condition applies, so no 2,000-token campaign was run.
Keep MTP-2 alone for the GTX 1660 Ti. The parameterized runner is retained for
SM61 and future GPUs, where larger verify batches may have a different cost.

## DFlash is not TriAttention

ik_llama's DFlash support builds a separate draft model architecture and
expects DFlash-specific target-feature metadata/tensors. The local Qwen3.6
GGUF does not expose that package, so DFlash cannot be enabled as a generic
KV eviction switch. TriAttention is the separate lossy token-selection idea
audited from Atomic TurboQuant.

## Offline KV eviction simulation

The Atomic reference at commit
`f67e13573ab344d98090ea7612056a0119fcc5ef` was inspected read-only. Its
documentation describes calibration-guided trigonometric scoring and its
runtime defaults to protecting every prefill token. Prompt KV reduction would
therefore require the explicit lossy `--triattention-no-protect-prefill` mode.
The documented `scripts/calibrate-triattention.py` file is not present in that
commit.

There is also a model compatibility issue. The local Qwen3.6 model has a
256-dimensional attention head but `n_rot=64`, IMRoPE, and sections
`[11,11,10,0]`. Atomic's calibration format fixes `freq_count=head_dim/2` and
its scorer inverts RoPE across the complete head. It cannot represent this
partial-RoPE layout correctly without a format and kernel extension.

`tools/turboquant-ref/simulate-triattention.py` is an independent,
non-mutating simulator. It validates an immutable post-RoPE Q/K capture,
learns only from tokens before a simulated prune boundary, and evaluates
later exact Q/K attention. It reports retained attention mass, old-token mass,
Top-1 retention, and Top-8 recall. Its fixed future-looking oracle is an upper
bound, not a deployable policy. A later capture-only mode adds a materialized
pre-attention V tensor and reconstructed output-vector error. Downstream logits
are still unavailable, so these measurements remain a screening proxy rather
than a quality claim.

Two Q8/Q8 captures were made with static S=33, no MTP, no adaptation, and no
physical eviction. Each records 6,174 tokens for full-attention layers 3, 7,
11, 15, 19, 23, 27, 31, 35, and 39. The first source is
`HYBRID_ANALYSIS.md`; the second is `README.md`. Both produced the same
one-token generated hash. Raw captures remain under `/tmp` and are excluded from
Git.

At a prune boundary of 4,096 tokens, a recent window of 128, an evaluation
window of 128, and a total keep budget of 2,048:

| Source / policy | Mean mass retained | P05 mass | Old mass retained | Top-1 retained | Top-8 recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Analysis / recency | 80.48% | 51.38% | 72.47% | 87.71% | 85.36% |
| Analysis / Atomic score, normalized | 76.99% | 50.84% | 67.55% | 89.21% | 83.49% |
| Analysis / history-128 | 91.36% | 75.19% | 87.81% | 98.85% | 96.85% |
| Analysis / future oracle | 93.15% | 78.58% | 90.33% | 99.59% | 98.46% |
| README / recency | 65.67% | 23.86% | 55.74% | 72.18% | 68.90% |
| README / Atomic score, normalized | 82.23% | 61.33% | 77.09% | 92.21% | 87.07% |
| README / history-128 | 93.09% | 75.83% | 91.09% | 98.75% | 97.39% |
| README / future oracle | 94.66% | 82.10% | 93.11% | 99.72% | 98.77% |

The causal history score sums exact attention received by old positions over
the previous 128 queries. It never reads beyond the prune point. On the first
capture it also beat recency at 2K, 3K, and 5K prune points under a 50% budget.
History lengths 32, 64, 128, 256, and 512 were compared at 4K; 128 won mean
and P05 retention, while 32 produced the best single-query minimum.

The no-eviction invariant (budget equals prefix length) returns exactly 1.0
for every retention metric and every policy. The simulator refuses to
overwrite an output and refuses to write inside a capture directory.

The stronger recursive test starts at occupancy 2,176, keeps 2,048 tokens,
and prunes every 128 tokens. Evicted positions cannot be selected again. All
31 complete windows through token 6,144 were evaluated:

| Source / recursive policy | Mean mass retained | P05 mass | Old mass retained | Top-1 retained | Top-8 recall | Last-window mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Analysis / recency | 75.55% | 30.40% | 66.64% | 80.28% | 78.51% | 58.29% |
| Analysis / history-128 | 86.05% | 53.48% | 80.96% | 93.86% | 90.81% | 68.38% |
| Analysis / step oracle | 87.14% | 56.28% | 82.45% | 94.55% | 91.78% | 70.12% |
| README / recency | 70.96% | 24.91% | 59.97% | 76.87% | 74.03% | 70.59% |
| README / history-128 | 85.80% | 52.93% | 80.43% | 93.50% | 89.80% | 81.79% |
| README / step oracle | 86.88% | 55.31% | 81.92% | 94.16% | 90.76% | 82.67% |

History-128 remains within about 1.1 percentage points of the step oracle on
both sources and beats recency by 10.5 and 14.8 points. This survives the
irreversibility gate. Minimum per-query retained mass is still below 0.1% for
all policies, so mean retention alone is not a sufficient quality gate.

Decision: do not port Atomic's current TriAttention runtime. Advance the
causal heavy-hitter concept to a second offline phase covering additional
prompt classes and attention-output/logit error. Physical eviction, MTP, and
server integration remain disabled.

An attempted default-off V observer was stopped and removed. Asking the
evaluation callback for the aliasing `Vcur-3` reshape deterministically caused
the existing CPU `MUL_MAT_ID` guard `e >= 0 && e < n_as` to abort twice before
the first capture event. A replacement capture-only path materializes V with a
dedicated `DUP` node that is consumed by attention. No-capture, Q/K-only, and
Q/K/V controls all produced `[271, 248068]` and hash `d72f773530d682db`; the
6,174-token long Q/K/V capture produced the same token and hash
`d9cdcdeb936dd0f7` as its earlier Q/K control. The normal inference graph is
unchanged unless `--capture-values` is explicitly requested.

The long capture retained 2,304 tokens from all ten full-attention layers.
At a single prune at token 2,048, the attention-output results were:

| Policy / retained prefix | Mass retained | Normalized RMSE | Mean relative L2 | P95 relative L2 | Mean cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recency / 1,024 | 80.04% | 12.00% | 16.84% | 54.62% | 97.32% |
| Atomic normalized / 1,024 | 74.62% | 10.19% | 15.32% | 43.28% | 98.02% |
| History-128 / 1,024 | 86.25% | 8.24% | 9.20% | 37.15% | 98.78% |
| Oracle / 1,024 | 91.33% | 4.13% | 5.32% | 18.20% | 99.69% |
| Recency / 1,536 | 91.46% | 9.69% | 12.06% | 46.49% | 98.07% |
| Atomic normalized / 1,536 | 90.28% | 5.41% | 6.62% | 23.05% | 99.42% |
| History-128 / 1,536 | 95.62% | 4.55% | 3.50% | 17.29% | 99.61% |
| Oracle / 1,536 | 97.89% | 1.21% | 1.44% | 5.68% | 99.97% |

The recursive 1,536-token test performed five irreversible prune events from
token 1,664 through 2,304. History-128 achieved 3.65% aggregate normalized
RMSE versus 11.42% for recency and 2.91% for the step oracle. Its per-event
RMSE rose from 0.45% to 6.85% at the last window. The policy is promising, but
the growing tail error and low worst-case cosine are still incompatible with
physical eviction or MTP enablement. The outputs are measured against captured
pre-cache F32 Q/K/V, not quantized-cache values or downstream logits.

A second 2,304-token Q/K/V capture used `TURBOQUANT_ANALYSIS.md`. At the same
recursive 1,536-token budget, History-128 improved to 2.27% aggregate RMSE and
3.56% in the final window; the step oracle reached 1.77%. A more conservative
1,792-token budget (77.8% of the final 2,304 positions) produced the following
cross-prompt result:

| Source / recursive policy | Aggregate RMSE | Mean relative L2 | P95 relative L2 | Final-window RMSE |
| --- | ---: | ---: | ---: | ---: |
| Hybrid analysis / history-128 | 2.86% | 1.51% | 6.44% | 4.13% |
| Hybrid analysis / step oracle | 1.92% | 0.95% | 3.76% | 3.08% |
| TurboQuant analysis / history-128 | 1.77% | 0.92% | 3.72% | 2.80% |
| TurboQuant analysis / step oracle | 1.12% | 0.61% | 2.47% | 1.88% |
| Sampling C++ source / history-128 | 1.66% | 1.59% | 6.02% | 1.61% |
| Sampling C++ source / step oracle | 1.26% | 1.01% | 3.89% | 1.15% |

The third capture used `tests/test-sampling.cpp`, tokenized to 7,293 prompt
tokens, and captured the same first 2,304 positions from all ten full-attention
layers. This is structurally different from the two Markdown sources. Its
generated token hash was `420a0bf869272c4c`; the capture completed without an
OOM or CUDA error. The conservative policy is much closer to the oracle on all
three inputs and may deserve a later logit-level gate, but it still has a 4.13%
last-window error on one source and large rare vector errors. It does not
justify runtime eviction yet.

The recursive report now also records per-layer metrics and raw squared error
and reference energy. This permits a correct offline estimate for selective
eviction: excluded layers contribute zero error but retain their reference
energy. At budget 1,792, layer 39 was the only consistently low-error layer
(maximum per-layer RMSE 1.18%); the next best layer already reached 4.48% on
one prompt. Evicting only layer 39 produced total normalized RMSE between
0.44% and 1.02%, but saves only about 2.2% of full-attention KV at this 22.2%
per-layer eviction ratio. At the milder 2,048 budget, layer-39-only error was
0.10% to 0.69% and the full-attention KV saving was only about 1.1%. The
quality/risk ratio is better, but the memory return is too small to justify a
runtime implementation before downstream-logit testing.

### Replay-only downstream logit gate

The capture executable can now truncate a prompt at a deterministic token
count, dump immutable raw F32 logits, and apply an offline attention-output
delta to one recorded graph/layer. This is not cache eviction: all K/V entries
remain present. The replay manifest validates model path and bytes, prompt
token hash, context, batch, ubatch, and K/V types before evaluation.

A no-prune layer-39 replay added an exactly zero delta and produced a
byte-identical 993,280-byte logit file. At budget 2,048 with prune point 2,176
and a 128-token evaluation window, three prompt prefixes produced:

| Source | Relative logit L2 | Maximum absolute logit error | Softmax KL | Top token | Top-10 overlap |
| --- | ---: | ---: | ---: | --- | ---: |
| Sampling C++ source | 0.630% | 0.0756 | 0.0001310 | same | 10/10 |
| TurboQuant analysis | 0.618% | 0.0760 | 0.0000004 | same | 10/10 |
| Hybrid analysis | 0.870% | 0.2286 | 0.0003961 | same | 10/10 |

One sampling-source screen reduced the budget further to 1,792. Its final
top token and Top-10 still matched, with 0.635% relative logit L2 and 0.000126
KL. This single final-position result does not clear the more aggressive
policy; the attention-output tail error and small one-layer memory return
still require a multi-position and live-mask gate.

That conservative gate is now complete. Across all 128 evaluation positions
and three prefixes, 381/384 Top-1 decisions matched and every row retained at
least 9/10 Top-10 entries. Aggregate relative logit L2 ranged from 0.648% to
1.497%, with a rare 10.06% row-level outlier. A zero delta was bit-identical
across all 31,784,960 logits. The actual layer-39 live-mask diagnostic retained
128/128 Top-1 decisions on the sampling prefix at 0.819% aggregate L2, while
leaving the Q8/Q8 KV cache allocated and unchanged. Feature-off output remained
byte-identical, as did a separate all-keep live mask. This is sufficient to
retain the diagnostic, but not to justify physical eviction for an
approximately 1.1% full-attention KV saving.

A 64-token greedy continuation with the fixed live mask also matched every
baseline token (`fd7a95ebca41dd36`) while the guard observed exactly 64 decode
graphs. This clears a bounded autoregressive sensitivity check, not the active
long-context quality, memory-reclamation, or MTP gates.

## Next gated work

1. Run the MTP-head Q4_K/Q5_K isolation screen natively on SM61 without losing
   fixed slots.
2. Expand weighted-K quality evaluation with a bounded reasoning prompt where
   Q8/Q8 itself reaches a correct final answer.
3. Add an autoregressive continuation and active long-context quality gate for
   the layer-39 live mask. Do not physically evict KV entries before quality,
   memory-benefit, multi-request, and MTP rollback checks pass.
4. Consider IQ/Trellis formats only with a separately quantized model and
   phase-matched CPU, CUDA, perplexity, and MTP evaluation.
