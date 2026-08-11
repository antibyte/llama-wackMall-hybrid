# TurboQuant experiments

Date: 2026-08-07

## Atomic SM75 build audit

Source commit: `f67e13573ab344d98090ea7612056a0119fcc5ef`

Configuration used native CUDA architecture 75. The build compiled 392 of 393
steps, including the Turbo3/Turbo4 quantizers, mixed Q8/Turbo FlashAttention
instances, TriAttention CUDA scorer, and `libggml-cuda`.

The final `llama-cli` link failed with:

```text
undefined reference to turbo_innerq_needs_tensor_update
undefined reference to turbo_innerq_mark_tensor_updated
```

The CUDA translation unit exports C++-mangled functions while the KV source
requests C symbols. The phase-1 implementation does not import InnerQ.

## Local reference build

```bash
cmake -S . -B build-turbo-ref -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_UI=OFF

cmake --build build-turbo-ref -j 8 \
    --target test-turboquant-ref llama-turboquant-ref

ctest --test-dir build-turbo-ref \
    -R '^test-turboquant-ref$' --output-on-failure
```

Result: passed.

A separate Debug build with AddressSanitizer and UndefinedBehaviorSanitizer also
passed with `ASAN_OPTIONS=detect_leaks=0`. LeakSanitizer itself cannot run under
the ptrace-based CLI environment, so leak detection was the only disabled
sanitizer component.

The existing `test-quantize-fns` suite also passed for every pre-existing GGML
type. Phase 1 does not add a global type and therefore does not change the
existing quantization dispatch table.

## Synthetic codec results

The test used 256 deterministic Gaussian query/key pairs for each format and
logical row width.

| Format | Width | Mean cosine | Mean relative L2 | Mean normalized dot error | Maximum dot error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Turbo3 | 64 | 0.984026 | 0.177823 | 0.0172885 | 0.0680579 |
| Turbo4 | 64 | 0.991661 | 0.125824 | 0.0130601 | 0.0541065 |
| Turbo3 | 128 | 0.983391 | 0.181625 | 0.0125160 | 0.0426251 |
| Turbo4 | 128 | 0.990925 | 0.132444 | 0.00963087 | 0.0532568 |
| Turbo3 | 256 | 0.982955 | 0.184314 | 0.00873873 | 0.0377624 |
| Turbo4 | 256 | 0.990946 | 0.133530 | 0.00630125 | 0.0332805 |

These are codec checks, not Qwen3.6 quality results. They establish deterministic
packing, reversible WHT, norm correction, padding behavior, and baseline error
bounds.

## Real Qwen3.6 post-RoPE capture

Hardware: NVIDIA GTX 1660 Ti, native SM75 build. Model: local Qwen3.6
35B-A3B Q4_K_M with MTP metadata, run without MTP for the capture. Target KV
was Q8_0/Q8_0, fixed expert placement was S=33, warm cache and adaptation were
disabled, and physical batch/ubatch were 128/128.

The dedicated capture executable does not register a TurboQuant runtime type.
It observes only post-RoPE Q/K tensors and writes bounded raw F32 diagnostic
files. The ordinary no-capture path has no evaluation callback.

### Observational-equivalence controls

| Workload | Control hash | Capture hash | Token IDs identical | Result |
| --- | --- | --- | --- | --- |
| `fft in c`, 8 generated tokens | `3f7b30abc1aea598` | `3f7b30abc1aea598` | yes | pass |
| Long repository analysis prompt, 1 generated token | `d9cdcdeb936dd0f7` | `d9cdcdeb936dd0f7` | yes | pass |

The long prompt used 22 physical prompt graphs at ubatch 128 and therefore
exceeded 2,000 input tokens. Capture was capped at 256 tokens per layer. It
found layers 3, 7, 11, 15, 19, 23, 27, 31, 35, and 39, for 2,560 captured
layer-tokens. No private prompt contents or model data are tracked by Git.

### Causal attention results

The corrected analysis quantizes only unique K rows. Every captured Q head is
compared with all causally visible K rows using the model's GQA head mapping.
There were 40,960 query distributions and 5,263,360 Q/K comparisons per
format.

| Format | Bits/value | K cosine | K relative L2 | Mean normalized dot error | Max normalized dot error | Mean scaled logit error | Max scaled logit error | Mean softmax KL | Max softmax KL | Mean max probability error | Max probability error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Turbo3 | 3.125 | 0.983363 | 0.182296 | 0.009456 | 0.068976 | 0.293008 | 2.472405 | 0.043618 | 0.348765 | 0.057609 | 0.354673 |
| Turbo4 | 4.25 | 0.991343 | 0.131348 | 0.006571 | 0.052184 | 0.203397 | 1.826764 | 0.020174 | 0.153296 | 0.038949 | 0.238998 |

Turbo4 wins every aggregate accuracy metric, but its worst-case errors are not
small enough to claim model-level quality. Layer 23 was the weakest Turbo4
layer in this capture. The result advances Turbo4 K/Q8 V to the first runtime
experiment and keeps Turbo3 out of the first runtime patch.

### Capture and analysis commands

Use a new path for every run; the tools intentionally refuse overwrite:

```bash
CAPTURE=/tmp/turboquant-qk-capture-01

LLAMA_CMOE_BATCH=128 \
LLAMA_CMOE_UBATCH=128 \
LLAMA_EXPERT_HOT="$PROFILE" \
LLAMA_EXPERT_S=33 \
LLAMA_EXPERT_ADAPT=0 \
LLAMA_EXPERT_STATS=0 \
LLAMA_EXPERT_WARM_SLOTS=0 \
LLAMA_EXPERT_STATIC_NO_SYNC=1 \
./build-turbo-sm75/bin/llama-turboquant-capture \
    --capture-dir "$CAPTURE" \
    --capture-max-tokens 256 \
    -m "$MODEL" -f "$PROMPT" -c 4096 -b 128 -ub 128 \
    -ctk q8_0 -ctv q8_0 -fa on -cmoe -t 8 -tb 8 \
    --offline --seed 1234 --temp 0 -n 1

python3 tools/turboquant-ref/analyze-capture.py \
    --capture-dir "$CAPTURE" \
    --analyzer build-turbo-sm75/bin/llama-turboquant-ref \
    --output /tmp/turboquant-qk-analysis.json
```

### Stop decision

Phase 1 is complete. Do not port Turbo3 first, do not import TriAttention, and
do not enable TurboQuant for MTP. The next bounded task is an SM75 Turbo4
K-only runtime spike with Q8 V, explicit Q8 per-layer fallback, and no MTP.
Quality evaluation remains separate from throughput evaluation.

## Phase 2 SM75 runtime spike

The runtime spike registers only `turbo4_k` as target K cache type. Q8 V,
FlashAttention, KQV offload, and no speculative context are mandatory. The
type remains opt-in and is not accepted for target V or draft K/V.

Build and focused tests:

```bash
cmake --build build-turbo-capture-sm75 -j 8 \
    --target llama-turboquant-capture test-turboquant-ref test-backend-ops

ctest --test-dir build-turbo-capture-sm75 \
    -R '^test-turboquant-ref$' --output-on-failure

./build-turbo-capture-sm75/bin/test-backend-ops test \
    -b CUDA0 -o SET_ROWS -p turbo4
```

The SM75 full build and reference test passed. Both focused CUDA `SET_ROWS`
cases passed against the CPU implementation, including I32/I64 indices,
broadcast dimensions, and a non-contiguous index view.

The first 128-token model run exposed an incompatible query layout in the
FlashAttention template. Turbo4 was initially classified like an integer
quant, while its centroid dot product reads the float query representation.
The resulting output repeated one token. The corrected template classifies
Turbo4 as a float-query K type. After rebuilding, all 128 generated token IDs
and hash `75f97ff7aef1c7c9` matched Q8/Q8.

Three alternating fresh processes per type at context 1024 produced:

| Target KV | Decode TPS runs | Median | Token hash |
| --- | --- | ---: | --- |
| Q8/Q8 | 28.479, 28.463, 29.290 | 28.479 | `75f97ff7aef1c7c9` |
| Turbo4-K/Q8-V | 27.539, 29.180, 28.348 | 28.348 | `75f97ff7aef1c7c9` |

The median delta is -0.46%, which is neutral for this short-context diagnostic
and not a speed claim.

The checked-in `ROUTER_LOOKAHEAD_ANALYSIS.md` tokenizes to 2,747 prompt tokens
with the target tokenizer. One phase-matched long diagnostic produced:

| Target KV | Prompt TPS | Decode TPS (32 tokens) | Hash |
| --- | ---: | ---: | --- |
| Q8/Q8 | 35.116 | 31.714 | `de9f61f20bde59c6` |
| Turbo4-K/Q8-V | 34.650 | 30.159 | `d5b581601a7a6b95` |

Turbo4 was 1.33% slower in this prefill and 4.90% slower in the short decode;
the first output divergence was zero-based token index 12. This is only one
long run, so it triggers more investigation rather than a final rejection.
MTP remains blocked. The next useful kernel experiment is a lower-cost
Turbo4 K/Q dot product; repeating the current float-centroid VEC kernel at 32K
would not explain or remove its exposed compute cost.

### DP4A K/Q experiment

`GGML_CUDA_TURBO4_KQ_DP4A` replaces the float centroid K/Q dot product with a
signed-int8 centroid approximation and a Q8 query. It is a compile-time,
default-off diagnostic. The least-squares centroid scale is
`0.0013702924565985558`; maximum absolute centroid error is about 0.000650.

The first implementation used four divergent constant-memory lookups per
`dp4a`. A second implementation reused CUDA's existing `prmt`-based 16-entry
table lookup, processes eight indices in each active lane, and removes those
divergent loads. Both implementations produced coherent text, but their
additional approximation and changed reduction order changed token hashes.

Results on the same SM75 host were:

| DP4A variant | 128-token Decode TPS | 2,747-token Prompt TPS |
| --- | ---: | ---: |
| Scalar lookup runs | 27.518, 28.370, 28.306, 27.448 | 34.617 |
| PRMT lookup | 29.141 diagnostic | 34.634 |
| Float centroid reference | 29.297, 29.218, 29.208 A/B runs | 34.650 |

The scalar DP4A median was 28.306 versus 29.218 for the interleaved float
reference (-3.12%). Autoregressive rates are not a pure kernel comparison once
the token streams diverge. The phase-matched prefill is decisive: PRMT DP4A
was still 0.05% slower than the float dot and 1.37% slower than Q8. DP4A is
therefore rejected for SM75 and remains compile-time disabled. No long quality
claim is made for it.

### CUDA timeline finding

Nsight Systems traces of the bit-identical short Q8 and float-centroid Turbo4
runs showed that Turbo4 WHT and quantization are small in one-token decode.
The important dispatch difference appears during larger prompt batches: Q8
can use the FP16 tile/MMA FlashAttention path, while the initial Turbo4 guard
forced every batch onto the vector kernel. The profiler reports are local
diagnostic artifacts under `/tmp` and are not tracked.

### Experimental FP16 prefill path

`GGML_CUDA_TURBO4_F16_PREFILL` is a separate default-off build option. Decode
and prompt fragments below `GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH` continue
to use the float-centroid vector path. The threshold defaults to 32. Larger
prompt batches dequantize rotated Turbo4 K to the existing temporary FP16
FlashAttention buffer and use the normal Turing tile/MMA dispatch. Stored KV
remains Turbo4 K plus Q8 V.

Three fresh processes per type used the identical checked-in 2,747-token
prompt and generated one token:

| Target KV / prefill path | Prompt TPS runs | Median | First-token hash |
| --- | --- | ---: | --- |
| Q8/Q8 normal dispatch | 35.116, 35.049, 35.052 | 35.052 | `d9cdcdeb936dd0f7` |
| Turbo4-K/Q8-V FP16 prefill | 35.191, 35.242, 35.102 | 35.191 | `d9cdcdeb936dd0f7` |

The Turbo4 median is 0.40% above Q8 and 1.56% above the earlier single
Turbo4-vector prefill result. A 32-token long-prompt diagnostic reached 35.242
Prompt TPS and 30.319 Decode TPS with coherent output and hash
`d5b581601a7a6b95`, matching the earlier Turbo4-vector hash for that workload.

The initial threshold of three query columns changed the short prompt hash
because FP16 prefill changes rounding. Raising the configurable threshold to
32 restored the established 128-token Turbo4/Q8 hash
`75f97ff7aef1c7c9`. A repeated 2,747-token run still used the large-batch path
and reached 35.151 Prompt TPS with the same first-token hash. At context 32768,
the same long prefill completed at 35.079 Prompt TPS; observed GPU allocation
was about 5,260 MiB with about 485 MiB free. This is an OOM smoke test, not a
32K quality or sustained-decode benchmark.

Focused validation after the threshold change passed the CPU golden codec,
both CUDA `SET_ROWS` Turbo4 cases, sampling, reasoning-budget, and grammar
parser tests. A complete all-target build remains blocked by the existing
`test-chat` target missing the `mtmd.h` include path in this configuration.
The network-dependent model-download test is also unavailable offline. These
limitations must not be reported as Turbo4 regressions or as a full-suite pass.

This path is promising for memory efficiency and prompt speed, but stays
default-off until perplexity, long reasoning, 64K memory, and no-MTP quality
tests pass.

### Initial model-level quality matrix

The checked-in 2,747-token analysis document was also used as an initial
model-level corpus. `llama-perplexity` evaluated one 1,024-token chunk with
batch/ubatch 128, static S=33, adaptation off, no MTP, and identical Q8 V where
the K format was the variable. This is a smoke gate rather than a substitute
for Wikitext or the private long-reasoning corpus.

| Target KV | PPL | Relative PPL vs Q8/Q8 | Mean KLD | Maximum KLD | Same top token | RMS probability delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8/Q8 | 11.0151 | baseline | 0 | 0 | 100% | 0% |
| Q8/Q4 | 11.0262 | +0.10% | 0.01242 | 0.19291 | 95.108% | 2.596% |
| Q4/Q8 | 11.1125 | +0.88% | 0.01479 | 0.22300 | 94.129% | 3.008% |
| Q4/Q4 | 11.1209 | +0.96% | 0.01508 | 0.17918 | 94.129% | 2.845% |
| Turbo4/Q8 | 11.0549 | +0.36% | 0.02023 | 0.39092 | 93.933% | 2.920% |
| Turbo4/Q8, layer 23 Q8 K | 11.0370 | +0.20% | 0.01857 | 0.26956 | 93.933% | 2.935% |
| Turbo4/Q8, layers 23/35/27 Q8 K | 11.0382 | +0.21% | 0.01553 | 0.25728 | 94.912% | 2.458% |

Turbo4/Q8 has substantially better perplexity than both Q4-K controls, but its
mean and tail KL divergence are worse. This mixed result does not satisfy the
quality gate. It motivates a larger corpus and a per-layer Q8 fallback study;
it does not justify enabling Turbo4 by default or lifting the MTP guard.

The default-off per-layer Q8 fallback then tested the offline worst Turbo4
layer (23) and the three worst layers (23, 35, 27). One fallback layer improved
PPL and both mean and maximum KLD. Three fallback layers reduced mean KLD to
near Q4/Q4 while improving top-token agreement beyond both Q4 controls. The
three-layer pass took 31.15 seconds versus 31.17 seconds for uniform Turbo4 in
this one-chunk run. This is a useful candidate, but the layer ranking must be
regenerated on a larger representative corpus before production use.

`scripts/bench_turboquant_quality.sh` reproduces the K controls plus the later
Q8/Turbo4 and Turbo4/Turbo4 V-cache cases from an externally supplied
`QUALITY_PROMPT_SOURCE`, preserves the input, and writes the large Q8 logit
reference only into a new result directory.

### Target-only MTP and FP16 verify crossover

Target-only Turbo4 K with draft Q8 K/Q4 V is guarded by
`LLAMA_TURBO4_MTP_EXPERIMENTAL=1`. The initial compile-time FP16 crossover of
32 left small MTP verify batches on different Turbo4 vector geometries. With
that configuration, MTP-1/2 matched each other, while MTP-3 first diverged at
token 31. CPU sampling reproduced the MTP-3 hash, excluding backend sampling
as the cause.

The FP16 crossover is now also runtime-selectable through
`GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH`. Setting it to 2 sends multi-column
verify attention through the existing FP16 tile/MMA path while retaining the
native Turbo4 vector path for one-token decode. In 4K, 128-token diagnostics,
MTP-2 and MTP-3 then produced identical output and token hashes. MTP-1 still
used a numerically distinct effective query geometry. MTP-2 was the fastest
of the tested production-relevant widths:

| MTP width | Decode TPS | Acceptance | Mean accepted | Hash relation |
| ---: | ---: | ---: | ---: | --- |
| 1 | 42.361 | 77.46% | 1.77 | distinct |
| 2 | 43.054 | 68.22% | 2.35 | matches MTP-3 |
| 3 | 38.662 | 56.03% | 2.68 | matches MTP-2 |

Three fresh 32K-context, 512-token MTP-2 processes established repeatability:

| Target KV | Effective fixed S | Decode TPS runs | Median | Acceptance | Hashes |
| --- | ---: | --- | ---: | ---: | --- |
| Turbo4/Q8, Q8 K layers 23/35/27 | 31 | 40.036, 39.982, 40.003 | 40.003 | 68.06% | 3/3 identical |
| Q8/Q8 | 30 | 39.115, 39.105, 38.768 | 39.105 | 67.36% | 3/3 identical |

Turbo4 was 2.30% faster in that phase-matched 512-token median and retained
one additional fixed expert slot. A subsequent single 2,000-token run changed
the ordering because the cache types generated different token streams and
therefore different speculative acceptance:

| Target KV | Decode TPS | Acceptance | Mean accepted | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| Turbo4/Q8 mixed | 38.593 | 74.94% | 2.50 | 5588 MiB |
| Q8/Q8 | 39.541 | 77.05% | 2.54 | 5572 MiB |

This one long run is not a phase-identical kernel comparison. It establishes
that target-only Turbo4 plus MTP-2 is stable at 32K and completes 2,000 tokens,
but not that it universally outperforms Q8. The measurable current advantage
is KV capacity and one extra fixed expert slot; sustained throughput depends
on the resulting token stream and MTP acceptance. MTP remains explicit opt-in.

At 64K the additional KV headroom became more valuable. Three fresh
512-token processes per type produced:

| Target KV | Effective fixed S | Decode TPS runs | Median | Acceptance | Hashes |
| --- | ---: | --- | ---: | ---: | --- |
| Turbo4/Q8 mixed | 27 | 39.016, 39.109, 39.077 | 39.077 | 69.39% | 3/3 identical |
| Q8/Q8 | 25 | 38.449, 38.416, 38.381 | 38.416 | 71.26% | 3/3 identical |

Turbo4 retained two additional fixed experts per layer and was 1.72% faster
by median. Both variants then completed a single 2,000-token 64K run:

| Target KV | Effective fixed S | Decode TPS | Acceptance | Mean accepted | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Turbo4/Q8 mixed | 27 | 38.241 | 74.63% | 2.49 | 5626 MiB |
| Q8/Q8 | 25 | 37.425 | 77.70% | 2.55 | 5598 MiB |

Despite lower speculative acceptance, Turbo4 was 2.18% faster in this long
64K pair. This supports the intended system-level use: compressed target K
preserves more fixed MoE residency as context grows. It remains a one-pair
long result; the 512-token medians are the repeatability evidence.

The MTP gate was integration-tested after these runs. Without
`LLAMA_TURBO4_MTP_EXPERIMENTAL=1`, server loading exits cleanly before model
load with status 1. With the flag, draft-mtp plus the neutral `none` registry
entry is accepted and an 8-token MTP-2 smoke test completes. The semantic
check requires draft-mtp to be present and rejects every non-neutral additional
speculator; it does not depend on an exact internal vector representation.

## Experimental Turbo4 V cache

Turbo4 target V is a separate default-off experiment. It requires
`LLAMA_TURBO4_V_EXPERIMENTAL=1`, FlashAttention, and either Q8 or Turbo4 target
K. Draft Turbo4 remains unsupported. With speculative decoding it additionally
requires the existing MTP-only guard. Negative tests returned status 1 both
without the V flag and without the MTP flag.

V is stored in randomized-WHT space. FlashAttention accumulates weighted V in
that linear domain, after which one inverse WHT reconstructs the attention
output. The first generic V dequantizer was correct enough to complete a model
request but reached only 4.810 token/s versus 27.713 for Q8/Q8 in the same
64-token no-MTP diagnostic. It was rejected. A dedicated vector-kernel loop
uses four cooperating lanes, eight adjacent values per lane, and a scaled
16-centroid table reused across the 128-value block. Q8-K/Turbo4-V then reached
36.524 token/s, and Turbo4-K/Turbo4-V reached 36.372 token/s. These short
figures have different token streams and are kernel diagnostics, not sustained
speed claims.

The phase-matched one-chunk quality test used the same 1,024-token portion of
`ROUTER_LOOKAHEAD_ANALYSIS.md` and Q8/Q8 logit reference as the K-only matrix:

| Target KV | PPL | Relative PPL vs Q8/Q8 | Mean KLD | Maximum KLD | Same top token | RMS probability delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8/Q8 | 11.0151 | baseline | 0 | 0 | 100% | 0% |
| Q8/Turbo4 | 11.1170 | +0.92% | 0.01966 | 0.43591 | 94.521% | 3.340% |
| Turbo4/Turbo4 | 11.1890 | +1.58% | 0.02801 | 0.48421 | 91.585% | 3.820% |

The symmetric cache therefore has a measurable quality cost and is not a
replacement for Q8/Q8 when maximum quality is required. The result is still
better characterized than Q4/Q4 by neither this small corpus nor a single PPL
number; the known long-reasoning quality prompt remains mandatory before any
production recommendation.

MTP-2 with runtime FP16 verify crossover 2 completed at 4K, 32K, and 64K.
A 4K, 128-token MTP-3 smoke produced the same output and token hashes as the
MTP-2 smoke, confirming stable multi-token verify geometry, but was slower at
35.374 versus 40.061 token/s because acceptance fell from 62.50% to 50.33%.
At 32K, three fresh 512-token processes produced identical hashes:

| Target KV | Effective fixed S | Decode TPS runs | Median | Acceptance | Mean accepted |
| --- | ---: | --- | ---: | ---: | ---: |
| Turbo4/Turbo4 | 33 | 42.097, 41.877, 41.918 | 41.918 | 75.86% | 2.52 |
| Q8/Q8 prior control | 30 | 39.115, 39.105, 38.768 | 39.105 | 67.36% | 2.34 |

The 512-token median delta was +7.19%, but token streams and MTP acceptance
differ. A 2,000-token Turbo4/Turbo4 run reached 39.634 token/s, 82.61%
acceptance, and mean accepted length 2.65. The existing Q8/Q8 long control was
39.541 token/s, so the sustained difference was only +0.24%. The robust 32K
benefit is three additional fixed expert slots, not a demonstrated throughput
win.

At 64K, where KV capacity displaces more fixed experts, the result was more
favorable:

| Target KV | Effective fixed S | Decode TPS runs | Median | Acceptance | Hashes |
| --- | ---: | --- | ---: | ---: | --- |
| Turbo4/Turbo4 | 30 | 39.985, 39.956, 39.951 | 39.956 | 67.51% | 3/3 identical |
| Q8/Q8 prior control | 25 | 38.449, 38.416, 38.381 | 38.416 | 71.26% | 3/3 identical |

The median delta was +4.01% and Turbo4 retained five additional fixed experts
per layer. A single 2,000-token run reached 38.852 token/s with 75.55%
acceptance and mean accepted length 2.51, versus the prior Q8/Q8 long control
at 37.425 token/s. This is +3.81%, but remains one long pair. Both 32K and 64K
refer to reserved context capacity with a short initial prompt, not a fully
populated long-context quality test.

Q8-K/Turbo4-V was also tested as a quality compromise. It retained only S=31
at 32K and reached a 37.772 token/s median with 65.76% MTP acceptance, below
the Q8/Q8 control. It is not a local throughput winner despite lower PPL
regression than the symmetric cache.

Current decision: retain Turbo4/Turbo4 as an explicit 64K capacity/performance
candidate, never a default. Require broader perplexity, the external reasoning
prompt, active long-context tests, and native SM61 validation. At 32K prefer
Q8/Q8 when quality matters because sustained performance is effectively tied.

### FFT chat quality control

The external `replay.txt` prompt requesting a fully explained, tested FFT was
run through the Jinja chat endpoint with MTP-2, temperature zero, seed 42, and
a 2,000-token cap. Turbo4/Turbo4 remained coherent but used the entire limit
for planning/reasoning and emitted no final answer. An identical fresh Q8/Q8
control did the same, including repeated refinement of its implementation
plan. The loop is therefore not attributable to Turbo4 on this control.

| Target KV | Finish | Final content | Decode TPS | Draft accepted / drafted |
| --- | --- | --- | ---: | ---: |
| Turbo4/Turbo4 | length at 2,000 | none | 33.525 | 1200 / 1596 |
| Q8/Q8 | length at 2,000 | none | 34.039 | 1224 / 1549 |

This prompt is useful as a regression control but cannot clear the quality
gate because the Q8 baseline itself exhibits the undesirable behavior. A
prompt on which Q8 produces a bounded correct final answer is still required
to compare reasoning length and self-correction behavior.

## Replay-only TriAttention logit gate

The diagnostic capture tool now supports a prompt-token limit, immutable raw
final-logit output, and a guarded one-layer attention-output delta replay. The
simulator exports pruned-minus-full F32 attention deltas from a captured
Q/K/V window. The replay adds the delta after actual Q8/Q8 FlashAttention and
before Qwen attention gating and output projection. It never deletes KV data
and is unavailable in the server.

The no-prune control used one zero-delta record for layer 39. Its 248,320 F32
logits were byte-identical to the capture baseline. A deliberately mismatched
128/128 replay was rejected before prompt processing. Three 2,304-token
prefixes then used History-128, prune point 2,176, budget 2,048, and one
layer-39 delta record:

| Prefix | Relative logit L2 | Maximum absolute error | Softmax KL | Top token | Top-10 |
| --- | ---: | ---: | ---: | --- | ---: |
| `tests/test-sampling.cpp` | 0.630% | 0.0756 | 0.0001310 | same | 10/10 |
| `TURBOQUANT_ANALYSIS.md` | 0.618% | 0.0760 | 0.0000004 | same | 10/10 |
| `HYBRID_ANALYSIS.md` | 0.870% | 0.2286 | 0.0003961 | same | 10/10 |

The sampling prefix also retained its final Top-10 at budget 1,792, with
0.635% relative logit L2 and 0.000126 KL. This is not a quality or throughput
win: only layer 39 was perturbed and one-layer eviction returns little KV
capacity.

### 128-position and live-mask gate

The capture tool can now request selected prompt logits through `batch.logits`
without changing the final 256-token batch. Each 128-row file contains
31,784,960 F32 logits (121.25 MiB) plus an immutable metadata sidecar. A
zero-delta replay produced the same SHA-256
`35b239c3d756df387ea7ef469eb9fcd979bfd98232dd596be3120cfe12cf7364`
as the baseline.

| Prefix | Aggregate L2 | Mean / P95 / max row L2 | Top-1 | Mean / min Top-10 | Max row KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tests/test-sampling.cpp` | 0.835% | 0.815 / 1.291 / 2.422% | 128/128 | 9.922 / 9 | 0.000461 |
| `TURBOQUANT_ANALYSIS.md` | 0.648% | 0.647 / 0.949 / 2.748% | 125/128 | 9.969 / 9 | 0.001711 |
| `HYBRID_ANALYSIS.md` | 1.497% | 1.062 / 2.136 / 10.059% | 128/128 | 9.906 / 9 | 0.004446 |

The three TurboQuant-prefix Top-1 changes were close alternatives and all
remained inside the unchanged Top-10 set. This means the final-position-only
gate was too optimistic, even though the conservative policy remains
numerically mild on most rows.

The first live-mask spike used the same sampling keep set. It modifies a
duplicate of only layer 39's F16 causal mask and leaves every K/V row present.
It applied one graph record, retained 128/128 Top-1 decisions, and measured
0.819% aggregate L2, 0.819% mean row L2, 1.196% P95 row L2, and 2.104% maximum
row L2 versus baseline. Its difference from the F32 offline-delta replay was
0.769% L2. A fresh feature-off run was byte-identical to the pre-spike
baseline. A 2,176/2,176 all-keep live mask was independently byte-identical as
well, proving that the duplicated mask and callback do not perturb results by
themselves. No server, MTP, cache compaction, or production path was enabled.

An optional continuation guard then reapplied the fixed 2,048/2,176 mask to
64 one-token decode graphs. All 64 greedy tokens matched the baseline and both
runs produced token hash `fd7a95ebca41dd36`; the callback reported exactly 64
continuation records. Baseline/live decode was 26.391/26.114 token/s in these
single runs. The approximately 1% diagnostic overhead is not a kernel or
production benchmark because every layer-39 mask is synchronously read and
written by the host callback.

Decision: keep the live mask as a diagnostic. Do not implement physical KV
eviction yet. The remaining one-layer memory return is too small, and the next
quality gate needs autoregressive continuation plus an active long-context
workload before the added request-local cache machinery is justified.

## MTP-2 Turbo4 kernel follow-up

Date: 2026-08-08. Hardware was the GTX 1660 Ti with a native SM75 Release
build. The target cache was Turbo4/Turbo4, the draft cache Q4_0/Q4_0, context
was reserved at 32K, fixed placement used S=33, MTP width was two, and the
runtime FP16 verify crossover was two. All comparisons used fresh server
processes, temperature zero, the same general profile with SHA-256
`cbb5aef750235862f04be083378975d6f8d9c1ec9eb2a6a7883ffa609445e194`,
376/376 maximum batches, eight target and draft CPU workers, backend sampling,
shared hot IDs, fused multi-token MoE Gate+Up, Q8 three-column rows=4, and Q6
rows=2/4.

### Profile-guided stop decisions

A fresh Nsight Systems trace showed that the Turbo4-to-F16 converter accounted
for only 0.3 percent of GPU kernel time and Turbo4 WHT for about 0.2 percent.
Two default-off converter schedules were correct but neutral/slower in the
256-token screen: the generic converter reached 42.606 token/s, one warp per
128-value block reached 42.455, and the two-warp variant reached 42.553. The
two-warp variant also reduced the 2,747-token prompt rate from 65.923 to 65.640
token/s. A default-off shuffle-based WHT reached 42.491 token/s. Neither option
is enabled by a winner configuration.

The dominant fused Q4_K Gate+Up and plain Q5_K Down kernels were then given
runtime-selectable one-, two-, and four-row geometries. Two rows remains the
default and local winner:

| Kernel geometry | 256-token decode | Difference from two rows | Correctness |
| --- | ---: | ---: | --- |
| fused Q4_K, rows=2 | 42.611 | baseline | reference hashes |
| fused Q4_K, rows=1 | 42.409 | -0.47% | identical |
| fused Q4_K, rows=4 | 42.544 | -0.16% | identical |
| plain Q5_K, rows=1 | 41.895 | -1.68% | identical |
| plain Q5_K, rows=4 | 42.614 | +0.01% | identical |

The overrides remain default-off so SM61 and future GPUs can tune them without
changing the established geometry.

The native Turbo4/Turbo4 vector attention path (`F16_PREFILL_MIN_BATCH=32`)
reached 42.025 token/s versus 42.611 for the FP16 verify bridge. Resource
inspection found a 336-byte local stack in the D=128, two-column native kernel.
Replacing the per-thread scaled-centroid table reduced the stack to 272 bytes,
but changed the numerical path, lowered MTP acceptance, and reached only 41.470
token/s. That source change was removed. The crossover of two remains the
correct local MTP setting.

### Combined MTP data-movement winner

The reproducible TurboQuant source tree now includes the previously measured,
default-off pinned H2D scheduler copy and flat non-contiguous dim-0 CONCAT
kernel. Enabling both preserves the FP16 Turbo4 verify bridge and avoids two
host/kernel overheads around the MTP target graph:

```text
GGML_CUDA_ASYNC_HOST_COPY=1
GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=256
GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1
```

Three fresh 512-token processes per side produced:

| Variant | Three results | Median | Difference |
| --- | --- | ---: | ---: |
| synchronous H2D, generic CONCAT | 42.997, 42.913, 42.911 | 42.913 | baseline |
| async pinned H2D, flat CONCAT | 44.395, 44.375, 44.379 | 44.379 | +3.42% |

The required sustained comparison then generated 2,000 tokens in every fresh
process:

| Variant | Three results | Median | Difference |
| --- | --- | ---: | ---: |
| synchronous H2D, generic CONCAT | 43.082, 43.144, 42.746 | 43.082 | baseline |
| async pinned H2D, flat CONCAT | 44.489, 44.432, 44.581 | 44.489 | +3.27% |

All six long runs produced output hash
`d643b475dd4bf99555565ce19fd61ff03ee3a90431b4dd2c5964fdd7aafbbccb`
and token hash
`564a14faa79383003b003716fbdb5cc8d6b72414a0030501e9480f8140af71c0`.
MTP acceptance was 0.755025 and mean accepted length was 2.51 throughout.
Peak VRAM remained 5,614 MiB. No CUDA error, invariant failure, or allocation
growth occurred.

This is the current 32K Turbo4/Turbo4 MTP-2 kernel winner on SM75. The same
comparison at 64K safely clamped the requested S=33 to an effective S=30 in
every process. The repeatability screen produced:

| 64K variant | Three 512-token results | Median | Difference |
| --- | --- | ---: | ---: |
| synchronous H2D, generic CONCAT | 44.773, 44.627, 44.569 | 44.627 | baseline |
| async pinned H2D, flat CONCAT | 45.456, 46.018, 46.075 | 46.018 | +3.12% |

The full 64K sustained gate then produced:

| 64K variant | Three 2,000-token results | Median | Difference |
| --- | --- | ---: | ---: |
| synchronous H2D, generic CONCAT | 42.566, 42.307, 42.114 | 42.307 | baseline |
| async pinned H2D, flat CONCAT | 43.640, 43.411, 43.697 | 43.640 | +3.15% |

All six 64K long runs produced output hash
`76775cc71c34134142a8d91e48e984325b3ea45c8aedbb2990d412b9977628e0`
and token hash
`f542444bb5b4732dcddeae61065b22aeb489cc918bfa482bb7583be2e98bdd56`.
MTP acceptance was 0.762476, mean accepted length was 2.52, peak VRAM was
5,606 MiB, and no allocation growth or CUDA failure occurred.

The H2D and CONCAT features remain individually default-off in the library
because their benefit is graph- and architecture-dependent. They are measured
winners for both 32K and 64K Turbo4/Turbo4 MTP-2 on SM75. SM61 needs its own
phase-matched screen. The remaining local quality/performance gate is an
actively populated long prompt rather than only reserved 64K capacity.

## Turbo4 in the MTP draft KV cache

Date: 2026-08-08. The draft parser and MTP context now accept Turbo4 K/V only
with `LLAMA_TURBO4_DRAFT_EXPERIMENTAL=1`, Flash Attention, KQV offload, and an
MTP-only speculative chain. MTP-1, MTP-2, and MTP-3 deterministic smoke tests
completed correctly on SM75 and released all VRAM after shutdown.

The phase-matched 32K, S=33, MTP-2, 376/376, 3x512 comparison was:

| Draft KV | Three results | Median | Acceptance | Mean accepted | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| Q4_0/Q4_0 | 44.533, 44.463, 44.507 | 44.507 | 62.20% | 2.24 | 5614 MiB |
| Turbo4/Turbo4 | 40.555, 40.442, 40.476 | 40.476 | 52.72% | 2.05 | 5608 MiB |

All three runs per side had identical output and token hashes, including
across the two draft formats. Draft Turbo4 therefore preserves routing-exact
target results in this test, but loses 9.06% throughput because its lower
draft acceptance outweighs the 6 MiB memory saving. It remains available
behind its guard, but is not enabled by `start.sh`; Q4_0/Q4_0 is the measured
SM75 performance winner and Q8_0/Q8_0 is the active quality-oriented setting.
