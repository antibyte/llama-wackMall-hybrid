# Hybrid experiments

This log distinguishes completed measurements from targets. No number below is
an estimate. The original repositories and their existing benchmark files were
read only.

## Reference evidence already present locally

The pre-existing files under the original worktree's `batch-benchmark/` contain
three 2,048-token MTP-2 runs for batch/ubatch 32/32:

| Run | Decode tok/s | Draft acceptance | Mean accepted length |
|---:|---:|---:|---:|
| 1 | 39.42 | 0.68403 | 2.37 |
| 2 | 39.00 | 0.68403 | 2.37 |
| 3 | 39.38 | 0.68403 | 2.37 |

The median is 39.38 tok/s. Those old runs used draft V `q8_0`, while the
requested hybrid protocol uses draft V `q4_0`; they are historical context, not
an exact control for the new measurements.

## Build and format validation

- CUDA release build completed for `CMAKE_CUDA_ARCHITECTURES=75` with tests on
  and without `GGML_CUDA_FORCE_MMQ`.
- `llama-server` and `llama-cli` linked successfully.
- The strict profile converter has 16 passing tests.
- Converter, profile-bank, and placement-optimizer Python suites have 31
  passing tests in total.
- The C++ warm-cache state machine test passes in the release build.
- The source Spark table has 40 layers, 256 experts/layer, top-8 routing,
  10,240 cells, 829,760 selections, 8,393 nonzero cells, and exactly 20,744
  selections per layer.
- Source and target GGUF metadata agree on architecture, base layer count,
  expert count, active expert count, file type, and quantization version. The
  target additionally declares one MTP layer.

## Runtime correctness smoke test

A real no-MTP model load with fixed `S=16`, synchronous `W=1`, and the raw
Spark seed completed without OOM or cache-invariant failure. It initialized 40
MoE layers and 120 tiered tensors. This was a two-token interactive smoke test,
not a throughput benchmark.

The JSON stats file parsed successfully and contained all 40 layers. It
recorded 200 synchronous admissions, 160 evictions, 382,197,760 copied bytes,
and 101.296 ms inside the tensor-copy calls.

## Synchronous warm-cache control

These two short runs used the same runner, prompt, seed, model, context, KV
types, `S=28`, no MTP, one four-token warm-up process, and a 16-token measured
process. A single repetition is enough to identify gross thrashing, but is not
a reportable sustained benchmark.

| W | Decode tok/s | Cold share | H2D copies | H2D bytes | H2D time |
|---:|---:|---:|---:|---:|---:|
| 0 | 35.919 | 53.65% | 0 | 0 | 0 ms |
| 2 sync | 16.652 | 48.70% | 1,749 | 3,335,053,312 | 856.046 ms |

The W=0 and W=2 output hashes and token hashes match exactly. Correctness held,
but throughput fell by 53.6% for only a 4.95 percentage-point cold-share
reduction. The synchronous cache is therefore retained only as a default-off
correctness MVP. It is rejected as a performance candidate under the requested
greater-than-3% regression stop rule.

The cause is graph-granular cache churn: W=1..4 is much smaller than top-8
routing, and copying one quantized expert across all three matrices costs about
1.9 MiB. LuceBox's published cache experiments used 16-48 slots/layer on a
24-GiB GPU; that operating point cannot be transferred directly to this card.

## Raw Spark profile: exact A/B MTP-2 comparison

Both valid runs used a fresh server for warm-up and another fresh server for
the 2,000-token measurement. Parameters otherwise match: batch/ubatch 32/32,
context 32,768, target KV q4_0/q4_0, draft K/V q8_0/q4_0, temperature zero,
seed 42, and MTP-2.

| Config | Seed | Decode tok/s | Cold share | Repins | Acceptance | Mean length |
|---|---|---:|---:|---:|---:|---:|
| A | none | 37.174 | 38.94% | 1,744 | 0.6680 | 2.34 |
| B | raw Spark counts | 35.721 | 60.66% | 297 | 0.7902 | 2.58 |

The raw imported profile regressed decode by 3.91% and substantially increased
the cold share for this prompt, so it fails the stop rule. The profile's large
lifetime counts also seed wackMall's long-term adaptation scores, explaining
the much lower repin count. LuceBox instead turns the table into a global,
layer-variable byte-budget placement; wackMall currently has uniform S per
layer. The CSV format alone cannot express that placement.

The A/B hashes differ. This is not accepted as a token-identity proof: changing
CPU/GPU expert placement changes floating-point paths, MTP acceptance, and in
this case the generated continuation. A quality/perplexity check was not run.

Only one valid repetition per A/B configuration was run. No median or success
claim is made.

## Placement-only seed experiment

The converter now has an explicit `--placement-slots S` mode. It preserves the
Spark top-S ordering but emits bounded scores S..1 and zero for other experts,
so raw lifetime counts cannot pin the online adapter for an entire request. The
original raw conversion remains the default; normalization is never silent.

With top-33 placement, the measured run initially sustained about 40.4 tok/s,
but the server segfaulted after 1,592 decoded tokens. The kernel located the
fault in `libggml-base.so`. The mapping-relative address used in the first
analysis was wrong: the kernel's file offset is `0x78464`, which resolves to
`dequantize_row_q4_0`, not `quantize_row_q3_K_impl`. The faulting instruction
stores dequantized floats through the destination pointer. One plausible
caller is the CPU flash-attention path for a q4_0 V cache, but no caller stack
was captured, so that attribution is not proven. There was no NVIDIA Xid and
the driver responded normally afterward. The failed run remains invalid.

A separate `RelWithDebInfo` CUDA build with AddressSanitizer reproduced the
same model, profile, MTP-2 mode, and request through 1,700 output tokens without
an ASan report. CUDA initialization under ASan required
`ASAN_OPTIONS=protect_shadow_gap=0`; without that option CUDA reported an
initialization OOM and the run was rejected before inference. The valid ASan
run reached 15.124 tok/s, 40.31% cold selections, 922 repins, 0.7736 draft
acceptance, and mean accepted length 2.55. Its throughput is diagnostic only
because sanitizer overhead changes timing substantially.

A subsequent release run used a preload-only SIGSEGV backtrace helper and
completed all 2,000 requested tokens without a fault. It measured 39.885 tok/s,
41.00% cold selections, 969 repins, 0.7770 draft acceptance, and mean accepted
length 2.55. This is one successful repetition, not a median. Together the two
non-reproductions show that the original fault is not deterministic under the
tested conditions; they do not prove it fixed. The helper is
`tools/debug_sigsegv_backtrace.cpp` and is not linked into normal binaries.
GDB 15.1 was installed under `~/.local/opt` for the next reproduction because
the system package was absent and passwordless sudo was unavailable.

Three additional release repetitions used the same top-33 placement profile,
MTP-2, 2,000 output tokens, and armed signal helper:

| Run | Decode tok/s | Cold share | Repins | Acceptance | Mean length |
|---:|---:|---:|---:|---:|---:|
| 1 | 39.619 | 41.00% | 969 | 0.7770 | 2.55 |
| 2 | 39.984 | 41.00% | 969 | 0.7770 | 2.55 |
| 3 | 39.906 | 41.00% | 969 | 0.7770 | 2.55 |

The median is 39.906 tok/s. Output and token hashes are identical across all
three runs. This makes the placement profile a stable measured candidate over
these repetitions, but the earlier nondeterministic q4_0 dequantization fault
is not proven fixed merely because it did not reproduce.

## Admission gating, no MTP

Immediate LRU copied too many 1.9-MiB experts to be useful on this GPU. Three
bounded policies were therefore compared with the same placement profile,
S=28, W=2, no MTP, and 64 output tokens:

| Policy | Decode tok/s | Cold share | Copies | H2D bytes | H2D time | W0 hash match |
|---|---:|---:|---:|---:|---:|---|
| W=0 control | 34.671 | 49.354% | 0 | 0 | 0 ms | reference |
| immediate | 16.565 | 43.961% | 4,536 | 8.646 GB | 2,241.571 ms | no |
| second-hit, window 8 | 21.266 | 44.768% | 2,713 | 5.166 GB | 1,334.014 ms | yes |
| frequency, half-life 8 | 30.386 | 44.592% | 926 | 1.766 GB | 460.232 ms | yes |

Frequency admission is 83.4% faster than immediate and reduces copies by
79.6%, but remains 12.4% below W=0 synchronously. The synchronous direction is
therefore stopped as a performance candidate. Second-hit and frequency keep
the exact deterministic W0 output/token hashes in these controls.

## Asynchronous prefetch, no MTP

The async implementation uses one dedicated CUDA backend stream, two global
in-flight jobs, pinned host staging, one completion event per job, and
publish-after-completion semantics. A pending expert stays sentinel-mapped and
uses CPU cold compute. The existing post-graph scheduler barrier remains.

The directly comparable 64-token placement-profile run measured:

| Config | Decode tok/s | Cold share | Warm hits | Copies | H2D bytes | H2D wait |
|---|---:|---:|---:|---:|---:|---:|
| W=0 | 34.671 | 49.354% | 0 | 0 | 0 | 0 ms |
| W=2 frequency sync | 30.386 | 44.592% | 1,618 | 926 | 1.766 GB | 460.232 ms |
| W=2 frequency async | 34.541 | 48.149% | 409 | 279 | 530.948 MB | 110.191 ms |

Async is 13.7% faster than the synchronous frequency variant and 0.38% below
W=0 in this single short run. Its output and token hashes match both controls
exactly. A prior async run used the raw-count profile by mistake and is retained
under `benchmark-results/async-f-w2-frequency8-nomtp-64`; its hash is not a
valid placement-profile comparison.

This is a correctness/performance smoke result, not a three-run median or a
2,000-token success claim.

The subsequent sustained control used three 2,000-token W=0 runs:

| Run | Decode tok/s | Cold share | Repins | Hashes |
|---:|---:|---:|---:|---|
| 1 | 35.767 | 44.102% | 881 | identical |
| 2 | 35.836 | 44.102% | 881 | identical |
| 3 | 35.632 | 44.102% | 881 | identical |

Median sustained decode is 35.767 tok/s. All output, token, and expert-count
results are identical across repetitions.

The first 2,000-token async W=2 run reached 34.624 tok/s, 44.902% cold share,
21,575 warm hits, 4,888 copies, 9.335 GB H2D, and 1,548.861 ms event-wait time.
This is 3.20% below the W=0 median and therefore crosses the requested stop
threshold. Its token stream first diverges from W=0 at output index 86. The
already-started second repetition was interrupted through the runner cleanup
trap; no partial run is included in `runs.csv`. No Xid, OOM, invariant failure,
or stuck server occurred.

Thus async prefetch is retained as a default-off prototype with passing short
state/hash controls, but it is not an accepted sustained performance candidate.
The 64-token equality did not extrapolate to a long numerically identical
continuation, and the cache did not reduce cold share for the changed long
continuation.

## Warmcache plus MTP-2 stop condition

MTP-2 was tested over 64 tokens with the same placement profile and S=28. The
experimental opt-in was used only to characterize the failure:

| Config | Effective W | Decode tok/s | Acceptance | Mean length | Token hash result |
|---|---:|---:|---:|---:|---|
| W=0 control | 0 | 40.636 | 0.7917 | 2.58 | reference |
| W=2 async experimental | 2 | 40.512 | 0.7917 | 2.58 | mismatch |
| W=2 sync experimental | 2 | 38.364 | 0.7115 | 2.42 | mismatch |
| W=2 requested, default guard | 0 | 40.477 | 0.7917 | 2.58 | exact reference |

The synchronous mismatch proves this is not solely an async stream/event race.
With token capture enabled, the synchronous warm run first diverges at output
token index 46. Different CPU/GPU expert arithmetic inside target MTP verify
batches is the leading explanation, but no logits-level proof has been made.

`configure_mtp()` now receives the parsed draft width before target-context
creation. Unless `LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL=1` is explicitly set, it
disables warm allocation while retaining fixed hot slots. The guard run reports
`mtp_n=2`, effective W=0, zero warm copies/hits, identical hashes, identical
acceptance, and mean accepted length. MTP-1 and MTP-3 warm tests are stopped
until this numerical equivalence issue is understood.

## Post-restart audit

After the Codex session disappeared, the hybrid worktree and benchmark files
were intact and `git diff --check` was clean. The available journal contained
no NVIDIA Xid, kernel panic, or OOM evidence. The only Apport SIGSEGV report was
for `/usr/bin/bash` and its command line proved it came from the intentional
signal-handler probe, where the test signal raced with a shell fork. The async
source edits had not been built or executed before the restart and therefore
could not have caused it. Current GPU health checks and all later runs were
normal.

## MTP multi-request failure found by the first runner

Before warm-up and measurement were separated, an MTP-2 server successfully
completed the 64-token warm-up but the second request failed in
`cublasSgemm_v2` with `unsupported value or parameter`. W was zero. The runner
now treats a stream without a final stop event as an error and uses a fresh
server process for the measured request. This avoids contaminating a benchmark,
but it does not prove the underlying request-reuse bug fixed.

## Not attempted after stop conditions

- Three-repetition, 2,000-token async medians.
- W=1 and W=4 long async runs.
- MTP-1 and MTP-3 warm-cache runs; the automatic MTP guard is active.
- Deferred/double-buffered adaptation or deeper CPU/GPU overlap changes.
- Perplexity/logits comparison for the MTP warm-path divergence.

The next useful warm-cache work is logits-level comparison of fixed CPU versus warm GPU
experts, first in a no-MTP single-token graph near divergence index 86 and then
in a multi-token verify graph. Only after deterministic controls pass should
the MTP guard be relaxed. For performance, copy admission needs a reuse/benefit
model stricter than the current frequency threshold before another sustained
async series. The later static no-sync upper-bound experiment and CPU/profile
tuning results are recorded below.

## Static no-sync upper bound

`LLAMA_EXPERT_STATIC_NO_SYNC=1` was tested with adaptation, warm slots, and all
expert count/stat outputs disabled. The fused cold op received no count tensor.
SB retained the otherwise useless global tier barrier; SC skipped only that
barrier. Both used the same raw Luce warmstart, S=33, W=0, MTP-2, eight default
auto-selected worker behavior, and three 2,000-token repetitions.

| Config | Adapt | Tier sync | Median tok/s | Acceptance | Mean length | Hashes |
|---|---:|---:|---:|---:|---:|---|
| SA | 1 | yes | 35.776 | 0.7902 | 2.58 | identical within SA |
| SB | 0 | yes | 33.642 | 0.7572 | 2.51 | identical within SB |
| SC | 0 | no | 33.848 | 0.7572 | 2.51 | identical within SC and equal to SB |

SC is only 0.612 percent faster than SB. This rejects the scheduler barrier as
the main steady-state bottleneck for a fully static tier on this GTX 1660 Ti.
The test initially appeared slower than the older 39.906 reference because it
used `luce-warmstart.csv` with 69.5 percent seed coverage. Re-running the old
`luce-warmstart-placement-s33.csv` reproduced 39.965 tok/s and the exact old
hash, acceptance, and mean length.

Artifacts: `benchmark-results/static-nosync-abc-20260802T004337Z` and
`benchmark-results/reproduce-placement-s33-balanced-20260802T100536Z`.

## CPU, power, and thread tuning

The existing cmoe auto policy selected 13 threads on the 8-core/16-thread
Ryzen 7 4800H, while long-run process utilization averaged about 4.3 logical
CPUs. The machine was on AC in the System76 `balanced` profile. Switching to
`performance` enabled observed boost clocks above 4 GHz. A deterministic
512-token sweep with the top-33 placement profile produced:

| Threads target/draft | Decode tok/s |
|---:|---:|
| 6/6 | 42.554 |
| 7/7 | 42.713 |
| 8/8 | 42.715 |
| 9/9 | 39.451 |
| 10/10 | 41.129 |
| 12/12 | 41.259 |
| 13/13 | 41.457 |
| 16/16 | 36.702 |

Every output/token hash in the sweep was identical. Three 2,000-token 8/8
runs measured 43.346, 43.137, and 43.133 tok/s, for a 43.137 median. They also
matched the old 39.906 candidate's exact hashes, 41.00 percent cold share, 969
repins, 0.7770 acceptance, and 2.55 mean accepted length. The combined thread
and power configuration improves the old median by 8.10 percent.

A split 8-target/4-draft candidate measured 43.283, 42.677, and 43.130 tok/s;
its 43.130 median is not better. Strict affinity mask `5555`, which selects one
logical CPU from every physical core, fell to 44.115 tok/s in the later learned
profile screen. Poll levels 0 and 100 measured 44.884 and 44.811 versus the
default 50 behavior and were not material. Observed screening temperatures
remained below about 64 C CPU and 49 C GPU.

Artifacts: `benchmark-results/thread-sweep-performance-20260802T100706Z`,
`benchmark-results/thread-refine-performance-20260802T101333Z`,
`benchmark-results/draft-thread-sweep-performance-20260802T101622Z`, and
`benchmark-results/placement-s33-mtp2-t8-performance-repeat3-20260802T100941Z`.

## Learned static placement result

One correct 2,000-token 8/8 adaptive run exported `LLAMA_EXPERT_USAGE` to:

```
benchmark-results/learn-profile-t8-performance-20260802T102338Z/B-run1.usage.csv
```

Using that learned profile reduced a 512-token adaptive run from 608 to 12
repins and raised its throughput to 44.577 tok/s. With adaptation and stats
disabled, the static no-sync path reached 44.874 tok/s in the short screen.
Three 2,000-token static repetitions then measured:

| Run | Decode tok/s | Acceptance | Mean length | Output/token hashes |
|---:|---:|---:|---:|---|
| 1 | 44.994 | 0.8064 | 2.61 | identical |
| 2 | 45.253 | 0.8064 | 2.61 | identical |
| 3 | 45.364 | 0.8064 | 2.61 | identical |

The median is 45.253 tok/s, 13.40 percent above the original stable 39.906
median. Configuration: System76 `performance`, target/draft threads 8/8,
S=33, W=0, MTP-2, adaptation/stat outputs off, and static no-sync on. VRAM
peaked at 5,538 MiB. No Xid, OOM, hang, or invariant failure occurred.

The saved response contains 7,209 streamed content bytes and stops at the
requested token limit. A later detokenization of the captured token-ID subset
produced coherent German architecture prose and Go code at both inspected
ends. That subset contains 1,983 IDs for 2,000 predicted tokens and reconstructs
17 replacement characters, so it is a plausibility check rather than an exact
reconstruction of the hashed SSE content.

Switching `system76-power` from `balanced` to `performance` emitted one NVIDIA
open-kernel-module internal `nvAssertFailedNoLog` at the exact profile-change
timestamp. It emitted no Xid, reset, CUDA error, or benchmark failure. All
subsequent runs completed and the idle GPU remained healthy. This driver/power
profile interaction is retained as a deployment risk even though it did not
invalidate the measurements.

This profile was learned from the same deterministic benchmark prompt. The
45.253 result is therefore a valid reproducibility/performance result for this
workload, but it is not evidence of equal performance on unrelated traffic.
For general service traffic, a profile accumulated across representative
requests with adaptation enabled is the safer choice; the bitidentical 43.137
adaptive candidate is the strongest measured general-purpose configuration.

One additional learning iteration from the new static token path lowered
top-33 coverage from 63.66 to 62.54 percent and screened at 43.952 tok/s, so
profile chasing stopped. The theoretical same-route top-33 headroom was only
1.48 percentage points.

## Fixed-tier MTP sweep

The best learned static profile was also checked without warm slots:

| MTP width | Tokens | Decode tok/s | Acceptance | Mean length | Result |
|---:|---:|---:|---:|---:|---|
| 1 | 2,000 | 43.267 | 0.8752 | 1.88 | slower |
| 2 | 2,000 x3 | 45.253 median | 0.8064 | 2.61 | winner |
| 3 | 2,000 | 40.406 | 0.6233 | 2.87 | stop: -10.7 percent |

MTP-3 was first screened at 41.541 tok/s over 512 tokens, then explicitly
confirmed with the full 2,000-token run. All modes completed correctly, but
their different accepted draft paths produce different deterministic hashes.

## Prompt-balanced profiles and layer-variable placement

Session-local usage export was validated over a 64-token MTP-2 run: the sparse
CSV contained 36,800 selections, exactly equal to JSON `selected_total`, and no
seed count leaked into it. A twelve-prompt corpus collector then produced
independent profiles for German/English reasoning, Go/Python code, Linux/Yocto,
tool JSON, product planning, quantitative reasoning, security, short answers,
translation, and structured extraction.

Eight 128-token training prompts were normalized independently and held out
four categories. On 512-token holdout traces their static top-33 coverage was:

| Profile | Mean top-33 coverage |
|---|---:|
| Luce placement warmstart | 38.746% |
| benchmark-prompt specialist | 34.017% |
| prompt-balanced train-128 | 44.477% |
| prompt-balanced train-512 | 43.378% |
| mixed train-128/train-512 | 44.256% |

Single 512-token static MTP-2 screens for the train-128 profile versus the
Luce placement measured +1.15% on security, -0.26% on concise QA, +2.10% on
translation, and +2.11% on structured extraction. These are screens, not
three-run medians, but they support a general profile bank rather than using
the benchmark-specific 45-TPS profile for unrelated traffic.

The strict layer-variable runtime was exercised with 1,321 fixed slots in a
16..43 range under the fixed-byte budget of uniform S=33. All 40 layer records,
36,800 selections, LUTs, and sentinels validated. The first 64-token MTP-2
control was hash-identical to uniform S=33. On a 512-token structured holdout,
variable placement reduced cold share from 63.130% to 60.088% but slowed from
41.031 to 39.647 tok/s and changed the token path. Without MTP it screened
0.80% faster over 256 tokens, also on a changed path. This general placement is
retained as experimental, not selected for production.

An offline capacity sweep demonstrates that the implementation is not tied to
6 GiB. Including one sentinel per layer, uniform-reference fixed budgets and
prompt-balanced holdout coverage were:

| Reference S | Tier GiB | Uniform coverage | Variable coverage |
|---:|---:|---:|---:|
| 33 | 2.419 | 44.477% | 44.505% |
| 64 | 4.627 | 62.501% | 62.580% |
| 96 | 6.905 | 75.133% | 75.218% |
| 128 | 9.182 | 83.412% | 83.739% |
| 160 | 11.461 | 89.833% | 90.162% |
| 192 | 13.739 | 94.573% | 94.917% |

These are offline coverage estimates, not throughput claims for untested GPUs.
They show why byte budgets, slot bounds, warm auto cap, and reserve remain
parameters. Larger cards should first spend VRAM on static fixed experts, then
re-evaluate a larger warmcache after measuring remaining churn.

## CPU cold timing and cost-based placement

`LLAMA_EXPERT_TIMING=1` measured 3,273.377 ms of fused CPU cold-node wall time
over a 512-token specialist run, or 6.393 ms per output token. Layer 0 used
159.182 ms; layers 2 and 1 used 143.665 and 140.067 ms. The q6_K down-weight
layers and several early layers had the highest time per cold selection.

A cost-weighted S=33-byte placement screened at 46.159 tok/s over 512 tokens
and lowered cold share from 38.542% to 31.908%. The required sustained probe,
however, reached only 44.445 tok/s over 2,000 tokens, 1.79% below the stable
45.253 median. Further repetitions stopped. The runtime and optimizer remain
useful for larger budgets, but this concrete 6-GiB layout is not a winner.

The fused op already uses scheduler-owned persistent scratch. A bit-identical
singleton row-chunk sweep measured 44.208, 44.872, 44.963, 44.683, and 44.707
tok/s for chunks 16, 32, 64, 128, and 256. The original/default 64 remains best
on the Ryzen 7 4800H. Average process CPU utilization does not imply only four
workers: all eight participate during the cold-node phases, while GPU work,
sampling, and waits lower the whole-request average.

## MTP parameter screens and new best result

MTP-2 `p_min` values 0.05 and 0.10 did not change draft behavior. Values 0.20
and 0.30 changed the path and fell to 43.073 and 42.587 tok/s. Moving draft
sampling from the backend to the host preserved hashes but fell to 43.687
tok/s. Both directions stopped.

Changing only the MTP draft KV from q8_0/q4_0 to q4_0/q4_0 preserved the exact
512-token hashes, acceptance, and mean accepted length while reducing peak VRAM
by about 8 MiB. The serious 3x2,000-token run measured:

| Run | Decode tok/s | Acceptance | Mean length | Peak VRAM |
|---:|---:|---:|---:|---:|
| 1 | 46.538 | 0.806409 | 2.61 | 5,530 MiB |
| 2 | 46.433 | 0.806409 | 2.61 | 5,530 MiB |
| 3 | 46.271 | 0.806409 | 2.61 | 5,530 MiB |

Median sustained decode is **46.433 tok/s**. All three output hashes and token
hashes are identical and equal to the prior q8_0/q4_0 specialist result. This
is 2.61% above the previous 45.253 median and 16.36% above the older 39.906
reference. Artifact:
`benchmark-results/mtp2-draft-kv-q4q4-2000-repeat3-20260802T150000Z`.

MTP-3 with q4_0/q4_0 reached only 41.292 tok/s in its 512-token screen
(acceptance 0.5924, mean length 2.78). The earlier full MTP-3 run was already
40.406 tok/s, so another long run is not justified. MTP-2 remains the winner.

## Cold phase decomposition and activation parallelism

The runtime timing mode was extended to decompose the fused CPU cold operation
into Gate/Up dot products, activation plus intermediate quantization, and Down
dot products. Two deterministic 256-token MTP-2 q4_0/q4_0 screens used the
specialist S=33 placement, W=0, eight target/draft threads, static no-sync, and
otherwise identical settings:

| Activation scheduling | Decode tok/s | Gate/Up ms | Activation ms | Down ms | Whole cold ms |
|---|---:|---:|---:|---:|---:|
| original | 43.097 | 1,072.500 | 40.004 | 715.545 | 1,920.669 |
| block-parallel | 43.192 | 1,070.198 | 32.114 | 716.290 | 1,911.099 |

Output and token hashes were identical. Block parallelism reduced its target
phase by 19.72%, but improved total throughput by only 0.22%, because activation
was about 2.1% of the measured cold-node wall time. The option remains
parameterized for CPUs with more cores and defaults off; it does not justify a
long benchmark on this CPU. Gate/Up and Down quantized dot products account for
the useful CPU optimization target. Artifacts:
`benchmark-results/cpu-act-phase-baseline-256-20260802T160500Z` and
`benchmark-results/cpu-act-phase-parallel-256-20260802T161000Z`.

This also explains the observed whole-process utilization of roughly four
logical CPUs. Eight workers participate in the cold dot-product phases, but
GPU execution, sampling, cross-backend waits, uneven expert work, and the short
activation phase reduce the request-wide average. Pinning one SMT thread per
physical core (`5555`) and using more than eight workers had already reduced
throughput, so maximizing the utilization display is not an optimization goal.

## CUDA/backend timeline

Nsight Systems 2022.4.2 captured one 64-token static MTP-2 q4_0/q4_0 request.
The profiler itself lowered decode to 39.64 tok/s, so absolute throughput from
this run is not comparable to uninstrumented results. Within the approximately
1.63-second decode window, the trace recorded:

- 10,082 `cudaStreamSynchronize` calls on the server thread, totaling 707.2 ms
  of host-call duration;
- about 112.5 ms summed CUDA-kernel duration on the two active streams;
- 1,892 H2D operations totaling 226.3 MiB and 38.4 ms of device-copy duration;
- 2,296 D2H operations totaling 94.1 MiB and 16.1 ms of device-copy duration.

The sums can overlap and include profiler overhead, so they are not additive.
They nevertheless disprove the idea that the previously tested tier-specific
post-ubatch barrier was the only synchronization cost. The generic scheduler
alternates CUDA and CPU splits around every cold layer. CPU backend execution
is synchronous, and cross-backend copies synchronize source or destination
backends, so the intended hot-GPU/cold-CPU graph independence does not yet
guarantee real overlap.

The imported report, SQLite export, and summaries are preserved at
`benchmark-results/nsys-static-mtp2-q4q4-64-20260802T162000Z`. The original
target-side `nsys` command could capture but not import; using the installed
host `QdstrmImporter` recovered the report. This evidence justifies a narrowly
guarded asynchronous CPU-split prototype, but not an unconditional scheduler
rewrite.

## Async overlap, model loading, and final memory screens

The guarded asynchronous CPU split preserved exact deterministic hashes but
did not improve this machine. With MTP-2, 64-token static S=33 q4_0/q4_0
controls measured 46.245 tok/s synchronously and 42.971 tok/s asynchronously
(-7.08%). The async run dispatched 1,120 cold jobs and still waited 583.698 ms
at their real output dependencies. Reducing target/draft workers to seven
reached only 39.685 tok/s over 256 tokens.

The no-MTP control separates the result from speculative scheduling:

| Mode | CPU async | Decode tok/s | Hash relationship |
|---|---:|---:|---|
| no MTP | 0 | 38.629 | identical |
| no MTP | 1 | 35.504 | identical |

The -8.09% no-MTP result confirms that simultaneous GPU work slows the cold
job and/or its scheduling more than the overlap saves on this 8-core system.
The mode remains parameterized for systems with a different CPU/GPU balance,
but defaults off and hit the local stop condition. Artifacts begin with
`benchmark-results/cpu-async-overlap-*` and
`benchmark-results/cpu-async-nomtp-*`.

The model loader's recommendation to avoid mmap for CPU tensor overrides was
also tested. `load_mode=none` auto-clamped the safe tier to S=32. At the same
S=32 and exact hashes, `none` reached 41.980 tok/s versus 42.780 tok/s for
`mmap` (-1.87%), while taking much longer to load. Forcing more slots was not
justified. `mmap` remains the recommendation on this machine; the runner keeps
the loading mode explicit for platforms with different VM behavior.

A Down-weight software-prefetch distance of one preserved hashes but reached
42.898 versus 43.517 tok/s for distance zero over 256 tokens (-1.42%). Larger
distances were not pursued after the negative first screen. The option remains
off by default for CPU-specific experiments.

Finally, lowering the reserve to 400 MiB allowed uniform S=34 with a 5,602 MiB
peak and no OOM/Xid. It reached only 42.416 tok/s versus 43.517 for the nearby
S=33 screen and changed the MTP path (acceptance 0.6486 versus 0.7095). S=33 is
therefore the measured 6-GiB optimum; merely filling VRAM is not sufficient.
Larger GPUs should still screen S=64/96/etc. because the offline coverage gains
are substantial, but each hardware/model pair needs a throughput and MTP gate.

After all feature-gated changes, a default-off 512-token regression control
measured 45.266 tok/s versus the earlier 45.280 tok/s (-0.03%), with exact
matching output/token hashes, acceptance, mean accepted length, and 5,530 MiB
peak VRAM. The conservative production recommendation remains static S=33,
W=0, MTP-2, draft KV q4_0/q4_0, eight target/draft threads, mmap, and
experimental CPU switches disabled. The later row-reuse experiment below is a
slightly faster local option but is not a portable default.

The final current-source 2,000-token validation measured 46.245, 46.419, and
46.231 tok/s, for a **46.245 tok/s median**. All three runs reproduced output
SHA-256 `49852a17cf69c3add0226110990beda292a590c083a23b8444886e7fda766fbe`,
token SHA-256 `1aaa24431d19af491551b7415fda4e30a2d18236e06b6bba698bf9ab11d5b66e`,
acceptance 0.806409, mean length 2.61, and 5,530 MiB peak VRAM. This current
median is 0.41% below the earlier 46.433 median and does not constitute a
regression under the 3% gate. Artifacts:
`benchmark-results/final-default-q4q4-2000-20260802T183000Z` and
`benchmark-results/final-default-q4q4-2000-repeat2-20260802T184000Z`.

## Final full-suite validation

The final `ctest -j8 --output-on-failure` run passed 54 of 59 registered tests.
Most importantly for the last backend change, `test-backend-ops` passed its
full CPU/CUDA matrix after 244.54 seconds. The targeted argument-parser,
warm-cache, placement, and save/load tests also passed.

The five non-passing registrations are outside the modified hybrid paths:

- `test-tokenizers-ggml-vocabs` sees Git-LFS pointer text (`vers`) instead of
  downloaded GGUF vocabulary data.
- `test-quant-type-selection` reports stale Qwen3.5 quantization snapshots
  (9/11 models pass, one is skipped).
- `test-generate-models` and `test-llama-archs` abort in the existing
  MiniMax-M3 `build_ffn` path at `llama-graph.cpp:1716`, before the modified
  MoE builder region.
- `test-recurrent-state-rollback` is not run because it depends on the failed
  generated-model test.

No failure points at the async coordinator, cold op, expert LUT, warm slots,
sentinel handling, placement parser, or MTP state paths changed here.

## MTP cold-weight row reuse

The final CPU-layout experiment reverses the inner row/column traversal only
when multiple tokens select the same cold expert. This keeps a quantized weight
row cache-resident across those token dot products. It is controlled by
`LLAMA_EXPERT_CPU_REUSE_ROWS=0|1` and defaults off.

With phase timing enabled, the 256-token MTP-2 A/B was hash-identical:

| Row reuse | Decode tok/s | Gate/Up ms | Activation ms | Down ms | Whole cold ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 43.242 | 1,061.206 | 39.176 | 717.474 | 1,909.885 |
| 1 | 43.205 | 1,062.443 | 38.983 | 701.934 | 1,895.692 |

Thus Down improved 2.17% and the whole cold node 0.74%, but timing overhead hid
the change in end-to-end throughput. A corresponding MTP-3 timing A/B was also
hash-identical and flat at 36.772 versus 36.767 tok/s; it does not change the
earlier conclusion that MTP-2 is faster on this hardware.

Without timing/statistics, a 512-token production-path screen reached 45.404
versus 44.800 tok/s (+1.35%). The required direct three-run 2,000-token A/B was:

| Mode | Runs (tok/s) | Median |
|---|---|---:|
| row reuse 0 | 46.365, 46.362, 46.361 | 46.362 |
| row reuse 1 | 46.593, 46.508, 46.335 | 46.508 |

The sustained median gain is 0.31%. A separate enabled probe reached 46.614
tok/s. Every 2,000-token run reproduced output SHA-256
`49852a17cf69c3add0226110990beda292a590c083a23b8444886e7fda766fbe`,
token SHA-256 `1aaa24431d19af491551b7415fda4e30a2d18236e06b6bba698bf9ab11d5b66e`,
acceptance 0.806409, mean length 2.61, and 5,530 MiB peak VRAM.

This is the fastest measured local option, but its margin is too small to
enable globally. It remains a portable knob for machines where repeated MTP
cold experts and CPU memory bandwidth are a larger fraction of decode time.
Artifacts begin with `benchmark-results/cpu-row-reuse-*`.

## Physical prefill ubatch sweep

The production override originally forced both maximum logical batch and
maximum physical ubatch to 32. A 2,051-token prompt confirmed 32-token progress
steps and only 19.829 prompt tok/s. Raising only the logical batch to 256 left
throughput unchanged at 19.833 tok/s. Raising the physical ubatch produced the
following single-run screens from the same profile seed, MTP-2, dynamic
request-boundary adaptation, S requested as 33, W=0, q4_0/q4_0 KV, and 256
output tokens:

| Batch / ubatch | Effective S | Prompt tok/s | TTFT ms | Decode tok/s | Peak VRAM MiB |
|---:|---:|---:|---:|---:|---:|
| 32 / 32 | 33 | 19.829 | 103,459 | 35.452 | 5,510 |
| 256 / 32 | 33 | 19.833 | 103,437 | 35.465 | 5,510 |
| 256 / 64 | 33 | 26.680 | 76,898 | 35.776 | 5,518 |
| 256 / 128 | 33 | 34.167 | 60,053 | 36.587 | 5,534 |
| 256 / 256 | 33 | 57.817 | 35,499 | 35.588 | 5,562 |
| 376 / 376 | 33 | 73.131 | 28,071 | 35.713 | 5,590 |
| 384 / 384 | 32 (auto-fit) | 73.532 | 27,918 | 35.432 | 5,518 |
| 512 / 512 | 32 (auto-fit) | 94.377 | 21,756 | 36.514 | 5,548 |
| 512 / 512, reserve 384 | 33 | 94.141 | 21,811 | 36.941 | 5,622 |

The reduced-reserve 512/S33 screen left only 127 MiB of real free VRAM and is
not the production default. With the default 512-MiB expert reserve, startup
probes found that ubatch 376 retains S=33 while 384 crosses the auto-fit
boundary. This threshold is model-, context-, KV-, MTP-, driver-, and
GPU-specific; the runner therefore exposes `CMOE_BATCH` and `CMOE_UBATCH`
rather than embedding a hardware value.

At 8,207 prompt tokens, 376/S33 reached 72.484 prompt tok/s and 113,261 ms TTFT.
The TTFT-oriented 512/S32 mode reached 89.511 prompt tok/s and 91,723 ms TTFT.
For a 2,000-token decode using the current mixed workload profile, 376/S33
measured 38.236, 38.033, and 37.458 tok/s (median 38.033) with identical output
and token hashes across the three runs. 512/S32 measured 37.709 tok/s in its
screen, 1.38% below the first matching 376/S33 run. Thus 376/S33 is the balanced
6-GiB production setting and 512/S32 remains an explicit TTFT-max option.

Different physical prefill batch shapes diverged after token 35 even with
adaptation completely disabled. This is normal floating-point batch-shape
dependence, not a mutable-tier race. A decode control whose prompt fits in one
32-token graph produced identical 2,000-token output/token hashes, identical
MTP acceptance 0.787371 and mean accepted length 2.57 at 32/32 and 376/376;
decode was 45.676 versus 45.778 tok/s. The large maximum ubatch therefore does
not force a large decode graph or regress identical-path decode.

Four consecutive dynamic requests at 376/376 passed with MTP-2. Every request
atomically produced a different cumulative profile hash, and decode improved
from 41.46 to 45.72 tok/s without a CUDA or invariant failure. MTP-3 also
passed four requests and wrote four distinct checkpoints, but its extra verify
memory auto-fits this 6-GiB card to S=32 even at the older small ubatches. A
2,000-token current-profile comparison measured 38.236 tok/s for MTP-2 versus
35.647 tok/s for MTP-3, so MTP-2 remains the production choice.

Artifacts begin with `benchmark-results/prefill-*`,
`benchmark-results/decode-control-*`, and
`benchmark-results/adaptive-multi-request-u376-*`.

## Dynamic CUDA-graph upper bound

Dynamic adaptation normally disables CUDA graph capture/reuse because a graph
used after a fixed-tier repin reproduced a next-request `cublasSgemm` abort.
Before implementing graph-cache invalidation, a safe upper-bound A/B ran only
the first request of every fresh process. No process was allowed to execute a
graph after publishing learned placement. Both sides used the same current
profile seed, 376/376, S=33, W=0, MTP-2 q4_0/q4_0, eight target/draft threads,
row reuse, a sub-32-token control prompt, and three deterministic 2,000-token
runs:

| Dynamic CUDA graphs | Runs (tok/s) | Median | Peak VRAM | Hashes |
|---:|---|---:|---:|---|
| 0 | 37.574, 37.619, 37.438 | 37.574 | 5,580 MiB | identical |
| 1, unsafe diagnostic | 37.760, 37.626, 37.731 | 37.731 | 5,600 MiB | identical |

The measured upper bound is only +0.42% and consumes about 20 MiB more VRAM.
This is below the threshold for implementing and validating targeted graph
invalidation on this GPU. `LLAMA_EXPERT_ADAPT_CUDA_GRAPHS=0` remains mandatory
for the multi-request learning service. Artifacts begin with
`benchmark-results/cuda-graph-upper-bound-{off,on}-u376-*`.

After this loop change, `test-backend-ops` was rerun: all 13,327 supported CUDA
operation cases passed and both registered backends completed successfully.

## Request-boundary dynamic adaptation repair

The OpenWebUI learning service originally completed one adaptive MTP-2 request
and aborted during the next request in `cublasSgemm_v2`. W was zero, no OOM or
Xid was present, and the direct-at-exit usage writer lost the current profile.
A matching static control processed two consecutive 128-token requests at
42.729 and 42.130 tok/s without failure.

The first repair separated graph-level count harvesting from fixed-tier
publication. Scores still consume every completed graph, but fixed slots now
change only before the next server request. A process-wide graph guard covers
target and MTP draft graph construction, compute, synchronization, count
harvesting, and publication. Fixed publication copies all weight slices first,
then publishes all LUTs and masks, then commits host ownership. A tested pure
policy selects one deterministic replacement per layer, and permanent fixed
invariants require a final sentinel, unique owners, LUT/mask agreement, and no
empty fixed slots.

Request boundaries alone did not fix the abort. With 39 valid fixed repins, the
next request still failed. Added failure-only CUDA diagnostics identified a
dense shared-expert gate in layer 33, not a hot expert matmul:

```text
src0=blk.33.ffn_gate_inp_shexp.weight f32 [2048,1,1,1]
src1=attn_post_norm-33 f32 [2048,32,1,1]
dst=shared_expert_gate-33 f32 [1,32,1,1]
m=1 n=32 k=2048 lda=2048 ldb=2048 ldc=1
```

All dimensions, leading dimensions, and pointers were non-null and valid.
Disabling CUDA graph capture/reuse eliminated the failure across repeated repins.
The runtime now disables the CUDA graph path before inference whenever
`LLAMA_EXPERT_ADAPT=1`; the CUDA backend no longer caches the environment
decision before expert-tier initialization. The explicit
`LLAMA_EXPERT_ADAPT_CUDA_GRAPHS=1` escape hatch is diagnostic and unsafe.

The repaired source was exercised in one server process per mode with a
placement-only S=33 seed, W=0, q4_0/q4_0 KV, 8/8 threads, and 128 output tokens
per request:

| Mode | Requests | Repins at later boundaries | Decode tok/s | Result |
|---|---:|---|---|---|
| no MTP | 3 | 40, 39 | 37.81, 37.34, 39.46 | pass |
| MTP-1 | 3 | 39, 38 | 42.87, 43.64, 43.20 | pass |
| MTP-2 | 4 | 39, 39, 38 | 43.52, 44.40, 43.73, 43.71 | pass |
| MTP-3 | 4 | 39, 39, 38 | 38.15, 38.23, 41.34, 41.72 | pass |

Every request produced a final stop event, the server health check passed after
every request, and every request atomically replaced the cumulative usage CSV.
The MTP-2 learned file ended at SHA-256
`6d4e25e94700e8a2dd51b18b3f988d87b65d3e6ebb1a8dde284cd8f811fdc6f8`;
the MTP-3 file ended at
`12223243dd24147fe45e48bbb116b4083fcfa4cde6f590865e40a8f958182d2e`.
These are correctness screens, not 2,000-token medians. Artifacts are under
`benchmark-results/adaptation-multi-request-20260802T161825Z`.

The tracked multi-request runner was then used for two consecutive 2,000-token
MTP-2 requests in one process. Request 1 sustained 38.26 tok/s with acceptance
0.72338 and mean accepted length 2.45. Its atomic checkpoint had SHA-256
`b84503b30b497e816eabd7407a7c1f907064a72c6141b00a01a39b79304fc2f4`.
The next boundary published 40 fixed repins. Request 2 sustained 38.35 tok/s
with acceptance 0.90640 and mean length 2.81; it completed all 2,000 tokens and
published profile SHA-256
`b9abe343cc772ba9f524295fbad126397ac0ff4dc19f8af05bcb4d8e7dcea98f`.
The server remained healthy with no CUDA, invariant, OOM, Xid, or missing-stop
failure. Output hashes differ because the fixed CPU/GPU placement changed at
the request boundary; no claim of bit identity is made. This is a two-request
correctness/stability result, not a three-run throughput median. Artifacts:
`benchmark-results/adaptation-mtp2-long-20260802T164157Z`.

The final CUDA backend regression was run after the graph-policy and
failure-diagnostic changes with the production SM75 build and no model process
occupying VRAM. All 13,327 supported CUDA operation cases passed against the
CPU reference, including `MUL_MAT_ID`, `MUL_MAT_ID_FUSION`, Q4_0/Q4_K Top-8
MoE cases, and Flash Attention. The CUDA backend and the overall backend suite
both reported `OK`.

The LAN/OpenWebUI service was then switched to the repaired binary with one
slot, MTP-2, S=33, W=0, request-boundary adaptation, and one shared persistent
input/output profile under `~/.local/state`. Two consecutive 128-token API
requests completed at 41.92 and 42.31 tok/s, both accepted 80 of 92 draft
tokens, and atomically changed the profile after each request. The PID stayed
constant, the restart counter stayed zero, and health remained `ok`. A
controlled service restart loaded the last checkpoint successfully; a further
128-token request completed, changed the profile again, and left the restarted
process healthy. This validates persistence and continuation across restart in
addition to the earlier repin-heavy regression tests.

### Excluding server warm-up from learned usage

A later no-request restart exposed a smaller persistence correctness issue:
the old server added exactly 1,280 selections to the cumulative profile at
shutdown despite serving no inference request.  The delta was 40 layers x 8
selected experts x 4 internal tokens.  Target and MTP context warm-up each
contributed 640 selections.  Repeated restarts would therefore teach the
placement from synthetic initialization graphs rather than only real usage.

The server now declares request-scoped collection before model/context load.
Completed graphs still clear their count buffers, but selections are discarded
until the first validated server request calls `request_begin()`.  CLI and
direct-library callers retain their previous context-lifetime collection
behavior.  A private-profile integration test produced:

```text
no API inference:
  discarded target warm-up = 640
  discarded MTP warm-up    = 640
  profile SHA before       = 8905adde740f35e2003bca76b80d601c1800bcbfd6c30b7d2d74ba0f6893b6b3
  profile SHA after        = 8905adde740f35e2003bca76b80d601c1800bcbfd6c30b7d2d74ba0f6893b6b3

one real 17+16-token MTP-2 request:
  profile SHA after        = fdd8db98455ebb1c0eb12f1c14743e243de58a7293e16435f24ecd15e5f78da4
  atomic request checkpoint present
  decode                   = 46.17 tok/s
```

The active production profile contains one already-persisted 1,280-selection
warm-up sample from discovering the issue (0.0063% of its then-current total).
It was documented rather than silently rewriting a user data file.  The
corrected production process logs both discarded 640-selection warm-ups and
does not add further restart samples.  Artifacts are under
`benchmark-results/startup-usage-filter-*`.

## Single-slot RAM prompt-cache branch restore

The Qwen3.6 hybrid context reports that KV shifting is unavailable, so
`--cache-reuse` is disabled by the server and cannot accelerate this model.
The separate RAM prompt cache can still save and restore complete target and
MTP draft states.  A deterministic A -> B -> A test used a roughly 3,500-token
synthetic prompt, one live slot, cache-prompt requests, 376/376, S=33, W=0,
q4_0/q4_0 KV, and 16 generated tokens per request.

Neither the ordinary live-slot behavior nor `--cache-idle-slots` helped with
`-np 1`: the selected single slot is already marked processing by the time the
idle-slot save loop runs.  A3 therefore re-evaluated about 3,500 tokens in
48.1 seconds.  Setting `--slot-prompt-similarity 0.70` deliberately routes the
about-0.59-similar branch through the existing LRU/RAM-cache path, preserving A
before B overwrites the live slot:

| Mode | A1 prompt / TTFT | B prompt / TTFT | A3 prompt / TTFT | A1/A3 hashes |
|---|---:|---:|---:|---|
| base | 3,492 / 47.86 s | 3,455 / 47.36 s | 3,509 / 48.20 s | not restored |
| RAM + idle save | 3,492 / 48.05 s | 3,442 / 47.18 s | 3,496 / 48.09 s | not restored |
| RAM + similarity 0.70 | 3,492 / 47.99 s | 3,442 / 47.35 s | 4 / 0.390 s | identical |
| same, dynamic MTP-2 | 3,492 / 48.19 s | 3,442 / 47.53 s | 4 / 0.396 s | identical |
| same, dynamic MTP-3 | 3,492 / 48.23 s | 3,442 / 47.61 s | 4 / 0.390 s | identical |

The dynamic MTP-2 run published 11 fixed repins before restored A3; MTP-3
published nine.  In both runs A1 and A3 had identical token and output hashes,
and identical MTP acceptance/mean length.  No CUDA, invariant, OOM, or server
failure occurred.  The active production profile was never used as an output;
each adaptive test learned into a private result-directory copy.

This is a TTFT/cache result, not a decode-TPS improvement.  It is most useful
when a one-slot OpenWebUI server switches between conversations and later
returns to one of them.  A linear continuation already benefits from its live
slot.  The 2-GiB RAM cache is bounded, and saving/loading adds roughly 0.1--0.2
seconds in this test.  Artifacts begin with
`benchmark-results/prompt-cache-{branch,ram-idle,ram-forced}-*`; the runner is
`scripts/test_prompt_cache.sh`.

## Target KV precision quality screen

After one production `"fft in c"` request spent 9,927 output tokens in a
self-correcting reasoning loop, target KV precision was isolated from draft KV
precision.  The original request had temperature 1, a random seed, no output
limit, and a 16,384-token reasoning budget, so it did not by itself establish a
KV-cache defect.  It nevertheless motivated a deterministic code-quality A/B.

All probes used the same learned profile, MTP-2, draft q4_0/q4_0, 32K context,
376/376, W=0, and no adaptation or usage output.  Startup fit was:

| Target K/V | Effective fixed S | Startup VRAM |
|---|---:|---:|
| q4_0/q4_0 | 33 | 5,578 MiB |
| q8_0/q4_0 | 33 | 5,486 MiB |
| q8_0/q8_0 | 30 | 5,518 MiB |

Thus q8_0/q4_0 permits a fair S=33 comparison on this card; q8_0/q8_0 changes
expert placement and was not used for the first quality decision.

A 1,024-token reasoning budget plus a 2,048-token total limit caused both
variants to hit the length stop and split the forced reasoning-to-content
transition while drafting code.  This exposed a reasoning-budget/total-budget
interaction rather than a clean KV comparison.  Reducing the reasoning budget
to 64 with a 1,024-token total limit produced visible code in both cases:

| Target K/V | Decode | MTP acceptance | Generated result |
|---|---:|---:|---|
| q4_0/q4_0 | 45.29 tok/s | 0.8652 | truncated multi-file answer; core fails to compile |
| q8_0/q4_0 | 42.04 tok/s | 0.8342 | complete single-file core and example; exact roundtrip under GNU-C99 |

The q4 core used a `double` as an array subscript, exchanged complex samples
through an `int`, and used an incorrect stage twiddle expression.  The q8/q4
program compiled under GNU-C99 and recovered `[1,2,3,4,4,3,2,1]` with zero
error at its printed precision.  Under strict C99 it still failed because the
generated answer used non-standard `M_PI`, and it omitted an explicit
power-of-two check.  The result is therefore a measurable improvement for this
prompt, not proof of production-grade generated code or a broad quality win.

The measured local cost was 7.2% decode throughput in the comparable 64-token
reasoning run.  Production was switched to target q8_0/q4_0 while retaining
draft q4_0/q4_0 and S=33.  To prevent another unbounded reasoning excursion,
the global fallback output limit remains 4,096 tokens.  The 64-token screen
produced a compact response but proved too terse in interactive use, so the
operational default was raised to 256 on 2026-08-02.  This is a deliberately
bounded quality-oriented compromise and is not yet a completed A/B result.  A
1,024-token budget was rejected as the default because the model had already
started drafting final code inside the reasoning block when the forced end
transition occurred.  Per-request client limits and explicitly different
reasoning budgets still take precedence.
Artifacts are under `benchmark-results/target-kv-{fit,quality}-*`.
