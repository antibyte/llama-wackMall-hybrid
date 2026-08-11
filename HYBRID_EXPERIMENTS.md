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

## Multi-output target backend sampling with a reasoning budget

The implementation from llama.cpp PR #25532 (tested head
`4cbb71d3b754af4c3f3c502cc227702686e2bdf9`) was ported onto the hybrid in an
isolated worktree.  It allows the target sampler to remain on its backend for
all output rows of an MTP verification graph.  This is distinct from the
already enabled draft backend sampler.

The upstream proposal rejected target backend sampling whenever a reasoning
budget sampler existed.  The local compatibility bridge keeps the budget state
on the CPU and treats it as one of two modes:

- while the budget is idle or counting, the backend-selected token remains
  authoritative;
- while the budget is forcing its configured closing sequence, the exact next
  forced token overrides the precomputed backend token before MTP comparison
  and acceptance.

Acceptance then advances both the reasoning-budget state and the transactional
backend RNG exactly once.  A forced token that was filtered from the compact
backend candidate list is represented by a one-entry CPU candidate array; the
model logits and the actual MTP routing/verification graph are not changed.
Grammar-constrained requests still disable backend sampling because their
valid token set cannot be handled by this small bridge.

During the semantic port, the initial implementation exposed an output-lifetime
bug: reshaped candidate views could be reused by the graph allocator and then
contain floating-point logit bits instead of token IDs.  Keeping the actual
`get_rows` result as the graph output, as in the upstream PR, fixed the issue.
The dedicated mixed-chain test now passes on both CPU and CUDA.

### Correctness probes

The real Qwen3.6 model was tested with target q8_0/q4_0, draft q4_0/q4_0,
S=33, W=0, temperature zero, seed 123, a 64-token reasoning budget, and 512
generated tokens.  MTP-2 and MTP-3 produced identical reasoning and answer
hashes with target backend sampling off and on.  Draft generated/accepted
counts were also identical.  MTP-2 was additionally checked at temperature
0.8 and seed 4242; CPU and transactional backend sampling again produced
identical hashes and 317/387 accepted draft tokens.  These runs cross the
forced reasoning-to-content transition inside normal speculative decoding.

### GTX 1660 Ti sustained result

The sustained screen used a fresh server per run, 32K context, q8_0/q4_0
target KV, q4_0/q4_0 draft KV, S=33, W=0, 376/376, eight CPU threads,
temperature zero, `ignore_eos`, 2,000 output tokens, and three repetitions.
The machine was in the `balanced` power profile, so the paired relative result
is the useful comparison rather than an older absolute record.

| MTP | Target backend sampling | Median decode | Acceptance | Mean accepted length | Peak VRAM |
|---|---|---:|---:|---:|---:|
| 2 | off | 37.571 tok/s | 0.7770 | 2.55 | 5,646 MiB |
| 2 | on  | 38.037 tok/s | 0.7770 | 2.55 | 5,672 MiB |
| 3 | off | 35.499 tok/s | 0.6678 | 3.00 | 5,708 MiB |
| 3 | on  | 35.873 tok/s | 0.6678 | 3.00 | 5,736 MiB |

This is a measured gain of 1.24% for MTP-2 and 1.05% for MTP-3.  All three
output hashes and token hashes were identical within every configuration and
also across off/on.  MTP-2 remains 6.0% faster than MTP-3 with backend sampling
on, so MTP-2 is still the production choice on this GPU.  The extra sampler
graphs cost about 26--28 MiB VRAM.

The result is positive but small on the GTX 1660 Ti.  It remains worth testing
on the GTX 1080: the PR reports roughly 4% on an sm_61 Tesla P40, and avoiding
host sampling can matter more with that machine's older CPU.  Because the
upstream PR is still open and under review, target backend sampling remains
default-off and is enabled explicitly with `-bs` or
`TARGET_BACKEND_SAMPLING=1` in the benchmark runner.

Artifacts are under
`benchmark-results/backend-sampling-mtp{2,3}-{off,on}-3x2000-20260804`.

## Shared hot IDs and multi-token MoE fusion

The hot tier originally mapped the same router IDs through separate Gate, Up,
and Down LUT tensors.  Besides launching redundant `CONT` and `GET_ROWS`
work, distinct mapped-ID tensor addresses prevented the existing CUDA
Gate+Up+GLU fusion predicate from matching.  The default-off
`LLAMA_EXPERT_SHARED_HOT_IDS=1` path builds the mapping once from the layer's
Gate store and passes it to all three expert matmuls.  Router IDs, masks,
sentinel handling, cold compute, and router weights are unchanged.

The first sm_75 screen used S=33, W=0, static no-sync, backend target sampling,
q4_0/q4_0 target KV, 128/128 physical batches, and a fresh server per run.
All compared output and token hashes were identical.

| Mode | Shared IDs off | Shared IDs on | Difference |
|---|---:|---:|---:|
| no MTP, 3x256 median | 35.647 tok/s | 36.544 tok/s | +2.52% |
| MTP-2, 3x256 median | 39.504 tok/s | 39.712 tok/s | +0.53% |

No-MTP benefits from both the removed LUT work and the existing single-token
Gate+Up+GLU fusion.  MTP-2 initially benefits only from the LUT change because
the dedicated multi-token MoE MMVQ kernel did not support fusion.

`GGML_CUDA_MOE_MULTI_FUSION=1` therefore adds a second default-off experiment.
For bias- and scale-free `MUL_MAT_ID` graphs on Turing or newer GPUs, the
dedicated kernel computes Gate and Up dot products together and applies GLU
before writing.  It accepts two through four target tokens so MTP-1/2/3 verify
graphs are covered.  Pascal remains guarded off pending an sm_61-specific
benchmark.

With shared IDs already enabled, the MTP-2 3x256 median increased from 39.712
to 39.905 tok/s, another +0.49%.  Relative to the original shared-ID-off
baseline, the combined short-screen increase was +1.02%.  A 128-token MTP-3
smoke increased from 36.763 to 37.159 tok/s after four-token graphs were added.
The MTP-3 value is an individual smoke, not a sustained result.  MTP-1, MTP-2,
and MTP-3 retained identical hashes, acceptance, and mean accepted length in
their correctness comparisons.

Artifacts are under `/tmp/hot-id-*` and `/tmp/moe-multi-fusion*` on the GTX
1660 Ti host.  Both controls remain default-off until counterbalanced 2,000
token runs and the GTX 1080 architecture screen are complete.

The subsequent sustained MTP-2 comparison used 32K context, 376/376 physical
batches, S=33, W=0, target backend sampling, three fresh servers per side, and
2,000 output tokens.  The baseline median was 38.427 tok/s and the combined
shared-ID plus multi-token-fusion median was 38.693 tok/s, a measured +0.69%.
All six outputs and token streams were hash-identical; acceptance was 0.752978
and mean accepted length was 2.51 in every run.  One optimized run was below
the baseline median, and the two three-run series were sequential rather than
counterbalanced.  The controls therefore remain default-off pending a paired
screen and the sm_61 result.  Artifacts are under
`/tmp/kernel-mtp2-{baseline,optimized}-3x2000-20260806`.

## AVX2 CPU cold-expert multi-row dots

The row-reuse experiment improved cache locality but still decoded each
quantized weight block separately for every repeated MTP expert selection.
`LLAMA_EXPERT_CPU_MULTI_ROW=1` adds a default-off x86 AVX2 path which decodes
one weight block and evaluates two through four Q8_K activation rows against
it. The first implementation intentionally covers only the model's dominant
expert types:

- all 41 Gate and Up stacks are Q4_K;
- 38 of 41 Down stacks are Q5_K;
- the three Q6_K Down stacks retain the existing single-row implementation.

The optimization is used only by the fused CPU cold-expert node. It implies
row-oriented traversal, groups Gate/Up only when quantized token rows are
physically consecutive, and otherwise falls back to the original call. The
feature is inert without AVX2. Other architectures and quant types preserve
their old paths.

A dedicated C++ test quantizes deterministic Q4_K/Q5_K weights and four Q8_K
input rows, then compares four ordinary dot products with one multi-row call.
Both types were bit-identical. Model-level No-MTP, MTP-1, MTP-2, and MTP-3
checks also retained output hashes, token hashes, acceptance, and mean accepted
length. The targeted expert tests passed 5/5.

One MTP-2 phase-timing pair at 256 output tokens showed that the kernel does
reduce the intended work:

| Measurement | Existing row reuse | Multi-row | Difference |
|---|---:|---:|---:|
| CPU cold total | 2132.898 ms | 2008.598 ms | -5.83% |
| Gate + Up | 1182.300 ms | 1137.849 ms | -3.76% |
| Activation | 43.923 ms | 42.069 ms | -4.22% |
| Down | 797.356 ms | 727.093 ms | -8.81% |

Without timing instrumentation, three 256-token MTP-2 runs moved from a
39.948 tok/s median to 40.102 tok/s (+0.39%). MTP-3, which offers more rows
per verify graph, moved from 37.755 to 38.075 tok/s (+0.85%). Hashes were
identical across each comparison, but MTP-3 remained slower than MTP-2.

The sustained test used the historical prompt-specific placement profile
with SHA-256 `351cf93b8776de0001f5b7a05d7a187c4a53e37ad4e1386fc678c0fdf6995b5c`,
32K context, S=33, W=0, 376/376 maximum batch sizes, q4_0/q4_0 target and draft
KV, MTP-2, eight CPU workers, target backend sampling, shared hot IDs,
multi-token CUDA MoE fusion, and the System76 performance profile. Three
fresh-server 2,000-token runs per side produced:

| CPU multi-row | Median decode | Acceptance | Mean accepted length | Hashes |
|---|---:|---:|---:|---|
| off | 46.734 tok/s | 0.780269 | 2.56 | identical |
| on | 46.688 tok/s | 0.780269 | 2.56 | identical |

The sustained result is -0.10%, so the CPU savings are not on the critical
path of this GTX 1660 Ti configuration. The feature remains default-off and
Q6_K specialization is deferred. It should be screened on the GTX 1080 host:
that machine's older quad-core i7 may expose more CPU cold time even though
the CUDA device is stronger and has more fixed slots. Enable it there only
for an A/B test with `LLAMA_EXPERT_CPU_MULTI_ROW=1`; do not change the stable
default based on the short-run gains.

Local artifacts are under `/tmp/cpu-multi-*20260806`. The power profile was
restored to `balanced` after the sustained comparison.

## Exact CUDA post-Down MoE combine fusion

The next kernel probe targeted the graph tail which previously materialized
the weighted expert tensor and then reduced its Top-8 dimension through seven
ordered F32 additions. `GGML_CUDA_MOE_COMBINE_FUSION=1` recognizes only the
closed named MoE subgraph

```text
MUL -> N VIEW nodes -> N-1 ordered ADD nodes
```

and replaces it with one CUDA kernel. The matcher validates tensor types,
shapes, sources, output ownership, and memory ranges. It accepts 2--32 routed
experts and any positive token count; the common Top-8 path is compile-time
unrolled. Router weights are broadcast through block-local shared memory.
Explicit round-to-nearest multiplication and addition preserve the separate
kernel's multiplication boundary and original left-to-right sum order.

Short model-level controls with no MTP and MTP-1/2/3 generated 32 tokens each.
Fusion off/on retained identical output hashes, token hashes, MTP acceptance,
and mean accepted length in all four pairs. The MTP-2 performance screen used
the historical prompt-specific profile (`351cf93...`), 32K context, S=33,
W=0, 376/376 batches, q4_0/q4_0 target and draft KV, eight CPU workers, target
backend sampling, shared hot IDs, the multi-token CUDA MoE fusion, and the
System76 performance profile:

| Variant | 3x256 median | Difference | Hashes |
|---|---:|---:|---|
| existing graph | 44.747 tok/s | baseline | identical |
| initial generic combine kernel | 44.640 tok/s | -0.24% | identical |
| shared-weight, unrolled Top-8 kernel | 44.777 tok/s | +0.07% | identical |

The optimized result is measurement noise rather than a practical gain, so a
3x2,000 run is not justified. The code remains default-off as a portable
experiment for architectures where launch overhead or F32 bandwidth balance
differs. It should be screened independently on sm_61; no GTX 1080 result is
claimed here. Local artifacts are under `/tmp/moe-combine-*20260806`.

## Q8_0 three-column MMVQ launch geometry

A node-level Nsight Systems trace was captured with MTP-2, the historical
prompt-specific placement profile, shared hot IDs, multi-token MoE fusion, and
target backend sampling. Within the decode interval, the dominant GPU kernel
categories were:

| Kernel category | Total GPU time | Calls | Share of decode GPU-kernel time |
|---|---:|---:|---:|
| Q8_0 MMVQ, three output columns | 79.478 ms | 2,838 | 25.10% |
| Q4_K fused multi-token MoE Gate+Up | 34.802 ms | 440 | 10.99% |
| Q6_K one-token output projection | 34.526 ms | 22 | 10.90% |
| Q5_K multi-token MoE Down | 24.753 ms | 407 | 7.82% |
| Q6_K three-token output projection | 23.287 ms | 11 | 7.35% |

The post-Down combine accounted for only a small fraction, explaining the
neutral combine-fusion result. The Q8_0 three-column path was therefore used
for a strictly launch-geometric experiment. `GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS`
accepts `1`, `2`, or `4`; unset/zero retains the existing automatic geometry.
Only the number of output rows assigned to a CUDA block changes. The quantized
dot products, warp reduction order, tensor shapes, and output stores remain the
same. The override applies only to non-ID Q8_0 calls with exactly three output
columns.

Short MTP-2 correctness runs at rows/block 1, 2, and 4 produced identical
output hashes, token hashes, acceptance, and mean accepted length. A 3x256
screen then produced:

| Rows/block | Median decode | Difference from automatic | Hashes |
|---:|---:|---:|---|
| automatic (2) | 44.758 tok/s | baseline | identical |
| 1 | 42.934 tok/s | -4.07% | identical |
| 4 | 45.255 tok/s | +1.11% | identical |

Because four rows/block cleared the short-screen threshold, it was compared
against automatic selection in three fresh-server 2,000-token runs. Conditions
were 32K context, S=33, W=0, 376/376 maximum batches, q4_0/q4_0 target and draft
KV, MTP-2, eight CPU workers, target backend sampling, shared hot IDs,
multi-token MoE fusion, static no-sync, the historical profile
`351cf93b8776de0001f5b7a05d7a187c4a53e37ad4e1386fc678c0fdf6995b5c`, and
the System76 performance profile.

| Rows/block | Three decode results | Median | Difference |
|---:|---|---:|---:|
| automatic (2) | 46.873, 46.831, 46.693 | 46.831 tok/s | baseline |
| 4 | 47.442, 47.436, 47.193 | 47.436 tok/s | +1.29% |

All six 2,000-token runs were hash-identical. Acceptance was 0.780269 and mean
accepted length was 2.56 throughout. The sm_75 result is therefore positive,
but the control remains default-off because launch geometry is
architecture-specific. The tree compiles for sm_61 with the override present;
the GTX 1080 still requires its own `0/1/4` A/B screen. Local artifacts are
under `/tmp/mmvq-q8-n3-*20260806`. The host power profile was restored to
`balanced` after the measurements.

## Q6_K output-projection MMVQ launch geometry

The same node-level decode trace attributed 34.526 ms (10.90%) to the Q6_K
one-column output projection and 23.287 ms (7.35%) to its three-column MTP-2
counterpart. Two independent default-off controls were added:

```text
GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS
GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS
```

Both accept `1`, `2`, or `4`; zero/unset retains automatic selection. The
override is restricted to plain non-ID Q6_K operations without fused bias,
scale, or gate inputs. Small-K one-column calls retain their specialized
dispatch. As with the Q8_0 experiment, only output-row grouping changes; dot
products and reductions keep the existing order.

MTP-2 correctness probes covered the baseline and `(ncols1,ncols3)` settings
`(2,0)`, `(4,0)`, `(0,1)`, and `(0,4)`. All generated identical output and
token hashes. The one-row three-column variant was visibly slower and was
discarded. Additional off/on checks for no-MTP, MTP-1, and MTP-3 were also
hash-identical, including unchanged acceptance and mean accepted lengths.

The 3x256 screen used the same historical profile and 32K MTP-2 conditions as
the Q8_0 experiment, with `GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=4` already active:

| Q6_K rows `(ncols1,ncols3)` | Median decode | Difference | Hashes |
|---|---:|---:|---|
| automatic `(0,0)` | 45.275 tok/s | baseline | identical |
| `(2,0)` | 45.425 tok/s | +0.33% | identical |
| `(0,4)` | 45.678 tok/s | +0.89% | identical |
| `(2,4)` | 45.861 tok/s | +1.29% | identical |

The combined `(2,4)` setting therefore advanced to three fresh-server
2,000-token runs:

| Q6_K rows | Three decode results | Median | Difference |
|---|---|---:|---:|
| automatic | 47.293, 47.438, 47.045 | 47.293 tok/s | baseline |
| `(2,4)` | 48.027, 47.794, 47.737 | 47.794 tok/s | +1.06% |

All six long outputs and token streams were identical. MTP acceptance was
0.780269 and mean accepted length was 2.56 throughout. This is a reproducible
sm_75 gain, but not yet a cross-architecture default. The modified source
builds successfully for sm_61 with MMQ disabled; the GTX 1080 must still run
its own geometry sweep. Local artifacts are under
`/tmp/mmvq-q6-k-*20260806`. The host power profile was restored to `balanced`
after the benchmark series.

## AMD Renoir iGPU layer offload

A CUDA+Vulkan build was used to test the Ryzen 7 4800H integrated Radeon as a
secondary layer device and as a full-model device. The experiment found real
NVIDIA VRAM savings but large throughput regressions: one leading layer on the
iGPU reduced decode from 39.497 to 32.725 tok/s, and full iGPU execution with
S=33 reached 10.015 tok/s. An aggressive full-GTT autofit selected S=181 and
12.96 GiB, completed its warm-up, then stopped making progress on the measured
request. It is therefore rejected on Renoir and remains default-off.

The test also fixed fixed-hot-tensor allocation for a model whose layers all
reside on one non-default GPU backend. Full setup, capability data, results,
reproduction parameters, and the stronger-APU retest policy are documented in
[`IGPU_LAYER_EXPERIMENTS.md`](IGPU_LAYER_EXPERIMENTS.md).

## Compiler optimization feasibility

The production build already uses O3, native CPU flags for ggml-cpu, native
sm_75 SASS, and CUDA fast math. Separate LTO, LTO plus whole-host native, and
LTO/native/no-semantic-interposition builds compile successfully. Six targeted
component tests pass. The CUDA library is byte-identical between these builds,
so any gain must come from CPU cold work or host/server overhead.

After the server was stopped, a controlled 3x256 screen compared O3, LTO,
O3+native, LTO+native, and LTO+native+no-semantic-interposition. O3 reached
45.771 tok/s; the candidates reached 45.749, 45.760, 45.687, and 45.671 tok/s.
All controlled outputs, token streams, MTP acceptance values, and mean accepted
lengths were identical. None cleared the one-percent promotion gate, so O3
remains the production compiler configuration and no 2,000-token compiler run
was warranted. The earlier allocation failure is still not a result. Detailed
flags, artifact paths, unsafe options, and remaining optimization priorities
are documented in
[`COMPILER_OPTIMIZATION_ANALYSIS.md`](COMPILER_OPTIMIZATION_ANALYSIS.md).

## Asynchronous pinned scheduler split copies

A target-decode-only Nsight Systems trace separated target verification,
prefill, and MTP draft work with default-off NVTX ranges. In the target ranges,
the dominant CUDA API wait was not the final scheduler barrier. Backtraces
attributed the long repeated waits to synchronous H2D tensor copies at CPU to
CUDA scheduler split boundaries. The trace contained 1,624 pinned H2D copies
for 28 target verification steps. Of these, 1,066 copies were 196,608 bytes,
the repeated per-layer cold contribution for the traced MTP shape.

`GGML_CUDA_ASYNC_HOST_COPY=1` is a default-off CUDA backend experiment. It is
restricted to source buffers allocated by the CUDA pinned-host buffer type and
CPU source backends. The copy is enqueued on the destination CUDA backend
stream, so the following graph remains ordered after the transfer without a
host-side `cudaStreamSynchronize()`. Pageable inputs and all other backend
pairs retain the original path.

Controlled GTX 1660 Ti MTP-2 runs used the archived profile with SHA-256
`351cf93b8776de0001f5b7a05d7a187c4a53e37ad4e1386fc678c0fdf6995b5c`,
S=33, W=0, static no-sync, 376/376 batching, eight target and draft threads,
Q4_0/Q4_0 target and draft KV, target and draft backend sampling, shared hot
IDs, multi-token MoE fusion, Q8 ncols=3 rows=4, and Q6 rows=2/4.

| Run length | synchronous median | asynchronous median | delta | correctness |
| --- | ---: | ---: | ---: | --- |
| 3 x 256 tokens | 45.458 tok/s | 46.196 tok/s | +1.62% | identical output/token hashes and MTP metrics |
| 3 x 2,000 tokens | 47.435 tok/s | 48.335 tok/s | +1.90% | identical output/token hashes and MTP metrics |

The 2K runs all produced output hash
`69de5b7a21d279b29febdab85f4d44c0c3040804fa84f555a6bfeaf71b0b9a7a`
and token hash
`07abeb9a3951592b2247a6a272eacc897849b7e16029f2a4944a23afce575263`.
MTP acceptance was 0.780269 and mean accepted length was 2.56 in both cases.
No CUDA error, invariant failure, VRAM increase, or output divergence occurred.

Artifacts:

- `benchmark-results/async-host-copy-control-3x256-20260807`
- `benchmark-results/async-host-copy-on-3x256-20260807`
- `benchmark-results/async-host-copy-control-3x2000-20260807`
- `benchmark-results/async-host-copy-on-3x2000-20260807`
- `/tmp/wackmall-target-decode-20260807.nsys-rep`
- `/tmp/wackmall-target-sync-backtrace-20260807.nsys-rep`

The implementation remains opt-in globally. The local GTX 1660 Ti launcher
enables it as a measured winner; the GTX 1080 launcher marks it as a candidate
that still requires the same phase-matched sm_61 validation.

The H2D winner was then combined with the already measured flat non-contiguous
CONCAT kernel. Three 2,000-token runs reached 49.187, 49.129, and 49.134 tok/s,
for a 49.134 tok/s median. This is 1.10% above the previous flat-CONCAT median
of 48.599 tok/s and 3.58% above the phase-matched synchronous/non-flat median
of 47.435 tok/s. Hashes and MTP metrics remained unchanged.

`GGML_SCHED_ASYNC_D2H_COPY=1` tests the corresponding CUDA-to-host boundary.
It queues the pinned readback on the source backend stream after source graph
work and synchronizes once at the CPU consumer, replacing the old source sync
followed by a separately synchronized copy stream. It is still exact and
default-off. With the H2D and CONCAT winners, three Q4_0/Q4_0 2K runs reached
49.656, 49.370, and 49.236 tok/s (49.370 median, +0.48%). Under the actual
quality-oriented q8_0/q4_0 target KV and general production profile, the 3x256
median changed only from 44.425 to 44.530 tok/s (+0.24%) and included one slow
candidate outlier. It therefore did not clear the one-percent promotion gate
and remains disabled in `start.sh` and `start1080.sh`.

Additional stop decisions from this series:

- S=34 was safely clamped to the effective S=33 by VRAM autofit.
- `--poll 100` was neutral (47.176 versus 47.190 tok/s in the D2H screen).
- Compiler LTO/native variants remain rejected; none improved on controlled O3.

Additional artifacts:

- `benchmark-results/async-host-copy-flat-concat-3x2000-20260807`
- `benchmark-results/async-d2h-combined-3x2000-20260807`
- `benchmark-results/q8q4-copy-control-3x256-20260807`
- `benchmark-results/q8q4-copy-d2h-3x256-20260807`
- `benchmark-results/s34-all-winners-smoke256-20260807`
- `benchmark-results/poll100-all-winners-3x256-20260807`

## Three-row Q4_K/Q5_K multi-token MoE geometry

After the target-decode trace identified the fused Q4_K Gate+Up and plain
Q5_K Down kernels as 25.22 percent of target CUDA-kernel time combined, a
strictly launch-geometric MTP-2 probe assigned three output rows to each
three-token block.  The experimental dispatch was limited to fused Q4_K
Gate+Up and plain Q5_K Down.  Dot products, warp reductions, routing, and
stores were unchanged.

A one-pair 256-token smoke was hash-identical and changed decode from 47.126
to 47.246 tok/s (+0.26 percent).  The subsequent fresh-server 3x512 screen
produced:

| Geometry | Three decode results | Median | Difference |
| --- | --- | ---: | ---: |
| existing two rows | 48.941, 48.942, 48.934 | 48.941 tok/s | baseline |
| experimental three rows | 49.090, 48.979, 49.009 | 49.009 tok/s | +0.14 percent |

All outputs and token streams were identical; MTP acceptance was 0.698824 and
mean accepted length was 2.39 in every 512-token run.  The result is noise and
does not meet the one-percent promotion gate.  No 2,000-token run was
performed, and the experimental dispatch was removed rather than adding a
non-winning production flag.  Artifacts are under
`/tmp/mmvq-moe-n3-{baseline,candidate}-{1x256,3x512}-20260807*`.

## Pascal DP4A single-token MMVQ tuning

Official llama.cpp pull request #25479 introduces a Pascal-specific MMVQ
parameter table and uses two rather than four warps for one-column quantized
matrix-vector products on compute capabilities 6.1 and 6.2. This is directly
relevant to the GTX 1080 decode path, but not to the local GTX 1660 Ti. The
port is isolated behind the default-off CMake option:

```text
GGML_CUDA_PASCAL_MMVQ_TUNING=OFF
```

When enabled, only `DP4A <= arch < Volta` and `ncols_dst == 1` select the new
two-warp table. Multi-column MTP verification and the existing type-specific
row controls are unchanged. Volta, Turing, later NVIDIA GPUs, AMD, and generic
dispatch retain their old configuration.

The default-off port compiled successfully on 2026-08-08 for SM 61 with
forced MMQ disabled. `llama-server`, `llama-cli`, and `test-backend-ops`
linked, and `cuobjdump` confirmed SM 61 cubins. This is a build result only:
the local SM 75 GPU cannot provide a GTX 1080 correctness or throughput
result. The target gate is a paired OFF/ON run in separate build directories,
followed by three fresh-server 512-token runs and only then 3x2,000 if the
short median gains at least one percent without hash, MTP, quality, VRAM, or
stability regression.

Source reference: https://github.com/ggml-org/llama.cpp/pull/25479

## Skip-sentinel MMVQ for cold expert slots

Date: 2026-08-08. Cold experts map through the hot-tier LUT to a zeroed
sentinel weight channel. The GPU previously still executed full Q4_K/Q5_K
matvecs against those zeros. `LLAMA_EXPERT_SKIP_SENTINEL=1` marks each
`MUL_MAT_ID` with `op_params[0]=1` and `op_params[1]=sentinel`, and the CUDA
MMVQ path (`mul_mat_vec_q` and `mul_mat_vec_q_moe`) early-exits those slots and
writes exact zeros. The mathematical result is identical to loading the zeroed
weights; cold work remains on the CPU fused path.

Controls:

- `LLAMA_EXPERT_SKIP_SENTINEL=0|1` (default off in the runtime until measured;
  `start.sh` / `start1080.sh` enable it for the first A/B)
- Fused Gate+Up multi-token launches pass the skip slot via
  `ggml_cuda_mm_fusion_args_{host,device}.skip_slot`

Side fix: shared hot-ID mapping no longer forces `ggml_cont` when the router
IDs are already contiguous.

Promoted on 2026-08-08 for the GTX 1660 Ti winner stack (SC, S=33, MTP-2,
q4_0/q4_0 target and draft KV, Q8/Q6 row overrides, async H2D, flat CONCAT,
shared hot IDs, multi-token MoE fusion, 376/376, 8/8 threads, profile
`cbb5aef7…`).

3×256 short gate (fresh server each run, 32-token warm-up):

| Skip | Three decode results | Median | Δ |
|---|---|---:|---:|
| 0 | 43.146, 42.107, 42.839 | 42.839 | baseline |
| 1 | 44.761, 44.760, 44.703 | 44.760 | **+4.48%** |

All six runs reproduced output hash
`87b87f1d8c52a0d64c48682675b01d3e458cc8c835098cab2ec98caca4ec577c`,
token hash
`f2739bc4de244337140dda6da246dcb436b05f3ca331e085506e722aa61b4ae5`,
acceptance 0.597403, mean accepted length 2.19, peak VRAM 5,624 MiB.

3×2,000 sustained gate (64-token warm-up):

| Skip | Three decode results | Median | Δ |
|---|---|---:|---:|
| 0 | 44.486, 44.820, 44.744 | 44.744 | baseline |
| 1 | 46.657, 46.350, 46.487 | **46.487** | **+3.90%** |

All six long runs reproduced output hash
`6d552602fadede320c5b4bfacec186158f5dccec650fdcb23d8a05412def2332`,
token hash
`9b315ca546d63c9f5b258415eb6e0b45445adf3943d8e56c06e5d6797919ef12`,
acceptance 0.766160, mean accepted length 2.53, peak VRAM 5,624 MiB.

`LLAMA_EXPERT_SKIP_SENTINEL=1` is therefore a measured sm_75 winner and is
enabled in `start.sh`. The GTX 1080 launcher keeps it as a candidate that still
needs its own phase-matched sm_61 A/B. Artifacts:
`/tmp/skip-sentinel-{off,on}-{3x256,3x2000}-20260808`.

### Rejected follow-up: multi-token MoE shared-weight K reassignment

A device-side fast path detected when every MTP token mapped to the same expert
channel and reassigned warps from "one per token" to "all on K with shared
weight loads". It compiled after shrinking the scratch to four tokens, but the
K-reduction reordering changed float accumulation order. Against the skip-
sentinel baseline the 3×256 screen produced different output/token hashes
(`5fc5623b…` / `038099e9…` vs `87b87f1d…` / `f2739bc4…`) and different MTP
metrics (0.6167 / 2.23 vs 0.5974 / 2.19). That path was removed.

### Bit-identical MoE weight staging (removed)

The former `GGML_CUDA_MMVQ_MOE_SHARE_WEIGHTS=1` path kept the original per-token
`kbx` schedule and warp reduction. When all tokens mapped to the same expert,
warp 0 copied each thread's weight block into shared memory and every token warp
reused it. Against
the skip-sentinel winner stack on 2026-08-08:

| Share weights | 3×256 decode | Median | Δ | Hashes / MTP |
|---|---|---:|---:|---|
| 0 (baseline) | 44.761 / 44.760 / 44.703 | 44.760 | — | `87b87f1d…` / `f2739bc4…`, 0.5974 / 2.19 |
| 1 | 37.375 / 37.300 / 37.314 | **37.314** | **−16.6%** | identical |

Correctness passes; throughput fails. Staging plus per-`kbx` `__syncthreads`
costs more than letting concurrent warps hit L1 on the same global weight
addresses. The implementation and runtime control were removed. Artifact:
`/tmp/moe-share-bitident-3x256-20260808`.

### Re-screen: async D2H with skip-sentinel stack

With skip-sentinel and the full sm_75 winner stack, `GGML_SCHED_ASYNC_D2H_COPY=1`
was hash-identical at 3×256 (hashes `87b87f1d…` / `f2739bc4…`, acceptance
0.5974) but the median was 44.762 vs 44.760 tok/s (+0.004%). Still below the
one-percent gate; remains disabled in `start.sh`.

## CUDA block-reduce shared-memory race fix

Official llama.cpp pull request #26385 fixes unsafe shared-memory reuse across
multiple multi-warp reductions. The port adds the required synchronization in
ordinary softmax and separate shared-memory regions for multi-stage single-row
softmax and GroupNorm variance reduction. Expert matmul arithmetic, routing,
MTP state, and KV formats are unchanged.

The SM 75 release build passed all 255 supported targeted CUDA `SOFT_MAX`,
`NORM`, `RMS_NORM`, and `GROUP_NORM` cases, including large softmax rows and
CUDA-graph reuse. Eight Q4_0/Q4_0 MTP-2 screens produced identical output and
token hashes with unchanged MTP metrics. Serial groups showed time-order
drift; an immediately paired comparison measured 44.246 versus 44.187 tok/s
(-0.13%). The change is retained as a correctness fix, not a throughput
winner, and no 2,000-token performance claim is made.

Artifacts remain outside the repository under
`/tmp/block-reduce-{base,patch}-*20260808*`.

Source reference: https://github.com/ggml-org/llama.cpp/pull/26385

## Compact hot-only MMVQ launches (skip-sentinel follow-up)

Date: 2026-08-09. Skip-sentinel still launched a full Top-k channel grid and
early-exited cold sentinel slots. With ~50% seed coverage that wasted launch
slots on zero work. The former `GGML_CUDA_MMVQ_COMPACT_SKIP=1` path built a
device-side packed list of non-sentinel channels (single-token) or
`(channel, token)` pairs (multi-token MoE), cleared the destination, read the
active count (one int D2H + stream sync), and launched only those work items.
Outputs scattered back to the original Top-k layout so the existing weighted
combine and CPU cold ADD stayed unchanged and bit-identical.

Historical controls:

- `GGML_CUDA_MMVQ_COMPACT_SKIP=0|1` (default 0)
- Requires `LLAMA_EXPERT_SKIP_SENTINEL=1` so `skip_slot` is set on MUL_MAT_ID
- While a CUDA graph is capturing, the path falls back to the full-grid
  skip-sentinel behavior (variable grid is not capture-safe)

Former implementation: `ggml/src/ggml-cuda/mmvq.cu`
(`mmvq_compact_*_kernel`, `mmvq_compact_args`, entry in
`ggml_cuda_mul_mat_vec_q`). It was wired in `start.sh` and
`scripts/bench_hybrid.sh` as an A/B flag.

### 3×256 gate (2026-08-09) — rejected

Phase-matched against skip-sentinel alone on the sm_75 winner stack (SC,
S=33, MTP-2, 376/376, q4_0/q4_0 target+draft, shared hot IDs, multi-token MoE
fusion, Q8 rows=4, Q6 rows=2/4, async H2D, flat CONCAT, static no-sync,
profile `cbb5aef7…`). Fresh server each run, 32-token warm-up.

| Compact skip | Three decode results | Median | Δ |
|---|---|---:|---:|
| 0 (baseline) | 39.979, 40.058, 39.830 | **39.979** | — |
| 1 | 39.928, 39.857, 39.871 | **39.871** | **−0.27%** |

All six runs reproduced output hash
`87b87f1d8c52a0d64c48682675b01d3e458cc8c835098cab2ec98caca4ec577c`,
token hash
`f2739bc4de244337140dda6da246dcb436b05f3ca331e085506e722aa61b4ae5`,
acceptance 0.597403, mean accepted length 2.19, peak VRAM 5,624 MiB.

Correctness passes; throughput fails the ≥1% promotion gate. The per-op
device compact plus one-int D2H stream sync costs more than the reduced
channel grid saves on this Top-8 / ~50% cold mix. No 3×2,000 follow-up.
The implementation and runtime control were removed.

Note: a first ON attempt aborted with
`GGML_ASSERT(ptr == pool_addr + pool_used)` because compact pool buffers were
freed out of VMM LIFO order; fixed by constructing/allocating
`n_active → ch → tok` after `src1_q8_1`.

Artifacts: `/tmp/compact-skip-off-3x256-20260809`,
`/tmp/compact-skip-on-3x256b-20260809`.

## Q8_0 ncols=1 and ncols=2 MMVQ geometry

Date: 2026-08-09. Nsight previously showed Q8_0 three-column MMVQ as ~25% of
decode GPU time; rows=4 is already the measured ncols=3 winner. `start.sh`
exported `GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS` but the CUDA path ignored it. The
kernel now honors:

- `GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS` = `0|2|4` (default auto = 1 row/block)
- `GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS` = `0|1|4` (default auto = 2 rows/block)
- existing `GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS` = `0|1|4`

Overrides apply only to non-ID Q8_0 launches (same restriction as ncols=3).

### 3×256 screen against the skip-sentinel winner stack

Fixed: SC, S=33, MTP-2, 376/376, q4_0/q4_0, shared hot IDs, multi-token MoE
fusion, Q8 n3=4, Q6 n1=2/n3=4, async H2D, flat CONCAT, static no-sync,
`SKIP_SENTINEL=1`, profile `cbb5aef7…`. Fresh server each run.

| Config | n1 | n2 | n3 | Three decode | Median | Δ vs base |
|---|---:|---:|---:|---|---:|---:|
| base | 0 | 0 | 4 | 40.072, 39.879, 39.484 | **39.879** | — |
| n1r2 | 2 | 0 | 4 | 39.905, 39.872, 39.888 | 39.888 | +0.02% |
| n1r4 | 4 | 0 | 4 | 39.885, 39.885, 39.849 | 39.885 | +0.02% |
| n2r1 | 0 | 1 | 4 | 39.806, 39.920, 39.847 | 39.847 | −0.08% |
| n2r4 | 0 | 4 | 4 | 39.913, 39.854, 39.847 | 39.854 | −0.06% |

All 15 runs shared output hash `87b87f1d…`, token hash `f2739bc4…`, acceptance
0.597403, mean length 2.19. Overrides appeared in server logs when set.

No candidate cleared the 1% gate. MTP-2 spends almost all Q8 work on
three-column verify graphs; ncols=1/2 geometry is not on the critical path.
Defaults remain auto (`0`) in `start.sh`. No 3×2,000 follow-up.

Artifacts: `/tmp/q8-geom-{base,n1r2,n1r4,n2r1,n2r4}-3x256-20260809`,
`/tmp/q8-geom-3x256-summary.txt`.

## Prompt-matched specialist profile and variable placement

Date: 2026-08-09. Seed coverage of the production general corpus profile is
only **51.6%** top-33 mass. A session-local usage capture on the standard
phase-1 bench prompt (512 decode tokens, `SAVE_EXPERT_USAGE=1`) was aggregated
into a specialist CSV and optional variable placements at a uniform S=33 byte
budget.

| Source | Mean top-33 mass | Notes |
|---|---:|---|
| general corpus profile | 51.64% | `profile-corpus-train8-512-…/general-profile.csv` |
| specialist (bench prompt) | **72.87%** | from `SC-run1.usage.csv` |
| variable placement @ S=33 budget (general) | 52.00% offline | slots 16–43 |
| variable placement @ S=33 budget (specialist) | 73.22% offline | slots 23–43 |

### 3×256 screen (same winner stack, skip-sentinel, q4_0/q4_0, MTP-2)

| Config | Seed cov | Median | Δ | Acc | Hash relation |
|---|---:|---:|---:|---:|---|
| gen-s33 (baseline) | 51.6% | 39.880 | — | 0.5974 | reference `87b87f1d…` |
| **spec-s33** | **72.9%** | **44.522** | **+11.64%** | 0.6545 | different (expected) |
| gen-var | validated | 41.962 | +5.22% | 0.6591 | different |
| spec-var | validated | 42.768 | +7.24% | 0.6087 | different |

### 3×2,000 sustained

| Config | Median | Δ | Acc | Mean len | Within-config hashes |
|---|---:|---:|---:|---:|---|
| gen-s33 | **42.375** | — | 0.7662 | 2.53 | identical |
| **spec-s33** | **43.222** | **+2.00%** | 0.7395 | 2.48 | identical |
| gen-var | 41.839 | **−1.27%** | 0.7488 | 2.50 | identical |

Peak VRAM stayed 5,624 MiB for every run.

### Decisions

- **spec-s33** clears the 1% sustained gate on this prompt (+2.00%) and improves
  short-run throughput strongly. Token streams diverge (different hot set →
  different float path and MTP acceptances). The specialist is **prompt-matched
  / overfit** to the phase-1 bench prompt and is **not** a drop-in replacement
  for multi-domain OpenWebUI traffic.
- **gen-var** wins the short screen but **loses sustained** (−1.27%); keep
  variable placement default-off for the portable general profile.
- Production `start.sh` keeps the general corpus profile and uniform S unless
  the operator deliberately installs a workload-matched specialist CSV.

How to rebuild a specialist for a real workload:

```bash
SAVE_EXPERT_USAGE=1  # one collection run
python3 tools/aggregate_expert_profiles.py \
  --input path/to/run.usage.csv --weight 1 \
  --model "$MODEL" --output specialist-profile.csv \
  --normalization per-layer --scale 10000 --top-k 33
# then set LLAMA_EXPERT_HOT=specialist-profile.csv and re-bench
```

Artifacts: `/tmp/profile-cov-artifacts-20260809`,
`/tmp/profile-cov-{gen-s33,spec-s33,gen-var,spec-var}-3x{256,2000}-20260809`,
`/tmp/profile-cov-3x{256,2000}-summary.txt`.

## MMVQ cleanup and scheduler destination-sync deduplication

Date: 2026-08-10. Two rejected MMVQ experiments were still compiled into the
normal Q4_K/Q5_K/Q6_K MoE kernels even when their runtime switches were off:
shared weight staging and compact skip packing. Their static shared arrays and
live ranges inflated the off-path kernel resources. The experiments, launch
plumbing, environment controls, and benchmark controls were removed while the
skip-sentinel fast path and all measured row overrides were retained.

| sm_75 rows=2 kernel | Before registers/shared | After registers/shared |
|---|---:|---:|
| Q4_K fused | 238 / 18,468 B | 64 / 0 B |
| Q5_K plain | 254 / 11,300 B | 57 / 0 B |
| Q6_K plain | 242 / 13,476 B | 64 / 0 B |

`MUL_MAT_ID` passed 790/790 backend cases and `MUL_MAT_ID_FUSION` passed
13/13. An exact old-binary versus cleaned-binary 3x256 A/B was neutral:
36.060 versus 36.031 token/s (-0.08%). Output hashes, token hashes, and MTP
acceptance were identical. The cleanup remains because it removes dead code and
restores kernel occupancy without making a throughput claim.

The same audit found a redundant destination synchronization at single-copy
CUDA scheduler split boundaries. A backend that was fully synchronized earlier
in the current `compute_splits` call does not need another full synchronization
before a later split input overwrites its copy slot, provided no graph has run
on that backend since. `GGML_SCHED_DEDUP_DST_SYNC=1` tracks this fact locally,
is active only for `n_copies == 1`, and invalidates it before every asynchronous
graph launch. Event-backed multi-copy scheduling is unchanged.

### Counterbalanced A/B

Fixed: specialist S=34, MTP-2, 64/64, q4_0/q4_0, skip sentinel, multi-token
MoE fusion, async pinned H2D, flat CONCAT, static no-sync, fresh server per run.

| Length | Off runs | On runs | Median off | Median on | Change |
|---|---|---|---:|---:|---:|
| 256 | 36.148, 36.008, 35.968 | 36.345, 36.175, 36.157 | 36.008 | 36.175 | +0.46% |
| 2,000 | 47.340, 47.664, 47.111 | 47.713, 47.878, 47.572 | 47.340 | 47.713 | +0.79% |

All paired deltas were positive. The 2,000-token output hash
`4bcf3885...`, token hash `1634df0a...`, acceptance 0.703614, and mean accepted
length 2.41 were identical. The synchronous-H2D fallback, no-MTP path, and two
consecutive requests in one server process also retained identical hashes. A
pre-change Nsight trace contained 7,622 sync-before-H2D pairs over 121 target
verifications; those synchronizations accounted for 89.603 ms, or about
0.736 ms per verification.

This is below the usual 1% promotion gate, but it is enabled in the private
max-TPS `start.sh` stack because both test lengths and every paired run agreed,
the removed wait is directly visible in the trace, and the implementation is a
small single-copy-only scheduling optimization. The benchmark harness keeps its
default off so controls remain explicit.

Two related candidates were rejected and removed:

- Packing 2/4/8 independent Q4_0-to-F16 converter warps per CUDA block was
  bit-identical and passed 336/336 Q4_0 flash-attention cases. At the observed
  1,441,792/1,572,864-value shapes, the best two-warp variant saved only
  0.25%/0.69% of a roughly 22-24 us kernel, with no useful end-to-end lever.
- Batching the two pinned CUDA-to-host inputs of a cold MoE split reduced
  synchronization calls but dropped a matched smoke run from 48.066 to
  30.256 token/s (-37.0%). The prototype was fully removed.

Artifacts: `/tmp/mmvq-cleanup-ab-20260810`,
`/tmp/sched-dedup-ab-20260810`, `/tmp/sched-dedup-ab-3x2000-20260810`,
`/tmp/sched-dedup-safety-20260810`.
