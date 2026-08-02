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
