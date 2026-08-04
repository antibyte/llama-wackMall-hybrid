# Router Lookahead Experiments

Date: 2026-08-03

This file is the append-only experiment record for routing-exact lookahead.
Results are recorded only after a reproducible run has completed.

## Environment discovery

### Repository

```text
path: /home/andi/src/llama-wackMall-hybrid
baseline: f337cf7d9e52f6986814c7923ea96b82223f376c
branch: codex/router-lookahead-prefetch
initial status: clean
```

The requested `/root/llama-wackMall-hybrid` path is for the future GTX 1080
host and does not exist on the current machine.

### Current GTX 1660 Ti host

```text
GPU: NVIDIA GeForce GTX 1660 Ti, 6144 MiB
architecture: sm_75
current nvcc: CUDA 12.0
gcc-13 and g++-13: present
driver during runtime tests: 595.84
idle after tests: 1 MiB, 0 percent, no compute process
```

The execution sandbox cannot see the NVIDIA device. Runtime tests were run outside the sandbox after verifying the exact model and profile paths. No model, profile, or prompt input was modified.

### Future GTX 1080 host

```text
expected architecture: sm_61
expected VRAM: 8 GiB
expected CUDA: 12.4
expected host compiler: GCC/G++ 13
expected repository: /root/llama-wackMall-hybrid
expected model: /root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf
```

These future-host values are supplied by the user and have not been locally
verified.

## Phase 0 source analysis

Status: complete.

Findings are recorded in `ROUTER_LOOKAHEAD_ANALYSIS.md` and
`ROUTER_LOOKAHEAD_DESIGN.md`. No productive prefetch code was added.

## Phase 1 planned matrix

Initial constraints:

```text
MTP: off
parallel sequences: 1
adaptation: off
warm slots: 0
temperature: 0
same seed per comparison
minimum output: 512 tokens
fresh server or CLI process for each serious comparison
```

Predictor matrix:

```text
points: post-attn, post-moe
norms: target, source diagnostic
distances: 1, then 2 only if distance 1 is measurable and useful
Top-M: 8, 12, 16, or the model expert-count clamp
prompts: multiple distinct public or user-provided external prompt files
```

For each trace run:

- compare output and token hashes with trace disabled,
- record prediction metrics per layer and overall,
- record trace overhead,
- record unavailable timing fields as unavailable,
- do not infer transfer readiness from recall alone.

The benchmark runner now has `L0` for the matching static no-MTP baseline and `LT` for the trace. `LO` is intentionally not present because Oracle scheduling belongs to Phase 2 and has not been simulated yet.

## Phase 1 results

Status: routing-quality collection complete on GTX 1660 Ti; timing gate open.

### Implementation checks

- Feature-off creates no lookahead graph branch.
- The final L0/LT 32-token pair has identical output and token hashes.
- Three 512-token L0/LT pairs have identical per-workload output and token hashes.
- Four 128-token predictor variants and the distance-2 diagnostic preserve the same deterministic output and token hashes.
- Every checked actual set contains eight unique valid IDs, every configured prediction contains 16 unique valid IDs, and every true weight is finite and non-negative.
- Two requests in one server process produced separate request IDs, separate files, and token indices restarting at zero.
- An MTP-2 server-start check logged the Phase-1 guard and produced no trace file.
- The final trace uses a pinned host buffer and reports predictor IDs, actual IDs, and actual weights on `CUDA0`.
- No process or VRAM growth remained after the runs.

Final 32-token hash pair:

```text
output: e2401739466140861c8fb0f3cb77990180701bb9cd59ed759356c59ae3d32cf3
tokens: 44016b281b4ddaeabd2caedfcde3e5bf79933a745ede2a78a59889edb7a199b0
```

### Predictor screening

The screening used one public FFT prompt, 127 one-token decode transitions, S=33, W=0, adaptation off, MTP off, distance 1, and one run per variant. These are routing measurements, not performance medians.

| Point | Norm | Top-M | Recall | Weighted coverage | Cold recall | Top-1 rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| post-attn | target | 8 | 0.8150 | 0.8753 | 0.7935 | 0.6996 |
| post-attn | target | 12 | 0.9216 | 0.9446 | 0.9090 | 0.6996 |
| post-attn | target | 16 | 0.9559 | 0.9673 | 0.9480 | 0.6996 |
| post-moe | target | 8 | 0.8474 | 0.8993 | 0.8344 | 0.7563 |
| post-moe | target | 12 | 0.9471 | 0.9629 | 0.9399 | 0.7563 |
| post-moe | target | 16 | 0.9704 | 0.9783 | 0.9656 | 0.7563 |
| post-attn | source | 8 | 0.8125 | 0.8731 | 0.7892 | 0.6889 |
| post-attn | source | 12 | 0.9206 | 0.9438 | 0.9084 | 0.6889 |
| post-attn | source | 16 | 0.9539 | 0.9659 | 0.9460 | 0.6889 |
| post-moe | source | 8 | 0.8441 | 0.8966 | 0.8305 | 0.7452 |
| post-moe | source | 12 | 0.9439 | 0.9605 | 0.9365 | 0.7452 |
| post-moe | source | 16 | 0.9688 | 0.9771 | 0.9640 | 0.7452 |

Post-moe with target norm wins on routing accuracy. Post-attn still has the larger theoretical lead window, so the final scheduling winner cannot be selected before timing is measured. Source norm is consistently worse and should remain diagnostic only.

Distance 2 was tested on the same 128-token workload with post-moe target norm. Top-16 fell to 0.9238 recall, 0.9424 weighted coverage, and 0.9133 cold recall. Distance 1 is the only current candidate for the timing phase.

### Three-workload 512-token trace

The selected routing-accuracy candidate, post-moe target norm at distance 1, was traced on public FFT, German architecture, and reasoning prompts. The pool contains 1,534 decoded transitions and 59,826 layer-token records.

These long traces used the first hash-stable explicit snapshot implementation, in which I32 snapshots were assigned to CPU. Predictor construction and all routing values are unchanged by the final CUDA-capable I32 `CONT` retention. The final retention path was revalidated on 128 and 32 tokens, including exact hashes and identical routing aggregates.

| Top-M | Recall | Weighted coverage | Cold recall | Useful cold candidate ratio | Candidate MiB/token | Useful MiB/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.8510 | 0.9015 | 0.8275 | 0.8070 | 325.8 | 262.9 |
| 12 | 0.9481 | 0.9644 | 0.9353 | 0.5636 | 527.4 | 297.2 |
| 16 | 0.9715 | 0.9798 | 0.9629 | 0.4111 | 744.2 | 305.9 |

Candidate bytes assume every predicted expert outside the fixed tier would be copied once for that layer-token. They are not measured transfers and do not account for a future arena retaining an expert long enough for reuse. The large byte counts make Top-16 especially questionable despite its recall.

The weakest pooled Top-16 weighted-coverage layers are:

```text
layer 2:  0.8582
layer 1:  0.8801
layer 39: 0.9346
layer 38: 0.9502
layer 7:  0.9633
```

Layers 1, 2, 38, and 39 should be excluded from an initial productive prototype unless Phase 2 timing provides a compelling counterexample. Middle layers 18, 19, 23, 25, and 26 are among the strongest Top-16 predictors.

### Trace overhead

The explicit snapshots and full per-token readback make LT deliberately expensive. In the final 32-token single runs, L0 measured 30.86 token/s and LT measured 3.99 token/s. This is not a production-prefetch estimate and is not a median. It only bounds the cost of the diagnostic implementation.

### Rejected observation variants

1. Marking only top-k views as graph outputs allowed allocator reuse and corrupted later-layer trace IDs.
2. I32 `DUP` snapshots moved IDs to CPU because the CUDA backend does not support I32 `DUP`.
3. Marking the argsort backing tensors as outputs removed snapshot kernels but changed graph allocation and the deterministic output hash. It was rejected even though the observed IDs were valid.
4. The final I32 `CONT` plus F32 `DUP` snapshots preserve hashes and keep all retained tensors on CUDA.

No H2D copy has been authorized. Predictor GPU time, predictor-ID D2H completion time, attention lead time, and timely-prefetch fraction remain unavailable rather than estimated.

### Build portability

The native sm_75 build and lookahead unit test pass. The requested sm_61 plus GCC-13 CUDA-host build cannot be configured with the local CUDA 12.0 toolkit because nvcc 12.0 rejects GCC newer than 12. A separate sm_61 build using GCC 13 for C/C++ and G++ 12 only as the nvcc host compiler is used as the local architecture compile check. The exact CUDA 12.4 plus GCC-13 configuration remains to be verified on the GTX 1080 host.

## Phase 2 Oracle and schedule simulation

Status: transport, resident-compute, and first routing replay complete on the
GTX 1660 Ti. Productive code remains blocked.

Phase 2 must use actual routing records plus measured pinned-transfer and
per-layer timing data. Assumed PCIe bandwidth is not acceptable. Required
outputs include useful prefetch ratio, useful bytes ratio, hidden copy ratio,
theoretical CPU saving, theoretical net saving, simultaneous scratch demand,
and an ideal Oracle under the same constraints.

The measured transport and compute work is documented in
`TRANSIENT_EXPERT_EXPERIMENTS.md`. With phase-matched no-MTP CPU timing and
resident GPU compute included, the best global predictor replay saves only
0.3123 ms/token under an optimistic always-ready transfer assumption while
scheduling about 63.0 MiB/token. The best perfect Oracle examined saves 0.6757
ms/token but requires four slots in every layer, about 291.6 MiB. Predictor
runtime, real exposed copy time, and contention are still excluded. This fails
the Phase 3 gate on the GTX 1660 Ti.

The GTX 1080 plus old i7 remains unevaluated. It must collect target-specific
CPU timing, transport, resident GPU compute, and fixed-residency traces before
the decision is reused.

## Productive prefetch

Status: not implemented.

The implementation stops after Phase 1 unless the requested measurement and
Oracle gates are positive. MTP lookahead remains disabled by default even if a
later no-MTP prototype is approved.

## Stop conditions

Any experiment stops on CUDA hang, Xid, OOM, invariant failure, token/hash
regression in observational mode, unexplained VRAM growth, quality regression,
or a sustained throughput regression beyond the configured acceptance limit.
