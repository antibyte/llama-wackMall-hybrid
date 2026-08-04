# Transient Expert Feasibility Experiments

Date: 2026-08-03 through 2026-08-04

Results in this file are measurements, not performance claims for untested
hardware.

## GTX 1660 Ti transport

Configuration:

```text
GPU: NVIDIA GeForce GTX 1660 Ti, SM 75, 6 GiB
active PCIe link during test: Gen3 x8
CUDA async engines: 3
mapped host memory: supported
working set: about 32 MiB
runs: 3 independent processes
samples per run and size: 200 after 20 warmups
```

Median of process medians:

| Segment | Bytes | Pinned H2D ms | Staging memcpy ms | Mapped read ms |
| --- | ---: | ---: | ---: | ---: |
| one Gate or Up matrix | 589,824 | 0.090976 | 0.029962 | 0.092816 |
| small Down matrix | 720,896 | 0.110528 | 0.036248 | 0.112608 |
| large Down matrix | 860,160 | 0.131328 | 0.042813 | 0.133280 |
| Gate plus Up | 1,179,648 | 0.179168 | 0.058178 | 0.181024 |
| small full expert | 1,900,544 | 0.287136 | 0.100188 | 0.288784 |
| large full expert | 2,039,808 | 0.307584 | 0.115937 | 0.309440 |

Pinned H2D sustains about 6.0 to 6.18 GiB/s. The full-expert numbers agree
with the earlier warm-cache transfer measurements.

With a requested 500-us synthetic window, the measured spin kernel lasted
about 0.417 ms. Between 92.4 and 97.8 percent of pinned copy time was hidden.
This proves copy-engine concurrency in isolation only.

Artifacts are under the ignored local directory:

```text
benchmark-results/transient-transport-sm75-median3-20260803T2210Z/
```

## GTX 1660 Ti resident expert compute

Three representative layers used expert 0. Values below are per-graph latency
medians with synchronization after each graph.

| Layer | Gate+Up ms | Down ms | Full ms | Full bytes |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.036702 | 0.019486 | 0.037156 | 1,900,544 |
| 18 | 0.025772 | 0.018578 | 0.033803 | 1,900,544 |
| 39 | 0.025771 | 0.020115 | 0.034956 | 2,039,808 |

Queued lower bounds are about 0.032 to 0.033 ms for a full expert. The latency
value is used by the simulator because a real layer dependency cannot queue an
unbounded number of independent copies of the same graph.

The phase-matched no-MTP control measured 37.496 token/s and 1,028.787 ms of
CPU cold work over 22,502 cold selections, or 0.04572 ms per selection. The
representative per-layer values are 0.05784 ms for layer 0, 0.04549 ms for
layer 18, and 0.07128 ms for layer 39. Thus resident GPU compute is faster for
these examples, but only by about 0.01 to 0.036 ms before transport.

## Exact on-demand gate

Representative optimistic comparisons:

```text
layer 0 full:
  CPU                 0.0578 ms
  pinned H2D + GPU    0.2871 + 0.0372 = 0.3243 ms

layer 18 full:
  CPU                 0.0455 ms
  pinned H2D + GPU    0.2871 + 0.0338 = 0.3209 ms

layer 39 full:
  CPU                 0.0713 ms
  pinned H2D + GPU    0.3076 + 0.0350 = 0.3425 ms
```

Gate+Up-only and Down-only transfers also lose. Routing-exact on-demand H2D is
therefore rejected for the current GTX 1660 Ti.

## Mapped-host lower bound

For the 128-token trace, direct mapped sequential reads would consume:

```text
full CPU phases:          742.02 ms
full mapped reads:      4,731.37 ms
optimistic difference: -3,989.35 ms
```

This already loses before the quantized GPU expert computation is added.
Mapped zero-copy is rejected for the current GPU.

## Routing replay

The replay input is one 128-token, no-MTP, post-moe, target-norm, distance-one
trace with S=33:

```text
layer-token records: 4,953
actual cold selections: 16,349
Top-12 recall: 0.9564
Top-12 weighted coverage: 0.9708
Top-12 cold recall: 0.9374
```

The current predictor rank probabilities are calibrated on that same trace
and are therefore explicitly labelled `in-sample-optimistic`. The negative
gate remains valid under this favorable assumption. GTX 1080 policy results
must use separate calibration and replay requests.

The first replay used an earlier MTP-2 CPU timing source. A phase-matched
no-MTP control was then collected because productive lookahead is initially
guarded to no-MTP. After resident GPU compute and that no-MTP CPU timing were
included, the best global predictor upper bound had:

```text
mode: full expert
global slots: 1 to 4 (same sequential result)
Top-M: 8
copies per selected layer: up to 1
copy budget: 64 MiB/token
ready cold recall: 0.2525
useful byte ratio: 0.9293
copied bytes: 63.0 MiB/token
scratch: 1.95 MiB; more global slots do not improve this sequential replay
net saving before predictor and copy exposure: 0.3123 ms/token
```

Even the perfect Oracle with four per-layer slots, about 291.6 MiB of scratch,
and completely hidden transfers saves only 0.6757 ms/token after resident GPU
compute. The predictor per-layer counterpart saves 0.6472 ms/token. Against
the 26.67-ms no-MTP control token time, the global predictor bound is about
1.17 percent and the impractical per-layer bound is about 2.43 percent. Both
remain below the 3 and 5 percent gates before predictor cost, real transfer
exposure, and contention are charged.

## GTX 1660 Ti decision

Current GTX 1660 Ti:

```text
transport and compute tools: keep
Oracle simulator: keep
productive transient path: do not implement
predictability-aware placement mutation: defer
single-layer event bridge: defer until another target passes the cost gate
```

This was not a rejection for the GTX 1080. The old quad-core CPU can make cold compute substantially more expensive. The following SM 61 run order was completed on 2026-08-04 before the runtime bridge was enabled.

## GTX 1080 run order

```text
1. build both microbench tools for native SM 61
2. run bench_expert_transport.py against the GTX 1080 model
3. run llama-expert-compute-bench for early, middle, and late layer layouts
4. collect per-layer CPU cold timing with the intended thread count
5. collect a no-MTP trace with the effective S=66/70 fixed placement
6. replay predictor and Oracle with the measured GPU compute
7. implement one routing-exact layer only if the gates pass
```

The earlier local SM 61 compile check passed with CUDA 12.0 and G++ 12 as the nvcc host compiler. Native verification with CUDA 12.4 and GCC/G++ 13 was subsequently completed on the GTX 1080 host as recorded below.

## GTX 1080 experiment update

Update date: 2026-08-04

This section records the complete GTX 1080 investigation, including rejected implementations and correctness failures. Results are local experimental measurements and are not general performance claims. No merge, commit, push, or upstream submission is planned.

### Reproducibility and input isolation

The prompt inputs are deliberately outside the Git repository and must remain there:

```text
CALIBRATION_PROMPT_SOURCE=/root/gtx1080-hybrid-inputs/calibration.txt
REPLAY_PROMPT_SOURCE=/root/gtx1080-hybrid-inputs/replay.txt
```

Their SHA-256 values are:

```text
calibration.txt  6e56a837d86a42880d0cdda79a7089e7ca5e15e71dc71a9adc1fa73ee610dcdf
replay.txt       68f3700281816732839b9d18ad2a297c38d3b467dfc3e948388a0c0a13899565
```

The exact calibration prompt is:

```text
Review the security architecture of a local automation daemon that accepts HTTP requests and launches constrained subprocesses. Analyze authentication, input validation, privilege boundaries, filesystem access, logging, cancellation, denial of service, and recovery after failures. Compare at least two implementation strategies and provide concrete tests.
```

The exact replay prompt is:

```text
Implement a correct iterative radix-2 Cooley-Tukey FFT in portable C. Include forward and inverse transforms, input validation, bit-reversal permutation, clear sign and normalization conventions, compilation instructions, and tests that detect common indexing and direction errors. Explain the important correctness decisions.
```

The placement/profile input is the local untracked file `1`, with SHA-256 `8bac1687ac121d930991c1086f6438919c409ceb0da381257bef4b00c1497ce5`. It is not to be committed.

The model was `/root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf`, with SHA-256 `56aeaf6a72cb2a538a7356941fe2983cff7c5efe91cac866336f33d144913cd8`.

All GTX 1080 result artifacts are outside the repository under `/root/gtx1080-hybrid-results/`.

### Host and build

```text
recorded starting HEAD: 3ac684552da9a47d4072c221cba0f73b4a2284cb
recorded branch: main
GPU: NVIDIA GeForce GTX 1080, SM 61, 8192 MiB
GPU async engines: 2
concurrent kernels: supported
mapped host memory: supported
unified addressing: supported
idle PCIe state: Gen1 x16, P8
active compute PCIe state: Gen3 x16, P2
CPU: Intel Core i7-4770 at 3.40 GHz, 4 cores / 8 threads, AVX2, 8 MiB L3
CUDA toolkit: 12.4.131
GCC/G++: 13.4.0
build directory: build-transient-sm61
CUDA architecture: native SM 61
```

The final inference configuration used a 65,536-token context, target KV types Q8_0/Q8_0, draft KV types Q8_0/Q4_0, four target threads, four draft threads, CMOE batch and ubatch 32/32, MTP-2, and disabled backend sampling for deterministic comparisons. The requested fixed expert count S=70 was clamped by the model/layout to effective S=65 in the early controls and later fixed at effective S=64 for the final MTP bridge comparisons.

Four CPU threads were retained. In the direct no-MTP thread check, four threads delivered 25.0946 token/s while eight threads delivered 22.2585 token/s. Eight threads raised average sampled CPU use from 207.2 to 379.6 percent without improving GPU use, so SMT oversubscription was counterproductive on the i7-4770.

The native SM 61 build completed successfully with CUDA 12.4 and GCC/G++ 13. The expert warm-cache, adaptation, placement, and lookahead tests passed. `git diff --check` was clean at the end of the recorded work.

### GTX 1080 transport calibration

The transport benchmark used three independent processes, 200 samples after 20 warmups per size, and an approximately 32 MiB working set.

| Segment | Bytes | Pinned H2D ms | Pageable staging ms | Staged H2D ms | Mapped read ms | Hidden fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one Gate or Up matrix | 589,824 | 0.055648 | 0.049071 | 0.067808 | 0.069856 | 0.9251 |
| large Down matrix | 860,160 | 0.080992 | 0.083973 | 0.098048 | 0.086496 | 0.9481 |
| Gate plus Up | 1,179,648 | 0.111712 | 0.118564 | 0.112064 | 0.119232 | 0.9627 |
| small full expert | 1,769,472 | 0.166816 | 0.181523 | 0.167584 | 0.174384 | 0.9748 |
| large full expert | 2,039,808 | 0.192096 | 0.210045 | 0.193296 | 0.197504 | 0.9782 |

Pinned H2D sustained approximately 9.84 to 9.89 GiB/s. The overlap test exposed only about 0.0042 ms of copy time under its synthetic compute window, demonstrating usable copy-engine concurrency in isolation.

The important Gate+Up transfer is 0.111712 ms. Layer 0 CPU Gate+Up timing was about 0.112243 ms, which is approximately break-even at one use. Other representative layers were roughly 0.069 to 0.073 ms, so they require at least two useful reuses to amortize one transfer. This rejected routing-exact synchronous copy but justified testing early asynchronous copy with exact consumption.

### Resident expert compute

The resident compute benchmark used three processes, five warmups, 30 latency repeats, and 100 queued repeats. The latency medians below are used because the productive path has layer dependencies.

| Layer | Gate+Up ms | Down ms | Full ms |
| ---: | ---: | ---: | ---: |
| 0 | 0.019728 | 0.012359 | 0.028300 |
| 20 | 0.018593 | 0.010726 | 0.024935 |
| 39 | 0.018567 | 0.011621 | 0.026585 |

Resident compute is much faster than the corresponding CPU phase. The problem is therefore transport timing and prediction usefulness, not resident kernel throughput.

### CPU timing and held-out routing replay

The phase-timed CPU controls used three repetitions and four threads:

| Context | Decode token/s | Cold share | CPU expert ms | Gate+Up ms | Activation ms | Down ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 24.9236 | 0.44759 | 9,718.257 | 6,529.059 | 146.369 | 2,847.393 |
| 65,536 | 24.5565 | 0.46330 | 10,022.069 | 6,734.143 | 149.031 | 2,959.208 |

Calibration and replay used separate prompts. Both traces contain 19,929 layer-token records. The 65,536 replay contains 73,137 actual cold selections; the 32,768 replay contains 70,848. This is held-out calibration, unlike the optimistic in-sample GTX 1660 Ti analysis.

For the 65,536 replay at distance one, the overall predictor metrics were:

| Candidate set | Top-1 rate | Recall | Weighted coverage | Cold recall | Estimated copied bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-8 | 0.76510 | 0.86187 | 0.91388 | 0.86544 | 10,543,841,280 |
| Top-12 | 0.76510 | 0.95844 | 0.97215 | 0.95827 | 16,144,072,704 |
| Top-16 | 0.76510 | 0.97746 | 0.98454 | 0.97739 | 22,114,172,928 |

The trace collection itself is intentionally not a throughput reference because instrumentation reduced replay decode speed to 18.4730 token/s at 65,536 context and 18.7249 token/s at 32,768 context.

### Streaming simulation

The simulator combined held-out routing records, fixed-residency booleans embedded in the trace, measured transport, measured resident compute, and measured CPU phase timing. It is explicitly an upper bound because predictor runtime is excluded and the optimistic-ready cases assume hidden copies.

At 65,536 context, the best full-expert predictor upper bound used four per-layer arena slots, Top-8, at most one copy per selected layer, and a 64 MiB/token budget. It reached 44.63 percent ready cold recall, an 89.87 percent useful-prefetch ratio, and a theoretical 2,926.928 ms total net saving over the 19,929 records. The corresponding perfect Oracle reached 44.73 percent ready recall and 2,933.387 ms theoretical net saving. Their closeness says the limiting factor in that optimistic point is arena/budget scheduling more than ranking quality.

The best 65,536 Oracle upper bounds by phase were:

| Mode | Ready recall | Copy ms | CPU saving ms | GPU compute ms | Net saving ms | Scratch bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.44726 | 3,105.911 | 3,787.184 | 853.797 | 2,933.387 | 304,742,400 |
| Gate+Up | 0.61374 | 3,109.392 | 3,540.475 | 834.562 | 2,705.913 | 188,743,680 |
| Down | 0.69456 | 2,233.205 | 1,728.644 | 576.751 | 1,151.893 | 115,998,720 |

The direct mapped-host lower bound remained negative before GPU computation: full CPU phases 8,579.227 ms versus 13,640.829 ms mapped sequential reads, Gate+Up 5,766.452 versus 8,720.271 ms, and Down 2,516.482 versus 5,747.385 ms. Mapped zero-copy was therefore rejected.

The broad simulator established feasibility but its best configurations require hundreds of MiB of scratch and unrealistically hidden work. The productive implementation was consequently narrowed to a small, explicit, one-layer bridge before any multi-layer expansion.

### Productive bridge design progression

The productive bridge is opt-in and experimental. It is inactive unless layer selection and a JSON output path are supplied. It is limited to MTP target verification graphs; MTP draft graphs are excluded.

The implementation progressed through these stages:

1. No-MTP singleton and MTP target-verification graph detection were validated without consuming results.
2. A target-only early router observation triggered a background worker with pinned host staging, a non-blocking CUDA stream, and device scratch.
3. Exact CPU claim/fetch used the tuple `(layer, expert, token)` so a result could only replace the matching CPU Gate+Up row.
4. A portable custom CUDA Q4_K x Q8_K kernel computed deterministic Gate and Up values for a copied expert.
5. K was generalized from one to at most three candidates. K=2 reuses one Q8 input quantization, copies contiguous candidate blocks, and launches one batched kernel.
6. The manager was generalized to at most four independent layer states, workers, streams, and scratch allocations.
7. Secondary-layer prediction was changed to graph-neutral recurrence after additional lookahead branches were proven to alter output.

The relevant implementation surface is the CUDA backend bridge and kernel, the CPU Gate+Up claim/fetch hook, target router observation in the graph/model path, and lookahead configuration. The work remains an uncommitted private experiment.

### Deterministic CUDA kernel validation

For layer 1, expert 0, `n_embd=2048`, and `n_ff=512`, the existing general CUDA comparison was not bit exact with the CPU reference: Gate matched 2/512 values with maximum absolute error 0.004440308, and Up matched 2/512 with maximum absolute error 0.003680289. That is normal floating-point variation but unsuitable for a hash-identical replacement path.

The custom deterministic kernel matched 512/512 Gate and 512/512 Up values with maximum absolute error 0. Device-side use of the bridge Q8 input also matched 512/512 for both matrices. In the short validation, deterministic compute took 0.023721 ms, device-input quantization plus compute took 0.073042 ms, and the resident Gate+Up check was 0.028879 ms.

### K=1 and K=2 smoke tests

Both 16-token smoke tests produced output SHA-256 `e6376cb4d68c50a7fe3817b974299d7b42b3a290ffab4f7cf46d5ca834112da9` and token SHA-256 `b020bcbc941d02edaf646dda131f701a09b67336e127684f24b7696ba4906c7a`.

| Mode | Batches | Copies | Useful/consumed | Repeated candidates | Decode token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| K=1 | 9 | 9 | 5 | 3 | 25.1548 |
| K=2 | 9 | 18 | 11 | 5 | 25.1406 |

K=2 increased useful coverage as intended. The smoke length was not sufficient to establish full-run correctness.

### Correctness failure: added graph tensors

The first 256-token baseline produced output SHA-256 `4f2ed06aec4d8b43d783836e0b1e5445c941aff2a0f5778142e1d4a024b980d3`, token SHA-256 `1300454af2cbdfe4553d9c4e1ae15df613377286f5c1930b4efc4736e0aaeea9`, MTP acceptance 0.782828, and 28.1174 token/s.

The early K=1 and K=2 implementations both produced the same incorrect alternative hashes: output `5cffccad9ca5b615c95fc5d6254585df37141b4d425cb19d35cb7edf717dfc56`, tokens `eb60cb3dbd28e8361160a9b30c25cec34fa884bfab9c9714d5d7b6bca24e1e95`. The first token mismatch was at output index 153, baseline token 1287 versus bridge token 310. Because K=1 and K=2 failed identically, batching was not the cause.

Verification mode computed each claimed row on the CPU, fetched the GPU result into separate scratch, compared every bit, and retained the CPU output. It verified 82 rows without a mismatch, but the generated stream still diverged at the same point. A pure observer that never consumed CUDA results also diverged. This isolated the failure from the custom math and from result consumption.

The root cause was the additional `CONT` snapshot tensors used to expose router values. Their presence changed graph scheduling and/or in-place allocation choices, which changed the numerical path. Any throughput recorded for these divergent runs is invalid as an optimization result.

The fix observes the already existing I32 top-k view tensor directly in the CUDA backend before the view/no-op fast path skips it. No additional graph tensor is created. The direct observer became hash-identical, and the direct-view K=2 256-token consume run reproduced both baseline hashes and the same MTP acceptance.

### Direct-view single-layer result

The direct-view layer-1 K=2 run used 102 batches, copied 204 candidates, consumed 139 useful rows, had zero late useful rows, copied 240,648,192 bytes, and recorded approximately 35.094 ms staging, 38.986 ms transfer, and 8.955 ms bridge compute in its representative run. Of 204 predicted candidates, 111 repeated; 12 repeated within one graph, 17 within two, 25 within four, and 32 within eight.

The first controlled three-run direct-view comparison gave a baseline median of 27.934939 token/s and a K=2 median of 28.153930 token/s, a measured increase of 0.783933 percent. All six outputs and token streams were hash-identical. Later manager measurements showed a smaller effect, so 0.78 percent must not be treated as a stable gain.

### Multi-layer failure and graph-neutral recurrence

The first two-layer manager ran lookahead branches for layers 1 and 2. Its 16-token smoke test passed, but the 256-token run diverged: output SHA-256 `1688d31a74ccbbf9b6d0ec8db60f6017945132d8338215e82583576518bdad67`, token SHA-256 `375edf0314f379e7aaadf6aab76c6b8b16f10ee43f53a7aeda45dc27816133c8`, and MTP acceptance 0.818653. Its apparent 28.5741 token/s is invalid because it did not execute the same token stream.

The safe multi-layer mechanism is recurrence. Actual top-k experts observed in graph N become candidates for graph N+1. The worker copies candidates and all current hidden-input rows while the job is active, then selects the matching token only after the actual router is known. It adds no predictor branch and therefore remains graph neutral.

All secondary layers in the manager are forced to recurrence automatically. The first listed layer may use direct target lookahead; explicitly selected recurrence layers also use recurrence. This restriction exists for correctness, not merely tuning.

Single-run exploration, all hash-identical to the baseline, was:

| Configuration | Layer useful rows | Decode token/s |
| --- | --- | ---: |
| layers 1+2, recurrence K=2/K=2 | 77 + 89 | 27.9898 |
| layer 1 lookahead K=2, layer 2 recurrence K=2 | 139 + 89 | 28.0493 |
| layer 1 lookahead K=2, layer 2 recurrence K=1 | 139 + 54 | 28.4291 |
| layer 1 lookahead K=2, layers 2+3 recurrence K=1 | 139 + 54 + 55 | 28.3374 |
| layer 1 lookahead K=2, layer 3 recurrence K=1 | 139 + 55 | 28.2610 |

These single runs selected layer-2 recurrence K=1 for repeated measurement. Adding layer 3 increased work without improving the measured result.

### Final controlled comparison

All final runs generated 256 tokens with output SHA-256 `4f2ed06aec4d8b43d783836e0b1e5445c941aff2a0f5778142e1d4a024b980d3`, token SHA-256 `1300454af2cbdfe4553d9c4e1ae15df613377286f5c1930b4efc4736e0aaeea9`, MTP acceptance 0.782828, and mean accepted length 2.57.

| Configuration | Run 1 | Run 2 | Run 3 | Median token/s | Versus baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| bridge disabled | 28.211832 | 27.931912 | 27.934939 | 27.934939 | reference |
| layer 1 lookahead K=2 | 27.946781 | 27.976870 | 28.102293 | 27.976870 | +0.150102% |
| layer 1 lookahead K=2 plus layer 2 recurrence K=1 | 28.156429 | 27.933494 | 28.057460 | 28.057460 | +0.438595% |

The final hybrid representative metrics were 204 layer-1 copies with 139 useful and consumed rows, plus 102 layer-2 copies with 54 useful and consumed rows. Both layers reported zero late useful rows. Total consumed Gate+Up rows were 193. Layer 1 copied 240,648,192 bytes and layer 2 copied 120,324,096 bytes.

The hybrid median was 0.288062 percent above the current single-layer manager median. The spread between runs is comparable to the gain, so the conclusion is a small measured positive direction, not a robust production-level speedup. Longer interleaved tests and more prompts are required before making a stronger claim.

### Recommended experimental configuration

The current best correctness-preserving configuration is:

```sh
GGML_CUDA_EXPERT_BRIDGE_LAYERS=1,2
GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_LAYERS=2
GGML_CUDA_EXPERT_BRIDGE_K=2
GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_K=1
GGML_CUDA_EXPERT_BRIDGE_CONSUME=1
GGML_CUDA_EXPERT_BRIDGE_JSON=/root/gtx1080-hybrid-results/bridge.json
```

`GGML_CUDA_EXPERT_BRIDGE_LAYER` remains a backward-compatible singular fallback. Candidate K is bounded to 1 through 3. The bridge manager supports at most four selected layers. Any selected layer after the first is forced to recurrence even if it is omitted from `GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_LAYERS`.

The bridge is explicit opt-in, produces per-layer v5 JSON metrics, and should remain disabled for ordinary runs. Verification can be enabled with `GGML_CUDA_EXPERT_BRIDGE_VERIFY=1` when validating math, at substantial diagnostic overhead.

### Artifact index

The main evidence directories and files are:

```text
/root/gtx1080-hybrid-results/transport-sm61/
/root/gtx1080-hybrid-results/compute-sm61/
/root/gtx1080-hybrid-results/cpu-ctx32k-q8/
/root/gtx1080-hybrid-results/cpu-ctx64k-q8/
/root/gtx1080-hybrid-results/trace-calibration-ctx32k-q8/
/root/gtx1080-hybrid-results/trace-calibration-ctx64k-q8/
/root/gtx1080-hybrid-results/trace-replay-ctx32k-q8/
/root/gtx1080-hybrid-results/trace-replay-ctx64k-q8/
/root/gtx1080-hybrid-results/simulation-ctx32k-q8.json
/root/gtx1080-hybrid-results/simulation-ctx64k-q8.json
/root/gtx1080-hybrid-results/direct-view-k2-expert-compute-check.json
/root/gtx1080-hybrid-results/mtp2-batch-k1-smoke-16/
/root/gtx1080-hybrid-results/mtp2-batch-k2-smoke-16/
/root/gtx1080-hybrid-results/mtp2-k2-baseline-256/
/root/gtx1080-hybrid-results/mtp2-early-target-consume-256/
/root/gtx1080-hybrid-results/mtp2-batch-k2-consume-256/
/root/gtx1080-hybrid-results/mtp2-batch-k2-verify-160/
/root/gtx1080-hybrid-results/mtp2-current-observer-160/
/root/gtx1080-hybrid-results/mtp2-direct-observer-160/
/root/gtx1080-hybrid-results/mtp2-direct-view-k2-consume-256-r3/
/root/gtx1080-hybrid-results/mtp2-multilayer-12-k2-256/
/root/gtx1080-hybrid-results/mtp2-recurrence-12-k2-256/
/root/gtx1080-hybrid-results/mtp2-hybrid-l1-r2-k2-256/
/root/gtx1080-hybrid-results/mtp2-hybrid-l1-r2k1-256/
/root/gtx1080-hybrid-results/mtp2-hybrid-l1-r23k1-256/
/root/gtx1080-hybrid-results/mtp2-hybrid-l1-r3k1-256/
/root/gtx1080-hybrid-results/mtp2-direct-view-baseline-256-r3/
/root/gtx1080-hybrid-results/mtp2-manager-l1-k2-256-r3/
/root/gtx1080-hybrid-results/mtp2-hybrid-l1-r2k1-256-r3/
```

Each inference result directory contains its run CSV, median CSV, response JSON, logs, and any bridge JSON emitted for that run. Host, compiler, Git starting state, GPU link, and input hash captures are stored directly under `/root/gtx1080-hybrid-results/`.

### Final decision and remaining work

```text
native SM 61 transport and compute calibration: passed
held-out predictor replay: passed feasibility gate
mapped zero-copy: rejected
synchronous routing-exact transfer: rejected
additional router snapshot tensors: rejected for correctness
multiple simultaneous lookahead branches: rejected for correctness
direct observation of existing router views: retained
deterministic Q4_K x Q8_K Gate+Up kernel: retained
single-layer K=2 bridge: retained as opt-in experiment
secondary-layer K=1 recurrence: retained as opt-in experiment
default enablement: rejected pending broader evidence
upstream merge or submission: not planned
```

The next useful optimization work is to reduce host staging and synchronization cost, interleave baseline and bridge repetitions to control thermal and clock drift, test longer generations and multiple held-out prompts, measure power and active PCIe state for every comparison, and explore reuse-aware candidate suppression. Any new path must preserve graph structure or prove full-stream hash identity before its throughput is considered.
