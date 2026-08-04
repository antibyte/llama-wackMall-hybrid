# Transient Expert Feasibility Experiments

Date: 2026-08-03

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

## Decision

Current GTX 1660 Ti:

```text
transport and compute tools: keep
Oracle simulator: keep
productive transient path: do not implement
predictability-aware placement mutation: defer
single-layer event bridge: defer until another target passes the cost gate
```

This is not a rejection for the GTX 1080. The old quad-core CPU can make cold
compute substantially more expensive. Tomorrow's SM 61 run must repeat the
same target-derived layout, three-process transport, representative resident
compute, CPU timing, fixed-residency trace, and replay before any runtime code
is enabled.

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

The local SM 61 compile check passes with CUDA 12.0 and G++ 12 as the nvcc host
compiler. The requested CUDA 12.4 plus GCC/G++ 13 combination still requires
verification on the GTX 1080 host.
