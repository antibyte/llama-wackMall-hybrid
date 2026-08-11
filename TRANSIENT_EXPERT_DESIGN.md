# Transient Expert Feasibility Design

Date: 2026-08-03

This document defines the measurement and simulation design. It does not add a
productive transient expert path.

## Invariant

The real router remains authoritative:

```text
actual_ids and actual_weights -> executed experts
predicted_ids                  -> optional transfer candidates only
```

The feature-off model graph and all fixed-tier, sentinel, CPU, KV, SSM, MTP,
rollback, and sampling behavior remain unchanged.

## Measurement tools

### llama-expert-transport-bench

The CUDA tool accepts repeatable `NAME=BYTES` segments. It has no model or GPU
size constants. It records:

- pageable H2D,
- pageable to pinned staging plus H2D,
- already-pinned H2D,
- sequential device-memory reads,
- sequential mapped-host reads,
- optional pinned-H2D overlap with a synthetic CUDA window.

The working set rotates across multiple source offsets. JSON records device
capabilities, medians, p95 values, bandwidth, and overlap diagnostics. Mapped
and device reads use the same checksum kernel, but they are not quantized
matmul measurements.

### bench_expert_transport.py

The wrapper reads GGUF metadata, derives all unique per-expert component,
Gate+Up, and full sizes, runs independent benchmark processes, and reports the
median of process medians. With an expert timing JSON it also reports an
optimistic transfer break-even. That break-even excludes GPU compute and is a
necessary condition only.

### llama-expert-compute-bench

The compute tool loads the model read-only through mmap, copies one selected
expert into a small GPU-resident graph, and measures:

- Gate+Up plus SiLU/multiply,
- Down,
- the complete Gate+Up, activation, and Down path.

Latency mode synchronizes after every graph. Queued mode amortizes submission
and synchronization and is a lower bound. Neither mode includes routing or
H2D. Results are parameterized by model, layer, expert, device, and repetition
count.

### simulate_expert_streaming.py

The replay consumes raw Phase 1 routing records, measured CPU timing, GGUF
layout, transport results, and optional GPU compute results. It compares:

- predictor versus perfect Oracle,
- full, Gate+Up, and Down-only streaming,
- global versus per-layer arenas,
- Top-8, Top-12, and Top-16 ranking pools,
- configurable slots, copies per layer, and bytes per token.

The fixed residency source is the boolean snapshot embedded in the trace. A
trace collected at S=33 cannot claim S=66 or S=70 behavior. The GTX 1080 must
collect its own trace with its effective fixed placement.

Rank precision should be calibrated with separate representative requests via
`--calibration-trace`. Without that option, the replay labels its predictor
probabilities `in-sample-optimistic`; this is useful as an upper bound but not
a production-policy estimate. The useful-byte metric counts a copy as useful
only when it hits the immediately predicted target invocation. A later cache
reuse contributes to ready hits and saved CPU time but does not retroactively
relabel the original copy, so the byte metric is conservative for retained
per-layer arenas.

Without `--lead-ms`, every scheduled copy is assumed ready and exposed copy
time is zero. This is explicitly an upper bound. An explicit lead value is
accepted only for controlled analysis; final decisions require measured
per-layer windows.

## Scratch options

The simulator reports exact bytes for:

```text
global K slots:    K * max(expert_part_bytes[layer])
per-layer K slots: K * sum(expert_part_bytes[layer])
```

A future global arena must attach owner layer, expert ID, generation, state,
and completion event to every slot. Pending bytes are not residency. A graph
may consume a slot only after publication, and a slot remains protected until
the consuming graph is complete.

## Gate to productive code

A single-layer routing-exact spike is allowed on a target only if all of these
hold under target measurements:

1. A relevant set of layers has `cpu_ms > resident_gpu_ms`.
2. On-demand `copy_ms + resident_gpu_ms` beats CPU, or measured lookahead hides
   enough copy time to produce the same result.
3. Oracle net saving is at least 5 percent.
4. Predictor net saving is at least 3 percent after GPU compute, predictor
   runtime, ID readback, exposed H2D, and false positives.
5. Useful bytes are at least 70 percent and late transfers are below 10
   percent.
6. Global scratch remains at or below 32 MiB for the first spike.
7. No global scheduler barrier is required per layer.

Failure on one GPU does not disable the tools or policy on another GPU. It
only blocks productive runtime code for the failed target.

## Future single-layer spike

If a target passes the gate, start with no MTP, one request, static placement,
adaptation off, W=0, one stream, and one selected layer. The actual cold set is
partitioned into disjoint fixed, transient-ready, and CPU-fallback sets. A late
copy remains CPU for that invocation. False positives are ignored. The three
contributions use the actual gate weights exactly once.

MTP remains guarded until the no-MTP path passes deterministic, logit, quality,
and long-run tests.
