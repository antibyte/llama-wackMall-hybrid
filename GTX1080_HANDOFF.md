# GTX 1080 Transient Expert Feasibility Handoff

Date: 2026-08-04

Status: completed. This document preserves the original run procedure. Final measurements, productive bridge configuration, rejected paths, and artifact locations are recorded in `TRANSIENT_EXPERT_EXPERIMENTS.md`.

This handoff is for Codex running on the GTX 1080 host. The first objective is to repeat the measured transport, resident GPU compute, CPU cold timing, routing trace, and Oracle replay on native SM 61. Do not implement productive transient transfers until the target-specific gates pass.

## Expected target

```text
repository: /root/llama-wackMall-hybrid
model: /root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf
GPU: NVIDIA GTX 1080, 8 GiB, SM 61
CUDA: 12.4
host compiler: GCC/G++ 13
CPU: older Intel i7 quad-core
primary KV: q8_0/q8_0
primary context: 65536
secondary context: 32768
MTP during feasibility work: disabled
```

The learned profile path is intentionally not assumed. Find it, inspect it, and set `PROFILE` to an explicit compatible file. Do not use a profile for a different GGUF without validation.

## Protected source state

The handoff commit is on `codex/transient-expert-feasibility` and descends from routing-trace commit `4d4f852be998d57b3968718e1d44c8e477e231f2`. The commit must be made available to the target machine by the user before this procedure starts.

Run these checks first:

```bash
cd /root/llama-wackMall-hybrid
git status --short
git branch --show-current
git rev-parse HEAD
git log -3 --oneline --decorate

test -f /root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf
find /root/atomic-nextn-good -type f \( -name '*.csv' -o -name '*.placement' \) -print
```

If the worktree is dirty, preserve and document every change. Do not reset, clean, or overwrite it. Do not modify anything below `/root/atomic-nextn-good`. Use a separate worktree if the existing checkout cannot be used safely.

Confirm that the following files exist at the selected revision:

```text
tools/expert-transport-bench/expert-transport-bench.cu
tools/expert-compute-bench/expert-compute-bench.cpp
tools/bench_expert_transport.py
tools/bench_expert_compute.py
tools/simulate_expert_streaming.py
TRANSIENT_EXPERT_ANALYSIS.md
TRANSIENT_EXPERT_DESIGN.md
TRANSIENT_EXPERT_EXPERIMENTS.md
```

## Record the target before measuring

Store outputs outside the model directory and outside Git-tracked paths:

```bash
RESULT_ROOT=/root/gtx1080-hybrid-results
mkdir -p "$RESULT_ROOT"

nvidia-smi > "$RESULT_ROOT/nvidia-smi.txt"
nvidia-smi --query-gpu=name,compute_cap,pci.bus_id,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv > "$RESULT_ROOT/gpu-link.csv"
lscpu > "$RESULT_ROOT/lscpu.txt"
nvcc --version > "$RESULT_ROOT/nvcc-version.txt"
gcc-13 --version > "$RESULT_ROOT/gcc-version.txt"
g++-13 --version > "$RESULT_ROOT/gxx-version.txt"
```

Record the active PCIe generation and width while a benchmark is running as well as at idle. Do not infer bandwidth from the nominal slot alone.

## Build native SM 61

Do not reuse an unrelated working build directory and do not enable forced MMQ:

```bash
cd /root/llama-wackMall-hybrid

cmake -S . -B build-transient-sm61 -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_NCCL=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_UI=OFF

cmake --build build-transient-sm61 -j 8 --target \
    llama-server llama-cli \
    llama-expert-transport-bench llama-expert-compute-bench \
    test-expert-adaptation test-expert-warm-cache \
    test-expert-placement test-expert-lookahead
```

### Optional Pascal MMVQ A/B build

The port of upstream llama.cpp PR #25479 is default-off and changes only
one-column MMVQ launch geometry on SM 61/62. Build it in a second directory;
do not overwrite the control build:

```bash
cmake -S . -B build-pascal-tuned-sm61 -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_NCCL=OFF \
    -DGGML_CUDA_PASCAL_MMVQ_TUNING=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_UI=OFF

cmake --build build-pascal-tuned-sm61 -j 8 --target \
    llama-server llama-cli test-backend-ops
```

Run identical fresh-server OFF/ON cases. The tuned build is only a candidate:
compile/link success was established locally, but no GTX 1080 performance
claim exists until the target machine completes hash, MTP, quality, VRAM, and
counterbalanced throughput checks.

If CUDA 12.4 rejects GCC 13, save the complete configure error. Use another installed host compiler only after documenting the exact compiler and starting a new build directory. Never add `GGML_CUDA_FORCE_MMQ` as a workaround.

Run the tests before benchmarks:

```bash
python3 -m unittest \
    tests/test-convert-luce-spark-profile.py \
    tests/test-aggregate-expert-profiles.py \
    tests/test-optimize-expert-placement.py \
    tests/test-expert-streaming-tools.py

ctest --test-dir build-transient-sm61 --output-on-failure \
    -R 'expert-(adaptation|warm-cache|placement|lookahead)'
```

## Set validated inputs

Use external prompt files. Do not copy private prompts into Git:

```bash
MODEL=/root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf
PROFILE=/absolute/path/to/validated-learned-profile.csv
CALIBRATION_PROMPT_SOURCE=/absolute/path/to/calibration-prompt.txt
REPLAY_PROMPT_SOURCE=/absolute/path/to/different-replay-prompt.txt
SERVER=/root/llama-wackMall-hybrid/build-transient-sm61/bin/llama-server

test -f "$MODEL"
test -f "$PROFILE"
test -f "$CALIBRATION_PROMPT_SOURCE"
test -f "$REPLAY_PROMPT_SOURCE"
test -x "$SERVER"
```

Calibration and replay prompts must differ. Use representative prompts rather than the profile-training prompt alone.

## Select the CPU thread count

The old i7 probably has four physical cores and eight threads, but do not optimize for utilization percentage. Compare four and eight workers with the same q8_0/q8_0, 64K, no-MTP workload:

```bash
for CPU_COUNT in 4 8; do
    CASES=L0 \
    REPEATS=1 \
    N_PREDICT=256 \
    WARMUP_TOKENS=32 \
    MODEL="$MODEL" \
    PROFILE="$PROFILE" \
    PROMPT_SOURCE="$REPLAY_PROMPT_SOURCE" \
    SERVER="$SERVER" \
    RESULTS_DIR="$RESULT_ROOT/thread-${CPU_COUNT}-ctx64k-q8" \
    STATIC_FIXED_S=70 \
    CPU_THREADS="$CPU_COUNT" \
    CMOE_BATCH=32 \
    CMOE_UBATCH=32 \
    CONTEXT=65536 \
    TARGET_TYPE_K=q8_0 \
    TARGET_TYPE_V=q8_0 \
    EXPERT_TIMING=0 \
    ./scripts/bench_hybrid.sh
done
```

Use the faster sustained decode result for all subsequent target measurements. Record the effective fixed slot count from `runs.csv` and the server log; requested S=70 may be safely clamped.

## Collect phase-matched CPU timing

Set `BEST_CPU_THREADS` to the winner. Use three fresh processes and select the median decode run, not the fastest run:

```bash
BEST_CPU_THREADS=4
CPU_RUN="$RESULT_ROOT/cpu-ctx64k-q8"

CASES=L0 \
REPEATS=3 \
N_PREDICT=512 \
WARMUP_TOKENS=32 \
MODEL="$MODEL" \
PROFILE="$PROFILE" \
PROMPT_SOURCE="$REPLAY_PROMPT_SOURCE" \
SERVER="$SERVER" \
RESULTS_DIR="$CPU_RUN" \
STATIC_FIXED_S=70 \
CPU_THREADS="$BEST_CPU_THREADS" \
CMOE_BATCH=32 \
CMOE_UBATCH=32 \
CONTEXT=65536 \
TARGET_TYPE_K=q8_0 \
TARGET_TYPE_V=q8_0 \
EXPERT_TIMING=1 \
./scripts/bench_hybrid.sh
```

Choose the median row from `runs.csv` and set both paths to its repetition:

```bash
CPU_REP=2
CPU_STATS="$CPU_RUN/L0-run${CPU_REP}.experts.json"
BASE_RESPONSE="$CPU_RUN/L0-run${CPU_REP}.response.json"
test -f "$CPU_STATS"
test -f "$BASE_RESPONSE"
```

Do not assume repetition 2 is the median just because there are three runs. Inspect and sort the sustained decode values first.

## Measure transport and mapped-host access

The wrapper derives every transfer size from the target GGUF and refuses to reuse an output directory:

```bash
TRANSPORT_RUN="$RESULT_ROOT/transport-sm61"

python3 tools/bench_expert_transport.py \
    --model "$MODEL" \
    --binary build-transient-sm61/bin/llama-expert-transport-bench \
    --output-dir "$TRANSPORT_RUN" \
    --stats-json "$CPU_STATS" \
    --runs 3 \
    --repeats 200 \
    --warmups 20 \
    --working-set-mib 32 \
    --overlap-us 500
```

The mapped-read kernel is a sequential lower bound, not a quantized expert matmul. Reject Zero-Copy if mapped reading alone is slower than the matching CPU phase.

## Measure resident expert GPU compute

Measure early, middle, and late layouts in independent processes:

```bash
COMPUTE_RUN="$RESULT_ROOT/compute-sm61"

python3 tools/bench_expert_compute.py \
    --model "$MODEL" \
    --binary build-transient-sm61/bin/llama-expert-compute-bench \
    --output-dir "$COMPUTE_RUN" \
    --runs 3 \
    --repeats 30 \
    --warmups 5 \
    --queued-iterations 100
```

Use latency medians for dependency-bound simulation. Queued medians are lower bounds only.

## Collect held-out calibration and replay traces

Use `post-moe`, target norm, distance one, and Top-16 capture. Top-M remains a ranking pool; it is not permission to copy all candidates.

```bash
CAL_TRACE_RUN="$RESULT_ROOT/trace-calibration-ctx64k-q8"

CASES=LT \
REPEATS=1 \
N_PREDICT=512 \
WARMUP_TOKENS=32 \
MODEL="$MODEL" \
PROFILE="$PROFILE" \
PROMPT_SOURCE="$CALIBRATION_PROMPT_SOURCE" \
SERVER="$SERVER" \
RESULTS_DIR="$CAL_TRACE_RUN" \
STATIC_FIXED_S=70 \
CPU_THREADS="$BEST_CPU_THREADS" \
CMOE_BATCH=32 \
CMOE_UBATCH=32 \
CONTEXT=65536 \
TARGET_TYPE_K=q8_0 \
TARGET_TYPE_V=q8_0 \
EXPERT_TIMING=0 \
LOOKAHEAD_DISTANCE=1 \
LOOKAHEAD_TOP_M=16 \
LOOKAHEAD_POINT=post-moe \
LOOKAHEAD_NORM=target \
./scripts/bench_hybrid.sh

REPLAY_TRACE_RUN="$RESULT_ROOT/trace-replay-ctx64k-q8"

CASES=LT \
REPEATS=1 \
N_PREDICT=512 \
WARMUP_TOKENS=32 \
MODEL="$MODEL" \
PROFILE="$PROFILE" \
PROMPT_SOURCE="$REPLAY_PROMPT_SOURCE" \
SERVER="$SERVER" \
RESULTS_DIR="$REPLAY_TRACE_RUN" \
STATIC_FIXED_S=70 \
CPU_THREADS="$BEST_CPU_THREADS" \
CMOE_BATCH=32 \
CMOE_UBATCH=32 \
CONTEXT=65536 \
TARGET_TYPE_K=q8_0 \
TARGET_TYPE_V=q8_0 \
EXPERT_TIMING=0 \
LOOKAHEAD_DISTANCE=1 \
LOOKAHEAD_TOP_M=16 \
LOOKAHEAD_POINT=post-moe \
LOOKAHEAD_NORM=target \
./scripts/bench_hybrid.sh
```

Compare the replay trace output and token hashes against every phase-matched L0 baseline. Any divergence means the trace is not observational and blocks further work.

## Replay predictor and Oracle policies

```bash
CAL_TRACE="$CAL_TRACE_RUN/LT-run1.lookahead-1.json"
REPLAY_TRACE="$REPLAY_TRACE_RUN/LT-run1.lookahead-1.json"
SIMULATION="$RESULT_ROOT/simulation-ctx64k-q8.json"

python3 tools/simulate_expert_streaming.py \
    --trace "$REPLAY_TRACE" \
    --calibration-trace "$CAL_TRACE" \
    --transport-json "$TRANSPORT_RUN/summary.json" \
    --layout-json "$TRANSPORT_RUN/model-layout.json" \
    --stats-json "$CPU_STATS" \
    --compute-json "$COMPUTE_RUN/summary.json" \
    --compute-timing latency \
    --output "$SIMULATION" \
    --top-m 8,12,16 \
    --arena global \
    --arena per-layer \
    --arena-slots 1,2,4 \
    --max-copies-per-layer 1,2 \
    --max-mib-per-token 8,16,32,64 \
    --mode full \
    --mode gate-up \
    --mode down
```

Omitting `--lead-ms` is intentional for the first replay. It produces an always-ready upper bound. It does not prove real overlap.

## Gates

Do not implement the transient runtime unless target measurements show all of the following:

```text
Oracle net saving: at least 5 percent of baseline decode time
held-out predictor net saving: at least 3 percent
useful byte ratio: at least 70 percent
late transfers: below 10 percent after real lead-time measurement
predictor plus ID handling: clearly below 1 ms/token
exposed copy wait: below 1.5 ms/token
global scratch for the first spike: at most 32 MiB
no global scheduler barrier per layer
```

The first simulation excludes predictor runtime, ID readback, real copy exposure, and CUDA contention. Passing it is necessary but not sufficient. If even the Oracle is below 5 percent, stop productive implementation and document the negative result.

If the upper bound passes, first measure a one-layer event bridge. Only after real predictor completion, ID availability, copy start/end, actual-router completion, event wait, and normal-graph interference are known may a one-layer routing-exact transient path be considered.

## Required one-layer invariants if the gate passes

```text
actual router IDs and weights remain authoritative
fixed, transient-ready, and CPU-fallback sets are disjoint
their union equals the actual expert set
pending bytes are never published as residency
late copies use CPU for the current invocation
false positives are not computed
every actual contribution is combined exactly once
owner layer, expert, generation, and completion event protect every slot
no MTP, adaptation, or warm cache in the first spike
feature default remains off
```

Do not add broad all-layer code first. Select one representative layer using measured CPU cost, resident GPU cost, predictor precision, and available lead.

## Secondary 32K comparison

After the 64K q8_0/q8_0 result is complete, repeat the phase-matched L0 and held-out LT collection with `CONTEXT=32768`. Reuse transport and resident-compute measurements only if the model, GPU, clocks, PCIe state, and binary are unchanged. The trace must be recollected because effective fixed residency can differ at 32K.

## MTP and quality remain separate

Do not enable MTP during this handoff. Do not claim that transient execution makes 64K q8_0/q8_0 practical from an Oracle result alone. MTP-2, KV quality, the private reasoning prompt, logit comparison, rollback, and long-run quality tests start only after a correct and faster no-MTP path exists.

## Stop conditions

Stop immediately on CUDA hang, NVIDIA Xid, OOM, invariant failure, output-hash divergence in trace-only mode, unexplained VRAM growth, more than 3 percent measured regression, mapped traffic slower than its CPU lower bound, or an Oracle result below the gate.

Do not commit model files, profiles, prompts, logs, responses, raw traces, or benchmark results. Preserve them under `RESULT_ROOT` for transfer back to the primary machine.

## Report back

Return these items separately for 64K and 32K:

1. Commit, compiler, CUDA, CPU, GPU, PCIe generation/width, and power state.
2. Requested and effective fixed slots.
3. Best CPU thread count and baseline decode TPS.
4. Per-size pinned H2D, staged H2D, mapped-read, and overlap measurements.
5. Early/middle/late resident GPU Gate+Up, Down, and Full timings.
6. Per-layer CPU cold timings and cold selections.
7. Held-out Recall@8/12/16 and weighted cold coverage.
8. Best global and per-layer predictor and Oracle policies.
9. Copied MiB/token, useful-byte ratio, ready recall, scratch MiB, and theoretical saving.
10. Whether the gate failed, requires an event-bridge measurement, or permits a one-layer prototype.

Keep quality conclusions and TPS conclusions separate.
