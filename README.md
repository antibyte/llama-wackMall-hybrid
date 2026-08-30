# llama-wackMall hybrid prototype

This branch keeps llama-wackMall as the execution engine and adds a profiled static expert tier, strict LuceBox Spark profile conversion, layer-variable placement, request-balanced profile collection, an experimental bounded warm cache, and guarded CPU/synchronization experiments. The normal llama.cpp CUDA path and wackMall MTP-1/2/3 decoding remain authoritative.

The safe defaults preserve existing wackMall behavior. Every experimental feature is controlled by an environment variable, so the same code can be tuned for cards with more VRAM without hard-coding GTX 1660 Ti limits.

## Current measured status

The fastest reproducible local configuration on an NVIDIA GTX 1660 Ti is MTP-2, a learned top-33 fixed placement, no warm slots, q4_0/q4_0 target and draft KV, and optional cold-row reuse. Three 2,000-token runs with row reuse reached 46.593, 46.508, and 46.335 token/s, for a median of 46.508 token/s. The matching control median was 46.362 token/s, so row reuse remains default-off because the gain is only 0.31% and can be hardware dependent.

MTP-3 is implemented and tested, but it was slower than MTP-2 on this machine. The warm cache is correct and remains available for larger GPUs, but W=1/2/4 caused copy churn and was slower on the 6-GiB card. Do not interpret either result as a limit for other GPUs; use the reproducible sizing and benchmark procedure below.

See [HYBRID_EXPERIMENTS.md](HYBRID_EXPERIMENTS.md) for measured runs and rejected approaches, [HYBRID_DESIGN.md](HYBRID_DESIGN.md) for flags and invariants, [HYBRID_ANALYSIS.md](HYBRID_ANALYSIS.md) for the architecture comparison, and [HYBRID_ATTRIBUTION.md](HYBRID_ATTRIBUTION.md) for LuceBox attribution.

## Repository contents

| Path | Purpose |
|---|---|
| `tools/convert_luce_spark_profile.py` | Strict Spark-to-wackMall profile conversion with GGUF validation |
| `tools/aggregate_expert_profiles.py` | Prompt-balanced aggregation of request-local usage profiles |
| `tools/optimize_expert_placement.py` | Exact byte-budget optimizer for layer-variable static placement |
| `tools/bench_expert_transport.py` | Model-derived repeated H2D and mapped-memory benchmark wrapper |
| `tools/bench_expert_compute.py` | Repeated early/middle/late resident GPU compute wrapper |
| `tools/simulate_expert_streaming.py` | Routing-trace Oracle and predictor replay under measured costs |
| `tools/expert-transport-bench/` | CUDA transport, staging, mapped-read, and overlap microbenchmark |
| `tools/expert-compute-bench/` | Resident per-expert GPU compute microbenchmark |
| `tools/turboquant-ref/` | Offline Turbo3/Turbo4 codec and raw float32 row quality analyzer |
| `tools/turboquant-capture/` | Deterministic post-RoPE Q/K capture for offline codec evaluation |
| `tools/turboquant-ref/simulate-triattention.py` | Non-mutating KV eviction and causal heavy-hitter simulator |
| `IK_LLAMA_EXPERIMENTS.md` | ik_llama concept audit, measured gates, and rejected imports |
| `GTX1080_HANDOFF.md` | Native SM 61 build, measurement, replay, and gate procedure |
| `scripts/collect_expert_profiles.sh` | Reproducible profile collection over the included JSONL corpus |
| `scripts/bench_hybrid.sh` | Fresh-process warm-up and repeated 2,000-token benchmark runner |
| `tests/test-expert-warm-cache.cpp` | LRU slot, in-flight copy, LUT, and sentinel state tests |
| `tests/test-expert-placement.cpp` | Placement manifest validation tests |

Generated builds, local profiles, model files, traces, and `benchmark-results/` are intentionally excluded by `.gitignore`.

TurboQuant Phase 1 is the offline reference and capture suite. Later phases
add experimental, default-off SM75 target Turbo4 K and target Turbo4 V paths.
Neither is enabled by the stable start scripts; MTP requires an additional
explicit guard. See
[`TURBOQUANT_ANALYSIS.md`](TURBOQUANT_ANALYSIS.md),
[`TURBOQUANT_DESIGN.md`](TURBOQUANT_DESIGN.md), and
[`TURBOQUANT_EXPERIMENTS.md`](TURBOQUANT_EXPERIMENTS.md) before attempting a
runtime use.

Build the phase-1 tools without changing the inference runtime:

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
    --target llama-turboquant-ref test-turboquant-ref
```

The capture executable needs the ordinary native CUDA build. Always run an
identical no-capture control first. The capture directory must not exist:

```bash
CAPTURE=/tmp/turboquant-qk-capture-01

./build-hybrid/bin/llama-turboquant-capture \
    -m "$MODEL" -f "$PROMPT" -c 4096 -b 128 -ub 128 \
    -ctk q8_0 -ctv q8_0 -n 8 --seed 1234 --temp 0 \
    --capture-dir "$CAPTURE" --capture-max-tokens 256

python3 tools/turboquant-ref/analyze-capture.py \
    --capture-dir "$CAPTURE" \
    --analyzer build-hybrid/bin/llama-turboquant-ref \
    --output /tmp/turboquant-qk-metrics.json
```

Compare the generated token IDs and FNV-1a hash printed by control and capture.
The analyzer validates every manifest dimension and file size, measures K-only
reconstruction, and evaluates each captured query against all causally visible
keys. Raw captures and metric JSON files are local artifacts and must not be
committed.

For attention-output diagnostics, add `--capture-values`. This enables a
capture-only materialized pre-attention V tensor; the ordinary graph remains
unchanged when the option is absent. Always compare its token hash with both
the no-capture and Q/K-only controls.

Long captures can also be evaluated without modifying the KV cache. The
Qwen3.6 model used here rotates only 64 of its 256 head dimensions, so
`--rope-dims 64` is required. NumPy is the only additional dependency.

```bash
python3 tools/turboquant-ref/simulate-triattention.py \
    --capture-dir /tmp/turboquant-qk-long \
    --output /tmp/kv-eviction-simulation.json \
    --train-tokens 4096 --eval-tokens 128 \
    --history-tokens 128 --recent-window 128 \
    --budgets 1024,2048,3072 \
    --rope-theta 10000000 --rope-dims 64 \
    --attention-output
```

The output compares recency, norm-only selection, Atomic-style
TriAttention scoring, a causal past-attention heavy-hitter score, and a
future-looking oracle. The oracle is an unattainable upper bound. With
`--attention-output`, a capture containing V tensors also reports normalized
RMSE, relative L2 error, and cosine similarity for the reconstructed attention
output after renormalizing the retained positions. This remains an offline
F32 proxy, not a downstream-logit or quality result. No physical KV eviction
is enabled by this tool.

Recursive JSON also includes per-layer metrics plus
`squared_error_sum`/`squared_reference_sum`. These allow selective-layer error
to be combined without incorrectly averaging layer RMSE values.

For the replay-only logit gate, first create a capture with an exact prompt
prefix and raw final logits:

```bash
./build-hybrid/bin/llama-turboquant-capture \
    -m "$MODEL" -f "$PROMPT" -c 2560 -b 256 -ub 256 \
    -ctk q8_0 -ctv q8_0 -fa on -cmoe --offline -n 1 \
    --prompt-max-tokens 2304 --capture-max-tokens 2304 --capture-values \
    --capture-dir /tmp/qkv-prefix --dump-final-logits /tmp/baseline.f32

python3 tools/turboquant-ref/simulate-triattention.py \
    --capture-dir /tmp/qkv-prefix --output /tmp/replay-simulation.json \
    --train-tokens 2176 --eval-tokens 128 --history-tokens 128 \
    --recent-window 128 --budgets 2048 --rope-dims 64 \
    --attention-output --export-replay-dir /tmp/replay-layer39 \
    --export-replay-layer 39 --export-replay-policy history_attention_global
```

Run the same prefix with `--attention-replay /tmp/replay-layer39` and a new
`--dump-final-logits` path, then compare it with:

```bash
python3 tools/turboquant-ref/compare-logits.py \
    --reference /tmp/baseline.f32 --candidate /tmp/replay.f32 \
    --output /tmp/logit-comparison.json
```

For a multi-position gate without changing the 256-token batch geometry, dump
the final 128 prompt rows in both runs:

```bash
--dump-logits-window /tmp/baseline-window.f32 --logits-window-start 2176
```

The tool writes a guarded JSON sidecar with row count, vocabulary width,
prompt hash, and token offset. `compare-logits.py` detects the sidecars and
reports per-row plus aggregate L2, KL, Top-1, and Top-10 metrics. The simulator
also exports `attention-keep.u8`. It can be tested against actual Q8/Q8
FlashAttention with the diagnostic-only option:

```bash
--attention-live-mask /tmp/replay-layer39 \
--dump-logits-window /tmp/live-window.f32 --logits-window-start 2176
```

Add `--attention-live-mask-continue` only for a no-MTP, single-sequence
autoregressive diagnostic. It reapplies the same fixed old-prefix mask to each
later decode graph and verifies the number of callback applications. This
synchronous mode is for quality comparison, not throughput measurement.

Replay and live-mask modes are mutually exclusive and confined to this
diagnostic executable. The live mask duplicates the selected layer's causal
mask and only adds `-inf` for exported evictions; it does not delete or compact
KV entries and is not available in `llama-server`. The manifest binds model
path and size, prompt-token hash, context, batch, ubatch, and cache types.
Mismatches abort before prompt evaluation. A zero-delta replay must be
byte-identical to its baseline.

Add `--recursive` to model repeated pruning where an evicted position cannot
return. In recursive mode the first prune occurs at `budget + recent-window`
and evaluation continues in `recent-window` increments. `--recursive-max-events`
provides a bounded smoke test.

```bash
python3 tools/turboquant-ref/simulate-triattention.py \
    --capture-dir /tmp/turboquant-qk-long \
    --output /tmp/heavy-hitter-recursive.json \
    --recursive --budgets 2048 \
    --history-tokens 128 --recent-window 128
```

The Phase 2 runtime is an explicit diagnostic only:

```bash
./build-hybrid/bin/llama-turboquant-capture \
    -m "$MODEL" -p "fft in c" -c 4096 -b 128 -ub 128 \
    -ctk turbo4_k -ctv q8_0 -fa on -cmoe \
    --offline --seed 1234 --temp 0 -n 128
```

Do not use `turbo4_k` for draft KV, FlashAttention-off operation, or CPU KQV.
Target V is a separate experiment and requires
`LLAMA_TURBO4_V_EXPERIMENTAL=1`. Target Turbo4 with the existing draft-mtp
path additionally requires `LLAMA_TURBO4_MTP_EXPERIMENTAL=1`. Target Turbo4
with draft-dflash instead requires `LLAMA_TURBO4_DFLASH_EXPERIMENTAL=1`;
other speculative methods stay rejected. Use a fresh Q8/Q8 control process
with identical inputs and compare hashes, quality, acceptance, and median timing.
An optional faster prompt/verify path can be built with:

```bash
cmake -S . -B build-turbo-sm75 -G Ninja \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DGGML_CUDA_TURBO4_F16_PREFILL=ON \
    -DGGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH=32
```

This option uses existing FP16 tile/MMA attention for larger prompt batches,
while stored K remains Turbo4 and decode remains on the float-centroid vector
path. The configurable minimum batch defaults to 32 so small prompt fragments
stay on the native Turbo4 path. A build with this option can override the
crossover per process with `GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH`; values
must be measured per GPU and MTP width. It improved
the local 2,747-token Prompt-TPS median by 0.40% over Q8/Q8. Target-only MTP-2
has completed repeatable 32K and 64K tests plus 2,000-token long runs, but this
is still not a production recommendation: broader quality data, rollback/state
tests, and SM61 validation remain outstanding. Do not enable
`GGML_CUDA_TURBO4_KQ_DP4A`; it did not improve SM75 throughput.

Run the no-MTP perplexity and KL quality matrix on an external representative
corpus with:

```bash
QUALITY_PROMPT_SOURCE=/absolute/path/to/corpus.txt \
PPL="$PWD/build-turbo-sm75/bin/llama-perplexity" \
MODEL="$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
PROFILE="$HOME/models/qwen3.6-35b-a3b-mtp/luce-warmstart.csv" \
TURBO4_Q8_FALLBACK_LAYERS="23,35,27" \
scripts/bench_turboquant_quality.sh
```

The runner refuses to overwrite results and compares Q8/Q8, Q8/Q4, Q4/Q8,
Q4/Q4, Turbo4/Q8, Q8/Turbo4, and Turbo4/Turbo4 against one saved Q8/Q8 logit
reference. Its default 1,024-token context requires a corpus that tokenizes to
at least 2,048 tokens. If `TURBO4_Q8_FALLBACK_LAYERS` is non-empty, it also
evaluates that mixed Turbo4/Q8-layer policy. The example IDs are local
measurements, not defaults.

An experimental per-layer Q8 K fallback can be selected only with Turbo4 K:

```bash
LLAMA_TURBO4_Q8_FALLBACK_LAYERS="23,35" \
    ./build-turbo-sm75/bin/llama-cli ... -ctk turbo4_k -ctv q8_0
```

The value accepts comma-separated layer IDs and inclusive ranges such as
`3,7,11-15`. It is empty by default, validates every ID against the loaded
model, and currently rejects cache sharing or layer reuse. Layer choices must
come from measured quality data; the example IDs are not a built-in policy.

For the first guarded MTP smoke test, compress only the target K cache. The
draft context retains its separately configured Q8/Q4 cache:

```bash
LLAMA_TURBO4_MTP_EXPERIMENTAL=1 \
LLAMA_TURBO4_Q8_FALLBACK_LAYERS="23,35,27" \
GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH=2 \
    ./build-turbo-sm75/bin/llama-server \
    ... -ctk turbo4_k -ctv q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q4_0
```

Draft Turbo4 is separately guarded by `LLAMA_TURBO4_DRAFT_EXPERIMENTAL=1`.
The runtime crossover of 2 made MTP-2 and MTP-3 hash-identical in the local short geometry screen;
MTP-1 remained numerically distinct. MTP-2 was fastest. At 64K, three
512-token runs reached a 39.077 token/s median versus 38.416 for Q8/Q8 while
retaining S=27 instead of S=25, and a single 2,000-token pair favored Turbo4
38.241 versus 37.425 token/s. Keep the flag absent from production scripts
until rollback/state and broader long-context quality tests pass.

For the symmetric target-cache experiment:

```bash
LLAMA_TURBO4_V_EXPERIMENTAL=1 \
LLAMA_TURBO4_MTP_EXPERIMENTAL=1 \
LLAMA_TURBO4_DRAFT_EXPERIMENTAL=1 \
GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH=2 \
    ./build-turbo-sm75/bin/llama-server \
    ... -ctk turbo4_k -ctv turbo4_k \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    --spec-draft-type-k turbo4_k --spec-draft-type-v turbo4_k
```

The DFlash path supports Turbo4 on the target and, with an extra guard, on the
draft sidecar as well. Q4_0 draft KV remains the measured SM75 throughput
default; draft Turbo4 is a capacity experiment:

```bash
LLAMA_TURBO4_V_EXPERIMENTAL=1 \
LLAMA_TURBO4_DFLASH_EXPERIMENTAL=1 \
LLAMA_TURBO4_DRAFT_EXPERIMENTAL=1 \
GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH=2 \
    ./build-main-sm75/bin/llama-server \
    ... -ctk turbo4_k -ctv turbo4_k \
    --spec-type draft-dflash --spec-draft-model "$DFLASH_MODEL" \
    --spec-draft-n-max 2 \
    --spec-draft-type-k turbo4_k --spec-draft-type-v turbo4_k
```

`LLAMA_TURBO4_DFLASH_EXPERIMENTAL=1` enables Turbo4 for the target with
draft-dflash. Draft Turbo4 additionally needs `LLAMA_TURBO4_DRAFT_EXPERIMENTAL=1`
(same flag as MTP draft Turbo4).

Draft Turbo4 uses the same exact-writing and Flash-Attention implementation as
the target cache, but the draft context contains only the single NextN layer.
On the GTX 1660 Ti it saved only 6 MiB at 32K. In a phase-matched 3x512 MTP-2
screen, Q4/Q4 draft KV reached a 44.507 token/s median with 62.20% acceptance,
while Turbo4/Turbo4 reached 40.476 token/s with 52.72% acceptance. All output
and token hashes remained identical. It is therefore a guarded capacity
experiment and not the SM75 throughput winner; Q4/Q4 remains the recommended
draft cache when speed matters.

On the local GTX 1660 Ti, Turbo4/Turbo4 retained S=33 instead of Q8/Q8 S=30
at 32K, but the 2,000-token rates were effectively tied (39.634 versus
39.541 token/s). At 64K it retained S=30 instead of S=25; the three-run
512-token median was 39.956 versus 38.416 token/s, and one 2,000-token run was
38.852 versus 37.425. The small quality corpus measured +1.58% PPL, mean KLD
0.02801, and 91.585% same-top-token agreement relative to Q8/Q8. This is a
capacity/performance option with a measurable quality cost, not a new default.
The 32K/64K figures reserve those context sizes but do not fill them with an
active long prompt. Native SM61 and active-long-context validation remain.

Two additional ik_llama-inspired experiments are available but remain
default-off:

```bash
# Low-perplexity Q4_0 block-scale experiment. Values: legacy, weighted,
# weighted-k, weighted-v. Production-compatible default: legacy.
LLAMA_KV_Q4_SCALE=weighted-k

# Create a smaller additional output matrix for MTP only. Values: none, q4_K,
# q5_K. The original output remains authoritative for the target graph.
LLAMA_MTP_REQUANTIZE_OUTPUT=q4_K
```

The MTP-only tensor consumes additional VRAM; lower fixed expert slots if
allocation fails. On the GTX 1660 Ti its isolated gain was below one percent
and the lost expert residency made it unattractive. See
[`IK_LLAMA_EXPERIMENTS.md`](IK_LLAMA_EXPERIMENTS.md) for quality metrics,
three-run medians, and the native-SM61 test gate. Neither variable is enabled
by the stable start scripts.

## 1. Requirements

The tested NVIDIA build needs Git, CMake, Ninja, GCC/G++ 12, Python 3, curl, the CUDA toolkit, and a compatible NVIDIA driver. On Pop!_OS or Ubuntu, install the ordinary build tools with:

```bash
sudo apt update
sudo apt install -y git cmake ninja-build build-essential gcc-12 g++-12 python3 curl
```

Install the CUDA toolkit using the packaging method appropriate for the host, then verify the driver and compiler before configuring:

```bash
nvidia-smi
nvcc --version
```

The benchmark monitor also requires `nvidia-smi`, `ps`, `sha256sum`, and curl. No Python packages outside the standard library are required by the hybrid tools.

## 2. Clone and build

Clone your fork into a new directory and keep model files outside the repository:

```bash
git clone https://github.com/YOUR_ACCOUNT/llama-wackMall.git llama-wackMall-hybrid
cd llama-wackMall-hybrid
git switch codex/hybrid-profile-placement
```

Use the native CUDA architecture for the target GPU. The GTX 1660 Ti uses SM 75:

```bash
cmake -S . -B build-hybrid -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
    -DLLAMA_BUILD_TESTS=ON

cmake --build build-hybrid -j 8 --target llama-server llama-cli
```

For another NVIDIA GPU, replace `75` with its native SM architecture and retain a separate build directory for each architecture. This project does not force `GGML_CUDA_FORCE_MMQ`; it was slower on SM 75.

## 3. Validate local inputs

Set paths explicitly and verify them before conversion or inference:

```bash
MODEL="$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
LUCE_MODEL="$HOME/models/qwen3.6-35b-a3b-luce/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
SPARK="$LUCE_MODEL.spark.csv"

test -f "$MODEL"
test -f "$LUCE_MODEL"
test -f "$SPARK"
```

Never commit `.gguf` files, private prompts, raw request profiles, or benchmark responses. The repository ignores standard local locations, but review `git status` before every commit.

## 4. Convert a LuceBox Spark profile

Create a local output directory and run the strict converter. It refuses to overwrite an existing output, validates layer/expert dimensions and model metadata, and reports conversion statistics:

```bash
mkdir -p local-profiles

python3 tools/convert_luce_spark_profile.py \
    --input "$SPARK" \
    --output local-profiles/luce-warmstart.csv \
    --model "$MODEL" \
    --source-model "$LUCE_MODEL"
```

Raw lifetime counts can dominate online adaptation. For a static top-33 placement seed, request the policy explicitly:

```bash
python3 tools/convert_luce_spark_profile.py \
    --input "$SPARK" \
    --output local-profiles/luce-placement-s33.csv \
    --model "$MODEL" \
    --source-model "$LUCE_MODEL" \
    --placement-slots 33
```

The Luce profile is only a warm start. A profile learned from the intended workload is normally a better production choice.

## 5. Learn a request-balanced profile

The included corpus covers several workload categories. The collector starts a fresh server for each selected prompt, writes session-only expert usage, validates it against JSON statistics, and normalizes every request before aggregation:

```bash
PROFILE="$PWD/local-profiles/luce-placement-s33.csv" \
MODEL="$MODEL" \
OUTPUT_DIR="$PWD/benchmark-results/profile-corpus-general" \
N_PREDICT=256 \
MTP_N=2 \
FIXED_S=33 \
CPU_THREADS=8 \
DRAFT_THREADS=8 \
./scripts/collect_expert_profiles.sh

GENERAL_PROFILE="$PWD/local-profiles/general-profile.csv"
if [[ -e "$GENERAL_PROFILE" ]]; then
    echo "Output already exists and was not overwritten: $GENERAL_PROFILE" >&2
else
    cp benchmark-results/profile-corpus-general/general-profile.csv "$GENERAL_PROFILE"
fi
```

Use `PROFILE_IDS=id1,id2,...` to build a category-specific profile from entries in `scripts/hybrid_profile_corpus.jsonl`. Use separate outputs such as `general-profile.csv`, `coding-profile.csv`, and `tools-profile.csv`, then select one only at server startup or between requests. Do not swap placement during an active MTP verification batch.

## 6. Run the stable MTP-2 configuration

This is the tested 6-GiB recipe. It uses eight physical CPU workers because the Ryzen 7 4800H has eight physical cores; seeing fewer than 16 logical CPUs fully occupied is not by itself an error. Test thread counts on the target host instead of optimizing for CPU utilization percentage.

```bash
PROFILE="$PWD/local-profiles/general-profile.csv"

CUDA_VISIBLE_DEVICES=0 \
LLAMA_CMOE_BATCH=376 \
LLAMA_CMOE_UBATCH=376 \
LLAMA_EXPERT_HOT="$PROFILE" \
LLAMA_EXPERT_S=33 \
LLAMA_EXPERT_ADAPT=0 \
LLAMA_EXPERT_WARM_SLOTS=0 \
LLAMA_EXPERT_STATIC_NO_SYNC=1 \
LLAMA_EXPERT_STATS=0 \
LLAMA_EXPERT_STATS_JSON=0 \
LLAMA_EXPERT_USAGE=0 \
LLAMA_EXPERT_CPU_REUSE_ROWS=1 \
./build-hybrid/bin/llama-server \
    -m "$MODEL" \
    --load-mode mmap \
    -c 32768 \
    -ctk q4_0 \
    -ctv q4_0 \
    -fa on \
    -cmoe \
    -np 1 \
    --threads 8 \
    --threads-batch 8 \
    --no-mmproj \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --spec-draft-ngl auto \
    --spec-draft-type-k q4_0 \
    --spec-draft-type-v q4_0 \
    --spec-draft-threads 8 \
    --spec-draft-threads-batch 8 \
    --reasoning-budget 16384 \
    --ctx-checkpoints 0 \
    --cache-ram 0 \
    --jinja \
    --offline \
    --host 127.0.0.1 \
    --port 8080 \
    -a qwen3.6-35b-a3b-hybrid
```

Set `LLAMA_EXPERT_CPU_REUSE_ROWS=0` for the conservative cross-platform behavior. Static no-sync activates only when adaptation, warm slots, statistics, timing, and usage output are all disabled; the server log must contain `expert_tier: static no-sync enabled`.

`LLAMA_CMOE_BATCH` and `LLAMA_CMOE_UBATCH` are maximum graph sizes, not
forced decode widths. A 1--3 token MTP decode still uses its actual small
graph, while a long prompt can use the larger physical ubatch. On the tested
6-GiB GTX 1660 Ti, `376/376` is the largest measured value that retains all 33
fixed slots with the default 512-MiB expert VRAM reserve. `512/512` is the
optional TTFT-oriented setting; auto-fit safely reduces it to 32 fixed slots.
Do not hard-code these values for another GPU: sweep physical ubatch sizes and
verify the logged effective fixed-slot count, peak VRAM, long decode throughput,
and output stability. Larger-memory GPUs should include 512, 1024, and the
upstream 2048/512 logical/physical defaults in that sweep.

An experimental, default-off `-np 1` phase cap is available for controlled
testing:

```bash
LLAMA_CMOE_BATCH=32 \
LLAMA_CMOE_UBATCH=32 \
LLAMA_CMOE_PREFILL_BATCH=128 \
LLAMA_CMOE_PREFILL_UBATCH=128 \
LLAMA_CMOE_DECODE_BATCH=32 \
LLAMA_CMOE_DECODE_UBATCH=32 \
./build-hybrid/bin/llama-server ...
```

The context still reserves the maximum pair, here 128/128. The server limits
prompt submission to the prefill pair and switches the runtime ubatch cap at
the existing prompt-to-generation boundary. This is not a production
recommendation: MTP decode already uses its actual small token count, and the
GTX 1660 Ti test did not recover decode throughput relative to global 128/128.
The flags are rejected as an active optimization for `-np > 1` because mixed
prefill/decode requests need an explicit scheduling policy.

Change `--spec-draft-n-max` to 1 or 3 for MTP-1/MTP-3, or remove all `--spec-*` options for a no-MTP control. Always compare token/output hashes and MTP acceptance, not throughput alone.

The benchmark runner can screen the already-supported self-speculative chain
without changing the production launcher. For example,
`SPEC_TYPES_OVERRIDE=ngram-mod,draft-mtp` enables N-gram fallback ahead of
MTP; `NGRAM_MOD_N_MATCH`, `NGRAM_MOD_N_MIN`, and `NGRAM_MOD_N_MAX` bound it.
The GTX 1660 Ti screens did not find a winner, so `start.sh` intentionally
keeps MTP-2 alone.

### Experimental hot-ID and MoE kernel fusion

Two default-off controls remove repeated hot-slot lookups and fuse quantized Gate+Up+GLU work for multi-token MoE verification graphs:

```bash
LLAMA_EXPERT_SHARED_HOT_IDS=1 \
GGML_CUDA_MOE_MULTI_FUSION=1 \
GGML_CUDA_MOE_COMBINE_FUSION=1 \
./build-kernel-sm75/bin/llama-server ...
```

`LLAMA_EXPERT_SHARED_HOT_IDS=1` reuses one mapped hot-ID tensor for Gate, Up, and Down within a layer. `GGML_CUDA_MOE_MULTI_FUSION=1` extends the existing quantized CUDA fusion to bias- and scale-free `MUL_MAT_ID` graphs with two to four target tokens. The multi-token kernel is guarded off on Pascal and older GPUs until separate architecture-specific measurements are available. Keep both controls disabled for an unmodified baseline.

`GGML_CUDA_MOE_COMBINE_FUSION=1` is a separate default-off experiment. It
replaces the post-Down F32 weighting plus the ordered Top-k reduction with one
CUDA kernel while retaining the exact router weights and addition order. It is
model-dimension independent (2--32 routed experts) and has an optimized Top-8
dispatch. The GTX 1660 Ti 3x256 screen was neutral (+0.07%), so it is not a
production default and needs an independent sm_61 screen before use on the GTX
1080.

### Experimental Q8_0 three-column MMVQ geometry

Node-level Nsight Systems profiling of the MTP-2 decode window identified the
dense Q8_0 three-column matrix-vector kernel as the largest GPU-kernel category
(25.1% of measured decode GPU time). The optional setting below changes only
how many output rows one CUDA block evaluates; it does not alter the dot-product
or reduction order:

```bash
GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=4 \
./build-kernel-sm75/bin/llama-server ...
```

Accepted values are `1`, `2`, and `4`; unset or `0` keeps the upstream-selected
geometry (`2` on the tested path). On the GTX 1660 Ti, three 2,000-token MTP-2
runs improved from a 46.831 tok/s median to 47.436 tok/s (+1.29%). All runs had
identical output/token hashes, MTP acceptance (0.780269), and mean accepted
length (2.56). One row per block regressed by about 4.1% in the short screen.
The override remains default-off and must be benchmarked independently on
Pascal; successful sm_61 compilation is not a GTX 1080 performance result.

The same profiler showed that the Q6_K output projection consumed another
18.3% of decode GPU-kernel time across one- and three-column launches. These
paths can be tuned independently:

```bash
GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2 \
GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=4 \
./build-kernel-sm75/bin/llama-server ...
```

With the Q8_0 four-row winner already enabled, the Q6_K pair improved the
GTX 1660 Ti three-run, 2,000-token MTP-2 median from 47.293 to 47.794 tok/s
(+1.06%). All output/token hashes, acceptance (0.780269), and mean accepted
length (2.56) remained identical. Short no-MTP, MTP-1, MTP-2, and MTP-3
off/on checks were also hash-identical. Both settings remain default-off and
architecture-specific; test `1`, `2`, and `4` independently on each GPU.

### Experimental target backend sampling

This tree contains a semantic port of llama.cpp PR #25532.  Add `-bs` to keep
target sampling on the backend for all output rows of an MTP verification
graph.  The existing `--spec-draft-backend-sampling` flag controls only the
draft sampler and is independent:

```bash
./build-backend-sampling-sm75/bin/llama-server \
    ... \
    -bs \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --reasoning-budget 256
```

The local reasoning-budget bridge preserves exact forced closing tokens even
when the budget expires in the middle of a multi-token MTP verification step.
It was checked with greedy and seeded stochastic sampling.  Grammar-constrained
requests still fall back to CPU sampling.  The feature remains default-off
because the upstream PR is open and the measured GTX 1660 Ti gain is modest:
1.24% for MTP-2 and 1.05% for MTP-3 over three 2,000-token runs.  MTP-2 remains
the faster mode on this card.

The benchmark runner exposes the two relevant controls without changing its
old defaults:

```bash
TARGET_BACKEND_SAMPLING=1 \
REASONING_BUDGET=256 \
MTP_OVERRIDE=2 \
./scripts/bench_hybrid.sh SB
```

See `HYBRID_EXPERIMENTS.md` for hashes, acceptance, VRAM, exact test settings,
and the GTX 1080 follow-up recommendation.

For a quality-oriented 32K configuration, target `q8_0/q4_0` is a tested
alternative to the fastest `q4_0/q4_0` recipe while the MTP draft cache remains
q4_0/q4_0:

```bash
-ctk q8_0 -ctv q4_0 \
--spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
--reasoning-budget 256 \
-n 4096
```

On the local GTX 1660 Ti this retained S=33 and improved the generated
`"fft in c"` implementation from a non-compiling/truncated q4 result to an
executing exact FFT/IFFT roundtrip under GNU-C99.  The comparable short screen
fell from 45.29 to 42.04 tok/s (-7.2%).  Strict C99 still exposed a generated
`M_PI` portability defect, so this is evidence for a quality preference, not a
claim that generated code no longer needs validation.  Larger GPUs should also
test target q8_0/q8_0; on the 6-GiB card it auto-fitted the fixed tier down from
S=33 to S=30.  The deterministic quality screen used a 64-token reasoning
budget.  Production currently uses 256 tokens as a quality-oriented compromise:
64 was visibly too terse in interactive use, while 1,024 had already started
drafting final code inside the reasoning block before the forced transition.
Clients can still request a different reasoning budget explicitly.

### Safe dynamic learning mode

Dynamic adaptation uses one stable fixed-tier mapping for an entire request.
Counts and scores are still harvested after every completed graph, but at most
one fixed expert per layer is published at the next request boundary. Target
and MTP draft contexts share a graph guard, so no graph can observe a partial
weight/LUT/mask transaction.

In server mode, collection is request-scoped.  Internal target/MTP context
warm-up graphs are cleared before the first real request, so restarts do not
pollute a persistent usage profile.  Look for two startup log messages that
discard 640 pre-request selections each on this 40-layer Top-8 model.  CLI and
direct libllama callers keep context-lifetime collection semantics.

CUDA graph capture/reuse is automatically disabled while mutable expert weights
are enabled. Its interaction with fixed-slot copies reproducibly caused
`cublasSgemm` to abort on the next request. Do not set
`LLAMA_EXPERT_ADAPT_CUDA_GRAPHS=1` outside controlled diagnostics.

Use a distinct output path for the learned cumulative profile. It is written
through a temporary file and atomically renamed after every completed or
cancelled server request, then written once more at orderly shutdown:

```bash
SEED="$PWD/local-profiles/luce-placement-s33.csv"
LEARNED="$PWD/local-profiles/workload-learned.csv"

LLAMA_EXPERT_HOT="$SEED" \
LLAMA_CMOE_BATCH=376 \
LLAMA_CMOE_UBATCH=376 \
LLAMA_EXPERT_S=33 \
LLAMA_EXPERT_ADAPT=1 \
LLAMA_EXPERT_ADAPT_INTERVAL=request \
LLAMA_EXPERT_WARM_SLOTS=0 \
LLAMA_EXPERT_STATIC_NO_SYNC=0 \
LLAMA_EXPERT_USAGE="$LEARNED" \
LLAMA_EXPERT_USAGE_MODE=cumulative \
LLAMA_EXPERT_USAGE_CHECKPOINT=request \
./build-hybrid/bin/llama-server ...
```

On the next server start, use `LLAMA_EXPERT_HOT="$LEARNED"` to continue from
the saved counts. Never enable static no-sync together with adaptation.

Run the single-process multi-request regression before deploying another GPU,
MTP width, slot count, or profile:

```bash
PROFILE="$SEED" \
MODEL="$MODEL" \
SERVER="$PWD/build-hybrid/bin/llama-server" \
CMOE_BATCH=376 \
CMOE_UBATCH=376 \
MTP_N=2 \
REQUESTS=4 \
N_PREDICT=128 \
./scripts/test_adaptive_multi_request.sh
```

### Optional RAM prompt cache for switching conversations

The model's hybrid KV layout cannot shift, so the server disables
`--cache-reuse`.  The bounded RAM prompt cache is still useful with one live
slot when a client switches between conversations and later returns.  The
tested configuration is:

```bash
./build-hybrid/bin/llama-server \
    ... \
    -np 1 \
    --slot-prompt-similarity 0.70 \
    --ctx-checkpoints 4 \
    --checkpoint-min-step 1024 \
    --cache-ram 2048 \
    --no-cache-idle-slots
```

The similarity threshold is important for one-slot branch switching.  It
causes a sufficiently divergent prompt to save the current slot into the RAM
bank before overwriting it.  `--cache-idle-slots` alone did not save the only
slot because it was already selected for the next request.  On the local
A -> B -> A test, restored A processed 4 rather than about 3,500 prompt tokens,
cutting TTFT from about 48.1 seconds to 0.39 seconds.  A1/A3 token and output
hashes matched with dynamic adaptation in both MTP-2 and MTP-3.

This does not accelerate the first unseen prompt or sustained decoding.  A
linear continuation normally reuses the live slot already; the RAM bank helps
when returning to an evicted conversation.  Size `--cache-ram` for available
system RAM, and benchmark the save/load overhead and hit rate on the actual
client workload.  Reproduce the branch test with:

```bash
PROFILE="$LEARNED" \
SERVER="$PWD/build-hybrid/bin/llama-server" \
CONFIGS=ram_forced \
ADAPT=1 \
MTP_N=2 \
./scripts/test_prompt_cache.sh
```

### LAN/OpenWebUI launcher

Fixed GPU references:

- [`start1660.sh`](start1660.sh) — GTX 1660 Ti (sm_75, 6 GiB) live stack; snapshot in [`START1660_REFERENCE.md`](START1660_REFERENCE.md) (45.04 tok/s / 3781 tok, peak 3s 57.48)
- [`start-ling-tiny.sh`](start-ling-tiny.sh) — Ling-3.0-tiny on the 1660 Ti; same knob surface as `start1660.sh` (KVFlash 8192, prefill 2048, ngram-simple, q8 KV)
- [`start-ling-tiny-1080.sh`](start-ling-tiny-1080.sh) — Ling-3.0-tiny starting recipe for GTX 1080 (sm_61, 8 GiB)
- [`start1080.sh`](start1080.sh) — measured GTX 1080 (sm_61, 8 GiB) production stack
- [`start.sh`](start.sh) — auto-generated baseline for the current machine
- [`start-turbollm.sh`](start-turbollm.sh) — TurboLLM with both GPU profiles registered (see [`tools/turbollm/README.md`](tools/turbollm/README.md))

The measured Ling launcher uses MLA K-only KVFlash, 2048/64 phase batching,
Q8 KV, and a four-row Q8_0 single-column MMVQ schedule. On the GTX 1660 Ti,
KVFlash plus phase batching raised long-prompt throughput from 481 to 592
tokens/s (+23%). The Ling grouped-router `TOP_K` path uses the ordered warp
kernel on CUDA toolkits without CUB `DeviceTopK`; together with the Q8_0
schedule, the measured decode uplift is about 2.1% with identical token hashes.

TurboLLM (UI + OpenAI API on port 6996):

```bash
./start-turbollm.sh
# Engines: Hybrid GTX 1660 Ti  |  Hybrid GTX 1080
# Then Load the MTP Qwen3.6-35B-A3B GGUF. Do not also run start1660.sh / start1080.sh.
```

Generate/refine `start.sh` with hardware detect + optional timed search:

```bash
python3 tools/hybrid_autotune/autotune.py detect
python3 tools/hybrid_autotune/autotune.py generate -y
python3 tools/hybrid_autotune/autotune.py optimize --mode quick   # ~10 min
# or: ./autotune
```

See [`tools/hybrid_autotune/README.md`](tools/hybrid_autotune/README.md).

The project-root launcher [`start.sh`](start.sh) contains the complete
environment configuration for the server, MTP, backend sampling, expert
tiering, warmcache, and lookahead controls. It binds to `0.0.0.0:8080` and
auto-detects the tested batch starting point (`376/376` on `sm_75`, `128/128`
on `sm_61`). The expert slot count is left to VRAM auto-fit so the same file
can be used on the GTX 1660 Ti, GTX 1080, and larger cards.

```bash
cd /path/to/llama-wackMall-hybrid
# edit the CONFIGURATION block in start.sh first
./start.sh
```

The complete list of parameters is in the editable block at the top of the
script, including `MODEL`, `PROFILE`, `HOST`, `PORT`, `MTP_N`, global and
phase-specific CMoE batches, `THREADS`, `EXPERT_S`,
`TARGET_BACKEND_SAMPLING`, `REASONING_BUDGET`, every `LLAMA_EXPERT_*` control,
and the lookahead/bridge flags. The launcher rejects command-line parameters
so the running configuration always comes from the checked-in file. It keeps
`WARM_SLOTS=0` and lookahead disabled by default. Without an API key, it prints
a warning because the HTTP API is unauthenticated on the LAN.

## 7. Run a reproducible benchmark

The runner keeps context, prompt, temperature, batch sizes, warm-up procedure, and process lifetime fixed. Case `SC` is a fixed static profile with no warm cache and guarded no-sync:

```bash
RESULTS_DIR="$PWD/benchmark-results/mtp2-s33-$(date -u +%Y%m%dT%H%M%SZ)" \
MODEL="$MODEL" \
PROFILE="$PWD/local-profiles/general-profile.csv" \
CASES=SC \
REPEATS=3 \
N_PREDICT=2000 \
WARMUP_TOKENS=64 \
STATIC_FIXED_S=33 \
CPU_THREADS=8 \
DRAFT_THREADS=8 \
DRAFT_TYPE_K=q4_0 \
DRAFT_TYPE_V=q4_0 \
CPU_REUSE_ROWS=1 \
CMOE_BATCH=376 \
CMOE_UBATCH=376 \
./scripts/bench_hybrid.sh
```

Use `MTP_OVERRIDE=0`, `1`, `2`, or `3` for comparable MTP screens. Case `SD` enables expert statistics and the required synchronization for diagnostics. Cases `SV` and `SVC` use a layer-variable placement manifest through `PLACEMENT=...`. Warm-cache cases remain experimental: MTP-2 warm residency is now allowed for controlled testing, while MTP-1 and MTP-3 retain the automatic guard.

The primary metric is median sustained decode token/s over 2,000 tokens. Reject a candidate if hashes unexpectedly diverge, output quality fails, CUDA hangs, NVIDIA Xid appears, VRAM OOM occurs, or throughput regresses more than the experiment's declared threshold.

## 8. Optimize placement for the target GPU

First benchmark fixed static placement with W=0. Leave `LLAMA_EXPERT_S` unset for auto-fit, then test a small integer range around the logged auto-fit value. Do not pass the literal value `auto` to `LLAMA_EXPERT_S`. Keep `LLAMA_EXPERT_VRAM_RESERVE_MIB=512` until the complete 32K-context configuration has demonstrated safe headroom.

Then allocate the same exact fixed-expert byte budget unevenly across layers:

```bash
python3 tools/optimize_expert_placement.py \
    --profile "$PWD/local-profiles/general-profile.csv" \
    --model "$MODEL" \
    --output "$PWD/local-profiles/placement-reference-s33.csv" \
    --reference-slots 33 \
    --min-slots 8 \
    --max-slots 96 \
    --objective counts-per-byte
```

Run the result with both `LLAMA_EXPERT_HOT` and `LLAMA_EXPERT_PLACEMENT`, with adaptation disabled and W=0. The runtime verifies the profile SHA-256, tensor sizes, byte budget, dimensions, architecture, and available VRAM before publishing the LUT.

To weight placement by measured cold CPU cost, first run a diagnostic case with timing enabled, then feed its per-layer JSON to a new optimizer output:

```bash
RESULTS_DIR="$PWD/benchmark-results/layer-timing" \
MODEL="$MODEL" \
PROFILE="$PWD/local-profiles/general-profile.csv" \
CASES=SD \
REPEATS=1 \
N_PREDICT=512 \
STATIC_FIXED_S=33 \
EXPERT_TIMING=1 \
./scripts/bench_hybrid.sh

python3 tools/optimize_expert_placement.py \
    --profile "$PWD/local-profiles/general-profile.csv" \
    --model "$MODEL" \
    --output "$PWD/local-profiles/placement-cost-s33.csv" \
    --reference-slots 33 \
    --min-slots 8 \
    --max-slots 96 \
    --objective counts-per-byte \
    --layer-stats-json "$PWD/benchmark-results/layer-timing/SD-run1.experts.json"
```

For a larger GPU, use `--fixed-budget-mib` or a larger `--reference-slots` value rather than copying the GTX 1660 Ti's S=33. The optimizer derives per-layer expert bytes from the GGUF, so heterogeneous layer sizes remain correctly budgeted. An optional `--layer-stats-json` weights avoided cold selections by measured CPU cost.

Only after static placement is optimized should a larger GPU test the warm tier. Start with bounded values such as W=4, 8, and 16, monitor copy volume and evictions, and compare against W=0. `LLAMA_EXPERT_WARM_SLOTS=auto` consumes remaining measured capacity, while `LLAMA_EXPERT_WARM_AUTO_MAX` raises the conservative default cap of four. If synchronous W improves throughput, test one async prefetch stream with a small in-flight limit before adding concurrency.

### Measure transient expert feasibility

Do not infer transient-streaming performance from routing recall or nominal
PCIe bandwidth. Build the measurement tools with the native CUDA architecture,
then derive every transfer size from the target model:

```bash
cmake --build build-hybrid -j 8 --target \
    llama-expert-transport-bench llama-expert-compute-bench

python3 tools/bench_expert_transport.py \
    --model "$MODEL" \
    --binary build-hybrid/bin/llama-expert-transport-bench \
    --output-dir benchmark-results/transport-$(date -u +%Y%m%dT%H%M%SZ) \
    --runs 3 \
    --repeats 200 \
    --working-set-mib 32
```

Measure resident GPU compute for representative early, middle, and late layer
layouts. The tool maps the source model read-only and refuses to overwrite its
JSON output:

```bash
python3 tools/bench_expert_compute.py \
    --model "$MODEL" \
    --binary build-hybrid/bin/llama-expert-compute-bench \
    --output-dir benchmark-results/expert-compute-$(date -u +%Y%m%dT%H%M%SZ) \
    --runs 3
```

Feed the resulting transport summary, model layout, CPU timing JSON, resident
compute summary, and raw Phase 1 traces to
`tools/simulate_expert_streaming.py`. Omit `--lead-ms` for a clearly labelled
always-ready upper bound. Do not enable productive transfers unless target
measurements pass the gates in `TRANSIENT_EXPERT_DESIGN.md`.

Use one or more separate `--calibration-trace` inputs when estimating a real
predictor policy. Without them, the simulator labels rank probabilities as an
optimistic in-sample estimate. Oracle results do not use predictor precision.

The current GTX 1660 Ti result is negative: full H2D plus resident GPU compute
takes about 0.32--0.34 ms per expert versus about 0.05--0.07 ms on the Ryzen
CPU. The tools remain hardware-parameterized because the GTX 1080 plus old i7,
and newer GPUs with faster interconnects or more fixed-tier VRAM, can have a
different crossover. See `TRANSIENT_EXPERT_ANALYSIS.md` and
`TRANSIENT_EXPERT_EXPERIMENTS.md`.

Recommended optimization order:

1. Representative request-balanced profile.
2. Fixed S and thread-count sweep with W=0.
3. Layer-variable placement under the same byte budget.
4. CPU-cost-weighted placement from timing statistics.
5. MTP-0/1/2/3 comparison with deterministic hashes.
6. Warm-cache sizing on GPUs with real remaining VRAM.
7. Async prefetch only after synchronous residency is beneficial.

Test physical-core and SMT thread counts such as 6, 8, 12, and 16. More occupied cores do not guarantee more token/s because target CPU experts, the MTP draft context, CUDA submission, and memory bandwidth compete for the same resources.

Key portable controls:

| Variable | Safe default | Purpose |
|---|---:|---|
| `LLAMA_EXPERT_HOT` | unset | Validated ranking/usage CSV used to choose fixed experts |
| `LLAMA_EXPERT_S` | unset (auto-fit) | Uniform fixed slots per layer; omit when using a placement manifest |
| `LLAMA_EXPERT_PLACEMENT` | unset | Validated layer-variable static slot manifest |
| `LLAMA_EXPERT_ADAPT` | 1 | Long-term online score updates and repinning |
| `LLAMA_EXPERT_ADAPT_INTERVAL` | request | Publish fixed repins at a request boundary; MTP rejects graph-level publication |
| `LLAMA_EXPERT_ADAPT_CUDA_GRAPHS` | 0 | Unsafe diagnostic opt-in for CUDA graph caching with mutable expert weights |
| `LLAMA_EXPERT_USAGE` | unset | Persist expert usage to a CSV atomically |
| `LLAMA_EXPERT_USAGE_MODE` | cumulative | Select cumulative or request/session-only export |
| `LLAMA_EXPERT_USAGE_CHECKPOINT` | request | Persist after each request or only at process exit |
| `LLAMA_EXPERT_STATS_JSON` | unset | Machine-readable counters and optional timing output |
| `LLAMA_EXPERT_TIMING` | 0 | Detailed CPU expert phase timing; use only for diagnosis |
| `LLAMA_EXPERT_WARM_SLOTS` | 0 | Additional bounded LRU slots per layer; integer or auto |
| `LLAMA_EXPERT_WARM_AUTO_MAX` | 4 | Safety cap for automatic warm slots; raise only after measuring a larger GPU |
| `LLAMA_EXPERT_VRAM_RESERVE_MIB` | 512 | Headroom retained after model and runtime allocations |
| `LLAMA_EXPERT_WARM_PREFETCH` | 0 | Experimental asynchronous H2D population of warm slots |
| `LLAMA_EXPERT_STATIC_NO_SYNC` | 0 | Skip only the expert-tier update barrier under strict immutable-tier conditions |
| `LLAMA_EXPERT_CPU_REUSE_ROWS` | 0 | Reuse quantized cold rows across repeated MTP expert selections |
| `LLAMA_EXPERT_CPU_MULTI_ROW` | 0 | AVX2 Q4_K/Q5_K multi-row dots for repeated MTP expert selections; implies row-oriented traversal |
| `LLAMA_EXPERT_CPU_FUSED_GATE_UP` | 0 | Experimental AVX2 Q4_K dual-dot sharing Q8_K activation loads between CPU-Cold Gate and Up; neutral on GTX 1660 Ti, retest on CPU-bound hosts |
| `LLAMA_EXPERT_SHARED_HOT_IDS` | 0 | Reuse one hot-slot ID mapping for Gate, Up, and Down in a layer |
| `GGML_CUDA_MOE_MULTI_FUSION` | 0 | Fuse quantized Gate+Up+GLU for two to four MoE target tokens on Turing or newer GPUs |
| `GGML_CUDA_MOE_COMBINE_FUSION` | 0 | Fuse exact F32 post-Down expert weighting and ordered Top-k reduction; experimental |
| `GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS` | 0 | Override rows/block for non-ID Q8_0 one-column MMVQ (`1`, `2`, or `4`); `0` preserves automatic selection |
| `GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS` | 0 | Override rows/block for non-ID Q8_0 three-column MMVQ (`1`, `2`, or `4`); `0` preserves automatic selection |
| `GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS` | 0 | Override rows/block for plain non-ID Q6_K one-column MMVQ (`1`, `2`, or `4`) |
| `GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS` | 0 | Override rows/block for plain non-ID Q6_K three-column MMVQ (`1`, `2`, or `4`) |
| `GGML_CUDA_MMVQ_MOE_FUSED_ROWS` | 0 | Tune rows/block (`1`, `2`, or `4`) for fused multi-token MoE Gate+Up; zero retains the current two-row default |
| `GGML_CUDA_MMVQ_MOE_PLAIN_ROWS` | 0 | Tune rows/block (`1`, `2`, or `4`) for plain multi-token MoE Down; zero retains the current two-row default |
| `GGML_CUDA_TURBO4_FAST_F16_CONVERT` | 0 | Experimental Turbo4-to-F16 converter schedule (`1` one warp/block, `2` two warps/block); both were slower on SM75 |
| `GGML_CUDA_TURBO4_WHT_SHUFFLE` | 0 | Experimental shuffle-first Turbo4 WHT; neutral/slower on SM75 |
| `LLAMA_TURBO4_DRAFT_EXPERIMENTAL` | 0 | Permit Turbo4 K/V in the MTP-only draft context; exact target verification remains authoritative |
| `GGML_CUDA_ASYNC_HOST_COPY` | 0 | Enqueue pinned CPU-to-CUDA scheduler split copies on the destination compute stream; measured GTX 1660 Ti MTP-2 winner |
| `GGML_SCHED_ASYNC_D2H_COPY` | 0 | Queue CUDA-to-pinned-host split readback after source compute and synchronize once at the CPU consumer; experimental |
| `GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE` | 0 | Non-contiguous CONCAT block override (`32`, `64`, `128`, `256`); zero keeps the CUDA default |
| `GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0` | 0 | Use the flat dim-0 non-contiguous CONCAT kernel; measured winner for local MTP-2 |
| `LLAMA_EXPERT_CPU_ASYNC` | 0 | Experimental CPU-cold/GPU-hot overlap; benchmark before use |

The benchmark runner also accepts `CONTEXT`, `TARGET_TYPE_K`, and `TARGET_TYPE_V`. Their defaults remain `32768`, `q4_0`, and `q4_0` so existing benchmark commands retain their behavior.

## 9. Correctness tests

Run the hybrid Python suites:

```bash
python3 -m unittest \
    tests/test-convert-luce-spark-profile.py \
    tests/test-aggregate-expert-profiles.py \
    tests/test-optimize-expert-placement.py
```

Build and run the targeted C++ tests:

```bash
cmake --build build-hybrid -j 8 --target test-expert-adaptation test-expert-warm-cache test-expert-placement test-moe-multi-row test-arg-parser test-backend-ops

./build-hybrid/bin/test-expert-adaptation
./build-hybrid/bin/test-expert-warm-cache
./build-hybrid/bin/test-expert-placement
./build-hybrid/bin/test-moe-multi-row
./build-hybrid/bin/test-arg-parser
./build-hybrid/bin/test-backend-ops
```

The full CTest suite may require Git LFS test data and generated upstream fixtures. Read [HYBRID_EXPERIMENTS.md](HYBRID_EXPERIMENTS.md) before interpreting failures outside the changed expert-tier paths.

## 10. Prepare a GitHub push

This checkout may still use the upstream repository as `origin`. Point a separate `origin` at your own fork and retain upstream under a distinct name:

```bash
git remote -v
git remote rename origin upstream
git remote add origin git@github.com:YOUR_ACCOUNT/llama-wackMall.git
```

Review exactly what Git would publish:

```bash
git status --short
git diff --check
git check-ignore -v benchmark-results/example local-profiles/example build-hybrid/CMakeCache.txt
git ls-files | grep -E '(^|/)(benchmark-results|local-profiles|build-hybrid)/' || true
git diff --cached --name-only | grep -E '\.(gguf|log|nsys-rep|qdstrm)$' || true
git diff --stat
```

Commit and push only after reviewing every changed line and understanding the implementation. Follow [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md); the upstream project explicitly forbids automated PR submission and AI-written PR descriptions or review responses.

---

## Original llama-wackMall documentation

This project is in active development. The public branch is a few days of commits behind because im trying to iron out a lot of the underlying systems before i publish as I am replacing mmap entirely. Feel free to contact me if you wish to help me test or contribute to the latest version at miltiadiskd@gmail.com

# llama-wackMall

> Expert-granular MoE tiering for llama.cpp: hot experts in VRAM, cold experts in RAM, adaptive online cache, zero-config auto-fit.

`llama-wackMall` is a research fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
that runs large sparse Mixture-of-Experts (MoE) models - Qwen3.5-122B-A10B,
Qwen3.6-35B-A3B, gemma-4-26B-A4B - at high token-generation throughput on
consumer GPUs with as little as 8 GB VRAM.

First public disclosure: 2026-07-26. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the full technical specification. Released under the Apache 2.0 License (see [LICENSE](LICENSE) for upstream MIT;
[NOTICE](NOTICE) for additions).

---

## The bottleneck

Stock llama.cpp offloads at layer granularity (`-ngl N`): an MoE layer with
128-256 expert tensors is an all-or-nothing block. On small GPUs, fitting a
28 GB MoE model means most layers fall back to CPU, and every generated token
streams its active experts through system RAM - the memory-bandwidth wall.

## What this engine does

Replaces layer-granular offload with **expert-granular offload**:

1. **Hot/cold expert split.** Per MoE layer, the S currently-hottest experts
   are pinned in VRAM; the rest stay in RAM and are computed on CPU only when
   actually routed. Per-token traffic drops from "all active experts" to
   "only the cold-selected ones".
2. **Hardware-aware auto-fit.** At startup the engine places dense weights,
   KV cache and compute buffers first, measures the remaining free VRAM, and
   computes the exact number of hot expert slots S that fits. No manual
   `-ngl` tuning.
3. **Adaptive online cache.** Router decisions are counted per token; a
   cumulative usage score with hysteresis (1.5x score ratio, 32-token
   minimum dwell) swaps hot/cold assignments online (optional exponential
   decay for recency weighting). No offline profiling required; optional
   warm-start seeds are supported.
4. **Zero-config entry point.** One flag, `-cmoe`, enables the whole stack.

The design is model-agnostic: all hooks live at the shared MoE graph-builder
level (no per-model branches), covering any llama.cpp MoE architecture whose
expert tensors use the standard `ffn_{gate,up,down}_exps` layout.

## Verified results

Hardware: RTX 3070 8 GB VRAM, 31 GB RAM, `--temp 0`, `-n 256 --ignore-eos`,
single run per config, same binary/session per A/B pair.

| Model | Quant (size) | Stock | wackMall | Speedup |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | IQ2_M (11 GB) | 42.70 tok/s | **74 tok/s** (server, n=1024) | +73% |
| Qwen3.6-35B-A3B | Q4_K_M (20 GB) | 26.89 tok/s | **49.93 tok/s** (S=64) | +86% |
| gemma-4-26B-A4B | Q5_K_S (17 GB) | 19.50 tok/s | **56 tok/s** (S=39) | +187% |
| Qwen3.5-122B-A10B | IQ2_M (28 GB) | ~8.0 tok/s (best layer-split config) | **10.60 tok/s** (S=28) | +33% |
| Long context (67k prompt) | - | CUDA OOM | **410.38 tok/s** (prompt eval) | runs cleanly |
| Qwen3.5-122B, 16 GB RAM cap | IQ3_XS (34 GB) | 2.40 tok/s | **2.84 tok/s** | +18% |

Latest rebased tree (upstream `0e4a036`, 2026-07-27): 35B IQ2_M server at
**63.25 tok/s** (n=1024, adaptation on, ~4250 online re-pins in one run),
with fresh-server determinism verified byte-identical across restarts.
Vulkan tiering verified: 35B Q4_K_M at **34.58 tok/s** on an RX 570 - use
K-quants on Vulkan, the IQ-quant MUL_MAT_ID shaders are not well optimized
there yet.

All numbers measured manually by the authors; single run per config,
same-flags stock baselines on the identical upstream base. Correctness:
perplexity within rounding noise of stock (9-chunk protocol); bit-identical
to stock when forced through identical compute paths; adaptive re-pin
bookkeeping machine-checked by a permanent invariant guard. Greedy output
can tie-flip vs stock (different-but-valid rounding, same class as changing
batch size); output quality is equivalent. Details in ARCHITECTURE.md
section 5.

## Quick start

Requirements: CMake 3.18+, GCC/Clang with OpenMP, CUDA toolkit for NVIDIA
builds.

```bash
mkdir build && cd build
cmake -DGGML_CUDA=ON ..
cmake --build . -j --target llama-server
```

Run (tiering on by default):

```bash
./bin/llama-server -m /path/to/moe-model.gguf -c 4096 --port 8080
```

Use `-no-cmoe` for stock llama.cpp behavior. `-t 10` for manual threads
(10 is the automatic default on 12-core CPUs; auto-threads picks 80% of
hardware when dense weights fit VRAM). KV cache defaults to f16; on K-quant
models at long context, add `-ctk q8_0 -ctv q8_0`.

Tiering also auto-configures auto-fit offloading, batch/ubatch 256, flash
attention and KV offload. Interactive use: `llama-cli` with the same flags
(add `--jinja` for architectures with custom chat templates, e.g. gemma4).
Vulkan: configure with `-DGGML_VULKAN=ON` and prefer K-quant models.

### Optional environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_EXPERT_S` | auto | Hot slots per layer (0 = all CPU, N = force N) |
| `LLAMA_EXPERT_HOT` | - | CSV heat seed for warm start |
| `LLAMA_EXPERT_ADAPT` | 1 | Online adaptation on/off |
| `LLAMA_EXPERT_DECAY` | 1.0 | Score decay per update (1.0 = cumulative) |
| `LLAMA_EXPERT_TMAX` | 16 | Max tokens/graph that takes the tiered path |
| `LLAMA_EXPERT_STATS` | - | 1 or path: dump cache statistics at exit |
| `LLAMA_EXPERT_USAGE` | - | Path: dump counts (reusable as next seed) |

## Status

Research preview. Verified on `qwen35moe` and `gemma4` architectures. On the
roadmap: disk as third tier for models exceeding RAM (mmap-based, see
ARCHITECTURE.md section 8), predictive expert prefetching (semantic seeding,
Markov correlation, learned router heads - also section 8), multi-GPU tier
priority, per-layer slot skew.

## Prior art notice

This repository and ARCHITECTURE.md are published to disclose the described
methods and systems as of the first publication date, with the intent that
this disclosure serve as prior art. The original code is MIT licensed; colibri-tier
additions are Apache 2.0 (see [NOTICE](NOTICE)). The authors grant no patent
rights and intend none to be asserted over the disclosed concepts.
