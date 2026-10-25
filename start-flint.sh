#!/usr/bin/env bash
# Flint Preview UD-Q4_K_XL (unsloth/flint-GGUF) on the GTX 1660 Ti.
# 49 layers, n_embd 2048, 256 experts / 6 used, 2 packed shared experts,
# 2 dense lead layers. Routed experts are mixed IQ/QK, so they stay CPU mmap.
#
# 2026-10-22, build-main-sm75, -ngl 99 -ncmoe 99 -fa on, f16 KV, t=8:
#   llama-bench tg64 36.37 +/- 0.27, pp512 138.37 +/- 1.00 (r=3).
#   Async host copy plus dedup dst sync is part of that number. q8 KV was 34.6 tg.
# Chat heat (S=32, TMAX=1, skip-sentinel), greedy, one sample each:
#   2+2 42.8 t/s, mmap note 41.0, held-out planets 40.1. Same answers as stock (~35 t/s).
# Keep TMAX at 1. A prompt that fits in TMAX takes the multi-token hot path and
# emits a run of "!" ; the pp512 near 180 measured with TMAX=512 is that failure.
# Unseeded auto-fit, t=16, the physical-core mask, readahead, ubatch 256,
# Pascal MMQ, and plain MMVQ row overrides did not beat this stack.
# 2026-10-25: CPU IQ4 and fused MMVQ tuning add about 1.6-1.7 t/s at 64512 context.
# 2026-10-25: cold-first plus deferred host inputs overlap hot and cold experts, +8.6% with identical tokens.
# CPU expert prefill avoids invalid GPU-prefill output; keep ubatch below 1024.
# Keep mmap. A 23.3 GiB anonymous resident set does not fit this host.
# Default context is 8192. The metadata context is 262144; 65536 has locked this machine.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SERVER="${FLINT_SERVER:-$PROJECT_ROOT/build-main-sm75/bin/llama-server}"
MODEL="${FLINT_MODEL:-$HOME/models/flint/flint-UD-Q4_K_XL.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
CONTEXT="${CONTEXT:-8192}"
BATCH="${BATCH:-512}"
THREADS="${THREADS:-8}"

[[ -x "$SERVER" ]] || { echo "start-flint.sh: missing $SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "start-flint.sh: missing $MODEL (run ./download-flint.sh)" >&2; exit 1; }

if [[ "${FLINT_IGNORE_GPU:-0}" != 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 { print; exit }')"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used > 512 )); then
    echo "start-flint.sh: GTX 1660 Ti is in use (${used} MiB). Stop agollm or set FLINT_IGNORE_GPU=1." >&2
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LLAMA_ARG_POWER_BUSY_CMD="${LLAMA_ARG_POWER_BUSY_CMD-busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Performance}"
export LLAMA_ARG_POWER_IDLE_CMD="${LLAMA_ARG_POWER_IDLE_CMD-busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Battery}"
export LLAMA_ARG_POWER_IDLE_DELAY="${LLAMA_ARG_POWER_IDLE_DELAY:-2000}"
export LLAMA_ARG_DECISION_SEQS="${LLAMA_ARG_DECISION_SEQS:-0}"
export GGML_CUDA_ASYNC_HOST_COPY=1
export GGML_SCHED_DEDUP_DST_SYNC=1
# Keep Flint mixed-quant expert prefill on CPU; GPU expert offload can produce invalid output.
export GGML_OP_OFFLOAD_MIN_BATCH=1024

# A set LLAMA_EXPERT_* var turns the tier on even with --no-cpu-moe.
# Clear inherited controls, then set only the measured decode pin.
unset LLAMA_EXPERT_HOT LLAMA_EXPERT_S LLAMA_EXPERT_PLACEMENT \
  LLAMA_EXPERT_ADAPT LLAMA_EXPERT_ADAPT_INTERVAL LLAMA_EXPERT_ADAPT_CUDA_GRAPHS \
  LLAMA_EXPERT_STATS LLAMA_EXPERT_USAGE LLAMA_EXPERT_USAGE_CHECKPOINT \
  LLAMA_EXPERT_STATS_JSON LLAMA_EXPERT_TIMING LLAMA_EXPERT_TMAX \
  LLAMA_EXPERT_CPU_CHUNK LLAMA_EXPERT_CPU_ACT_PARALLEL LLAMA_EXPERT_CPU_ASYNC \
  LLAMA_EXPERT_CPU_DOWN_PREFETCH LLAMA_EXPERT_CPU_REUSE_ROWS \
  LLAMA_EXPERT_CPU_MULTI_ROW LLAMA_EXPERT_CPU_FUSED_GATE_UP \
  LLAMA_EXPERT_SHARED_HOT_IDS LLAMA_EXPERT_SKIP_SENTINEL \
  LLAMA_EXPERT_WARM_SLOTS LLAMA_EXPERT_BW_PROFILE \
  LLAMA_EXPERT_STATIC_NO_SYNC LLAMA_EXPERT_LOOKAHEAD_TRACE LLAMA_EXPERT_COLD_FIRST \
  LLAMA_CMOE_BATCH LLAMA_CMOE_UBATCH \
  LLAMA_CMOE_PREFILL_BATCH LLAMA_CMOE_PREFILL_UBATCH \
  LLAMA_CMOE_DECODE_BATCH LLAMA_CMOE_DECODE_UBATCH \
  LLAMA_KVFLASH \
  GGML_CUDA_DISABLE_GRAPHS GGML_CUDA_MMVQ_INPUT_ALIAS GGML_SCHED_DEFER_HOST_INPUTS \
  GGML_CUDA_MMVQ_Q8_FUSED_NCOLS1_ROWS GGML_SCHED_ASYNC_D2H_COPY \
  GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS GGML_CUDA_MMVQ_Q8_NCOLS5_ROWS \
  GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS GGML_CUDA_MMVQ_Q5_K_NCOLS1_ROWS \
  GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS GGML_CUDA_MMVQ_Q6_K_NCOLS4_ROWS \
  GGML_CUDA_MMVQ_MOE_FUSED_ROWS GGML_CUDA_MMVQ_MOE_PLAIN_ROWS \
  || true

HEAT="${FLINT_HEAT:-$HOME/models/flint/flint-chat-heat.csv}"
if [[ -f "$HEAT" ]]; then
  export LLAMA_EXPERT_HOT="$HEAT"
  export LLAMA_EXPERT_S=32
  export LLAMA_EXPERT_ADAPT=0
  export LLAMA_EXPERT_TMAX=1
  export LLAMA_EXPERT_SKIP_SENTINEL=1
  export LLAMA_EXPERT_SHARED_HOT_IDS=1
  export LLAMA_EXPERT_STATIC_NO_SYNC=1
  export GGML_CUDA_MMVQ_INPUT_ALIAS=1
  export GGML_CUDA_MMVQ_Q8_FUSED_NCOLS1_ROWS=4
  export GGML_SCHED_ASYNC_D2H_COPY=1
  export LLAMA_EXPERT_CPU_DOWN_PREFETCH=4
  export LLAMA_EXPERT_CPU_ACT_PARALLEL=1
  export LLAMA_EXPERT_CPU_FUSED_GATE_UP=1
  export LLAMA_EXPERT_COLD_FIRST=1
  export GGML_SCHED_DEFER_HOST_INPUTS=1
else
  echo "start-flint.sh: no heat file at $HEAT; routed experts stay on CPU." >&2
fi

exec "$SERVER" \
  -m "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --alias flint \
  --no-ui --offline \
  --no-mmproj \
  --no-cpu-moe \
  -ngl 99 \
  -ncmoe 99 \
  -fa on \
  --cache-type-k f16 --cache-type-v f16 \
  -c "$CONTEXT" \
  -b "$BATCH" -ub "$BATCH" \
  -np 1 \
  -t "$THREADS" -tb "$THREADS" \
  --jinja \
  --reasoning-preserve \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --spec-type none \
  --load-mode mmap \
  "$@"
