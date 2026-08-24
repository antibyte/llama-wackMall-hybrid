#!/usr/bin/env bash
# LFM2.5-8B-A1B APEX I-Compact + DSpark sidecar on GTX 1660 Ti.
# Target: mudler/LFM2.5-8B-A1B-APEX-GGUF LFM2.5-8B-A1B-APEX-I-Compact.gguf
# Draft:  LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF LFM2.5-8B-A1B-DSpark-Q8_0.gguf
# Official DSpark uses n-max=10 (clamps to block_size=9). On 1660 Ti that
# 10-token MoE verify drops off the MMVQ/CUDA-graph path (limit 8) and
# is much slower than sequential decode. Default n-max=2 stays in-graph.
# GTX 1660 Ti 6 GiB: KVFlash is disabled for this hybrid, so keep -c modest.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SERVER="$PROJECT_ROOT/build-main-sm75/bin/llama-server"
MODEL_DIR="${LFM25_MODEL_DIR:-$HOME/models/lfm2.5-8b-a1b}"
MODEL="${LFM25_MODEL:-$MODEL_DIR/LFM2.5-8B-A1B-APEX-I-Compact.gguf}"
DRAFT="${LFM25_DRAFT:-$MODEL_DIR/LFM2.5-8B-A1B-DSpark-Q8_0.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CONTEXT="${CONTEXT:-32768}"
PREFILL="${PREFILL:-512}"
DECODE="${DECODE:-32}"
SPEC_N_MAX="${SPEC_N_MAX:-2}"
SPEC_TYPE="${SPEC_TYPE:-draft-dspark}"

[[ -x "$SERVER" ]] || { echo "missing $SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL" >&2; exit 1; }
[[ -f "$DRAFT" ]] || { echo "missing $DRAFT" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_MOE_MULTI_FUSION=1
export GGML_CUDA_MOE_COMBINE_FUSION=1
export GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS=4
export GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=4
export GGML_CUDA_MMVQ_Q3_K_NCOLS1_ROWS=2
export GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2
export GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2
export GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=4
export GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=128
export GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1
export GGML_CUDA_ASYNC_HOST_COPY=1
export GGML_SCHED_DEDUP_DST_SYNC=1
# export GGML_CUDA_REGISTER_HOST=1  # mmap pin; 21 GiB failed on 1660 Ti
# export GGML_SCHED_PREFETCH_EXPERTS=1  # n-cpu-moe prefill overlap; keep off without -ncmoe
export LLAMA_CMOE_PREFILL_BATCH="$PREFILL"
export LLAMA_CMOE_PREFILL_UBATCH="$PREFILL"
export LLAMA_CMOE_DECODE_BATCH="$DECODE"
export LLAMA_CMOE_DECODE_UBATCH="$DECODE"
export LLAMA_DFLASH_COMBINED=1
unset LLAMA_KVFLASH || true
unset LLAMA_EXPERT_S || true

exec "$SERVER" \
  -m "$MODEL" \
  -md "$DRAFT" \
  --host "$HOST" --port "$PORT" \
  --no-ui --offline \
  --no-cpu-moe \
  -ngl 99 \
  --spec-draft-ngl 99 \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --fit on --fit-target 70 --fit-ctx 2048 \
  -c "$CONTEXT" \
  -b "$DECODE" -ub "$DECODE" \
  -np 1 \
  --kv-unified \
  --cache-prompt --cache-reuse 16 --cache-ram 2048 \
  --ctx-checkpoints 4 --cache-idle-slots --cont-batching \
  -t 8 -tb 8 \
  --jinja --backend-sampling \
  --reasoning on --reasoning-budget 1000 \
  --spec-type "$SPEC_TYPE" \
  --spec-draft-n-max "$SPEC_N_MAX" --spec-draft-n-min 0 \
  --op-offload --load-mode mmap \
  --n-predict 8192 \
  --alias lfm2.5-8b-a1b \
  "$@"
