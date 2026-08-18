#!/usr/bin/env bash
# Ling-3.0-tiny (bailingmoe3) on GTX 1660 Ti — measured 2026-08-18.
# KVFlash 8192 + phase prefill 2048 / decode 64, all experts on GPU.
# No MTP in this GGUF. Turbo4 is incompatible with MLA k=576.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SERVER="$PROJECT_ROOT/build-main-sm75/bin/llama-server"
MODEL="${LING_MODEL:-$HOME/models/ling-3.0-tiny/Ling-3.0-tiny-Q4_K_M.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
PREFILL="${PREFILL:-2048}"
DECODE="${DECODE:-64}"
KVFLASH="${KVFLASH:-8192}"

[[ -x "$SERVER" ]] || { echo "missing $SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_MOE_MULTI_FUSION=1
export GGML_CUDA_MOE_COMBINE_FUSION=1
export GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=4
export GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2
export GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=4
export GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=128
export GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1
export GGML_CUDA_ASYNC_HOST_COPY=1
export GGML_SCHED_DEDUP_DST_SYNC=1
export LLAMA_CMOE_PREFILL_BATCH="$PREFILL"
export LLAMA_CMOE_PREFILL_UBATCH="$PREFILL"
export LLAMA_CMOE_DECODE_BATCH="$DECODE"
export LLAMA_CMOE_DECODE_UBATCH="$DECODE"
export LLAMA_KVFLASH="$KVFLASH"
export LLAMA_KVFLASH_MAX_POOL="$KVFLASH"
export LLAMA_KVFLASH_POLICY=lru
unset LLAMA_EXPERT_S || true

exec "$SERVER" \
  -m "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --no-ui --offline \
  --no-cpu-moe \
  -ngl 99 \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --fit on --fit-target 80 --fit-ctx 2048 \
  -c 131072 \
  -b "$DECODE" -ub "$DECODE" \
  -np 1 \
  --kv-unified \
  --cache-prompt --cache-reuse 16 --cache-ram 2048 \
  --ctx-checkpoints 4 --cache-idle-slots --cont-batching \
  -t 8 -tb 8 \
  --jinja --backend-sampling \
  --reasoning on --reasoning-preserve --reasoning-budget 1000 \
  --spec-type ngram-simple --spec-draft-n-max 4 \
  --op-offload --load-mode mmap \
  --n-predict 8192 \
  --alias ling-tiny \
  "$@"
