#!/usr/bin/env bash
# Ling-3.0-tiny (bailingmoe3) on GTX 1080 — starting recipe for 8 GiB / sm_61.
# Not a measured 1080 winner yet. All experts on GPU, no Turbo4 (MLA k=576),
# no MTP in the bartowski GGUF. Pascal kernel overrides stay off.
#
# Binary (first existing wins):
#   LING_SERVER
#   build-turbo-opt-sm61/bin/llama-server
#   build-pascal-tuned-sm61/bin/llama-server
#   build-transient-sm61/bin/llama-server
#   build-main-sm61/bin/llama-server
#
# Build on the 1080 host if none of those exist:
#   cmake -S . -B build-turbo-opt-sm61 -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
#     -DCMAKE_CUDA_ARCHITECTURES=61 -DLLAMA_BUILD_UI=OFF
#   cmake --build build-turbo-opt-sm61 -j8 --target llama-server
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODEL="${LING_MODEL:-$HOME/models/ling-3.0-tiny/Ling-3.0-tiny-Q4_K_M.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
# Extra ~2 GiB vs 1660 Ti: larger prefill + resident KV, Pascal decode 128.
PREFILL="${PREFILL:-2560}"
DECODE="${DECODE:-128}"
KVFLASH="${KVFLASH:-12288}"
THREADS="${THREADS:-4}"

if [[ -n "${LING_SERVER:-}" ]]; then
  SERVER="$LING_SERVER"
else
  SERVER=""
  for cand in \
    "$PROJECT_ROOT/build-turbo-opt-sm61/bin/llama-server" \
    "$PROJECT_ROOT/build-pascal-tuned-sm61/bin/llama-server" \
    "$PROJECT_ROOT/build-transient-sm61/bin/llama-server" \
    "$PROJECT_ROOT/build-main-sm61/bin/llama-server"
  do
    if [[ -x "$cand" ]]; then
      SERVER="$cand"
      break
    fi
  done
fi

[[ -n "$SERVER" && -x "$SERVER" ]] || {
  echo "start-ling-tiny-1080.sh: no sm_61 llama-server. Set LING_SERVER or build build-turbo-opt-sm61." >&2
  exit 1
}
[[ -f "$MODEL" ]] || { echo "start-ling-tiny-1080.sh: missing $MODEL" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_MOE_MULTI_FUSION=0
export GGML_CUDA_MOE_COMBINE_FUSION=0
export GGML_CUDA_ASYNC_HOST_COPY=1
export GGML_SCHED_DEDUP_DST_SYNC=1
# export GGML_CUDA_REGISTER_HOST=1  # mmap pin; 1660 21 GiB cudaHostRegister failed
# export GGML_SCHED_PREFETCH_EXPERTS=1  # n-cpu-moe prefill overlap; keep off without -ncmoe
export LLAMA_CMOE_PREFILL_BATCH="$PREFILL"
export LLAMA_CMOE_PREFILL_UBATCH="$PREFILL"
export LLAMA_CMOE_DECODE_BATCH="$DECODE"
export LLAMA_CMOE_DECODE_UBATCH="$DECODE"
export LLAMA_KVFLASH="$KVFLASH"
export LLAMA_KVFLASH_MAX_POOL="$KVFLASH"
export LLAMA_KVFLASH_POLICY=lru
unset LLAMA_EXPERT_S || true

printf 'start-ling-tiny-1080.sh: server=%s prefill=%s decode=%s kvflash=%s threads=%s\n' \
  "$SERVER" "$PREFILL" "$DECODE" "$KVFLASH" "$THREADS" >&2

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
  -t "$THREADS" -tb "$THREADS" \
  --jinja --backend-sampling \
  --reasoning on --reasoning-preserve --reasoning-budget 1000 \
  --spec-type ngram-simple --spec-draft-n-max 4 \
  --op-offload --load-mode mmap \
  --n-predict 8192 \
  --alias ling-tiny \
  "$@"
