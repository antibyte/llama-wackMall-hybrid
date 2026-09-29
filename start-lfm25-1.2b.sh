#!/usr/bin/env bash
#
# llama-wackMall-hybrid GTX 1660 Ti launcher for LFM2.5-1.2B-Instruct.
#
# Alle Einstellungen stehen in diesem Block. Keine Kommandozeilenparameter.
# Nach einer Aenderung: ./start-lfm25-1.2b.sh
#
# Unsloth Q4_K_M (1.17B, dense LFM2 hybrid, native ctx 32768).
# llama-bench 2026-09-26 GTX 1660 Ti, FA on, q8 KV, ngl 99, threads 8:
#   Pascal MMQ (this binary), no drafter: pp512 3428 t/s, pp2048 3228 t/s, tg128 219 t/s
#   sm75 with the same flags: pp512 1077 t/s, tg128 239 t/s (274 with MMVQ rows)
#   DSpark lowered prefill and decode on both builds. Spec stays none.
# Batch 2048 / ubatch 512 is the measured prefill shape.
# Sampling from the Liquid instruct card: temp 0.1, top-k 50, repeat-penalty 1.05.
# Direct answers, no think. No KVFlash: 32k q8 KV fits beside the weights.
#
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# ============================================================================
# EDITABLE CONFIGURATION -- only edit values in this section
# ============================================================================

SERVER="$PROJECT_ROOT/build-mmq-pascal/bin/llama-server"
MODEL="$HOME/models/lfm2.5-1.2b/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
CHAT_TEMPLATE_FILE="$PROJECT_ROOT/models/templates/LFM2.5-Instruct.jinja"

HOST="0.0.0.0"
PORT="8080"
CORS_ORIGINS="*"
API_KEY=""
API_KEY_FILE=""
MODEL_ALIAS="lfm2.5-1.2b"
CUDA_VISIBLE_DEVICES_VALUE="0"
N_PARALLEL="1"
N_GPU_LAYERS="99"
CPU_MOE="0"
UI="0"
CONT_BATCHING="1"
OP_OFFLOAD="1"
FIT="off"
FIT_TARGET="80"
FIT_CTX="2048"

CONTEXT="32768"
N_PREDICT="8192"
BATCH="2048"
UBATCH="512"
TARGET_TYPE_K="q8_0"
TARGET_TYPE_V="q8_0"
FLASH_ATTN="on"
KV_OFFLOAD="1"
LOAD_MODE="mmap"
OFFLINE="1"
KV_UNIFIED="1"

# 0 disables. Resident window is unused: the full 32k q8 cache fits in 6 GiB.
LLAMA_KVFLASH="0"
LLAMA_KVFLASH_MAX_POOL="8192"
LLAMA_KVFLASH_TAU="64"
LLAMA_KVFLASH_POLICY="lru"
LLAMA_KVFLASH_STATS="0"

SPEC_MODE="none"
SPEC_DRAFT_N_MAX="4"

TEMP="0.1"
TOP_K="50"
TOP_P="1.0"
MIN_P="0"
REPEAT_PENALTY="1.05"

REASONING="0"
REASONING_BUDGET="0"
REASONING_PRESERVE="0"
REASONING_FORMAT="none"
JINJA="1"

THREADS="8"
THREADS_BATCH="8"
TARGET_BACKEND_SAMPLING="1"

CMOE_BATCH="512"
CMOE_UBATCH="512"
CMOE_PREFILL_BATCH="2048"
CMOE_PREFILL_UBATCH="512"
CMOE_DECODE_BATCH="512"
CMOE_DECODE_UBATCH="512"

CTX_CHECKPOINTS="8"
CACHE_RAM="4096"
CACHE_PROMPT="1"
CACHE_REUSE="0"
CACHE_IDLE_SLOTS="1"

# ============================================================================
# End of editable configuration
# ============================================================================

die() {
    printf 'start-lfm25-1.2b.sh: %s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 0 ]]; then
    die "Keine Kommandozeilenparameter: Einstellungen oben in start-lfm25-1.2b.sh aendern."
fi

[[ -x "$SERVER" ]] || die "llama-server nicht ausfuehrbar: $SERVER"
[[ -f "$MODEL" ]] || die "Modell nicht gefunden: $MODEL"
[[ -f "$CHAT_TEMPLATE_FILE" ]] || die "Chat-Template nicht gefunden: $CHAT_TEMPLATE_FILE"
[[ "$JINJA" == 1 ]] || die "CHAT_TEMPLATE_FILE benoetigt JINJA=1."
case "$REASONING_FORMAT" in none|deepseek|deepseek-legacy|auto) ;; *) die "REASONING_FORMAT muss none, deepseek, deepseek-legacy oder auto sein." ;; esac
case "$SPEC_MODE" in none) ;; *) die "SPEC_MODE muss none sein. Ein Drafter hat den Prefill gesenkt." ;; esac
case "$FLASH_ATTN" in on) ;; *) die "FLASH_ATTN muss on sein." ;; esac
case "$FIT" in on|off|1|0|true|false) ;; *) die "FIT muss on oder off sein." ;; esac
[[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] || die "CONTEXT muss eine positive Ganzzahl sein."
[[ "$BATCH" =~ ^[1-9][0-9]*$ && "$UBATCH" =~ ^[1-9][0-9]*$ ]] || die "BATCH und UBATCH muessen positive Ganzzahlen sein."
if [[ "$LLAMA_KVFLASH" != 0 ]]; then
    die "LLAMA_KVFLASH bleibt 0. Der gemessene Prefill lief ohne KVFlash."
fi

case "$SPEC_MODE" in
    none) SPEC_TYPE="none" ;;
esac

cat <<EOF
llama-wackMall-hybrid start (lfm2.5-1.2b)
  project:   $PROJECT_ROOT
  server:    $SERVER
  model:     $MODEL
  listen:    $HOST:$PORT
  GPU:       $CUDA_VISIBLE_DEVICES_VALUE
  context:   $CONTEXT
  batch:     $BATCH/$UBATCH
  spec:      $SPEC_TYPE
  KV:        $TARGET_TYPE_K/$TARGET_TYPE_V
  ngl:       $N_GPU_LAYERS
  sampling:  temp=$TEMP top-k=$TOP_K top-p=$TOP_P min-p=$MIN_P repeat=$REPEAT_PENALTY
  reasoning: $REASONING format=$REASONING_FORMAT
  template:  $CHAT_TEMPLATE_FILE
EOF

env_args=(
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE"
    "LLAMA_ARG_HOST=$HOST"
    "LLAMA_ARG_PORT=$PORT"
    "LLAMA_ARG_CORS_ORIGINS=$CORS_ORIGINS"
    "LLAMA_ARG_ALIAS=$MODEL_ALIAS"
    "LLAMA_ARG_CTX_SIZE=$CONTEXT"
    "LLAMA_ARG_N_PREDICT=$N_PREDICT"
    "LLAMA_ARG_BATCH=$BATCH"
    "LLAMA_ARG_UBATCH=$UBATCH"
    "LLAMA_ARG_N_GPU_LAYERS=$N_GPU_LAYERS"
    "LLAMA_ARG_N_PARALLEL=$N_PARALLEL"
    "LLAMA_ARG_NO_CPU_MOE=1"
    "LLAMA_ARG_FLASH_ATTN=$FLASH_ATTN"
    "LLAMA_ARG_CACHE_TYPE_K=$TARGET_TYPE_K"
    "LLAMA_ARG_CACHE_TYPE_V=$TARGET_TYPE_V"
    "LLAMA_ARG_THREADS=$THREADS"
    "LLAMA_ARG_THREADS_BATCH=$THREADS_BATCH"
    "LLAMA_ARG_KV_UNIFIED=$KV_UNIFIED"
    "LLAMA_ARG_KV_OFFLOAD=$KV_OFFLOAD"
    "LLAMA_ARG_LOAD_MODE=$LOAD_MODE"
    "LLAMA_ARG_OFFLINE=$OFFLINE"
    "LLAMA_ARG_UI=$UI"
    "LLAMA_ARG_JINJA=$JINJA"
    "LLAMA_ARG_CHAT_TEMPLATE_FILE=$CHAT_TEMPLATE_FILE"
    "LLAMA_ARG_REASONING=$REASONING"
    "LLAMA_ARG_THINK_BUDGET=$REASONING_BUDGET"
    "LLAMA_ARG_REASONING_PRESERVE=$REASONING_PRESERVE"
    "LLAMA_ARG_THINK=$REASONING_FORMAT"
    "LLAMA_ARG_BACKEND_SAMPLING=$TARGET_BACKEND_SAMPLING"
    "LLAMA_ARG_SPEC_TYPE=$SPEC_TYPE"
    "LLAMA_ARG_SPEC_DRAFT_N_MAX=$SPEC_DRAFT_N_MAX"
    "LLAMA_ARG_CACHE_RAM=$CACHE_RAM"
    "LLAMA_ARG_CACHE_PROMPT=$CACHE_PROMPT"
    "LLAMA_ARG_CACHE_REUSE=$CACHE_REUSE"
    "LLAMA_ARG_CTX_CHECKPOINTS=$CTX_CHECKPOINTS"
    "LLAMA_ARG_CACHE_IDLE_SLOTS=$CACHE_IDLE_SLOTS"
    "LLAMA_ARG_CONT_BATCHING=$CONT_BATCHING"
    "LLAMA_ARG_FIT=$FIT"
    "LLAMA_ARG_FIT_TARGET=$FIT_TARGET"
    "LLAMA_ARG_FIT_CTX=$FIT_CTX"
    "LLAMA_CMOE_BATCH=$CMOE_BATCH"
    "LLAMA_CMOE_UBATCH=$CMOE_UBATCH"
    "LLAMA_CMOE_PREFILL_BATCH=$CMOE_PREFILL_BATCH"
    "LLAMA_CMOE_PREFILL_UBATCH=$CMOE_PREFILL_UBATCH"
    "LLAMA_CMOE_DECODE_BATCH=$CMOE_DECODE_BATCH"
    "LLAMA_CMOE_DECODE_UBATCH=$CMOE_DECODE_UBATCH"
)

server_args=(
    -m "$MODEL"
    --temp "$TEMP"
    --top-k "$TOP_K"
    --top-p "$TOP_P"
    --min-p "$MIN_P"
    --repeat-penalty "$REPEAT_PENALTY"
)
[[ "$OP_OFFLOAD" == 1 ]] && server_args+=(--op-offload)
[[ -n "$API_KEY" ]] && server_args+=(--api-key "$API_KEY")
[[ -n "$API_KEY_FILE" ]] && server_args+=(--api-key-file "$API_KEY_FILE")

exec env \
    -u LLAMA_ARG_SPEC_DRAFT_MODEL \
    -u LLAMA_ARG_MMPROJ \
    -u LLAMA_KVFLASH \
    -u LLAMA_EXPERT_S \
    "${env_args[@]}" \
    "$SERVER" "${server_args[@]}"
