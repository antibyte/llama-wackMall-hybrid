#!/usr/bin/env bash
#
# llama-wackMall-hybrid GTX 1660 Ti launcher for Spark-X2.5-4B (spark2_5).
#
# Alle Einstellungen stehen in diesem Block. Keine Kommandozeilenparameter,
# keine stillen Shell-Overrides. Nach einer Aenderung: ./startspark.sh
#
# Spark-X2.5-4B Q4_K_M, dense hybrid SWA (3 sliding + 1 full, window 512).
# llama-bench 2026-09-07 on GTX 1660 Ti: pp512 273.7 t/s, tg128 75.9 t/s.
# Tune 2026-09-07: keep q8 KV, FA on, Q4_K MMVQ rows=2, prefill 2048, spec none.
# ngram-simple did not draft on honest Jinja chat; turbo4_k KV garbled output.
# KVFlash stays off (ISWA, not a non-SWA hybrid). Native ctx is 1M tokens;
# 65536 is the 6-GiB starting point. Sampling from the model card:
# temp=1.0 top_p=0.95 top_k disabled. Chat template: Spark2.5.jinja.
# Qwen stack: start1660.sh. Thin alias: start-spark-x25.sh.
#
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# ============================================================================
# EDITABLE CONFIGURATION -- only edit values in this section
# ============================================================================

SERVER="$PROJECT_ROOT/build-main-sm75/bin/llama-server"
MODEL="$HOME/models/spark-x2.5-4b/Spark-X2.5-4B-Q4_K_M.gguf"
CHAT_TEMPLATE_FILE="$PROJECT_ROOT/models/templates/Spark2.5.jinja"

# Network / OpenWebUI
HOST="0.0.0.0"
PORT="8080"
CORS_ORIGINS="*"
API_KEY=""
API_KEY_FILE=""
MODEL_ALIAS="spark-x2.5-4b"
CUDA_VISIBLE_DEVICES_VALUE="0"
N_PARALLEL="1"
N_GPU_LAYERS="99"
CPU_MOE="0"
UI="0"
CONT_BATCHING="1"
OP_OFFLOAD="1"

# Context, KV, Flash Attention
CONTEXT="65536"
N_PREDICT="32768"
TARGET_TYPE_K="q8_0"
TARGET_TYPE_V="q8_0"
FLASH_ATTN="on"
KV_OFFLOAD="1"
LOAD_MODE="mmap"
OFFLINE="1"

# Speculative decoding: none | ngram
# Tune 2026-09-07: ngram-simple n_max=2/4/8 did not accept drafts on Jinja chat.
SPEC_MODE="none"
SPEC_DRAFT_N_MAX="4"
NGRAM_SIMPLE_SIZE_N=""
NGRAM_SIMPLE_SIZE_M=""
NGRAM_SIMPLE_MIN_HITS=""

# Sampling (Spark-X2.5 model card)
TEMP="1.0"
TOP_K="0"
TOP_P="0.95"
MIN_P="0"

# Reasoning / chat template
REASONING="1"
REASONING_BUDGET="4000"
REASONING_PRESERVE="1"
REASONING_FORMAT="deepseek"
JINJA="1"

THREADS="8"
THREADS_BATCH="8"
THREADS_HTTP=""
TARGET_BACKEND_SAMPLING="1"

# Phase batching
CMOE_BATCH="64"
CMOE_UBATCH="64"
CMOE_PREFILL_BATCH="2048"
CMOE_PREFILL_UBATCH="2048"
CMOE_DECODE_BATCH="64"
CMOE_DECODE_UBATCH="64"

# Prompt cache
CTX_CHECKPOINTS="8"
CACHE_RAM="4096"
CACHE_PROMPT="1"
CACHE_REUSE="0"
KV_UNIFIED="1"
CACHE_IDLE_SLOTS="1"

# sm_75 kernel knobs (same winners as start1660 / start-ling-tiny)
GGML_CUDA_MOE_MULTI_FUSION="1"
GGML_CUDA_MOE_COMBINE_FUSION="1"
GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS="4"
GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS="0"
GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS="4"
GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS="2"
GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS="2"
GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS="4"
GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE="128"
GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0="1"
GGML_CUDA_ASYNC_HOST_COPY="1"
GGML_SCHED_ASYNC_D2H_COPY="0"
GGML_SCHED_DEDUP_DST_SYNC="1"
GGML_CUDA_REGISTER_HOST="1"

# ============================================================================
# End of editable configuration
# ============================================================================

die() {
    printf 'startspark.sh: %s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 0 ]]; then
    die "Keine Kommandozeilenparameter: Einstellungen oben in startspark.sh aendern."
fi

[[ -x "$SERVER" ]] || die "llama-server nicht ausfuehrbar: $SERVER"
[[ -f "$MODEL" ]] || die "Modell nicht gefunden: $MODEL"
if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
    [[ -f "$CHAT_TEMPLATE_FILE" ]] || die "Chat-Template nicht gefunden: $CHAT_TEMPLATE_FILE"
    [[ "$JINJA" == 1 ]] || die "CHAT_TEMPLATE_FILE benoetigt JINJA=1."
fi
if [[ -n "$API_KEY_FILE" && ! -f "$API_KEY_FILE" ]]; then
    die "API-Key-Datei nicht gefunden: $API_KEY_FILE"
fi
case "$REASONING_FORMAT" in none|deepseek|deepseek-legacy|auto) ;; *) die "REASONING_FORMAT muss none, deepseek, deepseek-legacy oder auto sein." ;; esac
case "$SPEC_MODE" in none|ngram|ngram-simple) ;; *) die "SPEC_MODE muss none oder ngram sein." ;; esac
case "$CPU_MOE" in 0|1) ;; *) die "CPU_MOE muss 0 oder 1 sein." ;; esac
case "$FLASH_ATTN" in on|off|auto) ;; *) die "FLASH_ATTN muss on, off oder auto sein." ;; esac
[[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] || die "CONTEXT muss eine positive Ganzzahl sein."
[[ "$SPEC_DRAFT_N_MAX" =~ ^[1-9][0-9]*$ ]] || die "SPEC_DRAFT_N_MAX muss eine positive Ganzzahl sein."
[[ "$CMOE_PREFILL_BATCH" =~ ^[1-9][0-9]*$ ]] || die "CMOE_PREFILL_BATCH muss eine positive Ganzzahl sein."
[[ "$CMOE_DECODE_BATCH" =~ ^[1-9][0-9]*$ ]] || die "CMOE_DECODE_BATCH muss eine positive Ganzzahl sein."
if [[ "$CACHE_IDLE_SLOTS" == 1 && "$CACHE_RAM" == 0 ]]; then
    die "CACHE_IDLE_SLOTS=1 benoetigt CACHE_RAM ungleich 0."
fi

case "$SPEC_MODE" in
    none) SPEC_TYPE="none" ;;
    ngram|ngram-simple) SPEC_TYPE="ngram-simple" ;;
esac

if [[ -z "$API_KEY" && -z "$API_KEY_FILE" ]]; then
    printf 'WARNUNG: API_KEY/API_KEY_FILE ist leer; der Dienst ist ohne Authentifizierung im LAN erreichbar.\n' >&2
fi

cat <<EOF
llama-wackMall-hybrid start (spark-x2.5-4b)
  project:   $PROJECT_ROOT
  server:    $SERVER
  model:     $MODEL
  listen:    $HOST:$PORT
  GPU:       $CUDA_VISIBLE_DEVICES_VALUE
  context:   $CONTEXT
  spec:      $SPEC_MODE ($SPEC_TYPE) n_max=$SPEC_DRAFT_N_MAX
  KV:        $TARGET_TYPE_K/$TARGET_TYPE_V
  ngl:       $N_GPU_LAYERS  cpu-moe=$CPU_MOE
  phase:     prefill=$CMOE_PREFILL_BATCH/$CMOE_PREFILL_UBATCH decode=$CMOE_DECODE_BATCH/$CMOE_DECODE_UBATCH
  cache:     ram=$CACHE_RAM MiB prompt=$CACHE_PROMPT reuse=$CACHE_REUSE idle=$CACHE_IDLE_SLOTS ckpt=$CTX_CHECKPOINTS
  sampling:  temp=$TEMP top-k=$TOP_K top-p=$TOP_P min-p=$MIN_P
  reasoning: $REASONING_BUDGET format=$REASONING_FORMAT
  template:  ${CHAT_TEMPLATE_FILE:-<model metadata>}
EOF

env_args=(
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE"
    "LLAMA_ARG_HOST=$HOST"
    "LLAMA_ARG_PORT=$PORT"
    "LLAMA_ARG_CORS_ORIGINS=$CORS_ORIGINS"
    "LLAMA_ARG_ALIAS=$MODEL_ALIAS"
    "LLAMA_ARG_CTX_SIZE=$CONTEXT"
    "LLAMA_ARG_N_PREDICT=$N_PREDICT"
    "LLAMA_ARG_N_GPU_LAYERS=$N_GPU_LAYERS"
    "LLAMA_ARG_N_PARALLEL=$N_PARALLEL"
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
    "LLAMA_CMOE_BATCH=$CMOE_BATCH"
    "LLAMA_CMOE_UBATCH=$CMOE_UBATCH"
    "LLAMA_CMOE_PREFILL_BATCH=$CMOE_PREFILL_BATCH"
    "LLAMA_CMOE_PREFILL_UBATCH=$CMOE_PREFILL_UBATCH"
    "LLAMA_CMOE_DECODE_BATCH=$CMOE_DECODE_BATCH"
    "LLAMA_CMOE_DECODE_UBATCH=$CMOE_DECODE_UBATCH"
    "GGML_CUDA_MOE_MULTI_FUSION=$GGML_CUDA_MOE_MULTI_FUSION"
    "GGML_CUDA_MOE_COMBINE_FUSION=$GGML_CUDA_MOE_COMBINE_FUSION"
    "GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS"
    "GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS"
    "GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS"
    "GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=$GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS"
    "GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=$GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE"
    "GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=$GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0"
    "GGML_CUDA_ASYNC_HOST_COPY=$GGML_CUDA_ASYNC_HOST_COPY"
    "GGML_SCHED_ASYNC_D2H_COPY=$GGML_SCHED_ASYNC_D2H_COPY"
    "GGML_SCHED_DEDUP_DST_SYNC=$GGML_SCHED_DEDUP_DST_SYNC"
    "GGML_CUDA_REGISTER_HOST=$GGML_CUDA_REGISTER_HOST"
)

if [[ "$CPU_MOE" == 1 ]]; then
    env_args+=("LLAMA_ARG_CPU_MOE=1")
else
    env_args+=("LLAMA_ARG_NO_CPU_MOE=1")
fi
[[ -n "$CHAT_TEMPLATE_FILE" ]] && env_args+=("LLAMA_ARG_CHAT_TEMPLATE_FILE=$CHAT_TEMPLATE_FILE")
[[ -n "$THREADS_HTTP" ]] && env_args+=("LLAMA_ARG_THREADS_HTTP=$THREADS_HTTP")

server_args=(
    -m "$MODEL"
    --temp "$TEMP"
    --top-k "$TOP_K"
    --top-p "$TOP_P"
    --min-p "$MIN_P"
)
[[ "$OP_OFFLOAD" == 1 ]] && server_args+=(--op-offload)
[[ -n "$NGRAM_SIMPLE_SIZE_N" ]] && server_args+=(--spec-ngram-simple-size-n "$NGRAM_SIMPLE_SIZE_N")
[[ -n "$NGRAM_SIMPLE_SIZE_M" ]] && server_args+=(--spec-ngram-simple-size-m "$NGRAM_SIMPLE_SIZE_M")
[[ -n "$NGRAM_SIMPLE_MIN_HITS" ]] && server_args+=(--spec-ngram-simple-min-hits "$NGRAM_SIMPLE_MIN_HITS")
[[ -n "$API_KEY" ]] && server_args+=(--api-key "$API_KEY")
[[ -n "$API_KEY_FILE" ]] && server_args+=(--api-key-file "$API_KEY_FILE")

unset_args=(
    -u GGML_CUDA_DISABLE_GRAPHS
    -u LLAMA_EXPERT_S
    -u LLAMA_EXPERT_HOT
    -u LLAMA_EXPERT_PLACEMENT
    -u LLAMA_KVFLASH
    -u LLAMA_KVFLASH_MAX_POOL
    -u LLAMA_ARG_OVERRIDE_TENSOR
    -u LLAMA_ARG_SPEC_DRAFT_MODEL
    -u LLAMA_TURBOQUANT_LIVE_MASK_LAYER
    -u LLAMA_TURBOQUANT_CAPTURE_VALUES
)
if [[ "$CPU_MOE" == 1 ]]; then
    unset_args+=(-u LLAMA_ARG_NO_CPU_MOE)
else
    unset_args+=(-u LLAMA_ARG_CPU_MOE)
fi
[[ -n "$THREADS_HTTP" ]] || unset_args+=(-u LLAMA_ARG_THREADS_HTTP)

exec env "${unset_args[@]}" "${env_args[@]}" "$SERVER" "${server_args[@]}"
