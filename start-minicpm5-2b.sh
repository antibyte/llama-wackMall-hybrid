#!/usr/bin/env bash
#
# llama-wackMall-hybrid GTX 1660 Ti launcher for MiniCPM5-2B (Llama dense).
#
# Alle Einstellungen stehen in diesem Block. Keine Kommandozeilenparameter,
# keine stillen Shell-Overrides. Nach einer Aenderung: ./start-minicpm5-2b.sh
#
# MiniCPM5-2B Q4_K_M (1.56 GB), official openbmb/MiniCPM5-2B-GGUF.
# Standard LlamaForCausalLM, 42 layers, GQA 16/2, native ctx 131072.
# llama-bench 2026-09-08 GTX 1660 Ti:
#   default FA+q8 KV: pp512 516.8 t/s, tg128 114.1 t/s
#   winner: Q4_K ncols1=2 + Q6_K ncols1=2 -> tg128 123.1 t/s (+8%)
#   FA required for q8/q4 KV (FA off failed). turbo4_k garbled.
#   ngram-simple slower on Jinja think (108 vs 114 t/s). spec stays none.
#   pp2048 ubatch 1024 is 471 t/s vs 453 at 256; 2048 is within 0.4%.
# MoE/MTP/DFlash/DSpark knobs are inert. KVFlash loads (tested -c 32768).
# Sampling from the model card: temp=1.0 top_p=0.95. Think-only mode.
# Download: ./download-minicpm5-2b.sh
# Results: benchmark-results/minicpm5-2b-tune-20260908T203601Z/
#
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# ============================================================================
# EDITABLE CONFIGURATION -- only edit values in this section
# ============================================================================

SERVER="$PROJECT_ROOT/build-main-sm75/bin/llama-server"
MODEL="$HOME/models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf"
CHAT_TEMPLATE_FILE="$PROJECT_ROOT/models/templates/openbmb-MiniCPM5-2B.jinja"

# Network / OpenWebUI
HOST="0.0.0.0"
PORT="8080"
CORS_ORIGINS="*"
API_KEY=""
API_KEY_FILE=""
MODEL_ALIAS="minicpm5-2b"
CUDA_VISIBLE_DEVICES_VALUE="0"
N_PARALLEL="1"
N_GPU_LAYERS="99"
CPU_MOE="0"
UI="0"
CONT_BATCHING="1"
OP_OFFLOAD="1"
FIT="on"
FIT_TARGET="80"
FIT_CTX="2048"

# Context, KV, Flash Attention
CONTEXT="131072"
N_PREDICT="32768"
TARGET_TYPE_K="q8_0"
TARGET_TYPE_V="q8_0"
FLASH_ATTN="on"
KV_OFFLOAD="1"
LOAD_MODE="mmap"
OFFLINE="1"
# Resident KV tokens. 8192 keeps 131k logical ctx inside 6 GiB; 0 disables paging.
LLAMA_KVFLASH="8192"
LLAMA_KVFLASH_MAX_POOL="8192"
LLAMA_KVFLASH_TAU="64"
LLAMA_KVFLASH_POLICY="lru"
LLAMA_KVFLASH_STATS="0"

# Speculative decoding: none | ngram
SPEC_MODE="none"
SPEC_DRAFT_N_MAX="4"
NGRAM_SIMPLE_SIZE_N=""
NGRAM_SIMPLE_SIZE_M=""
NGRAM_SIMPLE_MIN_HITS=""

# Sampling (MiniCPM5-2B model card, think mode)
TEMP="1.0"
TOP_K="0"
TOP_P="0.95"
MIN_P="0"

# Reasoning / chat template
REASONING="1"
REASONING_BUDGET="8000"
REASONING_PRESERVE="1"
REASONING_FORMAT="auto"
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

# sm_75 kernel knobs (same winners as start1660 / startspark)
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
    printf 'start-minicpm5-2b.sh: %s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 0 ]]; then
    die "Keine Kommandozeilenparameter: Einstellungen oben in start-minicpm5-2b.sh aendern."
fi

[[ -x "$SERVER" ]] || die "llama-server nicht ausfuehrbar: $SERVER"
[[ -f "$MODEL" ]] || die "Modell nicht gefunden: $MODEL (zuerst ./download-minicpm5-2b.sh)"
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
case "$FIT" in on|off|1|0|true|false) ;; *) die "FIT muss on oder off sein." ;; esac
[[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] || die "CONTEXT muss eine positive Ganzzahl sein."
[[ "$SPEC_DRAFT_N_MAX" =~ ^[1-9][0-9]*$ ]] || die "SPEC_DRAFT_N_MAX muss eine positive Ganzzahl sein."
[[ "$CMOE_PREFILL_BATCH" =~ ^[1-9][0-9]*$ ]] || die "CMOE_PREFILL_BATCH muss eine positive Ganzzahl sein."
[[ "$CMOE_DECODE_BATCH" =~ ^[1-9][0-9]*$ ]] || die "CMOE_DECODE_BATCH muss eine positive Ganzzahl sein."
[[ "$FIT_TARGET" =~ ^[0-9]+$ ]] || die "FIT_TARGET muss eine nichtnegative Ganzzahl sein."
[[ "$FIT_CTX" =~ ^[1-9][0-9]*$ ]] || die "FIT_CTX muss eine positive Ganzzahl sein."
[[ "$LLAMA_KVFLASH" == 0 || "$LLAMA_KVFLASH" == auto || "$LLAMA_KVFLASH" =~ ^[1-9][0-9]*$ ]] || \
    die "LLAMA_KVFLASH muss 0, auto oder eine positive Ganzzahl sein."
if [[ "$LLAMA_KVFLASH" =~ ^[1-9][0-9]*$ ]] && (( LLAMA_KVFLASH < 512 )); then
    die "LLAMA_KVFLASH ist eine Tokenanzahl (mindestens 512), kein Boolean."
fi
[[ "$LLAMA_KVFLASH_MAX_POOL" =~ ^[1-9][0-9]*$ ]] || die "LLAMA_KVFLASH_MAX_POOL muss eine positive Ganzzahl sein."
case "$LLAMA_KVFLASH_POLICY" in lru) ;; *) die "LLAMA_KVFLASH_POLICY muss lru sein." ;; esac
if [[ "$LLAMA_KVFLASH" != 0 && "$FLASH_ATTN" != on ]]; then
    die "KVFlash benoetigt FLASH_ATTN=on."
fi
if [[ "$LLAMA_KVFLASH" != 0 && "$N_PARALLEL" != 1 ]]; then
    die "KVFlash benoetigt N_PARALLEL=1."
fi
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
llama-wackMall-hybrid start (minicpm5-2b)
  project:   $PROJECT_ROOT
  server:    $SERVER
  model:     $MODEL
  listen:    $HOST:$PORT
  GPU:       $CUDA_VISIBLE_DEVICES_VALUE
  context:   $CONTEXT
  spec:      $SPEC_MODE ($SPEC_TYPE) n_max=$SPEC_DRAFT_N_MAX
  KV:        $TARGET_TYPE_K/$TARGET_TYPE_V
  KVFlash:   $LLAMA_KVFLASH (max=$LLAMA_KVFLASH_MAX_POOL policy=$LLAMA_KVFLASH_POLICY)
  ngl:       $N_GPU_LAYERS  cpu-moe=$CPU_MOE  fit=$FIT/$FIT_TARGET/$FIT_CTX
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
    "LLAMA_ARG_FIT=$FIT"
    "LLAMA_ARG_FIT_TARGET=$FIT_TARGET"
    "LLAMA_ARG_FIT_CTX=$FIT_CTX"
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
if [[ "$LLAMA_KVFLASH" != 0 ]]; then
    env_args+=(
        "LLAMA_KVFLASH=$LLAMA_KVFLASH"
        "LLAMA_KVFLASH_MAX_POOL=$LLAMA_KVFLASH_MAX_POOL"
        "LLAMA_KVFLASH_TAU=$LLAMA_KVFLASH_TAU"
        "LLAMA_KVFLASH_POLICY=$LLAMA_KVFLASH_POLICY"
        "LLAMA_KVFLASH_STATS=$LLAMA_KVFLASH_STATS"
    )
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
    -u LLAMA_ARG_OVERRIDE_TENSOR
    -u LLAMA_ARG_SPEC_DRAFT_MODEL
    -u LLAMA_TURBOQUANT_LIVE_MASK_LAYER
    -u LLAMA_TURBOQUANT_CAPTURE_VALUES
)
if [[ "$LLAMA_KVFLASH" == 0 ]]; then
    unset_args+=(
        -u LLAMA_KVFLASH
        -u LLAMA_KVFLASH_MAX_POOL
        -u LLAMA_KVFLASH_TAU
        -u LLAMA_KVFLASH_POLICY
        -u LLAMA_KVFLASH_STATS
    )
fi
if [[ "$CPU_MOE" == 1 ]]; then
    unset_args+=(-u LLAMA_ARG_NO_CPU_MOE)
else
    unset_args+=(-u LLAMA_ARG_CPU_MOE)
fi
[[ -n "$THREADS_HTTP" ]] || unset_args+=(-u LLAMA_ARG_THREADS_HTTP)

exec env "${unset_args[@]}" "${env_args[@]}" "$SERVER" "${server_args[@]}"
