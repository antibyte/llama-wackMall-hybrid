#!/usr/bin/env bash
#
# llama-wackMall-hybrid GTX 1660 Ti launcher for Spark-X2.6-4B Q6_K (spark2_5).
#
# Alle Einstellungen stehen in diesem Block. Keine Kommandozeilenparameter,
# keine stillen Shell-Overrides. Nach einer Aenderung: ./start-spark-x26.sh
#
# Same arch as Spark-X2.5-4B: 36 layers, 3 sliding + 1 full, window 512.
# llama-bench 2026-10-23, build-main-sm75, r=3, f16 KV, FA on, t=8,
# Q6_K MMVQ ncols1=2 ncols3=2: pp512 832.78 +/- 7.43, tg128 62.94 +/- 0.02.
# Without the Q6 row override tg128 was 54.83. q8_0 KV 60.05, q4_0 KV 59.19,
# FA off 57.35. t=4 62.15, t=16 62.03. pp2048 ubatch 512 and 1024 tied (~773);
# ubatch 2048 was 758.75. f16 at depth 8192: pp512 608.36, tg128 57.59.
# Context 49152 leaves 489 MiB VRAM free (non-SWA KV 1728 MiB, SWA 108 MiB).
# Do not use 65536: X2.5 f16 already OOM'd there, and Q6 weighs more.
# Sampling from the X2.6 card: temp=1 top_k=-1 top_p=0.95
# repetition_penalty=1 presence_penalty=0 frequency_penalty=0.
# Chat template: Spark2.6.jinja, thinking_effort high.
# KVFlash stays off (ISWA). Spec none. Stop agollm before starting; this binds :8080.
#
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# ============================================================================
# EDITABLE CONFIGURATION -- only edit values in this section
# ============================================================================

SERVER="$PROJECT_ROOT/build-main-sm75/bin/llama-server"
MODEL="$HOME/models/spark-x2.6-4b/Spark-X2.6-4B.Q6_K.gguf"
CHAT_TEMPLATE_FILE="$PROJECT_ROOT/models/templates/Spark2.6.jinja"

# Network / OpenWebUI
HOST="0.0.0.0"
PORT="8080"

# Power profile: Performance before model work, Battery once no llama-server of
# this user has worked for POWER_IDLE_DELAY ms. Battery also dims the panel to
# 10%; use ...PowerDaemon Balanced as idle command to avoid that. Empty
# commands disable switching. Servers coordinate via /tmp/llama-power-$UID.
POWER_BUSY_CMD="busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Performance"
POWER_IDLE_CMD="busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Battery"
POWER_IDLE_DELAY="2000"  # milliseconds without model work before POWER_IDLE_CMD runs
DECISION_SEQS="0"  # POST /v1/decision side context; agollm serves decisions on its own worker
CORS_ORIGINS="*"
API_KEY=""
API_KEY_FILE=""
MODEL_ALIAS="spark-x2.6-4b"
CUDA_VISIBLE_DEVICES_VALUE="0"
N_PARALLEL="1"
N_GPU_LAYERS="99"
CPU_MOE="0"
UI="0"
CONT_BATCHING="1"
OP_OFFLOAD="1"

# Context, KV, Flash Attention
# 2026-10-23: f16 KV beats q8_0 and q4_0 on this Q6_K. 49152 leaves 489 MiB free.
CONTEXT="49152"
N_PREDICT="32768"
TARGET_TYPE_K="f16"
TARGET_TYPE_V="f16"
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

# Sampling (XHToken Spark-X2.6 card)
TEMP="1.0"
TOP_K="-1"
TOP_P="0.95"
MIN_P="0"
REPEAT_PENALTY="1"
PRESENCE_PENALTY="0"
FREQUENCY_PENALTY="0"
CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"high"}'

# Reasoning / chat template
REASONING="1"
REASONING_BUDGET="4000"
REASONING_PRESERVE="1"
REASONING_FORMAT="deepseek"
JINJA="1"

THREADS="8"
THREADS_BATCH="8"
THREADS_HTTP=""
TARGET_BACKEND_SAMPLING="0"

# Phase batching
CMOE_BATCH="64"
CMOE_UBATCH="64"
CMOE_PREFILL_BATCH="2048"
CMOE_PREFILL_UBATCH="512"
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
GGML_CUDA_MMVQ_Q8_NCOLS5_ROWS="4"
GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS="2"
GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS="2"
GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS="2"
GGML_CUDA_MMVQ_Q6_K_NCOLS4_ROWS="4"
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
    printf 'start-spark-x26.sh: %s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 0 ]]; then
    die "Keine Kommandozeilenparameter: Einstellungen oben in start-spark-x26.sh aendern."
fi

[[ -x "$SERVER" ]] || die "llama-server nicht ausfuehrbar: $SERVER"
[[ -f "$MODEL" ]] || die "Modell nicht gefunden: $MODEL"
if [[ "${SPARK26_IGNORE_GPU:-0}" != 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 { print; exit }')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used > 512 )); then
        die "GTX 1660 Ti is in use (${used} MiB). Stop agollm or set SPARK26_IGNORE_GPU=1."
    fi
fi
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
llama-wackMall-hybrid start (spark-x2.6-4b)
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
  sampling:  temp=$TEMP top-k=$TOP_K top-p=$TOP_P min-p=$MIN_P repeat=$REPEAT_PENALTY presence=$PRESENCE_PENALTY frequency=$FREQUENCY_PENALTY thinking=high
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
    "LLAMA_ARG_CHAT_TEMPLATE_KWARGS=$CHAT_TEMPLATE_KWARGS"
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
    "GGML_CUDA_MMVQ_Q8_NCOLS5_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS5_ROWS"
    "GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=$GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS4_ROWS=$GGML_CUDA_MMVQ_Q6_K_NCOLS4_ROWS"
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
    --repeat-penalty "$REPEAT_PENALTY"
    --presence-penalty "$PRESENCE_PENALTY"
    --frequency-penalty "$FREQUENCY_PENALTY"
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

[[ -n "$POWER_BUSY_CMD" ]] && env_args+=("LLAMA_ARG_POWER_BUSY_CMD=$POWER_BUSY_CMD")
[[ -n "$POWER_IDLE_CMD" ]] && env_args+=("LLAMA_ARG_POWER_IDLE_CMD=$POWER_IDLE_CMD")
env_args+=("LLAMA_ARG_POWER_IDLE_DELAY=$POWER_IDLE_DELAY")
env_args+=("LLAMA_ARG_DECISION_SEQS=$DECISION_SEQS")

exec env "${unset_args[@]}" "${env_args[@]}" "$SERVER" "${server_args[@]}"
