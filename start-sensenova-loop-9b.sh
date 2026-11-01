#!/usr/bin/env bash
#
# llama-wackMall-hybrid GTX 1660 Ti launcher for SenseNova-Loop-9B-Preview Q3_K_M.
#
# Alle Einstellungen stehen in diesem Block. Keine Kommandozeilenparameter,
# keine stillen Shell-Overrides. Nach einer Aenderung: ./start-sensenova-loop-9b.sh
#
# SenseNova-Loop-9B-Preview Q3_K_M (4.32 GiB), guideboardlabs GGUF of
# sensenova/SenseNova-Loop-9B-Preview. Arch sensenova_loop: Qwen3.5 9B hybrid
# trunk (linear attention, full attention every 4th layer), layers 4..27 run
# twice (56 passes from 32 stored layers), two mHC residual streams with a
# sinkhorn router around every sublayer. Text only. The guideboardlabs GGUF has
# no MTP head; download-sensenova-loop-9b.sh appends the trained head from the
# HF repo (mtp.safetensors) as block 56 (Q4_0, +131 MiB).
# Thinking is the template default.
# Bench 2026-11-01 GTX 1660 Ti, build-main-sm75 (agollm stopped):
#   tg128 23.2 t/s, pp512 257 t/s. Before the Q3_K matvec changes (SWAR byte
#   subtract, VDR 2) and Q3_K rows=2 it was 20.4 t/s.
#   Q3_K MMVQ ncols1 rows: 0 20.7, 1 20.0, 2 23.2, 4 20.9 t/s.
#   Prefill ub 256/512/1024: 254-258 t/s (compute-bound dp4a).
#   KV f16 vs q8_0: tg128 23.2/22.0, tg64 at 16k depth 21.2/18.8, at 32k q8_0 16.4.
#   ctx 20480 f16: 17.5k prompt 221 t/s, decode 20.9 t/s, VRAM peak 5468 MiB.
#   ctx 24576 f16 peaked at 5696 MiB (too tight); q8_0 ctx 32768: 5392 MiB, 17.4 t/s.
#   ngram-simple: code edit 63.8 t/s vs 22.7 (spec none), prose -1..-3 %.
#   ngram-mod n-match 16: code edit 47.9 t/s.
#   MTP (ctx 8192): n_max 1: edit 31.0, German prose 24.7, code 29.5 t/s (none 22.5).
#   n_max 2: 33.1 / 21.0 / 30.8 (German acceptance 34 %); p_min 0.5/0.75 no better.
#   ngram-simple + MTP n_max 1: edit 62.4, German 24.9, code 28.6 t/s.
#   MTP head Q4_0 = Q8_0 acceptance (154 vs 155 drafts), 116 MiB less VRAM.
#   MTP_PROCESS_CHUNK 8 -> 0: 13k prompt 199.8 -> 221.8 t/s (ngram only 225.4),
#   edit 63.2 -> 70.1 t/s; the 8-token catch-up decodes re-read the LM head.
#   Server sampling (temp 1.0) ngram vs ngram-mtp: edit 64.6/70.3, German
#   22.6/23.9, code 22.2/29.5, thinking 21.6/28.3 and 20.8/27.7 t/s.
#   VRAM with MTP (head 131 + rollback state 88 + KV + compute MiB): ctx 20480
#   5712 MiB (too tight), 18432 5590, 16384 5470-5516. Prefill ub 256 = 512 speed,
#   70 MiB less.
#   The router kernels cost 1.6 ms of 43 ms per token; a split reduction and an
#   overlapped sinkhorn were not faster.
# Sampling: generation_config.json temp=1.0 top_p=0.95 (top_k 20 as Qwen3.5).
# Download: ./download-sensenova-loop-9b.sh
# Results: benchmark-results/sensenova-loop-9b-20261101/
#
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# ============================================================================
# EDITABLE CONFIGURATION -- only edit values in this section
# ============================================================================

SERVER="$PROJECT_ROOT/build-main-sm75/bin/llama-server"
MODEL="$HOME/models/sensenova-loop-9b/SenseNova-Loop-9B-Preview-Q3_K_M-mtp-q4_0.gguf"  # trunk + MTP head
CHAT_TEMPLATE_FILE=""  # GGUF Jinja; generation prompt opens <think>

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
MODEL_ALIAS="sensenova-loop-9b"
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

# Context, KV, Flash Attention
CONTEXT="16384"  # f16 KV + MTP: 18432 peaked at 5590, 20480 at 5712 of 5748 MiB
N_PREDICT="16384"
TARGET_TYPE_K="f16"
TARGET_TYPE_V="f16"
FLASH_ATTN="on"
KV_OFFLOAD="1"
LOAD_MODE="mmap"
OFFLINE="1"
# Resident KV tokens; 0 disables paging. The whole 20k context fits, and LRU
# paging drops evicted chunks from attention.
LLAMA_KVFLASH="0"
LLAMA_KVFLASH_MAX_POOL="8192"
LLAMA_KVFLASH_TAU="64"
LLAMA_KVFLASH_POLICY="lru"
LLAMA_KVFLASH_STATS="0"

# Speculative decoding: none | ngram | mtp | ngram-mtp
# ngram-mtp: ngram-simple drafts first (edits), the MTP head when no n-gram matches.
SPEC_MODE="ngram-mtp"
SPEC_DRAFT_N_MAX="1"  # MTP draft tokens; ngram-simple drafts up to its size_m (48)
SPEC_DRAFT_P_MIN=""   # MTP stops drafting below this top-1 probability; empty = server default 0
MTP_PROCESS_CHUNK="0" # tokens per MTP catch-up decode, 0 = draft ubatch (fork default 8)
NGRAM_SIMPLE_SIZE_N=""
NGRAM_SIMPLE_SIZE_M=""
NGRAM_SIMPLE_MIN_HITS=""

# Sampling (generation_config.json, top_k from Qwen3.5)
TEMP="1.0"
TOP_K="20"
TOP_P="0.95"
MIN_P="0"
REPEAT_PENALTY="1.0"
PRESENCE_PENALTY="0"

# Reasoning / chat template
REASONING="1"
REASONING_BUDGET="4000"
REASONING_PRESERVE="1"
REASONING_FORMAT="auto"
JINJA="1"

THREADS="8"
THREADS_BATCH="8"
THREADS_HTTP=""
TARGET_BACKEND_SAMPLING="0"

# Phase batching
CMOE_BATCH="64"
CMOE_UBATCH="64"
CMOE_PREFILL_BATCH="2048"
CMOE_PREFILL_UBATCH="256"  # 256/512/1024 within 2 %; 256 saves 70 MiB for the MTP context
CMOE_DECODE_BATCH="64"
CMOE_DECODE_UBATCH="64"

# Prompt cache
CTX_CHECKPOINTS="8"
CACHE_RAM="4096"
CACHE_PROMPT="1"
CACHE_REUSE="0"
KV_UNIFIED="1"
CACHE_IDLE_SLOTS="1"

# sm_75 kernel knobs. Q3_K rows measured for this model, the rest as start1660.
GGML_CUDA_MMVQ_Q3_K_NCOLS1_ROWS="2"
GGML_CUDA_MOE_MULTI_FUSION="1"
GGML_CUDA_MOE_COMBINE_FUSION="1"
GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS="4"
GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS="0"
GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS="4"
GGML_CUDA_MMVQ_Q8_NCOLS5_ROWS="4"
GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS="2"
GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS="2"
GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS="4"
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
    printf 'start-sensenova-loop-9b.sh: %s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 0 ]]; then
    die "Keine Kommandozeilenparameter: Einstellungen oben in start-sensenova-loop-9b.sh aendern."
fi

[[ -x "$SERVER" ]] || die "llama-server nicht ausfuehrbar: $SERVER"
[[ -f "$MODEL" ]] || die "Modell nicht gefunden: $MODEL (zuerst ./download-sensenova-loop-9b.sh)"
if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
    [[ -f "$CHAT_TEMPLATE_FILE" ]] || die "Chat-Template nicht gefunden: $CHAT_TEMPLATE_FILE"
    [[ "$JINJA" == 1 ]] || die "CHAT_TEMPLATE_FILE benoetigt JINJA=1."
fi
if [[ -n "$API_KEY_FILE" && ! -f "$API_KEY_FILE" ]]; then
    die "API-Key-Datei nicht gefunden: $API_KEY_FILE"
fi
case "$REASONING_FORMAT" in none|deepseek|deepseek-legacy|auto) ;; *) die "REASONING_FORMAT muss none, deepseek, deepseek-legacy oder auto sein." ;; esac
case "$SPEC_MODE" in none|ngram|ngram-simple|mtp|ngram-mtp) ;; *) die "SPEC_MODE muss none, ngram, mtp oder ngram-mtp sein." ;; esac
case "$CPU_MOE" in 0|1) ;; *) die "CPU_MOE muss 0 oder 1 sein." ;; esac
case "$FLASH_ATTN" in on|off|auto) ;; *) die "FLASH_ATTN muss on, off oder auto sein." ;; esac
case "$FIT" in on|off|1|0|true|false) ;; *) die "FIT muss on oder off sein." ;; esac
[[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] || die "CONTEXT muss eine positive Ganzzahl sein."
[[ "$SPEC_DRAFT_N_MAX" =~ ^[1-9][0-9]*$ ]] || die "SPEC_DRAFT_N_MAX muss eine positive Ganzzahl sein."
[[ "$MTP_PROCESS_CHUNK" =~ ^[0-9]+$ ]] || die "MTP_PROCESS_CHUNK muss eine nichtnegative Ganzzahl sein."
[[ -z "$SPEC_DRAFT_P_MIN" || "$SPEC_DRAFT_P_MIN" =~ ^(0|1|0?\.[0-9]+)$ ]] || die "SPEC_DRAFT_P_MIN muss leer oder zwischen 0 und 1 sein."
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
    mtp) SPEC_TYPE="draft-mtp" ;;
    ngram-mtp) SPEC_TYPE="ngram-simple,draft-mtp" ;;
esac

if [[ -z "$API_KEY" && -z "$API_KEY_FILE" ]]; then
    printf 'WARNUNG: API_KEY/API_KEY_FILE ist leer; der Dienst ist ohne Authentifizierung im LAN erreichbar.\n' >&2
fi

cat <<EOF
llama-wackMall-hybrid start (sensenova-loop-9b)
  project:   $PROJECT_ROOT
  server:    $SERVER
  model:     $MODEL
  listen:    $HOST:$PORT
  GPU:       $CUDA_VISIBLE_DEVICES_VALUE
  context:   $CONTEXT
  spec:      $SPEC_MODE ($SPEC_TYPE) n_max=$SPEC_DRAFT_N_MAX p_min=${SPEC_DRAFT_P_MIN:-default}
  KV:        $TARGET_TYPE_K/$TARGET_TYPE_V
  KVFlash:   $LLAMA_KVFLASH (max=$LLAMA_KVFLASH_MAX_POOL policy=$LLAMA_KVFLASH_POLICY)
  ngl:       $N_GPU_LAYERS  cpu-moe=$CPU_MOE  fit=$FIT/$FIT_TARGET/$FIT_CTX
  phase:     prefill=$CMOE_PREFILL_BATCH/$CMOE_PREFILL_UBATCH decode=$CMOE_DECODE_BATCH/$CMOE_DECODE_UBATCH
  cache:     ram=$CACHE_RAM MiB prompt=$CACHE_PROMPT reuse=$CACHE_REUSE idle=$CACHE_IDLE_SLOTS ckpt=$CTX_CHECKPOINTS
  sampling:  temp=$TEMP top-k=$TOP_K top-p=$TOP_P min-p=$MIN_P repeat=$REPEAT_PENALTY presence=$PRESENCE_PENALTY
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
    "LLAMA_MTP_PROCESS_CHUNK=$MTP_PROCESS_CHUNK"
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
    "GGML_CUDA_MMVQ_Q8_NCOLS5_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS5_ROWS"
    "GGML_CUDA_MMVQ_Q3_K_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q3_K_NCOLS1_ROWS"
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
[[ -n "$SPEC_DRAFT_P_MIN" ]] && env_args+=("LLAMA_ARG_SPEC_DRAFT_P_MIN=$SPEC_DRAFT_P_MIN")

server_args=(
    -m "$MODEL"
    --temp "$TEMP"
    --top-k "$TOP_K"
    --top-p "$TOP_P"
    --min-p "$MIN_P"
    --repeat-penalty "$REPEAT_PENALTY"
    --presence-penalty "$PRESENCE_PENALTY"
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
    -u LLAMA_ARG_MMPROJ
    -u LLAMA_ARG_MMPROJ_URL
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
[[ -n "$SPEC_DRAFT_P_MIN" ]] || unset_args+=(-u LLAMA_ARG_SPEC_DRAFT_P_MIN)

[[ -n "$POWER_BUSY_CMD" ]] && env_args+=("LLAMA_ARG_POWER_BUSY_CMD=$POWER_BUSY_CMD")
[[ -n "$POWER_IDLE_CMD" ]] && env_args+=("LLAMA_ARG_POWER_IDLE_CMD=$POWER_IDLE_CMD")
env_args+=("LLAMA_ARG_POWER_IDLE_DELAY=$POWER_IDLE_DELAY")
env_args+=("LLAMA_ARG_DECISION_SEQS=$DECISION_SEQS")

exec env "${unset_args[@]}" "${env_args[@]}" "$SERVER" "${server_args[@]}"
