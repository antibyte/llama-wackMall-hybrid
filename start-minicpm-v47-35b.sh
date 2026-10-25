#!/usr/bin/env bash
# MiniCPM-V-4.7-35B-A3B AD-Q4_K-IQ4_XS (qwen35moe) on the GTX 1660 Ti, text only.
# Mirrors the agollm recipe config/llama/minicpm-v4.7-35b-a3b.yaml.
# 40 hot experts per layer from profiles/minicpm-v47-chat.csv sit in VRAM, cold experts
# run on the CPU from the mmap. LLAMA_EXPERT_COLD_FIRST plus GGML_SCHED_DEFER_HOST_INPUTS
# run the hot and shared experts while the CPU computes the cold ones.
# 2026-10-25, temp 1.0, ABBA against the old CPU-MoE + DSpark recipe: 31.8 -> 41.0 t/s.
# The DSpark draft is off: its context needs ~1.6 GiB, which holds 24 more hot slots per layer.
# Peak VRAM 5.27 GiB, 5.38 GiB with a 9k-token prompt (pp 371 t/s). S is clamped to free VRAM.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

SERVER="${MINICPM_V47_SERVER:-$PROJECT_ROOT/build-main-sm75/bin/llama-server}"
MODEL="${MINICPM_V47_MODEL:-$HOME/models/minicpm-v-4.7-35b-a3b/MiniCPM-V-4.7-35B-A3B-AD-Q4_K-IQ4_XS.gguf}"
HOST="${MINICPM_V47_HOST:-127.0.0.1}"
PORT="${MINICPM_V47_PORT:-8092}"
CONTEXT="${MINICPM_V47_CONTEXT:-262144}"
THREADS="${MINICPM_V47_THREADS:-8}"
ALIAS="${MINICPM_V47_ALIAS:-minicpm-v4.7-35b-a3b}"

die() {
    printf 'start-minicpm-v47-35b.sh: %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die "use MINICPM_V47_* environment variables instead of arguments"
[[ -x "$SERVER" ]] || die "llama-server is not executable: $SERVER"
[[ -f "$MODEL" ]] || die "model not found: $MODEL"

if [[ "${MINICPM_V47_IGNORE_GPU:-0}" != 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 { print; exit }')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used > 512 )); then
        die "GTX 1660 Ti is in use (${used} MiB). Stop agollm or set MINICPM_V47_IGNORE_GPU=1."
    fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LLAMA_ARG_POWER_BUSY_CMD="${LLAMA_ARG_POWER_BUSY_CMD-busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Performance}"
export LLAMA_ARG_POWER_IDLE_CMD="${LLAMA_ARG_POWER_IDLE_CMD-busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Battery}"
export LLAMA_ARG_POWER_IDLE_DELAY="${LLAMA_ARG_POWER_IDLE_DELAY:-2000}"
export LLAMA_ARG_DECISION_SEQS="${LLAMA_ARG_DECISION_SEQS:-0}"

# Base batch stays 2048; prefill uses one 2048 ubatch (expert copies over PCIe), decode 64.
export LLAMA_CMOE_BATCH="${LLAMA_CMOE_BATCH:-2048}"
export LLAMA_CMOE_UBATCH="${LLAMA_CMOE_UBATCH:-2048}"
export LLAMA_CMOE_PREFILL_BATCH="${LLAMA_CMOE_PREFILL_BATCH:-2048}"
export LLAMA_CMOE_PREFILL_UBATCH="${LLAMA_CMOE_PREFILL_UBATCH:-2048}"
export LLAMA_CMOE_DECODE_BATCH="${LLAMA_CMOE_DECODE_BATCH:-64}"
export LLAMA_CMOE_DECODE_UBATCH="${LLAMA_CMOE_DECODE_UBATCH:-64}"
# Static hot set. ADAPT=0 keeps CUDA graphs; S=32 plus 8 warm slots measured 44.1 vs 44.5 t/s.
export LLAMA_EXPERT_HOT="${LLAMA_EXPERT_HOT-$PROJECT_ROOT/profiles/minicpm-v47-chat.csv}"
export LLAMA_EXPERT_S="${LLAMA_EXPERT_S:-40}"
export LLAMA_EXPERT_VRAM_RESERVE_MIB="${LLAMA_EXPERT_VRAM_RESERVE_MIB:-512}"
export LLAMA_EXPERT_ADAPT="${LLAMA_EXPERT_ADAPT:-0}"
export LLAMA_EXPERT_WARM_SLOTS="${LLAMA_EXPERT_WARM_SLOTS:-0}"
export LLAMA_EXPERT_STATS="${LLAMA_EXPERT_STATS:-0}"
export LLAMA_EXPERT_SHARED_HOT_IDS="${LLAMA_EXPERT_SHARED_HOT_IDS:-1}"
export LLAMA_EXPERT_COLD_FIRST="${LLAMA_EXPERT_COLD_FIRST:-1}"
export GGML_SCHED_DEFER_HOST_INPUTS="${GGML_SCHED_DEFER_HOST_INPUTS:-1}"
export GGML_CUDA_ASYNC_HOST_COPY="${GGML_CUDA_ASYNC_HOST_COPY:-1}"
export GGML_SCHED_ASYNC_D2H_COPY="${GGML_SCHED_ASYNC_D2H_COPY:-1}"
export GGML_SCHED_DEDUP_DST_SYNC="${GGML_SCHED_DEDUP_DST_SYNC:-1}"
# 10 full-attention layers; the KVFlash pool keeps 262144 context within VRAM.
export LLAMA_KVFLASH="${LLAMA_KVFLASH:-8192}"
export LLAMA_KVFLASH_MAX_POOL="${LLAMA_KVFLASH_MAX_POOL:-8192}"
export LLAMA_KVFLASH_TAU="${LLAMA_KVFLASH_TAU:-64}"
export LLAMA_KVFLASH_POLICY="${LLAMA_KVFLASH_POLICY:-lru}"
export LLAMA_KVFLASH_STATS="${LLAMA_KVFLASH_STATS:-0}"
# Same server settings as the agollm recipe.
export LLAMA_ARG_REASONING="${LLAMA_ARG_REASONING:-1}"
export LLAMA_ARG_THINK="${LLAMA_ARG_THINK:-auto}"
export LLAMA_ARG_THINK_BUDGET="${LLAMA_ARG_THINK_BUDGET:-8000}"
export LLAMA_ARG_REASONING_PRESERVE="${LLAMA_ARG_REASONING_PRESERVE:-1}"
export LLAMA_ARG_CTX_CHECKPOINTS="${LLAMA_ARG_CTX_CHECKPOINTS:-8}"
export LLAMA_ARG_CACHE_RAM="${LLAMA_ARG_CACHE_RAM:-4096}"
export LLAMA_ARG_CACHE_PROMPT="${LLAMA_ARG_CACHE_PROMPT:-1}"
export LLAMA_ARG_CACHE_REUSE="${LLAMA_ARG_CACHE_REUSE:-0}"

exec "$SERVER" \
    -m "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --alias "$ALIAS" \
    --no-ui \
    --offline \
    --no-mmproj \
    -cmoe \
    -ngl 99 \
    -ncmoe 99 \
    -c "$CONTEXT" \
    -n 32768 \
    -b 2048 \
    -ub 2048 \
    -ctk q4_0 \
    -ctv q4_0 \
    -fa on \
    -np 1 \
    -t "$THREADS" \
    -tb "$THREADS" \
    --spec-type none \
    --op-offload \
    --jinja \
    --temp 1.0 --top-k 0 --top-p 0.95 --min-p 0 \
    --load-mode mmap
