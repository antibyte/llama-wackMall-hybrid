#!/usr/bin/env bash
# Gemma 4 26B-A4B Q4_K_M launcher tuned for the local GTX 1660 Ti / Ryzen 4800H.
# Benchmark date: 2026-08-19. The default profile is balanced across 12 prompts.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

SERVER="${GEMMA4_SERVER:-$PROJECT_ROOT/build-main-sm75/bin/llama-server}"
MODEL="${GEMMA4_MODEL:-$HOME/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
PROFILE="${GEMMA4_PROFILE:-$PROJECT_ROOT/profiles/gemma4-26b-a4b-general-20260819.csv}"
HOST="${GEMMA4_HOST:-127.0.0.1}"
PORT="${GEMMA4_PORT:-8080}"
CONTEXT="${GEMMA4_CONTEXT:-8192}"
THREADS="${GEMMA4_THREADS:-8}"
ALIAS="${GEMMA4_ALIAS:-gemma-4-26b-a4b-q4-hybrid}"

die() {
    printf 'start-gemma4-26b-a4b.sh: %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die "use GEMMA4_* environment variables instead of arguments"
[[ -x "$SERVER" ]] || die "llama-server is not executable: $SERVER"
[[ -f "$MODEL" ]] || die "model not found: $MODEL"
[[ -f "$PROFILE" ]] || die "expert profile not found: $PROFILE"

# The S=21 path peaks near 5.22 GiB on a 5.61-GiB usable GPU. The runtime
# clamps S when less VRAM is free, while 256 MiB remains reserved for graphs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LLAMA_EXPERT_HOT="$PROFILE"
export LLAMA_EXPERT_S="21"
export LLAMA_EXPERT_VRAM_RESERVE_MIB="256"
export LLAMA_EXPERT_ADAPT="0"
export LLAMA_EXPERT_STATS="0"
export LLAMA_EXPERT_STATS_JSON="0"
export LLAMA_EXPERT_USAGE="0"
export LLAMA_EXPERT_WARM_SLOTS="0"
export LLAMA_EXPERT_STATIC_NO_SYNC="1"
export LLAMA_EXPERT_CPU_CHUNK="64"
export LLAMA_EXPERT_CPU_ACT_PARALLEL="0"
export LLAMA_EXPERT_CPU_ASYNC="0"
export LLAMA_EXPERT_CPU_DOWN_PREFETCH="0"
export LLAMA_EXPERT_CPU_REUSE_ROWS="0"
export LLAMA_EXPERT_CPU_MULTI_ROW="0"
export LLAMA_EXPERT_CPU_FUSED_GATE_UP="1"
export LLAMA_EXPERT_SHARED_HOT_IDS="1"
export LLAMA_EXPERT_SKIP_SENTINEL="1"

export LLAMA_CMOE_PREFILL_BATCH="512"
export LLAMA_CMOE_PREFILL_UBATCH="512"
export LLAMA_CMOE_DECODE_BATCH="32"
export LLAMA_CMOE_DECODE_UBATCH="32"

export GGML_CUDA_MOE_COMBINE_FUSION="1"
export GGML_CUDA_ASYNC_HOST_COPY="1"
export GGML_SCHED_DEDUP_DST_SYNC="1"
# export GGML_CUDA_REGISTER_HOST=1  # mmap pin; 21 GiB failed on 1660 Ti
# export GGML_SCHED_PREFETCH_EXPERTS=1  # n-cpu-moe prefill overlap; keep off without -ncmoe

# Fixed download provenance:
#   Hugging Face revision c099eb48e663fd284577b04978a94ffccb261841
#   model SHA-256 f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f
#   profile SHA-256 71941eb3e6a902da470ee1ed1c1d1c924478b92bd2372c2f10649f2561a0a67d
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
    -c "$CONTEXT" \
    -b 64 \
    -ub 64 \
    -ctk q8_0 \
    -ctv q8_0 \
    -fa on \
    -np 1 \
    -t "$THREADS" \
    -tb "$THREADS" \
    --backend-sampling \
    --op-offload \
    --load-mode none
