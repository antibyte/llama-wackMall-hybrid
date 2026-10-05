#!/usr/bin/env bash
# Kolibri-1 Q4_K_M on the local GTX 1660 Ti.
# The Q4 file is larger than RAM. Routed experts stay in the mmap and fault
# from the SSD. Attention, norms, the router and the shared expert sit in VRAM,
# plus the hottest routed experts of a chat usage profile (auto-fit, S=28 on 6 GiB).
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

SERVER="${KOLIBRI_SERVER:-$PROJECT_ROOT/build-main-sm75/bin/llama-server}"
MODEL="${KOLIBRI_MODEL:-$HOME/models/kolibri-1/Kolibri-1-Q4_K_M.gguf}"
HOST="${KOLIBRI_HOST:-127.0.0.1}"
PORT="${KOLIBRI_PORT:-8091}"
CONTEXT="${KOLIBRI_CONTEXT:-8192}"
THREADS="${KOLIBRI_THREADS:-8}"
ALIAS="${KOLIBRI_ALIAS:-kolibri-1-q4}"

die() {
    printf 'start-kolibri-1.sh: %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die "use KOLIBRI_* environment variables instead of arguments"
[[ -x "$SERVER" ]] || die "llama-server is not executable: $SERVER"
[[ -f "$MODEL" ]] || die "model not found: $MODEL (run ./download-kolibri-1.sh)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LLAMA_ARG_POWER_BUSY_CMD="${LLAMA_ARG_POWER_BUSY_CMD-busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Performance}"
export LLAMA_ARG_POWER_IDLE_CMD="${LLAMA_ARG_POWER_IDLE_CMD-busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Battery}"
export LLAMA_ARG_POWER_IDLE_DELAY="${LLAMA_ARG_POWER_IDLE_DELAY:-2000}"
export LLAMA_ARG_DECISION_SEQS="${LLAMA_ARG_DECISION_SEQS:-0}"

# Do not MAP_POPULATE or pin the 44 GiB mapping. Cold experts fault from disk.
# Readahead stays on. RANDOM advice turned each slab into 4 KiB reads.
export LLAMA_MMAP_PREFETCH=0
# Read missing expert slabs ahead in parallel before the CPU matmul touches them.
export GGML_CPU_EXPERT_READAHEAD="${GGML_CPU_EXPERT_READAHEAD:-1}"
# Static hot set from profiles/kolibri-1-chat.csv; S is auto-fit unless LLAMA_EXPERT_S is set.
# ADAPT=0 keeps CUDA graphs. A 256 MiB reserve left no room to regrow the
# prefill buffer after decode; the server then fell back to ubatch 64.
export LLAMA_EXPERT_HOT="${LLAMA_EXPERT_HOT-$PROJECT_ROOT/profiles/kolibri-1-chat.csv}"
export LLAMA_EXPERT_ADAPT="${LLAMA_EXPERT_ADAPT:-0}"
export LLAMA_EXPERT_VRAM_RESERVE_MIB="${LLAMA_EXPERT_VRAM_RESERVE_MIB:-500}"
export LLAMA_EXPERT_STATS="${LLAMA_EXPERT_STATS:-0}"
export LLAMA_EXPERT_STATS_JSON="${LLAMA_EXPERT_STATS_JSON:-0}"
export LLAMA_EXPERT_USAGE="${LLAMA_EXPERT_USAGE:-0}"
export LLAMA_EXPERT_WARM_SLOTS="${LLAMA_EXPERT_WARM_SLOTS:-0}"
export LLAMA_EXPERT_STATIC_NO_SYNC="${LLAMA_EXPERT_STATIC_NO_SYNC:-1}"
export LLAMA_EXPERT_CPU_CHUNK="${LLAMA_EXPERT_CPU_CHUNK:-64}"
export LLAMA_EXPERT_CPU_FUSED_GATE_UP="${LLAMA_EXPERT_CPU_FUSED_GATE_UP:-1}"
export LLAMA_EXPERT_SHARED_HOT_IDS="${LLAMA_EXPERT_SHARED_HOT_IDS:-1}"
export LLAMA_EXPERT_SKIP_SENTINEL="${LLAMA_EXPERT_SKIP_SENTINEL:-1}"
# Read ahead the top-8 predicted cold experts of the next layer: +23% on the first
# tokens after a cold start, -2% over a 1024-token answer (extra GPU router work).
export LLAMA_EXPERT_PAGE_PREFETCH="${LLAMA_EXPERT_PAGE_PREFETCH:-0}"
# Every ubatch reads nearly all experts, so prefill wants one large ubatch.
export LLAMA_CMOE_PREFILL_BATCH="${LLAMA_CMOE_PREFILL_BATCH:-2048}"
export LLAMA_CMOE_PREFILL_UBATCH="${LLAMA_CMOE_PREFILL_UBATCH:-2048}"
export LLAMA_CMOE_DECODE_BATCH="${LLAMA_CMOE_DECODE_BATCH:-64}"
export LLAMA_CMOE_DECODE_UBATCH="${LLAMA_CMOE_DECODE_UBATCH:-64}"

# --no-op-offload: copying the used experts over PCIe (x8, ~6.6 GB/s) was slower
# than computing them on the CPU from RAM, even at ubatch 2048.
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
    -ctk q8_0 \
    -ctv q8_0 \
    -fa on \
    -np 1 \
    -t "$THREADS" \
    -tb "$THREADS" \
    --no-op-offload \
    --chat-template-file "$PROJECT_ROOT/models/templates/kolibri-1.jinja" \
    --load-mode mmap
