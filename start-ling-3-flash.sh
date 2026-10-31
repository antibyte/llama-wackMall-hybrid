#!/usr/bin/env bash
# Ling-3.0-flash Q4_K_M on the local GTX 1660 Ti.
# 127B MoE (512 experts top-8 + 1 shared, KDA + MLA hybrid, one MTP layer), 72.5 GiB:
# far larger than RAM. Routed experts stay in the mmap and fault from the SSD;
# attention, KDA, norms, routers, shared experts and the dense layers sit in VRAM,
# plus the hottest routed experts of a chat usage profile (auto-fit S=12 on 6 GiB).
# Bench 2026-10-30 (benchmark-results/ling-3-flash-20261030): decode ~7.5 t/s,
# prefill ~37 t/s for a 3.9k-token prompt, 7-11 t/s for short prompts; the SSD
# runs at its 0.9-1.2 GB/s limit while decoding.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

SERVER="${LING3_SERVER:-$PROJECT_ROOT/build-main-sm75/bin/llama-server}"
MODEL="${LING3_MODEL:-$HOME/models/ling-3-flash/Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf}"
HOST="${LING3_HOST:-127.0.0.1}"
PORT="${LING3_PORT:-8094}"
CONTEXT="${LING3_CONTEXT:-32768}"
THREADS="${LING3_THREADS:-8}"
ALIAS="${LING3_ALIAS:-ling-3-flash}"

die() {
    printf 'start-ling-3-flash.sh: %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die "use LING3_* environment variables instead of arguments"
[[ -x "$SERVER" ]] || die "llama-server is not executable: $SERVER"
[[ -f "$MODEL" ]] || die "model not found: $MODEL (run ./download-ling-3-flash.sh)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LLAMA_ARG_POWER_BUSY_CMD="${LLAMA_ARG_POWER_BUSY_CMD-busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Performance}"
export LLAMA_ARG_POWER_IDLE_CMD="${LLAMA_ARG_POWER_IDLE_CMD-busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Battery}"
export LLAMA_ARG_POWER_IDLE_DELAY="${LLAMA_ARG_POWER_IDLE_DELAY:-2000}"
export LLAMA_ARG_DECISION_SEQS="${LLAMA_ARG_DECISION_SEQS:-0}"

# Do not MAP_POPULATE or pin the mapping. Cold experts fault from disk.
export LLAMA_MMAP_PREFETCH=0
# Read missing expert slabs ahead in parallel before the CPU matmul touches them.
export GGML_CPU_EXPERT_READAHEAD="${GGML_CPU_EXPERT_READAHEAD:-1}"
# Static hot set from profiles/ling-3-flash-chat.csv; S is auto-fit unless LLAMA_EXPERT_S is set.
# bailingmoe3 tiers only with a hot set or an explicit S. ADAPT=1 measured 7.41 vs 7.52 t/s.
export LLAMA_EXPERT_HOT="${LLAMA_EXPERT_HOT-$PROJECT_ROOT/profiles/ling-3-flash-chat.csv}"
export LLAMA_EXPERT_ADAPT="${LLAMA_EXPERT_ADAPT:-0}"
export LLAMA_EXPERT_VRAM_RESERVE_MIB="${LLAMA_EXPERT_VRAM_RESERVE_MIB:-500}"
# No expert mlock: 12 GiB gave 7.29 and 18 GiB 7.19 vs 7.43 t/s without; the page
# cache keeps the chat's experts better than the static profile.
export LLAMA_EXPERT_RAM_LOCK_GIB="${LLAMA_EXPERT_RAM_LOCK_GIB:-0}"
ulimit -l unlimited 2>/dev/null || ulimit -l "$(ulimit -Hl)" 2>/dev/null || true
export LLAMA_EXPERT_STATS="${LLAMA_EXPERT_STATS:-0}"
export LLAMA_EXPERT_STATS_JSON="${LLAMA_EXPERT_STATS_JSON:-0}"
export LLAMA_EXPERT_USAGE="${LLAMA_EXPERT_USAGE:-0}"
export LLAMA_EXPERT_WARM_SLOTS="${LLAMA_EXPERT_WARM_SLOTS:-0}"
export LLAMA_EXPERT_STATIC_NO_SYNC="${LLAMA_EXPERT_STATIC_NO_SYNC:-1}"
export LLAMA_EXPERT_CPU_CHUNK="${LLAMA_EXPERT_CPU_CHUNK:-64}"
export LLAMA_EXPERT_CPU_FUSED_GATE_UP="${LLAMA_EXPERT_CPU_FUSED_GATE_UP:-1}"
export LLAMA_EXPERT_SHARED_HOT_IDS="${LLAMA_EXPERT_SHARED_HOT_IDS:-1}"
export LLAMA_EXPERT_SKIP_SENTINEL="${LLAMA_EXPERT_SKIP_SENTINEL:-1}"
export LLAMA_EXPERT_PAGE_PREFETCH="${LLAMA_EXPERT_PAGE_PREFETCH:-0}"
# Copy the CPU splits' inputs from the GPU asynchronously, one sync per split:
# decode 7.50 -> 7.58 t/s, same tokens (SSD-bound; Kolibri +2.5 %).
export GGML_SCHED_ASYNC_D2H_COPY="${GGML_SCHED_ASYNC_D2H_COPY:-1}"
export GGML_SCHED_DEDUP_DST_SYNC="${GGML_SCHED_DEDUP_DST_SYNC:-1}"
# Compute buffers shrink only to at most RATIO and by at least MIN_MIB (prefill -> decode still frees the
# prefill peak; small shape switches no longer reallocate). 0 / 1 = shrink on any decrease.
export GGML_ALLOC_SHRINK_MIN_MIB="${GGML_ALLOC_SHRINK_MIN_MIB:-64}"
export GGML_ALLOC_SHRINK_RATIO="${GGML_ALLOC_SHRINK_RATIO:-0.5}"
# One kept graph per small ubatch shape; only speculative decoding switches shapes. SHAPES 0 = off.
export LLAMA_GRAPH_SMALL_TOKENS="${LLAMA_GRAPH_SMALL_TOKENS:-16}"
export LLAMA_GRAPH_SMALL_SHAPES="${LLAMA_GRAPH_SMALL_SHAPES:-8}"
# Every ubatch reads nearly all experts, so prefill wants one large ubatch. 4096 left
# room for only S=8 and prefilled at 15 t/s.
export LLAMA_CMOE_PREFILL_BATCH="${LLAMA_CMOE_PREFILL_BATCH:-2048}"
export LLAMA_CMOE_PREFILL_UBATCH="${LLAMA_CMOE_PREFILL_UBATCH:-2048}"
export LLAMA_CMOE_DECODE_BATCH="${LLAMA_CMOE_DECODE_BATCH:-64}"
export LLAMA_CMOE_DECODE_UBATCH="${LLAMA_CMOE_DECODE_UBATCH:-64}"

# MTP draft (--spec-type draft-mtp) is not used: n=1 accepted every draft and changed
# the greedy output (verify path suspect), n=2 was slower (6.85 t/s).
# --no-op-offload: copying the used experts over PCIe (x8, ~6.6 GB/s) lost to the
# CPU path on Kolibri.
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
    -fa on \
    -np 1 \
    -t "$THREADS" \
    -tb "$THREADS" \
    --no-op-offload \
    --jinja \
    --load-mode mmap
