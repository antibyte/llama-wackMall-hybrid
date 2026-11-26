#!/usr/bin/env bash
# Ling-3.0-flash Q4_K_M on the local GTX 1660 Ti.
# 127B MoE (512 experts top-8 + 1 shared, KDA + MLA hybrid, one MTP layer), 72.5 GiB:
# far larger than RAM. Routed experts stay in the mmap and fault from the SSD;
# attention, KDA, norms, routers, shared experts and the dense layers sit in VRAM,
# plus the hottest routed experts of a chat usage profile (auto-fit S=12 on 6 GiB).
# Bench 2026-10-30 (benchmark-results/ling-3-flash-20261030): decode ~7.5 t/s,
# prefill ~37 t/s for a 3.9k-token prompt, 7-11 t/s for short prompts; the SSD
# runs at its 0.9-1.2 GB/s limit while decoding.
# Default LING3_QUANT=expq3k loads the file from ./download-ling-3-flash-q3.sh (routed experts
# Q3_K, rest Q4_K_M, 52.7 GiB); q4km loads the plain splits the lines above describe. Bench 2026-11-14 (benchmark-results/ling-dspark-20261114, cold page
# and drive cache, A/B/A/B): decode 14.3 vs 7.5-8.4 t/s, after a 5.2k-token prompt 12.4 vs 7.6,
# its prefill 38.7 vs 29.1 t/s, auto-fit S=19. KLD 0.095 against Q4_K_M, same top token 91.3 %.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

SERVER="${LING3_SERVER:-$PROJECT_ROOT/build-main-sm75/bin/llama-server}"
QUANT="${LING3_QUANT:-expq3k}"
case "$QUANT" in
    q4km)   MODEL_DEFAULT="$HOME/models/ling-3-flash/Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf" ;;
    expq3k) MODEL_DEFAULT="$HOME/models/ling-3-flash/Ling-3.0-flash-Q4_K_M-expq3k.gguf" ;;
    *) printf 'start-ling-3-flash.sh: LING3_QUANT must be q4km or expq3k\n' >&2; exit 1 ;;
esac
MODEL="${LING3_MODEL:-$MODEL_DEFAULT}"
HOST="${LING3_HOST:-127.0.0.1}"
PORT="${LING3_PORT:-8094}"
DSPARK="${LING3_DSPARK:-0}"
# The draft keeps f16 K/V for 5 layers of 32 heads: 2.56 GiB at 32768 cells, 640 MiB at 8192.
if [[ "$DSPARK" == 1 ]]; then
    CONTEXT="${LING3_CONTEXT:-8192}"
else
    CONTEXT="${LING3_CONTEXT:-32768}"
fi
THREADS="${LING3_THREADS:-8}"
ALIAS="${LING3_ALIAS:-ling-3-flash}"

die() {
    printf 'start-ling-3-flash.sh: %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die "use LING3_* environment variables instead of arguments"
[[ -x "$SERVER" ]] || die "llama-server is not executable: $SERVER"
[[ -f "$MODEL" ]] || die "model not found: $MODEL (run ./download-ling-3-flash.sh or ./download-ling-3-flash-q3.sh)"

# DSpark draft (inclusionAI/Ling-3.0-flash-dspark, ./download-ling-3-flash-dspark.sh). Off by default:
# 2026-11-14, cold cache, A/B/A/B, 192 greedy tokens: 6.8-7.4 vs 7.6 t/s on Q4_K_M and 12.7-13.0
# vs 13.8 t/s on expq3k with n_max 3 (80-90 % of the drafts accepted). A k-token verify computes
# and reads k tokens' cold experts; only the GPU part is shared. With every expert in RAM
# (same prompt repeated) n_max 3 gives 19 vs 15.7 t/s, n_max 8 only 13.5.
# A Q4_K_M draft and q8_0 draft K/V (S 7-10 instead of 5) stay at 12.5-13.3 vs 14.2-14.4 t/s.
# Draft length 2 (acceptance 90-95 %) stays at 13.1-13.6 vs 13.7-14.3 t/s.
DRAFT="${LING3_DRAFT:-$HOME/models/ling-3-flash/Ling-3.0-flash-DSpark-Q8_0.gguf}"
case "$DSPARK" in 0|1) ;; *) die "LING3_DSPARK must be 0 or 1" ;; esac
if [[ "$DSPARK" == 1 ]]; then
    [[ -f "$DRAFT" ]] || die "draft not found: $DRAFT (run ./download-ling-3-flash-dspark.sh)"
    export LLAMA_ARG_SPEC_TYPE=draft-dspark
    export LLAMA_ARG_SPEC_DRAFT_MODEL="$DRAFT"
    export LLAMA_ARG_N_GPU_LAYERS_DRAFT=99
    export LLAMA_ARG_SPEC_DRAFT_N_MAX="${LLAMA_ARG_SPEC_DRAFT_N_MAX:-3}"
    export LLAMA_ARG_SPEC_DRAFT_N_MIN="${LLAMA_ARG_SPEC_DRAFT_N_MIN:-1}"
    # The confidence head predicts 1 - TV (rejection sampling), so it is conservative for greedy.
    export LLAMA_ARG_SPEC_DRAFT_P_MIN="${LLAMA_ARG_SPEC_DRAFT_P_MIN:-0.25}"
    export LLAMA_DFLASH_COMBINED="${LLAMA_DFLASH_COMBINED:-1}"
    # The expert auto-fit runs before the draft loads; leave room for its weights, K/V and buffers.
    export LLAMA_EXPERT_VRAM_RESERVE_MIB="${LLAMA_EXPERT_VRAM_RESERVE_MIB:-1900}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LLAMA_ARG_POWER_BUSY_CMD="${LLAMA_ARG_POWER_BUSY_CMD-busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Performance}"
export LLAMA_ARG_POWER_IDLE_CMD="${LLAMA_ARG_POWER_IDLE_CMD-busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Battery}"
export LLAMA_ARG_POWER_IDLE_DELAY="${LLAMA_ARG_POWER_IDLE_DELAY:-2000}"
export LLAMA_ARG_DECISION_SEQS="${LLAMA_ARG_DECISION_SEQS:-0}"

# Do not MAP_POPULATE or pin the mapping. Cold experts fault from disk.
export LLAMA_MMAP_PREFETCH=0
# Read missing expert slabs ahead in parallel before the CPU matmul touches them.
export GGML_CPU_EXPERT_READAHEAD="${GGML_CPU_EXPERT_READAHEAD:-1}"
# 2026-11-09 (Ling winners, A/B/A/B, same tokens): hints through the file and no touch pass,
# so the compute pulls the slabs in and the reads overlap it. Ling:
# fadvise decode +3.7 % (2026-11-06), no touch pass +13.6 % in the service (2026-11-08).
export LLAMA_EXPERT_FADVISE="${LLAMA_EXPERT_FADVISE:-1}"
export GGML_CPU_EXPERT_FAULT_TOUCH="${GGML_CPU_EXPERT_FAULT_TOUCH:-0}"
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
# Read the next layer's predicted cold experts ahead. 2026-11-14 with 8 predictions, decode only
# (MAX_TOKENS 1): expq3k 13.0-13.2 vs 13.8 t/s. On Q4_K_M 8.3 vs 7.6-7.8 t/s with 8 predictions
# and 5.3 t/s with 16 (both without the token limit). Off.
export LLAMA_EXPERT_PAGE_PREFETCH="${LLAMA_EXPERT_PAGE_PREFETCH:-0}"
export LLAMA_EXPERT_PAGE_PREFETCH_MAX_TOKENS="${LLAMA_EXPERT_PAGE_PREFETCH_MAX_TOKENS:-1}"
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
export LLAMA_CMOE_PREFILL_STEPDOWN="${LLAMA_CMOE_PREFILL_STEPDOWN:-1}"  # prefill ubatch that does not fit: halve it before decode takes over
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
