#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible no-MTP KV quality comparison. The input corpus stays external;
# this script never modifies it and refuses to overwrite a result directory.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PPL="${PPL:-$ROOT/build-hybrid/bin/llama-perplexity}"
MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PROFILE="${PROFILE:-$HOME/models/qwen3.6-35b-a3b-mtp/luce-warmstart.csv}"
QUALITY_PROMPT_SOURCE="${QUALITY_PROMPT_SOURCE:-}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/benchmark-results/turboquant-quality-$(date -u +%Y%m%dT%H%M%SZ)}"

CONTEXT="${CONTEXT:-1024}"
BATCH="${BATCH:-128}"
UBATCH="${UBATCH:-128}"
CHUNKS="${CHUNKS:-1}"
FIXED_S="${FIXED_S:-33}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TURBO4_Q8_FALLBACK_LAYERS="${TURBO4_Q8_FALLBACK_LAYERS:-}"
QUALITY_CASES="${QUALITY_CASES:-q8-q4,q4-q8,q4-q4,turbo4-q8,q8-turbo4,turbo4-turbo4}"

if [[ -z "$QUALITY_PROMPT_SOURCE" ]]; then
    echo "Set QUALITY_PROMPT_SOURCE to a representative text corpus." >&2
    exit 1
fi

for path in "$PPL" "$MODEL" "$PROFILE" "$QUALITY_PROMPT_SOURCE"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing file: $path" >&2
        exit 1
    fi
done

if [[ -e "$RESULTS_DIR" ]]; then
    echo "Result directory already exists; refusing to overwrite: $RESULTS_DIR" >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR"

BASE_LOGITS="$RESULTS_DIR/q8-q8-base.kld"
COMMON_ARGS=(
    -m "$MODEL"
    -c "$CONTEXT"
    -b "$BATCH"
    -ub "$UBATCH"
    -fa on
    -cmoe
    --offline
    --chunks "$CHUNKS"
    --no-warmup
)

run_ppl() {
    local name="$1"
    local type_k="$2"
    local type_v="$3"
    local fallback_layers="$4"
    local q4_scale="$5"
    local attn_rot_disable="$6"
    shift 6

    local extra_env=()
    if [[ "$type_v" == turbo4_k ]]; then
        extra_env+=(LLAMA_TURBO4_V_EXPERIMENTAL=1)
    fi
    if [[ -n "$fallback_layers" ]]; then
        if [[ "$type_k" != turbo4_k ]]; then
            echo "Internal error: Q8 fallback requires turbo4_k" >&2
            return 1
        fi
        extra_env+=("LLAMA_TURBO4_Q8_FALLBACK_LAYERS=$fallback_layers")
    fi
    if [[ -n "$q4_scale" ]]; then
        extra_env+=("LLAMA_KV_Q4_SCALE=$q4_scale")
    fi
    if [[ -n "$attn_rot_disable" ]]; then
        extra_env+=("LLAMA_ATTN_ROT_DISABLE=$attn_rot_disable")
    fi

    echo "Running $name (K=$type_k, V=$type_v)"
    env \
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
        LLAMA_CMOE_BATCH="$BATCH" \
        LLAMA_CMOE_UBATCH="$UBATCH" \
        LLAMA_EXPERT_HOT="$PROFILE" \
        LLAMA_EXPERT_S="$FIXED_S" \
        LLAMA_EXPERT_ADAPT=0 \
        LLAMA_EXPERT_STATS=0 \
        "${extra_env[@]}" \
        "$PPL" \
        "${COMMON_ARGS[@]}" \
        -ctk "$type_k" \
        -ctv "$type_v" \
        "$@" 2>&1 | tee "$RESULTS_DIR/$name.log"
}

case_requested() {
    local wanted="$1"
    [[ ",$QUALITY_CASES," == *",$wanted,"* ]]
}

# Supplying the base path without --kl-divergence records the Q8/Q8 logits.
run_ppl q8-q8 q8_0 q8_0 "" "" "" \
    -f "$QUALITY_PROMPT_SOURCE" \
    --kl-divergence-base "$BASE_LOGITS"

if case_requested q8-q4; then
    run_ppl q8-q4 q8_0 q4_0 "" legacy "" \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

if case_requested q4-q8; then
    run_ppl q4-q8 q4_0 q8_0 "" legacy "" \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

if case_requested q4-q4; then
    run_ppl q4-q4 q4_0 q4_0 "" legacy "" \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

if case_requested q4-weighted-q4-weighted; then
    run_ppl q4-weighted-q4-weighted q4_0 q4_0 "" weighted "" \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

if case_requested q4-weighted-k-q4; then
    run_ppl q4-weighted-k-q4 q4_0 q4_0 "" weighted-k "" \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

if case_requested q4-q4-weighted-v; then
    run_ppl q4-q4-weighted-v q4_0 q4_0 "" weighted-v "" \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

if case_requested q4-no-rot; then
    run_ppl q4-no-rot q4_0 q4_0 "" legacy 1 \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

if case_requested turbo4-q8; then
    run_ppl turbo4-q8 turbo4_k q8_0 "" "" "" \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

if case_requested q8-turbo4; then
    run_ppl q8-turbo4 q8_0 turbo4_k "" "" "" \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

if case_requested turbo4-turbo4; then
    run_ppl turbo4-turbo4 turbo4_k turbo4_k "" "" "" \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

if [[ -n "$TURBO4_Q8_FALLBACK_LAYERS" ]] && case_requested turbo4-mixed-q8; then
    run_ppl turbo4-mixed-q8 turbo4_k q8_0 "$TURBO4_Q8_FALLBACK_LAYERS" "" "" \
        --kl-divergence-base "$BASE_LOGITS" \
        --kl-divergence
fi

{
    echo "TurboQuant KV quality summary"
    echo "model=$MODEL"
    echo "corpus=$QUALITY_PROMPT_SOURCE"
    echo "context=$CONTEXT batch=$BATCH ubatch=$UBATCH chunks=$CHUNKS fixed_s=$FIXED_S"
    echo "quality_cases=$QUALITY_CASES"
    echo "turbo4_q8_fallback_layers=${TURBO4_Q8_FALLBACK_LAYERS:-none}"
    echo
    for log in "$RESULTS_DIR"/*.log; do
        echo "[$(basename "$log" .log)]"
        grep -E 'Final estimate|Mean PPL|Mean    KLD|Maximum KLD|RMS|Same top p' "$log" || true
        echo
    done
} > "$RESULTS_DIR/summary.txt"

echo "Results: $RESULTS_DIR"
echo "Summary: $RESULTS_DIR/summary.txt"
