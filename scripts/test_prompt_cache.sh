#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVER="${SERVER:-$ROOT/build-hybrid/bin/llama-server}"
CLIENT="${CLIENT:-$ROOT/tools/bench_hybrid_client.py}"
MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PROFILE="${PROFILE:-}"
PREFIX_FILE="${PREFIX_FILE:-$ROOT/scripts/prompts/prefill-control.txt}"
BRANCH_A_FILE="${BRANCH_A_FILE:-$ROOT/scripts/prompts/cache-branch-a.txt}"
BRANCH_B_FILE="${BRANCH_B_FILE:-$ROOT/scripts/prompts/cache-branch-b.txt}"
PREFIX_REPEAT="${PREFIX_REPEAT:-54}"
BRANCH_REPEAT="${BRANCH_REPEAT:-54}"
N_PREDICT="${N_PREDICT:-16}"
CMOE_BATCH="${CMOE_BATCH:-376}"
CMOE_UBATCH="${CMOE_UBATCH:-376}"
CPU_THREADS="${CPU_THREADS:-8}"
MTP_N="${MTP_N:-2}"
ADAPT="${ADAPT:-0}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/benchmark-results/prompt-cache-$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${PORT:-18084}"
CONFIGS="${CONFIGS:-base,ram}"

if [[ -z "$PROFILE" ]]; then
    echo "PROFILE must name a validated expert profile CSV." >&2
    exit 1
fi
for path in "$SERVER" "$CLIENT" "$MODEL" "$PROFILE" "$PREFIX_FILE" "$BRANCH_A_FILE" "$BRANCH_B_FILE"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done
if [[ -e "$RESULTS_DIR" ]]; then
    echo "Results directory already exists; refusing to overwrite: $RESULTS_DIR" >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR"

RUN_PROFILE="$PROFILE"
if [[ "$ADAPT" == "1" ]]; then
    RUN_PROFILE="$RESULTS_DIR/runtime-expert-usage.csv"
    cp -- "$PROFILE" "$RUN_PROFILE"
elif [[ "$ADAPT" != "0" ]]; then
    echo "ADAPT must be 0 or 1." >&2
    exit 1
fi

PID=""
cleanup() {
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
        kill -INT "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    PID=""
}
trap cleanup EXIT INT TERM

wait_ready() {
    local log_file="$1"
    for _ in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$PID" 2>/dev/null; then
            tail -120 "$log_file" >&2
            return 1
        fi
        sleep 1
    done
    echo "Server did not become ready." >&2
    return 1
}

printf '%s\n' 'config,request,branch,prompt_n,prompt_tps,prompt_ms,ttft_ms,decode_tps,wall_ms,output_sha256,token_sha256' > "$RESULTS_DIR/summary.csv"

IFS=',' read -r -a config_list <<< "$CONFIGS"
for config in "${config_list[@]}"; do
    log_file="$RESULTS_DIR/$config.server.log"
    server_args=(
        -m "$MODEL"
        --load-mode mmap
        -c 32768
        -ctk q4_0
        -ctv q4_0
        -fa on
        -cmoe
        -np 1
        --threads "$CPU_THREADS"
        --threads-batch "$CPU_THREADS"
        --no-mmproj
        --spec-type draft-mtp
        --spec-draft-n-max "$MTP_N"
        --spec-draft-ngl auto
        --spec-draft-type-k q4_0
        --spec-draft-type-v q4_0
        --spec-draft-threads "$CPU_THREADS"
        --spec-draft-threads-batch "$CPU_THREADS"
        --reasoning-budget 16384
        --jinja
        --offline
        --host 127.0.0.1
        --port "$PORT"
        -a "prompt-cache-$config"
    )
    case "$config" in
        base)
            server_args+=(--ctx-checkpoints 0 --cache-ram 0)
            ;;
        ram)
            server_args+=(
                --ctx-checkpoints 4
                --checkpoint-min-step 1024
                --cache-ram 2048
                --no-cache-idle-slots
            )
            ;;
        ram_idle)
            server_args+=(
                --ctx-checkpoints 4
                --checkpoint-min-step 1024
                --cache-ram 2048
                --cache-idle-slots
            )
            ;;
        ram_forced)
            # Force a divergent branch below this similarity threshold through
            # the RAM prompt-cache path.  This measures the cache's upper bound
            # for a single live slot without changing server code.
            server_args+=(
                --slot-prompt-similarity 0.70
                --ctx-checkpoints 4
                --checkpoint-min-step 1024
                --cache-ram 2048
                --no-cache-idle-slots
            )
            ;;
        *)
            echo "Unknown cache configuration: $config" >&2
            exit 1
            ;;
    esac

    if [[ "$ADAPT" == "1" ]]; then
        adapt_env=(
            LLAMA_EXPERT_ADAPT=1
            LLAMA_EXPERT_ADAPT_INTERVAL=request
            LLAMA_EXPERT_ADAPT_CUDA_GRAPHS=0
            LLAMA_EXPERT_STATIC_NO_SYNC=0
            LLAMA_EXPERT_USAGE="$RUN_PROFILE"
            LLAMA_EXPERT_USAGE_MODE=cumulative
            LLAMA_EXPERT_USAGE_CHECKPOINT=request
        )
    else
        adapt_env=(
            LLAMA_EXPERT_ADAPT=0
            LLAMA_EXPERT_STATIC_NO_SYNC=1
            LLAMA_EXPERT_USAGE=0
        )
    fi

    env \
        CUDA_VISIBLE_DEVICES=0 \
        LLAMA_CMOE_BATCH="$CMOE_BATCH" \
        LLAMA_CMOE_UBATCH="$CMOE_UBATCH" \
        LLAMA_EXPERT_HOT="$RUN_PROFILE" \
        LLAMA_EXPERT_S=33 \
        LLAMA_EXPERT_WARM_SLOTS=0 \
        LLAMA_EXPERT_STATS=0 \
        LLAMA_EXPERT_CPU_REUSE_ROWS=1 \
        LLAMA_EXPERT_VRAM_RESERVE_MIB=512 \
        "${adapt_env[@]}" \
        "$SERVER" "${server_args[@]}" > "$log_file" 2>&1 &
    PID=$!
    wait_ready "$log_file"

    python3 "$CLIENT" \
        --url "http://127.0.0.1:$PORT" \
        --prompt-file "$ROOT/scripts/prompts/decode-control.txt" \
        --n-predict 1 \
        --output "$RESULTS_DIR/$config.warmup.json"

    request_index=0
    for branch in a b a; do
        request_index=$((request_index + 1))
        if [[ "$branch" == a ]]; then
            suffix_file="$BRANCH_A_FILE"
        else
            suffix_file="$BRANCH_B_FILE"
        fi
        response_file="$RESULTS_DIR/$config.request-$request_index-$branch.json"
        python3 "$CLIENT" \
            --url "http://127.0.0.1:$PORT" \
            --prompt-file "$PREFIX_FILE" \
            --prompt-repeat "$PREFIX_REPEAT" \
            --prompt-suffix-file "$suffix_file" \
            --prompt-suffix-repeat "$BRANCH_REPEAT" \
            --cache-prompt \
            --n-predict "$N_PREDICT" \
            --output "$response_file"
        python3 - "$config" "$request_index" "$branch" "$response_file" >> "$RESULTS_DIR/summary.csv" <<'PY'
import json
import sys
from pathlib import Path

config, request, branch, response_path = sys.argv[1:]
response = json.loads(Path(response_path).read_text())
timings = response.get("timings", {})
row = [
    config,
    request,
    branch,
    timings.get("prompt_n", 0),
    timings.get("prompt_per_second", 0),
    timings.get("prompt_ms", 0),
    response.get("ttft_ms", 0),
    timings.get("predicted_per_second", 0),
    response.get("wall_ms", 0),
    response.get("output_sha256", ""),
    response.get("token_sha256", ""),
]
print(",".join(str(value) for value in row))
PY
    done

    cleanup
done

python3 - "$RESULTS_DIR/summary.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    print(
        f"{row['config']:>4} request={row['request']} branch={row['branch']} "
        f"prompt_n={row['prompt_n']} prompt_ms={float(row['prompt_ms']):.1f} "
        f"ttft_ms={float(row['ttft_ms']):.1f}"
    )
PY

echo "Prompt-cache results: $RESULTS_DIR"
