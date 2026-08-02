#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVER="${SERVER:-$ROOT/build-hybrid/bin/llama-server}"
CLIENT="${CLIENT:-$ROOT/tools/bench_hybrid_client.py}"
MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PROFILE="${PROFILE:-}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/scripts/prompts/adaptive-regression.txt}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/benchmark-results/adaptive-multi-request-$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${PORT:-18082}"
MTP_N="${MTP_N:-2}"
REQUESTS="${REQUESTS:-4}"
N_PREDICT="${N_PREDICT:-128}"
FIXED_S="${FIXED_S:-33}"
CPU_THREADS="${CPU_THREADS:-8}"
CMOE_BATCH="${CMOE_BATCH:-32}"
CMOE_UBATCH="${CMOE_UBATCH:-32}"

if [[ -z "$PROFILE" ]]; then
    echo "PROFILE must name a validated expert usage/placement CSV." >&2
    exit 1
fi
for path in "$SERVER" "$CLIENT" "$MODEL" "$PROFILE" "$PROMPT_FILE"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done
if [[ ! "$MTP_N" =~ ^[0-3]$ ]]; then
    echo "MTP_N must be 0, 1, 2, or 3." >&2
    exit 1
fi
if [[ -e "$RESULTS_DIR" ]]; then
    echo "Results directory already exists; refusing to overwrite: $RESULTS_DIR" >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR"

PID=""
cleanup() {
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
        kill -INT "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

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
    --reasoning-budget 16384
    --ctx-checkpoints 0
    --cache-ram 0
    --jinja
    --offline
    --host 127.0.0.1
    --port "$PORT"
    -a "adaptive-mtp-$MTP_N"
)
if (( MTP_N > 0 )); then
    server_args+=(
        --spec-type draft-mtp
        --spec-draft-n-max "$MTP_N"
        --spec-draft-ngl auto
        --spec-draft-type-k q4_0
        --spec-draft-type-v q4_0
        --spec-draft-threads "$CPU_THREADS"
        --spec-draft-threads-batch "$CPU_THREADS"
    )
fi

env \
    CUDA_VISIBLE_DEVICES=0 \
    LLAMA_CMOE_BATCH="$CMOE_BATCH" \
    LLAMA_CMOE_UBATCH="$CMOE_UBATCH" \
    LLAMA_EXPERT_HOT="$PROFILE" \
    LLAMA_EXPERT_S="$FIXED_S" \
    LLAMA_EXPERT_ADAPT=1 \
    LLAMA_EXPERT_ADAPT_INTERVAL=request \
    LLAMA_EXPERT_WARM_SLOTS=0 \
    LLAMA_EXPERT_STATIC_NO_SYNC=0 \
    LLAMA_EXPERT_STATS=1 \
    LLAMA_EXPERT_STATS_JSON="$RESULTS_DIR/experts.json" \
    LLAMA_EXPERT_USAGE="$RESULTS_DIR/learned.csv" \
    LLAMA_EXPERT_USAGE_MODE=cumulative \
    LLAMA_EXPERT_USAGE_CHECKPOINT=request \
    LLAMA_EXPERT_CPU_REUSE_ROWS=1 \
    "$SERVER" "${server_args[@]}" > "$RESULTS_DIR/server.log" 2>&1 &
PID=$!

for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        tail -120 "$RESULTS_DIR/server.log" >&2
        exit 1
    fi
    sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

for run in $(seq 1 "$REQUESTS"); do
    "$CLIENT" \
        --url "http://127.0.0.1:$PORT" \
        --prompt-file "$PROMPT_FILE" \
        --n-predict "$N_PREDICT" \
        --output "$RESULTS_DIR/request-$run.json" \
        --timeout 1200
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null
    test -s "$RESULTS_DIR/learned.csv"
    sha256sum "$RESULTS_DIR/learned.csv" >> "$RESULTS_DIR/profile-checkpoints.sha256"
done

if rg -n "CUDA error|FATAL .*invariant|without a final stop event" "$RESULTS_DIR/server.log"; then
    echo "Adaptive multi-request regression detected a fatal log entry." >&2
    exit 1
fi

echo "Adaptive multi-request test passed: $RESULTS_DIR"
