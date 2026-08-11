#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVER="${SERVER:-$ROOT/build-hybrid/bin/llama-server}"
CLIENT="${CLIENT:-$ROOT/tools/bench_hybrid_client.py}"
AGGREGATOR="${AGGREGATOR:-$ROOT/tools/aggregate_expert_profiles.py}"
MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PROFILE="${PROFILE:-$HOME/models/qwen3.6-35b-a3b-mtp/luce-warmstart-placement-s33.csv}"
CORPUS="${CORPUS:-$ROOT/scripts/hybrid_profile_corpus.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/benchmark-results/profile-corpus-$(date -u +%Y%m%dT%H%M%SZ)}"
N_PREDICT="${N_PREDICT:-256}"
MTP_N="${MTP_N:-2}"
FIXED_S="${FIXED_S:-auto}"
CPU_THREADS="${CPU_THREADS:-8}"
DRAFT_THREADS="${DRAFT_THREADS:-$CPU_THREADS}"
PORT="${PORT:-18082}"
PROFILE_IDS="${PROFILE_IDS:-}"

for path in "$SERVER" "$CLIENT" "$AGGREGATOR" "$MODEL" "$PROFILE" "$CORPUS"; do
    if [[ ! -e "$path" ]]; then
        echo "Fehlender Pfad: $path" >&2
        exit 1
    fi
done
for command in curl python3; do
    command -v "$command" >/dev/null || {
        echo "Fehlendes Programm: $command" >&2
        exit 1
    }
done
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Ausgabeverzeichnis existiert bereits; nichts wird ueberschrieben: $OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR/prompts" "$OUTPUT_DIR/usage" "$OUTPUT_DIR/responses" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/stats"

python3 - "$CORPUS" "$OUTPUT_DIR" "$PROFILE_IDS" <<'PY'
import json
import re
import shutil
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
requested_raw = sys.argv[3]
requested = [value for value in requested_raw.split(",") if value] if requested_raw else []
if len(requested) != len(set(requested)):
    raise SystemExit("PROFILE_IDS contains duplicates")
requested_set = set(requested)
seen = set()
rows = []
for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
    if not raw:
        continue
    try:
        item = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid corpus JSON at line {line_number}: {exc}")
    identifier = item.get("id")
    category = item.get("category")
    prompt = item.get("prompt")
    weight = str(item.get("weight", "1"))
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identifier):
        raise SystemExit(f"invalid corpus id at line {line_number}")
    if identifier in seen:
        raise SystemExit(f"duplicate corpus id {identifier!r}")
    if not isinstance(category, str) or not category or not isinstance(prompt, str) or not prompt:
        raise SystemExit(f"invalid category or prompt at line {line_number}")
    try:
        if Decimal(weight) <= 0:
            raise InvalidOperation
    except InvalidOperation:
        raise SystemExit(f"invalid positive weight at line {line_number}")
    seen.add(identifier)
    prompt_path = output / "prompts" / f"{identifier}.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    if not requested_set or identifier in requested_set:
        rows.append((identifier, category, weight, prompt_path))
if not rows:
    raise SystemExit("selected corpus is empty")
if requested_set:
    missing = sorted(requested_set - {row[0] for row in rows})
    if missing:
        raise SystemExit(f"PROFILE_IDS contains unknown corpus ids: {missing}")
shutil.copyfile(source, output / "corpus.jsonl")
with (output / "selection.txt").open("w", encoding="utf-8", newline="\n") as stream:
    stream.write("\n".join(row[0] for row in rows) + "\n")
with (output / "index.tsv").open("w", encoding="utf-8", newline="\n") as stream:
    for row in rows:
        stream.write("\t".join(map(str, row)) + "\n")
PY

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
            tail -100 "$log_file" >&2
            return 1
        fi
        sleep 1
    done
    echo "Server wurde nicht rechtzeitig bereit." >&2
    tail -100 "$log_file" >&2
    return 1
}

aggregate_args=()
while IFS=$'\t' read -r identifier category weight prompt_file; do
    log_file="$OUTPUT_DIR/logs/$identifier.log"
    usage_file="$OUTPUT_DIR/usage/$identifier.csv"
    stats_file="$OUTPUT_DIR/stats/$identifier.json"
    response_file="$OUTPUT_DIR/responses/$identifier.json"
    echo "=== Profil $identifier ($category), n_predict=$N_PREDICT, S=$FIXED_S, MTP=$MTP_N ==="

    env_args=(
        CUDA_VISIBLE_DEVICES=0
        LLAMA_CMOE_BATCH=32
        LLAMA_CMOE_UBATCH=32
        "LLAMA_EXPERT_HOT=$PROFILE"
        LLAMA_EXPERT_ADAPT=0
        LLAMA_EXPERT_STATS=0
        "LLAMA_EXPERT_STATS_JSON=$stats_file"
        "LLAMA_EXPERT_USAGE=$usage_file"
        LLAMA_EXPERT_USAGE_MODE=session
        LLAMA_EXPERT_WARM_SLOTS=0
        LLAMA_EXPERT_STATIC_NO_SYNC=0
    )
    if [[ "$FIXED_S" != auto ]]; then
        env_args+=("LLAMA_EXPERT_S=$FIXED_S")
    fi
    server_args=(
        -m "$MODEL"
        -c 32768
        -ctk q4_0
        -ctv q4_0
        -fa on
        -cmoe
        -np 1
        --no-mmproj
        --reasoning-budget 16384
        --ctx-checkpoints 0
        --cache-ram 0
        --jinja
        --offline
        --threads "$CPU_THREADS"
        --threads-batch "$CPU_THREADS"
        --host 127.0.0.1
        --port "$PORT"
        -a "profile-$identifier"
    )
    if [[ "$MTP_N" -gt 0 ]]; then
        server_args+=(
            --spec-type draft-mtp
            --spec-draft-n-max "$MTP_N"
            --spec-draft-ngl auto
            --spec-draft-type-k q8_0
            --spec-draft-type-v q4_0
            --spec-draft-threads "$DRAFT_THREADS"
            --spec-draft-threads-batch "$DRAFT_THREADS"
        )
    fi

    env "${env_args[@]}" "$SERVER" "${server_args[@]}" > "$log_file" 2>&1 &
    PID=$!
    wait_ready "$log_file"
    python3 "$CLIENT" \
        --url "http://127.0.0.1:$PORT" \
        --prompt-file "$prompt_file" \
        --n-predict "$N_PREDICT" \
        --output "$response_file"
    cleanup

    python3 - "$usage_file" "$stats_file" <<'PY'
import csv
import json
import sys
from pathlib import Path

usage_path, stats_path = map(Path, sys.argv[1:])
with usage_path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
if not rows or set(rows[0]) != {"layer", "expert", "count"}:
    raise SystemExit(f"invalid or empty session usage: {usage_path}")
layer_totals = {}
for row in rows:
    layer = int(row["layer"])
    layer_totals[layer] = layer_totals.get(layer, 0) + int(row["count"])
stats = json.loads(stats_path.read_text())
usage_total = sum(layer_totals.values())
if usage_total != stats["selected_total"]:
    raise SystemExit(
        f"session usage total {usage_total} != stats selected_total {stats['selected_total']}"
    )
if len(layer_totals) != len(stats["layers"]) or min(layer_totals.values()) <= 0:
    raise SystemExit("session usage does not cover every tiered layer")
PY
    aggregate_args+=(--input "$usage_file" --weight "$weight")
done < "$OUTPUT_DIR/index.tsv"

python3 "$AGGREGATOR" \
    "${aggregate_args[@]}" \
    --model "$MODEL" \
    --output "$OUTPUT_DIR/general-profile.csv" \
    --normalization per-layer

echo "Profile und Einzelstatistiken: $OUTPUT_DIR"
