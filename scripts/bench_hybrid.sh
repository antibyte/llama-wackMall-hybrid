#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVER="${SERVER:-$ROOT/build-hybrid/bin/llama-server}"
CLIENT="${CLIENT:-$ROOT/tools/bench_hybrid_client.py}"
MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PROFILE="${PROFILE:-${7:-$HOME/models/qwen3.6-35b-a3b-mtp/luce-warmstart.csv}}"
PLACEMENT="${PLACEMENT:-}"
RESULTS_DIR="${RESULTS_DIR:-${5:-$ROOT/benchmark-results/hybrid-$(date -u +%Y%m%dT%H%M%SZ)}}"
PROMPT_SOURCE="${PROMPT_SOURCE:-}"
PORT="${PORT:-18081}"
REPEATS="${REPEATS:-${2:-3}}"
N_PREDICT="${N_PREDICT:-${3:-2000}}"
WARMUP_TOKENS="${WARMUP_TOKENS:-${4:-64}}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-10}"
HYBRID_FIXED_S="${HYBRID_FIXED_S:-28}"
STATIC_FIXED_S="${STATIC_FIXED_S:-33}"
CPU_THREADS="${CPU_THREADS:-}"
DRAFT_THREADS="${DRAFT_THREADS:-$CPU_THREADS}"
CMOE_BATCH="${CMOE_BATCH:-32}"
CMOE_UBATCH="${CMOE_UBATCH:-32}"
PROMPT_REPEAT="${PROMPT_REPEAT:-1}"
WARMUP_PROMPT_REPEAT="${WARMUP_PROMPT_REPEAT:-1}"
ADAPT_CUDA_GRAPHS="${ADAPT_CUDA_GRAPHS:-0}"
DRAFT_P_MIN="${DRAFT_P_MIN:-}"
DRAFT_BACKEND_SAMPLING="${DRAFT_BACKEND_SAMPLING:-}"
EFFECTIVE_DRAFT_BACKEND_SAMPLING="${DRAFT_BACKEND_SAMPLING:-1}"
DRAFT_TYPE_K="${DRAFT_TYPE_K:-q8_0}"
DRAFT_TYPE_V="${DRAFT_TYPE_V:-q4_0}"
SAVE_EXPERT_USAGE="${SAVE_EXPERT_USAGE:-0}"
EXPERT_TIMING="${EXPERT_TIMING:-0}"
CPU_CHUNK="${CPU_CHUNK:-64}"
CPU_ACT_PARALLEL="${CPU_ACT_PARALLEL:-0}"
CPU_ASYNC="${CPU_ASYNC:-0}"
LOAD_MODE="${LOAD_MODE:-mmap}"
CPU_DOWN_PREFETCH="${CPU_DOWN_PREFETCH:-0}"
CPU_REUSE_ROWS="${CPU_REUSE_ROWS:-0}"
CPU_MASK="${CPU_MASK:-}"
CPU_POLL="${CPU_POLL:-}"
BEST_W="${BEST_W:-${6:-2}}"
WARM_ADMISSION="${WARM_ADMISSION:-immediate}"
WARM_ADMISSION_WINDOW="${WARM_ADMISSION_WINDOW:-8}"
WARM_AUTO_MAX="${WARM_AUTO_MAX:-4}"
VRAM_RESERVE_MIB="${VRAM_RESERVE_MIB:-512}"
PREFETCH_STREAMS="${PREFETCH_STREAMS:-1}"
PREFETCH_MAX_INFLIGHT="${PREFETCH_MAX_INFLIGHT:-2}"
MTP_OVERRIDE="${MTP_OVERRIDE:-}"
PREFETCH_OVERRIDE="${PREFETCH_OVERRIDE:-}"
SIGSEGV_PRELOAD="${SIGSEGV_PRELOAD:-}"
WARM_MTP_EXPERIMENTAL="${WARM_MTP_EXPERIMENTAL:-0}"
# Safe default after the documented stop conditions. C-J remain selectable
# explicitly for later debugging once the repin crash is resolved.
CASES="${CASES:-${1:-A}}"

PROMPT_FILE="$RESULTS_DIR/prompt.txt"
RUNS_CSV="$RESULTS_DIR/runs.csv"
MEDIANS_CSV="$RESULTS_DIR/medians.csv"

if [[ -e "$RESULTS_DIR" ]]; then
    echo "Ergebnisverzeichnis existiert bereits; nichts wird ueberschrieben: $RESULTS_DIR" >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR"
if [[ -n "$PROMPT_SOURCE" ]]; then
    if [[ ! -f "$PROMPT_SOURCE" ]]; then
        echo "Fehlende Promptdatei: $PROMPT_SOURCE" >&2
        exit 1
    fi
    cp -- "$PROMPT_SOURCE" "$PROMPT_FILE"
else
    printf '%s\n' \
      'Entwirf detailliert eine robuste Architektur fuer einen lokalen KI-Agenten in Go. Behandle Nebenlaeufigkeit, Werkzeuge, Speicher, Fehlerbehandlung, Tests und Sicherheit.' \
      > "$PROMPT_FILE"
fi

for path in "$SERVER" "$CLIENT" "$MODEL"; do
    if [[ ! -e "$path" ]]; then
        echo "Fehlender Pfad: $path" >&2
        exit 1
    fi
done
needs_profile=0
needs_placement=0
IFS=',' read -r -a CONFIGS <<< "$CASES"
for config in "${CONFIGS[@]}"; do
    if [[ "$config" != A ]]; then
        needs_profile=1
    fi
    if [[ "$config" == SV || "$config" == SVC ]]; then
        needs_placement=1
    fi
done
if [[ "$needs_profile" == 1 && ! -e "$PROFILE" ]]; then
    echo "Fehlender Pfad: $PROFILE" >&2
    exit 1
fi
if [[ "$needs_placement" == 1 && ( -z "$PLACEMENT" || ! -f "$PLACEMENT" ) ]]; then
    echo "Konfiguration SV benoetigt eine vorhandene PLACEMENT-Datei: ${PLACEMENT:-<leer>}" >&2
    exit 1
fi
for command in curl python3 nvidia-smi ps sha256sum; do
    command -v "$command" >/dev/null || {
        echo "Fehlendes Programm: $command" >&2
        exit 1
    }
done
PROFILE_SHA256=none
if [[ "$needs_profile" == 1 ]]; then
    PROFILE_SHA256=$(sha256sum "$PROFILE")
    PROFILE_SHA256=${PROFILE_SHA256%% *}
fi
PLACEMENT_SHA256=none
if [[ "$needs_placement" == 1 ]]; then
    PLACEMENT_SHA256=$(sha256sum "$PLACEMENT")
    PLACEMENT_SHA256=${PLACEMENT_SHA256%% *}
    MANIFEST_PROFILE_SHA256=$(sed -n 's/^# profile_sha256=//p' "$PLACEMENT")
    if [[ "$MANIFEST_PROFILE_SHA256" != "$PROFILE_SHA256" ]]; then
        echo "Placement wurde fuer ein anderes Profil erzeugt: $MANIFEST_PROFILE_SHA256 != $PROFILE_SHA256" >&2
        exit 1
    fi
fi
POWER_PROFILE=unknown
if command -v system76-power >/dev/null; then
    POWER_PROFILE=$(system76-power profile 2>/dev/null | sed -n 's/^Power Profile: //p' | tr '[:upper:]' '[:lower:]')
    POWER_PROFILE=${POWER_PROFILE:-unknown}
fi

printf '%s\n' \
  'config,rep,fixed_s,effective_fixed_s,warm_s,mtp_n,cpu_threads,draft_threads,draft_type_k,draft_type_v,draft_p_min,draft_backend_sampling,cpu_chunk,cpu_act_parallel,cpu_async,cpu_down_prefetch,cpu_reuse_rows,load_mode,cmoe_batch,cmoe_ubatch,prompt_repeat,profile_sha256,placement_sha256,power_profile,cpu_mask,cpu_poll,adapt,adapt_cuda_graphs,static_no_sync_requested,static_no_sync_active,prompt_tps,ttft_ms,decode_tps,sustained_decode_tps,mtp_acceptance,mean_accepted_length,hot_hits,warm_hits,cold_hits,cold_share,repins,warm_promotions,warm_evictions,h2d_copies,h2d_bytes,h2d_ms,cpu_expert_ms,cpu_gate_up_ms,cpu_activation_ms,cpu_down_ms,cpu_async_jobs,cpu_async_wait_ms,gpu_expert_ms,sync_wait_ms,vram_peak_mib,ram_peak_mib,gpu_util_avg,cpu_util_avg,predicted_tokens,output_sha256,token_sha256' \
  > "$RUNS_CSV"

PID=""
MONITOR_PID=""
cleanup() {
    if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    MONITOR_PID=""
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
        kill -INT "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    PID=""
}
trap cleanup EXIT INT TERM

case_settings() {
    local config="$1"
    USE_PROFILE=1
    FIXED_S="$HYBRID_FIXED_S"
    WARM_S=0
    PREFETCH=0
    MTP_N=2
    ADAPT=1
    STATIC_NO_SYNC=0
    EXPERT_STATS=1
    case "$config" in
        A) USE_PROFILE=0; FIXED_S=auto ;;
        B) FIXED_S=auto ;;
        C) WARM_S=1 ;;
        D) WARM_S=2 ;;
        E) WARM_S=4 ;;
        F) WARM_S="$BEST_W"; PREFETCH=1 ;;
        G) WARM_S="$BEST_W"; MTP_N=0 ;;
        H) WARM_S="$BEST_W"; MTP_N=1 ;;
        I) WARM_S="$BEST_W"; MTP_N=2 ;;
        J) WARM_S="$BEST_W"; MTP_N=3 ;;
        SA) FIXED_S="$STATIC_FIXED_S"; ADAPT=1; EXPERT_STATS=0 ;;
        SB) FIXED_S="$STATIC_FIXED_S"; ADAPT=0; EXPERT_STATS=0 ;;
        SC) FIXED_S="$STATIC_FIXED_S"; ADAPT=0; STATIC_NO_SYNC=1; EXPERT_STATS=0 ;;
        SD) FIXED_S="$STATIC_FIXED_S"; ADAPT=0; EXPERT_STATS=1 ;;
        SV) FIXED_S=variable; ADAPT=0; EXPERT_STATS=1 ;;
        SVC) FIXED_S=variable; ADAPT=0; STATIC_NO_SYNC=1; EXPERT_STATS=0 ;;
        *) echo "Unbekannte Konfiguration: $config" >&2; return 1 ;;
    esac
    if [[ -n "$MTP_OVERRIDE" ]]; then
        MTP_N="$MTP_OVERRIDE"
    fi
    if [[ -n "$PREFETCH_OVERRIDE" ]]; then
        PREFETCH="$PREFETCH_OVERRIDE"
    fi
}

start_monitor() {
    local sample_file="$1"
    (
        printf 'unix_s,gpu_util_pct,vram_mib,cpu_pct,rss_kib\n'
        while kill -0 "$PID" 2>/dev/null; do
            local_gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || printf ',')"
            local_ps="$(ps -p "$PID" -o %cpu=,rss= 2>/dev/null || printf '0 0')"
            printf '%s,%s,%s\n' "$(date +%s.%N)" "${local_gpu// /}" "$(tr -s ' ' ',' <<< "${local_ps# }")"
            sleep 0.5
        done
    ) > "$sample_file" &
    MONITOR_PID=$!
}

wait_ready() {
    local log_file="$1"
    for _ in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "Serverstart fehlgeschlagen:" >&2
            tail -100 "$log_file" >&2
            return 1
        fi
        sleep 1
    done
    echo "Server wurde nicht rechtzeitig bereit." >&2
    tail -100 "$log_file" >&2
    return 1
}

for config in "${CONFIGS[@]}"; do
    status=0
    case_settings "$config" || status=$?
    if [[ "$status" != 0 ]]; then
        if [[ "$status" == 2 ]]; then
            echo "Konfiguration F uebersprungen."
            continue
        fi
        exit "$status"
    fi

    for rep in $(seq 1 "$REPEATS"); do
        stem="${config}-run${rep}"
        log_file="$RESULTS_DIR/$stem.log"
        response_file="$RESULTS_DIR/$stem.response.json"
        stats_file="$RESULTS_DIR/$stem.experts.json"
        samples_file="$RESULTS_DIR/$stem.samples.csv"
        warmup_file="$RESULTS_DIR/$stem.warmup.json"
        warmup_log="$RESULTS_DIR/$stem.warmup.log"
        warmup_stats="$RESULTS_DIR/$stem.warmup.experts.json"
        usage_file="$RESULTS_DIR/$stem.usage.csv"

        echo "=== $config, Lauf $rep/$REPEATS: fixed=$FIXED_S warm=$WARM_S mtp=$MTP_N batch=$CMOE_BATCH/$CMOE_UBATCH prompt_repeat=$PROMPT_REPEAT draft_kv=$DRAFT_TYPE_K/$DRAFT_TYPE_V pmin=${DRAFT_P_MIN:-default} backend_sampling=$EFFECTIVE_DRAFT_BACKEND_SAMPLING cpu_chunk=$CPU_CHUNK cpu_act_parallel=$CPU_ACT_PARALLEL cpu_async=$CPU_ASYNC cpu_down_prefetch=$CPU_DOWN_PREFETCH cpu_reuse_rows=$CPU_REUSE_ROWS load_mode=$LOAD_MODE prefetch=$PREFETCH threads=${CPU_THREADS:-auto}/${DRAFT_THREADS:-auto} power=$POWER_PROFILE adapt=$ADAPT adapt_cuda_graphs=$ADAPT_CUDA_GRAPHS static_no_sync=$STATIC_NO_SYNC ==="

        env_args=(
            CUDA_VISIBLE_DEVICES=0
            "LLAMA_CMOE_BATCH=$CMOE_BATCH"
            "LLAMA_CMOE_UBATCH=$CMOE_UBATCH"
            "LLAMA_EXPERT_STATS=$EXPERT_STATS"
            LLAMA_EXPERT_STATS_JSON=0
            LLAMA_EXPERT_USAGE=0
            "LLAMA_EXPERT_ADAPT=$ADAPT"
            "LLAMA_EXPERT_ADAPT_CUDA_GRAPHS=$ADAPT_CUDA_GRAPHS"
            "LLAMA_EXPERT_TIMING=$EXPERT_TIMING"
            "LLAMA_EXPERT_CPU_CHUNK=$CPU_CHUNK"
            "LLAMA_EXPERT_CPU_ACT_PARALLEL=$CPU_ACT_PARALLEL"
            "LLAMA_EXPERT_CPU_ASYNC=$CPU_ASYNC"
            "LLAMA_EXPERT_CPU_DOWN_PREFETCH=$CPU_DOWN_PREFETCH"
            "LLAMA_EXPERT_CPU_REUSE_ROWS=$CPU_REUSE_ROWS"
            "LLAMA_EXPERT_STATIC_NO_SYNC=$STATIC_NO_SYNC"
            LLAMA_EXPERT_DECAY=1.0
            "LLAMA_EXPERT_WARM_SLOTS=$WARM_S"
            LLAMA_EXPERT_WARM_POLICY=lru
            LLAMA_EXPERT_WARM_RESET=request
            "LLAMA_EXPERT_WARM_ADMISSION=$WARM_ADMISSION"
            "LLAMA_EXPERT_WARM_ADMISSION_WINDOW=$WARM_ADMISSION_WINDOW"
            "LLAMA_EXPERT_WARM_AUTO_MAX=$WARM_AUTO_MAX"
            "LLAMA_EXPERT_VRAM_RESERVE_MIB=$VRAM_RESERVE_MIB"
            "LLAMA_EXPERT_WARM_PREFETCH=$PREFETCH"
            "LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL=$WARM_MTP_EXPERIMENTAL"
        )
        if [[ "$USE_PROFILE" == 1 ]]; then
            env_args+=("LLAMA_EXPERT_HOT=$PROFILE")
        fi
        if [[ "$FIXED_S" != auto ]]; then
            if [[ "$FIXED_S" == variable ]]; then
                env_args+=("LLAMA_EXPERT_PLACEMENT=$PLACEMENT")
            else
                env_args+=("LLAMA_EXPERT_S=$FIXED_S")
            fi
        fi
        if [[ "$PREFETCH" == 1 ]]; then
            env_args+=(
                "LLAMA_EXPERT_PREFETCH_STREAMS=$PREFETCH_STREAMS"
                "LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT=$PREFETCH_MAX_INFLIGHT"
            )
        fi
        if [[ -n "$SIGSEGV_PRELOAD" ]]; then
            if [[ ! -f "$SIGSEGV_PRELOAD" ]]; then
                echo "Fehlender SIGSEGV-Preload: $SIGSEGV_PRELOAD" >&2
                exit 1
            fi
            env_args+=("LD_PRELOAD=$SIGSEGV_PRELOAD")
        fi

        server_args=(
            -m "$MODEL"
            --load-mode "$LOAD_MODE"
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
            --host 127.0.0.1
            --port "$PORT"
            -a "hybrid-$config"
        )
        if [[ "$MTP_N" -gt 0 ]]; then
            server_args+=(
                --spec-type draft-mtp
                --spec-draft-n-max "$MTP_N"
                --spec-draft-ngl auto
                --spec-draft-type-k "$DRAFT_TYPE_K"
                --spec-draft-type-v "$DRAFT_TYPE_V"
            )
            if [[ -n "$DRAFT_P_MIN" ]]; then
                server_args+=(--spec-draft-p-min "$DRAFT_P_MIN")
            fi
            if [[ "$DRAFT_BACKEND_SAMPLING" == 0 ]]; then
                server_args+=(--no-spec-draft-backend-sampling)
            elif [[ -n "$DRAFT_BACKEND_SAMPLING" && "$DRAFT_BACKEND_SAMPLING" != 1 ]]; then
                echo "DRAFT_BACKEND_SAMPLING muss 0, 1 oder leer sein" >&2
                exit 1
            fi
        fi
        if [[ -n "$CPU_THREADS" ]]; then
            server_args+=(
                --threads "$CPU_THREADS"
                --threads-batch "$CPU_THREADS"
            )
        fi
        if [[ -n "$DRAFT_THREADS" ]]; then
            server_args+=(
                --spec-draft-threads "$DRAFT_THREADS"
                --spec-draft-threads-batch "$DRAFT_THREADS"
            )
        fi
        if [[ -n "$CPU_MASK" ]]; then
            server_args+=(
                --cpu-mask "$CPU_MASK"
                --cpu-mask-batch "$CPU_MASK"
                --cpu-strict 1
                --cpu-strict-batch 1
                --spec-draft-cpu-mask "$CPU_MASK"
                --spec-draft-cpu-mask-batch "$CPU_MASK"
                --spec-draft-cpu-strict 1
                --spec-draft-cpu-strict-batch 1
            )
        fi
        if [[ -n "$CPU_POLL" ]]; then
            server_args+=(
                --poll "$CPU_POLL"
                --spec-draft-poll "$CPU_POLL"
            )
        fi

        warmup_env_args=("${env_args[@]}")
        if [[ "$EXPERT_STATS" == 1 ]]; then
            warmup_env_args+=("LLAMA_EXPERT_STATS_JSON=$warmup_stats")
        fi
        env "${warmup_env_args[@]}" \
            "$SERVER" "${server_args[@]}" > "$warmup_log" 2>&1 &
        PID=$!
        wait_ready "$warmup_log"

        python3 "$CLIENT" \
            --url "http://127.0.0.1:$PORT" \
            --prompt-file "$PROMPT_FILE" \
            --prompt-repeat "$WARMUP_PROMPT_REPEAT" \
            --n-predict "$WARMUP_TOKENS" \
            --output "$warmup_file"

        # A fresh measured process avoids cross-request KV/speculative state,
        # while the warm-up still primes model pages and the CUDA runtime.
        cleanup
        measured_env_args=("${env_args[@]}")
        if [[ "$EXPERT_STATS" == 1 ]]; then
            measured_env_args+=("LLAMA_EXPERT_STATS_JSON=$stats_file")
        fi
        if [[ "$SAVE_EXPERT_USAGE" == 1 ]]; then
            measured_env_args+=(
                "LLAMA_EXPERT_USAGE=$usage_file"
                LLAMA_EXPERT_USAGE_MODE=session
            )
        fi
        env "${measured_env_args[@]}" \
            "$SERVER" "${server_args[@]}" > "$log_file" 2>&1 &
        PID=$!
        wait_ready "$log_file"

        start_monitor "$samples_file"
        python3 "$CLIENT" \
            --url "http://127.0.0.1:$PORT" \
            --prompt-file "$PROMPT_FILE" \
            --prompt-repeat "$PROMPT_REPEAT" \
            --n-predict "$N_PREDICT" \
            --output "$response_file"

        cleanup

        if [[ "$STATIC_NO_SYNC" == 1 ]] && ! grep -q 'expert_tier: static no-sync enabled' "$log_file"; then
            echo "Static-No-Sync wurde angefordert, aber nicht aktiviert:" >&2
            grep -E 'static no-sync|expert tiering' "$log_file" >&2 || true
            exit 1
        fi
        if [[ "$FIXED_S" == variable ]] && ! grep -q 'variable placement validated:' "$log_file"; then
            echo "Variable Placement wurde angefordert, aber nicht aktiviert:" >&2
            grep -E 'placement|expert tiering' "$log_file" >&2 || true
            exit 1
        fi

        python3 - "$config" "$rep" "$FIXED_S" "$WARM_S" "$MTP_N" "${CPU_THREADS:-auto}" "${DRAFT_THREADS:-auto}" "$DRAFT_TYPE_K" "$DRAFT_TYPE_V" "${DRAFT_P_MIN:-default}" "$EFFECTIVE_DRAFT_BACKEND_SAMPLING" "$CPU_CHUNK" "$CPU_ACT_PARALLEL" "$CPU_ASYNC" "$CPU_DOWN_PREFETCH" "$CPU_REUSE_ROWS" "$LOAD_MODE" "$CMOE_BATCH" "$CMOE_UBATCH" "$PROMPT_REPEAT" "$PROFILE_SHA256" "$PLACEMENT_SHA256" "$POWER_PROFILE" "${CPU_MASK:-auto}" "${CPU_POLL:-default}" "$ADAPT" "$ADAPT_CUDA_GRAPHS" "$STATIC_NO_SYNC" \
            "$response_file" "$stats_file" "$samples_file" "$log_file" >> "$RUNS_CSV" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

config, rep, fixed_s, warm_s, mtp_n, cpu_threads, draft_threads, draft_type_k, draft_type_v, draft_p_min, draft_backend_sampling, cpu_chunk, cpu_act_parallel, cpu_async, cpu_down_prefetch, cpu_reuse_rows, load_mode, cmoe_batch, cmoe_ubatch, prompt_repeat, profile_sha256, placement_sha256, power_profile, cpu_mask, cpu_poll, adapt, adapt_cuda_graphs, static_requested, response_path, stats_path, samples_path, log_path = sys.argv[1:]
response = json.loads(Path(response_path).read_text())
stats_file = Path(stats_path)
stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
timings = response.get("timings", {})
log = Path(log_path).read_text(errors="replace")
static_active = "expert_tier: static no-sync enabled" in log

def last(pattern, default=0.0):
    found = re.findall(pattern, log, re.MULTILINE)
    return found[-1] if found else default

def numbers(column):
    with Path(samples_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        try:
            result.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            pass
    return result

effective_fixed_s = last(r"expert tiering on:\s*(\d+)\s+fixed", fixed_s)

gpu = numbers("gpu_util_pct")
vram = numbers("vram_mib")
cpu = numbers("cpu_pct")
rss = numbers("rss_kib")
draft_n = float(timings.get("draft_n", 0) or 0)
draft_accepted = float(timings.get("draft_n_accepted", 0) or 0)
total = float(stats.get("selected_total", 0) or 0)
cold = float(stats.get("cold_hits", 0) or 0)

row = [
    config, rep, fixed_s, effective_fixed_s, warm_s, mtp_n, cpu_threads, draft_threads, draft_type_k, draft_type_v, draft_p_min, draft_backend_sampling, cpu_chunk, cpu_act_parallel, cpu_async, cpu_down_prefetch, cpu_reuse_rows, load_mode, cmoe_batch, cmoe_ubatch, prompt_repeat, profile_sha256, placement_sha256, power_profile, cpu_mask, cpu_poll, adapt, adapt_cuda_graphs, static_requested, int(static_active),
    timings.get("prompt_per_second", 0),
    response.get("ttft_ms", 0),
    timings.get("predicted_per_second", 0),
    timings.get("predicted_per_second", 0),
    draft_accepted / draft_n if draft_n else 0,
    last(r"mean len\s*=\s*([0-9.]+)"),
    stats.get("hot_hits", 0), stats.get("warm_hits", 0), stats.get("cold_hits", 0),
    cold / total if total else 0,
    stats.get("repins", 0), stats.get("warm_promotions", 0), stats.get("warm_evictions", 0),
    stats.get("h2d_copies", 0), stats.get("h2d_bytes", 0), stats.get("h2d_copy_ms", 0),
    stats.get("cpu_expert_ms", 0), stats.get("cpu_gate_up_ms", 0),
    stats.get("cpu_activation_ms", 0), stats.get("cpu_down_ms", 0),
    stats.get("cpu_async_jobs", 0), stats.get("cpu_async_wait_ms", 0),
    stats.get("gpu_expert_ms", 0), stats.get("sync_wait_ms", 0),
    max(vram, default=0), max(rss, default=0) / 1024.0,
    statistics.fmean(gpu) if gpu else 0, statistics.fmean(cpu) if cpu else 0,
    timings.get("predicted_n", response.get("tokens_predicted", 0)),
    response.get("output_sha256", ""), response.get("token_sha256", ""),
]
print(",".join(str(value) for value in row))
PY

        tail -n 1 "$RUNS_CSV"
        sleep "$COOLDOWN_SECONDS"
    done
done

python3 - "$RUNS_CSV" "$MEDIANS_CSV" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

source, destination = sys.argv[1:]
with open(source, newline="") as handle:
    rows = list(csv.DictReader(handle))
groups = defaultdict(list)
for row in rows:
    groups[row["config"]].append(row)

numeric = [
    "prompt_tps", "ttft_ms", "decode_tps", "sustained_decode_tps",
    "mtp_acceptance", "mean_accepted_length",
    "hot_hits", "warm_hits", "cold_hits", "cold_share", "repins",
    "warm_promotions", "warm_evictions", "h2d_copies", "h2d_bytes", "h2d_ms",
    "cpu_expert_ms", "cpu_gate_up_ms", "cpu_activation_ms", "cpu_down_ms",
    "cpu_async_jobs", "cpu_async_wait_ms",
    "gpu_expert_ms", "sync_wait_ms",
    "vram_peak_mib", "ram_peak_mib", "gpu_util_avg", "cpu_util_avg",
    "predicted_tokens",
]
with open(destination, "w", newline="") as handle:
    fields = ["config", "repeats", "fixed_s", "effective_fixed_s", "warm_s", "mtp_n", "cpu_threads", "draft_threads", "draft_type_k", "draft_type_v", "draft_p_min", "draft_backend_sampling", "cpu_chunk", "cpu_act_parallel", "cpu_async", "cpu_down_prefetch", "cpu_reuse_rows", "load_mode", "cmoe_batch", "cmoe_ubatch", "prompt_repeat", "profile_sha256", "placement_sha256", "power_profile", "cpu_mask", "cpu_poll", "adapt", "adapt_cuda_graphs", "static_no_sync_requested", "static_no_sync_active"] + numeric + [
        "output_hashes_identical", "token_hashes_identical", "output_sha256", "token_sha256",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for config in sorted(groups):
        group = groups[config]
        output_hashes = {row["output_sha256"] for row in group}
        token_hashes = {row["token_sha256"] for row in group}
        out = {
            "config": config,
            "repeats": len(group),
            "fixed_s": group[0]["fixed_s"],
            "effective_fixed_s": group[0]["effective_fixed_s"],
            "warm_s": group[0]["warm_s"],
            "mtp_n": group[0]["mtp_n"],
            "cpu_threads": group[0]["cpu_threads"],
            "draft_threads": group[0]["draft_threads"],
            "draft_type_k": group[0]["draft_type_k"],
            "draft_type_v": group[0]["draft_type_v"],
            "draft_p_min": group[0]["draft_p_min"],
            "draft_backend_sampling": group[0]["draft_backend_sampling"],
            "cpu_chunk": group[0]["cpu_chunk"],
            "cpu_act_parallel": group[0]["cpu_act_parallel"],
            "cpu_async": group[0]["cpu_async"],
            "cpu_down_prefetch": group[0]["cpu_down_prefetch"],
            "cpu_reuse_rows": group[0]["cpu_reuse_rows"],
            "load_mode": group[0]["load_mode"],
            "cmoe_batch": group[0]["cmoe_batch"],
            "cmoe_ubatch": group[0]["cmoe_ubatch"],
            "prompt_repeat": group[0]["prompt_repeat"],
            "profile_sha256": group[0]["profile_sha256"],
            "placement_sha256": group[0]["placement_sha256"],
            "power_profile": group[0]["power_profile"],
            "cpu_mask": group[0]["cpu_mask"],
            "cpu_poll": group[0]["cpu_poll"],
            "adapt": group[0]["adapt"],
            "adapt_cuda_graphs": group[0]["adapt_cuda_graphs"],
            "static_no_sync_requested": group[0]["static_no_sync_requested"],
            "static_no_sync_active": group[0]["static_no_sync_active"],
            "output_hashes_identical": len(output_hashes) == 1,
            "token_hashes_identical": len(token_hashes) == 1,
            "output_sha256": next(iter(output_hashes)) if len(output_hashes) == 1 else "MISMATCH",
            "token_sha256": next(iter(token_hashes)) if len(token_hashes) == 1 else "MISMATCH",
        }
        for field in numeric:
            out[field] = statistics.median(float(row[field]) for row in group)
        writer.writerow(out)
PY

echo "Einzellaeufe: $RUNS_CSV"
echo "Mediane:      $MEDIANS_CSV"
