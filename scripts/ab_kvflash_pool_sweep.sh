#!/usr/bin/env bash
# Prefill sweep over KVFlash pool sizes on the 1660 Ti.
# Compares off / 512 / 1024 / 2048 / auto at one S and prefill ubatch.
#
# Usage:
#   ./scripts/ab_kvflash_pool_sweep.sh
#   POOLS="off 512 auto" REPEATS=2 ./scripts/ab_kvflash_pool_sweep.sh
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/kvflash-pool-${STAMP}}"
mkdir -p "$OUT"

SERVER="${SERVER:-$ROOT/build-main-sm75/bin/llama-server}"
CLIENT="${CLIENT:-$ROOT/tools/bench_hybrid_client.py}"
MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"
PROMPT_SOURCE="${PROMPT_SOURCE:-$ROOT/ROUTER_LOOKAHEAD_ANALYSIS.md}"
PORT="${PORT:-18083}"
REPEATS="${REPEATS:-3}"
S="${S:-32}"
PREFILL="${PREFILL:-512}"
CONTEXT="${CONTEXT:-24576}"
CMOE_BASE="${CMOE_BASE:-64}"
THREADS="${THREADS:-8}"
POOLS="${POOLS:-off 512 1024 2048 auto}"
POLICY="${POLICY:-lru}"

for p in "$SERVER" "$CLIENT" "$MODEL" "$PROFILE" "$PROMPT_SOURCE"; do
  [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }
done

cp -f "$PROMPT_SOURCE" "$OUT/prompt.txt"
printf '%s\n' 'pool,rep,prompt_n,prompt_ms,prompt_tps,ttft_ms,vram_peak_mib,page_outs,moved_mib,status' > "$OUT/runs.csv"

PID=""
cleanup() {
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    kill -INT "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
  PID=""
}
trap cleanup EXIT INT TERM

wait_ready() {
  local logf="$1"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "server died:" >&2
      tail -80 "$logf" >&2
      return 1
    fi
    sleep 1
  done
  echo "timeout waiting for health" >&2
  tail -80 "$logf" >&2
  return 1
}

start_server() {
  local pool="$1" logf="$2"
  cleanup
  sleep 1

  local kv_env=()
  if [[ "$pool" == "off" ]]; then
    kv_env=("LLAMA_KVFLASH=0")
  else
    kv_env=("LLAMA_KVFLASH=$pool" "LLAMA_KVFLASH_STATS=1" "LLAMA_KVFLASH_POLICY=$POLICY")
  fi

  env -u LLAMA_EXPERT_PLACEMENT \
    CUDA_VISIBLE_DEVICES=0 \
    LLAMA_CMOE_BATCH="$CMOE_BASE" \
    LLAMA_CMOE_UBATCH="$CMOE_BASE" \
    LLAMA_CMOE_PREFILL_BATCH="$PREFILL" \
    LLAMA_CMOE_PREFILL_UBATCH="$PREFILL" \
    LLAMA_ARG_CTX_SIZE="$CONTEXT" \
    LLAMA_ARG_THREADS="$THREADS" \
    LLAMA_ARG_N_PARALLEL=1 \
    LLAMA_ARG_CPU_MOE=1 \
    LLAMA_ARG_CACHE_TYPE_K=q4_0 \
    LLAMA_ARG_CACHE_TYPE_V=q4_0 \
    LLAMA_ARG_FLASH_ATTN=on \
    LLAMA_EXPERT_HOT="$PROFILE" \
    LLAMA_EXPERT_S="$S" \
    LLAMA_EXPERT_ADAPT=0 \
    "${kv_env[@]}" \
    "$SERVER" \
      -m "$MODEL" \
      --load-mode mmap \
      -c "$CONTEXT" \
      -ctk q4_0 -ctv q4_0 \
      -fa on -cmoe -np 1 --no-mmproj \
      --jinja --offline \
      --host 127.0.0.1 --port "$PORT" \
      --ctx-checkpoints 0 --cache-ram 0 \
      --threads "$THREADS" \
      -a "kvflash-${pool}" \
    >"$logf" 2>&1 &
  PID=$!
  wait_ready "$logf"
}

vram_now() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' '
}

echo "Results -> $OUT"
echo "POOLS=$POOLS POLICY=$POLICY S=$S PREFILL=$PREFILL REPEATS=$REPEATS"

for pool in $POOLS; do
  logf="$OUT/pool-${pool}.server.log"
  echo ""
  echo "=== KVFlash pool=$pool ==="
  if ! start_server "$pool" "$logf"; then
    echo "$pool,0,,,,,,OOM_or_fail" >> "$OUT/runs.csv"
    cleanup
    continue
  fi
  grep -E 'kvflash|KVFlash' "$logf" | head -20 || true
  peak_vram="$(vram_now)"

  for rep in $(seq 1 "$REPEATS"); do
    resp="$OUT/pool-${pool}-run${rep}.json"
    status=ok
    if ! python3 "$CLIENT" \
        --url "http://127.0.0.1:${PORT}" \
        --prompt-file "$OUT/prompt.txt" \
        --n-predict 1 \
        --output "$resp" \
        --timeout 600; then
      status=fail
    fi
    python3 - "$pool" "$rep" "$peak_vram" "$status" "$resp" "$logf" "$OUT/runs.csv" <<'PY'
import json, re, sys
pool, rep, vram, status, resp, logf, csvp = sys.argv[1:8]
pn = pms = pps = ttft = 0
page_outs = moved = ""
if status == "ok":
    d = json.loads(open(resp).read())
    t = d.get("timings") or {}
    pn = t.get("prompt_n") or t.get("n_prompt") or 0
    pms = t.get("prompt_ms") or 0
    pps = t.get("prompt_per_second") or 0
    ttft = d.get("ttft_ms") or 0
    if (not pps or pps == 0) and pms and pn:
        pps = float(pn) / (float(pms) / 1000.0)
text = open(logf).read()
m = re.findall(r"page_outs=(\d+).*moved=([0-9.]+) MiB", text)
if m:
    page_outs, moved = m[-1]
with open(csvp, "a") as f:
    f.write(f"{pool},{rep},{pn},{pms},{pps},{ttft},{vram},{page_outs},{moved},{status}\n")
print(f"  rep {rep}: prompt_tps={pps} vram={vram} page_outs={page_outs} status={status}")
PY
    sleep 2
  done
  cleanup
  sleep 3
done

echo "Done: $OUT"
echo "Compare prompt_tps and vram_peak_mib across pools before opening cut 4."
