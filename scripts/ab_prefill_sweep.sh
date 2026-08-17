#!/usr/bin/env bash
# Prefill-only sweep on GTX 1660 Ti: S × prefill_ubatch.
# Measures prompt_tps with n_predict=1 (no meaningful decode).
# Fixed: q4_0/q4_0, MTP=0, specialist profile, start.sh kernel knobs.
#
# Usage:
#   ./scripts/ab_prefill_sweep.sh
#   REPEATS=2 S_LIST="32 34" PREFILL_LIST="256 512 768" ./scripts/ab_prefill_sweep.sh
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/prefill-sweep-${STAMP}}"
mkdir -p "$OUT"

SERVER="${SERVER:-$ROOT/build-main-sm75/bin/llama-server}"
CLIENT="${CLIENT:-$ROOT/tools/bench_hybrid_client.py}"
MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"
PROMPT_SOURCE="${PROMPT_SOURCE:-$ROOT/ROUTER_LOOKAHEAD_ANALYSIS.md}"
PORT="${PORT:-18082}"
REPEATS="${REPEATS:-3}"
S_LIST="${S_LIST:-32 34}"
PREFILL_LIST="${PREFILL_LIST:-256 512 768}"
CONTEXT="${CONTEXT:-24576}"
CMOE_BASE="${CMOE_BASE:-64}"
THREADS="${THREADS:-8}"

for p in "$SERVER" "$CLIENT" "$MODEL" "$PROFILE" "$PROMPT_SOURCE"; do
  [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }
done

cp -f "$PROMPT_SOURCE" "$OUT/prompt.txt"
printf '%s\n' 's,prefill_ubatch,rep,effective_s,prompt_n,prompt_ms,prompt_tps,ttft_ms,vram_peak_mib,status' > "$OUT/runs.csv"

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
  local s="$1" pref="$2" logf="$3"
  cleanup
  sleep 1

  local env_args=(
    "CUDA_VISIBLE_DEVICES=0"
    "LLAMA_CMOE_BATCH=$CMOE_BASE"
    "LLAMA_CMOE_UBATCH=$CMOE_BASE"
    "LLAMA_CMOE_PREFILL_BATCH=$pref"
    "LLAMA_CMOE_PREFILL_UBATCH=$pref"
    "LLAMA_ARG_CTX_SIZE=$CONTEXT"
    "LLAMA_ARG_BATCH=$CMOE_BASE"
    "LLAMA_ARG_UBATCH=$CMOE_BASE"
    "LLAMA_ARG_THREADS=$THREADS"
    "LLAMA_ARG_THREADS_BATCH=$THREADS"
    "LLAMA_ARG_N_PARALLEL=1"
    "LLAMA_ARG_CPU_MOE=1"
    "LLAMA_ARG_CACHE_TYPE_K=q4_0"
    "LLAMA_ARG_CACHE_TYPE_V=q4_0"
    "LLAMA_ARG_FLASH_ATTN=on"
    "LLAMA_ARG_JINJA=1"
    "LLAMA_ARG_OFFLINE=1"
    "LLAMA_ARG_LOAD_MODE=mmap"
    "LLAMA_ARG_HOST=127.0.0.1"
    "LLAMA_ARG_PORT=$PORT"
    "LLAMA_ARG_CTX_CHECKPOINTS=0"
    "LLAMA_ARG_CACHE_RAM=0"
    "LLAMA_ARG_CACHE_PROMPT=0"
    "LLAMA_ARG_SPEC_TYPE=none"
    "LLAMA_EXPERT_HOT=$PROFILE"
    "LLAMA_EXPERT_S=$s"
    "LLAMA_EXPERT_ADAPT=0"
    "LLAMA_EXPERT_STATS=0"
    "LLAMA_EXPERT_USAGE=0"
    "LLAMA_EXPERT_WARM_SLOTS=0"
    "LLAMA_EXPERT_STATIC_NO_SYNC=1"
    "LLAMA_EXPERT_SHARED_HOT_IDS=1"
    "LLAMA_EXPERT_SKIP_SENTINEL=1"
    "LLAMA_EXPERT_VRAM_RESERVE_MIB=400"
    "LLAMA_KV_Q4_SCALE=weighted-k"
    "GGML_CUDA_MOE_MULTI_FUSION=1"
    "GGML_CUDA_MOE_COMBINE_FUSION=1"
    "GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=4"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2"
    "GGML_CUDA_ASYNC_HOST_COPY=1"
    "GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1"
    "GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=128"
  )

  env -u LLAMA_EXPERT_PLACEMENT "${env_args[@]}" \
    "$SERVER" \
      -m "$MODEL" \
      --load-mode mmap \
      -c "$CONTEXT" \
      -ctk q4_0 -ctv q4_0 \
      -fa on -cmoe -np 1 --no-mmproj \
      --jinja --offline \
      --host 127.0.0.1 --port "$PORT" \
      --ctx-checkpoints 0 --cache-ram 0 \
      --threads "$THREADS" --threads-batch "$THREADS" \
      -bs \
      -a "prefill-s${s}-p${pref}" \
    >"$logf" 2>&1 &
  PID=$!
  wait_ready "$logf"
  # extract effective S from log
  grep -E 'AUTO-FIT|expert tiering on|fixed' "$logf" | head -5 || true
}

vram_now() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' '
}

echo "Results → $OUT"
echo "S_LIST=$S_LIST PREFILL_LIST=$PREFILL_LIST REPEATS=$REPEATS"

for s in $S_LIST; do
  for pref in $PREFILL_LIST; do
    stem="s${s}-p${pref}"
    logf="$OUT/${stem}.server.log"
    echo ""
    echo "=== arm S=$s prefill_ubatch=$pref ==="
    if ! start_server "$s" "$pref" "$logf"; then
      echo "$s,$pref,0,,,,OOM_or_fail" >> "$OUT/runs.csv"
      cleanup
      continue
    fi
    effective="$(grep -oE 'expert tiering on: [0-9]+ fixed|set S = [0-9]+' "$logf" | head -1 || true)"
    peak_vram="$(vram_now)"

    for rep in $(seq 1 "$REPEATS"); do
      resp="$OUT/${stem}-run${rep}.json"
      status=ok
      if ! python3 "$CLIENT" \
          --url "http://127.0.0.1:${PORT}" \
          --prompt-file "$OUT/prompt.txt" \
          --n-predict 1 \
          --output "$resp" \
          --timeout 600; then
        status=fail
        echo "$s,$pref,$rep,$effective,,,,,,$status" >> "$OUT/runs.csv"
        continue
      fi
      # parse timings
      python3 - "$s" "$pref" "$rep" "$effective" "$peak_vram" "$status" "$resp" "$OUT/runs.csv" <<'PY'
import json, sys
s, pref, rep, eff, vram, status, resp, csvp = sys.argv[1:9]
d = json.loads(open(resp).read())
t = d.get("timings") or {}
pn = t.get("prompt_n") or t.get("n_prompt") or 0
pms = t.get("prompt_ms") or 0
pps = t.get("prompt_per_second") or 0
ttft = d.get("ttft_ms") or 0
# fallback estimate
if (not pps or pps == 0) and pms and pn:
    pps = float(pn) / (float(pms) / 1000.0)
with open(csvp, "a") as f:
    f.write(f"{s},{pref},{rep},{eff!s},{pn},{pms},{pps},{ttft},{vram},{status}\n")
print(f"  rep {rep}: prompt_n={pn} prompt_ms={pms:.1f} prompt_tps={pps:.2f} ttft_ms={ttft:.0f} vram={vram}")
PY
      sleep 2
    done
    cleanup
    sleep 3
  done
done

python3 - "$OUT" <<'PY'
import csv, statistics
from pathlib import Path
from collections import defaultdict
out = Path(__file__).resolve().parent if False else Path(__import__('sys').argv[1])
rows = list(csv.DictReader((out / "runs.csv").open()))
by = defaultdict(list)
for r in rows:
    if r.get("status") != "ok":
        continue
    try:
        by[(r["s"], r["prefill_ubatch"])].append(float(r["prompt_tps"]))
    except Exception:
        pass

md = ["# Prefill-only sweep (1660 Ti)", "",
      f"Dir: `{out}`", "",
      "Fixed: q4_0/q4_0, MTP off, specialist profile, decode batch 64, n_predict=1.",
      "",
      "| S | prefill_ubatch | n | prompt_tps median | min | max |",
      "| ---: | ---: | ---: | ---: | ---: | ---: |"]
for key in sorted(by.keys(), key=lambda x: (int(x[0]), int(x[1]))):
    xs = sorted(by[key])
    md.append(f"| {key[0]} | {key[1]} | {len(xs)} | {statistics.median(xs):.2f} | {xs[0]:.2f} | {xs[-1]:.2f} |")
if not by:
    md.append("| — | — | 0 | no successful runs | | |")
md += ["", "## Notes", "",
       "- Higher S ⇒ more MoE on GPU during prefill (VRAM tradeoff).",
       "- Higher prefill_ubatch ⇒ larger PP chunks (peak VRAM tradeoff).",
       "- Compare medians to ~250 tok/s GTX1080 (S≈70, batch 128, 180W, 8GiB).",
       ""]
(out / "summary.md").write_text("\n".join(md) + "\n")
print((out / "summary.md").read_text())
PY

echo "Done: $OUT"
