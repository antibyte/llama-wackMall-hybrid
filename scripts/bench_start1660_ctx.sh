#!/usr/bin/env bash
# Benchmark start1660 "new defaults" at large context on GTX 1660 Ti.
#
# Base stack matches start1660.sh:
#   turbo4 target+draft KV, S=26, n_max=5 n_min=2 p_min=0.75,
#   prefill 1024, combined DFlash, CONTEXT=24576
#
# Cases (override via CASES_LIST):
#   prod_new       — exact new launcher draft params, short prompt
#   prod_new_mid   — same + prompt_repeat (mid-context fill)
#   prod_new_long  — same + larger prompt_repeat
#   nmax3_legacy   — n_max=3 n_min=0 (previous production-ish) at same ctx/KV
#   nmax5_nmin0    — n_max=5 n_min=0 p_min=0.75 (ablate n_min=2)
#
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-$ROOT/benchmark-results/start1660-ctx-$STAMP}"
N_PREDICT="${N_PREDICT:-256}"
REPEATS="${REPEATS:-1}"
WARMUP_TOKENS="${WARMUP_TOKENS:-32}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-8}"
PORT_BASE="${PORT_BASE:-18410}"

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
SUMMARY_MD="$OUT_ROOT/SUMMARY.md"

# --- start1660 base env ---
export SPEC_MODE=dflash
export DFLASH_COMBINED=1
export SPEC_DRAFT_BACKEND_SAMPLING=1
export SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf}"
export MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
export PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"
export CONTEXT="${CONTEXT:-24576}"
export CMOE_BATCH="${CMOE_BATCH:-64}"
export CMOE_UBATCH="${CMOE_UBATCH:-64}"
export CMOE_PREFILL_BATCH="${CMOE_PREFILL_BATCH:-1024}"
export CMOE_PREFILL_UBATCH="${CMOE_PREFILL_UBATCH:-1024}"
export TARGET_TYPE_K="${TARGET_TYPE_K:-turbo4_k}"
export TARGET_TYPE_V="${TARGET_TYPE_V:-turbo4_k}"
export DRAFT_TYPE_K="${DRAFT_TYPE_K:-turbo4_k}"
export DRAFT_TYPE_V="${DRAFT_TYPE_V:-turbo4_k}"
export STATIC_FIXED_S="${STATIC_FIXED_S:-26}"
export SKIP_SENTINEL="${SKIP_SENTINEL:-1}"
export MOE_MULTI_FUSION="${MOE_MULTI_FUSION:-1}"
export CUDA_ASYNC_HOST_COPY="${CUDA_ASYNC_HOST_COPY:-1}"
export CONCAT_NONCONT_FLAT_DIM0="${CONCAT_NONCONT_FLAT_DIM0:-1}"
export TARGET_BACKEND_SAMPLING="${TARGET_BACKEND_SAMPLING:-1}"
export REASONING_BUDGET="${REASONING_BUDGET:-1024}"
export LOAD_MODE="${LOAD_MODE:-mmap}"
export TURBO4_MTP_EXPERIMENTAL=1
export TURBO4_DFLASH_EXPERIMENTAL=1
export TURBO4_V_EXPERIMENTAL=1
export TURBO4_DRAFT_EXPERIMENTAL=1
export TURBO4_F16_PREFILL_MIN_BATCH="${TURBO4_F16_PREFILL_MIN_BATCH:-2}"
export LLAMA_DFLASH_DDTREE=0
export LLAMA_DFLASH_TREE_VERIFY=0
export CASES=SC
export SERVER="${SERVER:-$ROOT/build-main-sm75/bin/llama-server}"
export N_PREDICT REPEATS WARMUP_TOKENS COOLDOWN_SECONDS

# name|n_max|n_min|p_min|prompt_repeat
CASES_LIST="${CASES_LIST:-prod_new|5|2|0.75|1
prod_new_mid|5|2|0.75|16
prod_new_long|5|2|0.75|48
nmax3_legacy|3|0|0.75|1
nmax5_nmin0|5|0|0.75|1}"

printf '%s\n' \
  'case,n_max,n_min,p_min,prompt_repeat,context,decode_tps,sustained_tps,prompt_tps,ttft_ms,spec_acceptance,mean_accepted_length,draft_ms,inject_ms,verify_ms,n_verify_calls,predicted_n,vram_peak_mib,results_dir' \
  > "$SUMMARY_CSV"

{
  echo "# start1660 large-context benches — $STAMP"
  echo
  echo "- CONTEXT=$CONTEXT  n_predict=$N_PREDICT  S=$STATIC_FIXED_S"
  echo "- KV target/draft: $TARGET_TYPE_K / $DRAFT_TYPE_K"
  echo "- prefill batch: $CMOE_PREFILL_BATCH  decode batch: $CMOE_BATCH"
  echo "- OUT: \`$OUT_ROOT\`"
  echo
} | tee "$OUT_ROOT/run.log" > "$SUMMARY_MD"

port_i=0
while IFS='|' read -r case_name n_max n_min p_min prepeat; do
  [[ -z "${case_name// }" ]] && continue
  [[ "$case_name" =~ ^# ]] && continue
  port_i=$((port_i + 1))
  port=$((PORT_BASE + port_i))
  res="$OUT_ROOT/${case_name}"
  echo "=== case=$case_name n_max=$n_max n_min=$n_min p_min=$p_min prompt_repeat=$prepeat port=$port ===" \
    | tee -a "$OUT_ROOT/run.log"

  export SPEC_DRAFT_N_MAX="$n_max"
  export SPEC_DRAFT_N_MIN="$n_min"
  export SPEC_DRAFT_P_MIN="$p_min"
  export PROMPT_REPEAT="$prepeat"
  export WARMUP_PROMPT_REPEAT=1
  export PORT="$port"
  export RESULTS_DIR="$res"

  if ! bash "$ROOT/scripts/bench_hybrid.sh" SC "$REPEATS" "$N_PREDICT" "$WARMUP_TOKENS" \
      2>&1 | tee -a "$OUT_ROOT/run.log"; then
    echo "FAILED case=$case_name" | tee -a "$OUT_ROOT/run.log"
    printf '%s\n' "$case_name,$n_max,$n_min,$p_min,$prepeat,$CONTEXT,,,,,,,,,,,,,FAIL,$res" >> "$SUMMARY_CSV"
    continue
  fi

  python3 - "$case_name" "$n_max" "$n_min" "$p_min" "$prepeat" "$CONTEXT" "$res" "$SUMMARY_CSV" <<'PY'
import json, re, sys
from pathlib import Path

case, n_max, n_min, p_min, prepeat, ctx, res, summary = sys.argv[1:9]
res = Path(res)
summary = Path(summary)

cands = sorted(res.glob("*-run*.response.json"))
if not cands:
    summary.open("a").write(f"{case},{n_max},{n_min},{p_min},{prepeat},{ctx},,,,,,,,,,,,,NORESP,{res}\n")
    raise SystemExit(0)
resp = json.loads(cands[0].read_text())
t = resp.get("timings") or {}
log_text = ""
logs = sorted(res.glob("*-run*.log"))
if logs:
    log_text = logs[-1].read_text(errors="replace")

def last(pat, default=""):
    m = re.findall(pat, log_text)
    return m[-1] if m else default

acc = last(r"draft acceptance = ([0-9.]+)")
mean_len = last(r"mean len =\s*([0-9.]+)")
draft_ms = t.get("draft_ms", "")
inject_ms = t.get("inject_ms", "")
verify_ms = t.get("verify_ms", "")
n_verify = t.get("n_verify_calls", "")
# fall back to log timings if client JSON lacks phase fields
if not draft_ms:
    draft_ms = last(r"draft=\s*([0-9.]+)\s*ms")
if not verify_ms:
    # "spec verify time =   1947.11 ms /    31 calls"
    m = re.findall(r"spec verify time =\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*calls", log_text)
    if m:
        verify_ms, n_verify = m[-1]

# runs.csv has the full row
runs = res / "runs.csv"
decode = sustained = prompt_tps = ttft = predicted = vram = ""
if runs.exists():
    import csv
    rows = list(csv.DictReader(runs.open()))
    if rows:
        r = rows[-1]
        decode = r.get("decode_tps", "")
        sustained = r.get("sustained_decode_tps", "")
        prompt_tps = r.get("prompt_tps", "")
        ttft = r.get("ttft_ms", "")
        predicted = r.get("predicted_tokens", "")
        vram = r.get("vram_peak_mib", "")
        if not acc:
            acc = r.get("spec_acceptance", "")
        if not mean_len:
            mean_len = r.get("mean_accepted_length", "")

line = ",".join([
    case, str(n_max), str(n_min), str(p_min), str(prepeat), str(ctx),
    str(decode), str(sustained), str(prompt_tps), str(ttft),
    str(acc), str(mean_len),
    str(draft_ms), str(inject_ms), str(verify_ms), str(n_verify),
    str(predicted), str(vram), str(res),
])
summary.open("a").write(line + "\n")
print(line)
PY

  sleep "$COOLDOWN_SECONDS"
done <<< "$CASES_LIST"

{
  echo
  echo "## Results"
  echo
  echo '```'
  column -t -s, "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
  echo '```'
  echo
  echo "CSV: \`$SUMMARY_CSV\`"
} | tee -a "$SUMMARY_MD" | tee -a "$OUT_ROOT/run.log"

echo "done: $OUT_ROOT"
