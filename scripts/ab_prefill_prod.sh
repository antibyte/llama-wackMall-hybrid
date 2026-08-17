#!/usr/bin/env bash
# Phase-0/1 production prefill sweep (GTX 1660 Ti / start1660 stack).
#
# Primary: Mid (prompt_repeat=16) + Long (48) prompt_tps with DFlash.
# Isolation: same cells with SPEC_MODE=none.
# Decode gate: optional second pass with N_PREDICT=256.
#
# Usage:
#   ./scripts/ab_prefill_prod.sh
#   QUICK=1 ./scripts/ab_prefill_prod.sh          # fewer cells, REPEATS=1
#   MODE_LIST=dflash PREFILL_LIST="1024 1536" S_LIST="30 28" ./scripts/ab_prefill_prod.sh
#   N_PREDICT=256 REPEATS=3 ./scripts/ab_prefill_prod.sh   # e2e decode gate
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-$ROOT/benchmark-results/prefill-prod-${STAMP}}"
mkdir -p "$OUT_ROOT"

# --- start1660 production defaults (override via env) ---
export SERVER="${SERVER:-$ROOT/build-main-sm75/bin/llama-server}"
export MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
export SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf}"
export PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"
export CONTEXT="${CONTEXT:-24576}"
export TARGET_TYPE_K="${TARGET_TYPE_K:-turbo4_k}"
export TARGET_TYPE_V="${TARGET_TYPE_V:-turbo4_k}"
export DRAFT_TYPE_K="${DRAFT_TYPE_K:-turbo4_k}"
export DRAFT_TYPE_V="${DRAFT_TYPE_V:-turbo4_k}"
export CMOE_BATCH="${CMOE_BATCH:-64}"
export CMOE_UBATCH="${CMOE_UBATCH:-64}"
export CMOE_DECODE_BATCH="${CMOE_DECODE_BATCH:-64}"
export CMOE_DECODE_UBATCH="${CMOE_DECODE_UBATCH:-64}"
export SKIP_SENTINEL="${SKIP_SENTINEL:-1}"
export MOE_MULTI_FUSION="${MOE_MULTI_FUSION:-1}"
export CUDA_ASYNC_HOST_COPY="${CUDA_ASYNC_HOST_COPY:-1}"
export CONCAT_NONCONT_FLAT_DIM0="${CONCAT_NONCONT_FLAT_DIM0:-1}"
export TARGET_BACKEND_SAMPLING="${TARGET_BACKEND_SAMPLING:-1}"
export REASONING_BUDGET="${REASONING_BUDGET:-1000}"
export LOAD_MODE="${LOAD_MODE:-mmap}"
export TURBO4_MTP_EXPERIMENTAL=1
export TURBO4_DFLASH_EXPERIMENTAL=1
export TURBO4_V_EXPERIMENTAL=1
export TURBO4_DRAFT_EXPERIMENTAL=1
export TURBO4_F16_PREFILL_MIN_BATCH="${TURBO4_F16_PREFILL_MIN_BATCH:-2}"
export DFLASH_COMBINED="${DFLASH_COMBINED:-1}"
export SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-6}"
export SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-1}"
export SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.75}"
export SPEC_DRAFT_BACKEND_SAMPLING=1
export LLAMA_DFLASH_DDTREE=0
export LLAMA_DFLASH_TREE_VERIFY=0
export LLAMA_KVFLASH="${LLAMA_KVFLASH:-12288}"
export LLAMA_KVFLASH_MAX_POOL="${LLAMA_KVFLASH_MAX_POOL:-12288}"
export LLAMA_KVFLASH_TAU="${LLAMA_KVFLASH_TAU:-64}"
export LLAMA_KVFLASH_POLICY="${LLAMA_KVFLASH_POLICY:-lru}"
export CASES=SC
export N_PREDICT="${N_PREDICT:-1}"
export REPEATS="${REPEATS:-1}"
export WARMUP_TOKENS="${WARMUP_TOKENS:-16}"
export COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-4}"
export WARMUP_PROMPT_REPEAT=1
export PORT_BASE="${PORT_BASE:-18510}"

if [[ "${QUICK:-0}" == 1 ]]; then
  PREFILL_LIST="${PREFILL_LIST:-1024 1536 2048}"
  S_LIST="${S_LIST:-30 28}"
  MODE_LIST="${MODE_LIST:-dflash none}"
  REGIME_LIST="${REGIME_LIST:-mid|16 long|48}"
  REPEATS="${REPEATS:-1}"
else
  PREFILL_LIST="${PREFILL_LIST:-1024 1280 1536 2048}"
  S_LIST="${S_LIST:-30 28}"
  MODE_LIST="${MODE_LIST:-dflash none}"
  REGIME_LIST="${REGIME_LIST:-mid|16 long|48}"
fi

SUMMARY_CSV="$OUT_ROOT/summary.csv"
SUMMARY_MD="$OUT_ROOT/SUMMARY.md"
RUN_LOG="$OUT_ROOT/run.log"

printf '%s\n' \
  'mode,s,prefill,regime,prompt_repeat,n_predict,prompt_n,prompt_tps,ttft_ms,decode_tps,vram_peak_mib,effective_s,status,results_dir' \
  > "$SUMMARY_CSV"

{
  echo "# Prefill prod sweep — $STAMP"
  echo
  echo "- CONTEXT=$CONTEXT  n_predict=$N_PREDICT  REPEATS=$REPEATS"
  echo "- KV: $TARGET_TYPE_K/$TARGET_TYPE_V  draft=$DRAFT_TYPE_K  KVFlash=$LLAMA_KVFLASH"
  echo "- decode batch: $CMOE_DECODE_BATCH/$CMOE_DECODE_UBATCH  DFlash n_max=$SPEC_DRAFT_N_MAX n_min=$SPEC_DRAFT_N_MIN p_min=$SPEC_DRAFT_P_MIN"
  echo "- PREFILL_LIST=$PREFILL_LIST"
  echo "- S_LIST=$S_LIST"
  echo "- MODE_LIST=$MODE_LIST"
  echo "- REGIME_LIST=$REGIME_LIST"
  echo "- OUT: \`$OUT_ROOT\`"
  echo
} | tee "$RUN_LOG" > "$SUMMARY_MD"

port_i=0
failed=0

for mode in $MODE_LIST; do
  for s in $S_LIST; do
    for pref in $PREFILL_LIST; do
      # one server config; run all regimes against it (fresh process per regime via bench_hybrid)
      for regime_spec in $REGIME_LIST; do
        IFS='|' read -r regime prepeat <<<"$regime_spec"
        port_i=$((port_i + 1))
        port=$((PORT_BASE + port_i))
        case_name="${mode}_s${s}_p${pref}_${regime}"
        res="$OUT_ROOT/${case_name}"
        echo "=== $case_name prompt_repeat=$prepeat port=$port ===" | tee -a "$RUN_LOG"

        export SPEC_MODE="$mode"
        export STATIC_FIXED_S="$s"
        export CMOE_PREFILL_BATCH="$pref"
        export CMOE_PREFILL_UBATCH="$pref"
        export PROMPT_REPEAT="$prepeat"
        export PORT="$port"
        export RESULTS_DIR="$res"
        export N_PREDICT REPEATS WARMUP_TOKENS COOLDOWN_SECONDS

        status=ok
        if ! bash "$ROOT/scripts/bench_hybrid.sh" SC "$REPEATS" "$N_PREDICT" "$WARMUP_TOKENS" \
            >>"$RUN_LOG" 2>&1; then
          status=fail
          failed=$((failed + 1))
          echo "FAILED $case_name" | tee -a "$RUN_LOG"
          printf '%s\n' \
            "$mode,$s,$pref,$regime,$prepeat,$N_PREDICT,,,,,,,$status,$res" \
            >> "$SUMMARY_CSV"
          continue
        fi

        python3 - "$mode" "$s" "$pref" "$regime" "$prepeat" "$N_PREDICT" "$res" "$SUMMARY_CSV" <<'PY'
import json, re, sys
from pathlib import Path

mode, s, pref, regime, prepeat, npred, res, summary = sys.argv[1:9]
res = Path(res)
summary = Path(summary)

cands = sorted(res.glob("*-run*.response.json"))
if not cands:
    with summary.open("a") as f:
        f.write(f"{mode},{s},{pref},{regime},{prepeat},{npred},,,,,,,fail_no_json,{res}\n")
    print("  no response json", flush=True)
    raise SystemExit(0)

# median over repeats if multiple
rows = []
for p in cands:
    d = json.loads(p.read_text())
    t = d.get("timings") or {}
    pn = float(t.get("prompt_n") or 0)
    pms = float(t.get("prompt_ms") or 0)
    pps = float(t.get("prompt_per_second") or 0)
    if (not pps) and pms and pn:
        pps = 1000.0 * pn / pms
    ttft = float(d.get("ttft_ms") or 0)
    dps = float(t.get("predicted_per_second") or 0)
    rows.append((pn, pps, ttft, dps))

rows.sort(key=lambda r: r[1])
med = rows[len(rows) // 2]
pn, pps, ttft, dps = med

# vram / effective S from server log
vram = ""
eff_s = ""
for log in sorted(res.glob("*.log")):
    text = log.read_text(errors="replace")
    m = re.search(r"vram[_\s]*peak[^\d]*(\d+(?:\.\d+)?)", text, re.I)
    # samples csv last vram
    break
samples = sorted(res.glob("*-run*.samples.csv"))
if samples:
    import csv
    last = None
    with samples[-1].open() as f:
        for row in csv.DictReader(f):
            last = row
    if last and last.get("vram_mib"):
        vram = last["vram_mib"]

# medians.csv if present
med_csv = res / "medians.csv"
if med_csv.exists():
    import csv
    with med_csv.open() as f:
        r = next(csv.DictReader(f), None)
        if r:
            if r.get("prompt_tps"):
                pps = float(r["prompt_tps"])
            if r.get("ttft_ms"):
                ttft = float(r["ttft_ms"])
            if r.get("decode_tps"):
                dps = float(r["decode_tps"])
            if r.get("vram_peak_mib"):
                vram = r["vram_peak_mib"]
            if r.get("effective_fixed_s"):
                eff_s = r["effective_fixed_s"]
            if r.get("prompt_n") or False:
                pass

# effective S from log
for log in list(res.glob("*.log")) + list(res.glob("**/*.log")):
    text = log.read_text(errors="replace")
    m = re.search(r"expert tiering on:\s*(\d+)\s*fixed", text)
    if m:
        eff_s = m.group(1)
        break
    m = re.search(r"set S\s*=\s*(\d+)", text)
    if m:
        eff_s = m.group(1)
        break

with summary.open("a") as f:
    f.write(
        f"{mode},{s},{pref},{regime},{prepeat},{npred},"
        f"{pn:.0f},{pps:.4f},{ttft:.2f},{dps:.4f},{vram},{eff_s},ok,{res}\n"
    )
print(
    f"  prompt_n={pn:.0f} prompt_tps={pps:.2f} ttft={ttft:.0f}ms "
    f"decode={dps:.2f} vram={vram} S_eff={eff_s}",
    flush=True,
)
PY
      done
    done
  done
done

python3 - "$OUT_ROOT" <<'PY'
import csv
from pathlib import Path
from collections import defaultdict

out = Path(__import__("sys").argv[1])
rows = [r for r in csv.DictReader((out / "summary.csv").open()) if r.get("status") == "ok"]

# baseline: dflash s=30 prefill=1024 per regime
base = {}
for r in rows:
    if r["mode"] == "dflash" and r["s"] == "30" and r["prefill"] == "1024":
        base[r["regime"]] = float(r["prompt_tps"])

lines = ["", "## Results (ok only)", "",
         "| mode | S | prefill | regime | prompt_tps | vs base | ttft_ms | decode_tps | vram |",
         "|------|--:|--------:|--------|-----------:|--------:|--------:|-----------:|-----:|"]

by_key = defaultdict(list)
for r in rows:
    by_key[(r["mode"], r["s"], r["prefill"], r["regime"])].append(r)

for key in sorted(by_key.keys(), key=lambda k: (k[0], int(k[1]), int(k[2]), k[3])):
    r = by_key[key][0]
    pps = float(r["prompt_tps"])
    b = base.get(r["regime"])
    delta = f"{(pps / b - 1) * 100:+.1f}%" if b and b > 0 else "—"
    lines.append(
        f"| {r['mode']} | {r['s']} | {r['prefill']} | {r['regime']} | "
        f"{pps:.2f} | {delta} | {float(r['ttft_ms']):.0f} | "
        f"{float(r['decode_tps'] or 0):.2f} | {r.get('vram_peak_mib','')} |"
    )

# best per regime for dflash
lines += ["", "## Best DFlash vs baseline (S=30 p=1024)", ""]
for regime in ("mid", "long"):
    cands = [r for r in rows if r["mode"] == "dflash" and r["regime"] == regime]
    if not cands:
        continue
    best = max(cands, key=lambda r: float(r["prompt_tps"]))
    b = base.get(regime)
    pps = float(best["prompt_tps"])
    delta = (pps / b - 1) * 100 if b else 0
    gate = "PASS ≥10%" if b and delta >= 10 else ("below 10%" if b else "no baseline")
    lines.append(
        f"- **{regime}**: best S={best['s']} prefill={best['prefill']} "
        f"→ {pps:.2f} tok/s ({delta:+.1f}% vs base) — {gate}"
    )

# isolation none vs dflash at same geometry
lines += ["", "## Isolation (none vs dflash) at same S/prefill", ""]
for r in rows:
    if r["mode"] != "none":
        continue
    match = next(
        (x for x in rows
         if x["mode"] == "dflash" and x["s"] == r["s"]
         and x["prefill"] == r["prefill"] and x["regime"] == r["regime"]),
        None,
    )
    if not match:
        continue
    pn, pd = float(r["prompt_tps"]), float(match["prompt_tps"])
    lines.append(
        f"- S={r['s']} p={r['prefill']} {r['regime']}: "
        f"none={pn:.2f} dflash={pd:.2f} (dflash {(pd/pn-1)*100:+.1f}% vs none)"
        if pn > 0 else f"- S={r['s']} p={r['prefill']} {r['regime']}: n/a"
    )

text = "\n".join(lines) + "\n"
with (out / "SUMMARY.md").open("a") as f:
    f.write(text)
print(text)
print(f"Wrote {out / 'SUMMARY.md'}")
print(f"CSV: {out / 'summary.csv'}")
PY

echo "Done. failed=$failed OUT=$OUT_ROOT"
exit "$failed"
