#!/usr/bin/env bash
# A/B: CTX_CHECKPOINTS 0 vs 4 (mid+long TTFT) and DFlash n_max 2/3/6 (short decode).
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-$ROOT/benchmark-results/ckpt-nmax-${STAMP}}"
mkdir -p "$OUT_ROOT"

export SERVER="${SERVER:-$ROOT/build-main-sm75/bin/llama-server}"
export MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
export SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf}"
export PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"
export CONTEXT=24576
export TARGET_TYPE_K=turbo4_k TARGET_TYPE_V=turbo4_k
export DRAFT_TYPE_K=turbo4_k DRAFT_TYPE_V=turbo4_k
export CMOE_BATCH=64 CMOE_UBATCH=64
export CMOE_PREFILL_BATCH=1856 CMOE_PREFILL_UBATCH=1856
export CMOE_DECODE_BATCH=64 CMOE_DECODE_UBATCH=64
export STATIC_FIXED_S=28
export SKIP_SENTINEL=1 MOE_MULTI_FUSION=1
export CUDA_ASYNC_HOST_COPY=1 CONCAT_NONCONT_FLAT_DIM0=1
export TARGET_BACKEND_SAMPLING=1 REASONING_BUDGET=1000 LOAD_MODE=mmap
export TURBO4_MTP_EXPERIMENTAL=1 TURBO4_DFLASH_EXPERIMENTAL=1
export TURBO4_V_EXPERIMENTAL=1 TURBO4_DRAFT_EXPERIMENTAL=1
export TURBO4_F16_PREFILL_MIN_BATCH=2
export DFLASH_COMBINED=1 SPEC_MODE=dflash
export SPEC_DRAFT_N_MIN=1 SPEC_DRAFT_P_MIN=0.75 SPEC_DRAFT_BACKEND_SAMPLING=1
export LLAMA_DFLASH_DDTREE=0 LLAMA_DFLASH_TREE_VERIFY=0
export LLAMA_KVFLASH=12288 LLAMA_KVFLASH_MAX_POOL=12288
export LLAMA_KVFLASH_TAU=64 LLAMA_KVFLASH_POLICY=lru
export CASES=SC
export N_PREDICT="${N_PREDICT:-256}"
export REPEATS="${REPEATS:-3}"
export WARMUP_TOKENS=32
export COOLDOWN_SECONDS=4
export WARMUP_PROMPT_REPEAT=1

SUMMARY="$OUT_ROOT/summary.csv"
printf '%s\n' \
  'kind,ckpt,n_max,regime,prompt_repeat,prompt_n,prompt_tps,ttft_ms,decode_tps,acc,mean_len,vram,status,dir' \
  > "$SUMMARY"

run_case() {
  local kind="$1" ckpt="$2" nmax="$3" regime="$4" prepeat="$5"
  local name="${kind}_ckpt${ckpt}_nmax${nmax}_${regime}"
  local res="$OUT_ROOT/$name"
  local port="$6"
  echo "=== $name port=$port ===" | tee -a "$OUT_ROOT/run.log"
  export CTX_CHECKPOINTS="$ckpt"
  export CACHE_RAM
  if [[ "$ckpt" == 0 ]]; then
    export CACHE_RAM=0
  else
    export CACHE_RAM=2048
  fi
  export SPEC_DRAFT_N_MAX="$nmax"
  export PROMPT_REPEAT="$prepeat"
  export PORT="$port"
  export RESULTS_DIR="$res"
  local status=ok
  if ! bash "$ROOT/scripts/bench_hybrid.sh" SC "$REPEATS" "$N_PREDICT" "$WARMUP_TOKENS" \
      >>"$OUT_ROOT/run.log" 2>&1; then
    status=fail
  fi
  python3 - "$kind" "$ckpt" "$nmax" "$regime" "$prepeat" "$res" "$status" "$SUMMARY" <<'PY'
import csv, json, sys
from pathlib import Path
kind, ckpt, nmax, regime, prepeat, res, status, summary = sys.argv[1:9]
res = Path(res)
summary = Path(summary)
pp = ttft = dps = acc = ml = vram = pn = ""
if status == "ok":
    med = res / "medians.csv"
    if med.exists():
        r = next(csv.DictReader(med.open()), {})
        pp = r.get("prompt_tps", "")
        ttft = r.get("ttft_ms", "")
        dps = r.get("decode_tps", "")
        acc = r.get("spec_acceptance", r.get("mtp_acceptance", ""))
        ml = r.get("mean_accepted_length", "")
        vram = r.get("vram_peak_mib", "")
    for p in sorted(res.glob("*-run*.response.json")):
        d = json.loads(p.read_text())
        t = d.get("timings") or {}
        pn = t.get("prompt_n", pn)
        if not pp:
            pp = t.get("prompt_per_second", "")
        if not dps:
            dps = t.get("predicted_per_second", "")
        if not ttft:
            ttft = d.get("ttft_ms", "")
        break
with summary.open("a") as f:
    f.write(f"{kind},{ckpt},{nmax},{regime},{prepeat},{pn},{pp},{ttft},{dps},{acc},{ml},{vram},{status},{res}\n")
print(f"  {kind} ckpt={ckpt} nmax={nmax} {regime}: pp={pp} ttft={ttft} dec={dps} acc={acc} al={ml} {status}", flush=True)
PY
}

echo "OUT=$OUT_ROOT" | tee "$OUT_ROOT/run.log"

port=19310
# --- checkpoints: mid + long, n_max=6 (prod) ---
for ckpt in 0 4; do
  for spec in "mid|16" "long|48"; do
    IFS='|' read -r regime prepeat <<<"$spec"
    port=$((port + 1))
    run_case ckpt "$ckpt" 6 "$regime" "$prepeat" "$port"
  done
done

# --- n_max: short prompt, checkpoints off ---
for nmax in 2 3 6; do
  port=$((port + 1))
  run_case nmax 0 "$nmax" short 1 "$port"
done

python3 - "$OUT_ROOT" <<'PY'
from pathlib import Path
import csv
out = Path(__import__("sys").argv[1])
rows = list(csv.DictReader((out / "summary.csv").open()))
lines = ["", "# ckpt + n_max A/B", "", "## Checkpoints (n_max=6)", "",
         "| ckpt | regime | prompt_tps | ttft_ms | decode_tps | status |",
         "|-----:|--------|-----------:|--------:|-----------:|--------|"]
ck = [r for r in rows if r["kind"] == "ckpt"]
for r in ck:
    lines.append(f"| {r['ckpt']} | {r['regime']} | {r.get('prompt_tps','')} | {r.get('ttft_ms','')} | {r.get('decode_tps','')} | {r['status']} |")
base = {(r["regime"]): r for r in ck if r["ckpt"] == "0" and r["status"] == "ok"}
lines += ["", "### vs ckpt=0", ""]
for r in ck:
    if r["ckpt"] != "4" or r["status"] != "ok":
        continue
    b = base.get(r["regime"])
    if not b:
        continue
    def pct(a, b):
        try:
            return (float(a) / float(b) - 1) * 100
        except Exception:
            return None
    pp, tt, dd = pct(r["prompt_tps"], b["prompt_tps"]), pct(r["ttft_ms"], b["ttft_ms"]), pct(r["decode_tps"], b["decode_tps"])
    lines.append(f"- **{r['regime']}** ckpt4 vs 0: prompt {pp:+.1f}%  ttft {tt:+.1f}%  decode {dd:+.1f}%" if pp is not None else f"- {r['regime']}: n/a")

lines += ["", "## n_max short (ckpt=0)", "",
          "| n_max | decode_tps | acc | mean_len | prompt_tps | status |",
          "|------:|-----------:|----:|---------:|-----------:|--------|"]
nm = [r for r in rows if r["kind"] == "nmax"]
for r in nm:
    lines.append(f"| {r['n_max']} | {r.get('decode_tps','')} | {r.get('acc','')} | {r.get('mean_len','')} | {r.get('prompt_tps','')} | {r['status']} |")
text = "\n".join(lines) + "\n"
(out / "SUMMARY.md").write_text(text)
print(text)
print("CSV", out / "summary.csv")
PY
echo "Done OUT=$OUT_ROOT"
