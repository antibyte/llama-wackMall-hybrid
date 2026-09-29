#!/usr/bin/env bash
# A/B: production sm_75 vs Pascal MMQ (61-virtual + FORCE_MMQ) on GTX 1660 Ti.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/pascal-mmq-${STAMP}}"
mkdir -p "$OUT"

SM75="${SM75:-$ROOT/build-main-sm75/bin/llama-bench}"
PASCAL="${PASCAL:-$ROOT/build-mmq-pascal/bin/llama-bench}"
THREADS="${THREADS:-8}"
REPS="${REPS:-3}"

MINICPM="${MINICPM:-$HOME/models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf}"
LFM="${LFM:-$HOME/models/lfm2.5-vl-3b/LFM2.5-VL-3B-Q4_K_M.gguf}"
NEX="${NEX:-$HOME/models/nex-n2.5-mini/Nex-N2.5-mini-Q4_K_M.gguf}"

[[ -x "$SM75" ]] || { echo "missing $SM75" >&2; exit 1; }
[[ -x "$PASCAL" ]] || { echo "missing $PASCAL" >&2; exit 1; }

csv="$OUT/runs.csv"
printf '%s\n' 'bin,model,label,n_prompt,n_gen,avg_ts,stddev_ts,status' > "$csv"
echo "results: $OUT"

run_one() {
    local bin_tag="$1" bin="$2" model_tag="$3" model="$4" label="$5" extra_args="$6" extra_env="$7"
    local jsonl="$OUT/${bin_tag}-${model_tag}-${label}.jsonl"
    local log="$OUT/${bin_tag}-${model_tag}-${label}.log"
    echo "BENCH $bin_tag $model_tag $label"
    # shellcheck disable=SC2086
    if env CUDA_VISIBLE_DEVICES=0 $extra_env "$bin" \
        -m "$model" -ngl 99 -t "$THREADS" -r "$REPS" \
        -fa on -ctk q8_0 -ctv q8_0 \
        -o jsonl -oe md \
        $extra_args \
        >"$jsonl" 2>"$log"
    then
        python3 - "$csv" "$bin_tag" "$model_tag" "$label" "$jsonl" <<'PY'
import json, sys
csv, btag, model, label, path = sys.argv[1:6]
with open(path) as f:
    lines = [ln for ln in f if ln.strip()]
if not lines:
    open(csv, "a").write(f"{btag},{model},{label},,,,,empty\n")
    raise SystemExit(0)
for ln in lines:
    r = json.loads(ln)
    open(csv, "a").write(
        f"{btag},{model},{label},{r.get('n_prompt',0)},{r.get('n_gen',0)},"
        f"{r.get('avg_ts','')},{r.get('stddev_ts','')},ok\n"
    )
PY
    else
        echo "${bin_tag},${model_tag},${label},,,,,fail" >> "$csv"
        echo "FAIL $bin_tag $model_tag $label" >&2
        tail -25 "$log" >&2 || true
    fi
}

PROD="GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2"

for bin_tag in sm75 pascal; do
    if [[ "$bin_tag" == sm75 ]]; then
        bin="$SM75"
    else
        bin="$PASCAL"
    fi
    if [[ -f "$MINICPM" ]]; then
        run_one "$bin_tag" "$bin" minicpm "$MINICPM" def "-p 64,512 -n 128" ""
        run_one "$bin_tag" "$bin" minicpm "$MINICPM" prod "-p 64,512 -n 128" "$PROD"
    fi
    if [[ -f "$LFM" ]]; then
        run_one "$bin_tag" "$bin" lfmvl "$LFM" def "-p 64,512 -n 128" ""
        run_one "$bin_tag" "$bin" lfmvl "$LFM" prod "-p 64,512 -n 128" "$PROD"
    fi
    if [[ -f "$NEX" ]]; then
        run_one "$bin_tag" "$bin" nex "$NEX" prod "-p 128 -n 64 -ncmoe 99" "$PROD"
    fi
done

python3 - "$OUT" <<'PY'
import csv, sys
from collections import defaultdict
from pathlib import Path
out = Path(sys.argv[1])
rows = list(csv.DictReader(open(out / "runs.csv")))

def f(x):
    try:
        return float(x)
    except Exception:
        return None

base = {}
for r in rows:
    if r["bin"] == "sm75" and r["status"] == "ok":
        ts = f(r["avg_ts"])
        if ts:
            base[(r["model"], r["label"], r["n_prompt"], r["n_gen"])] = ts

lines = [
    "# Pascal MMQ vs sm_75 (GTX 1660 Ti)",
    "",
    "Candidate: `CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual` + `GGML_CUDA_FORCE_MMQ=ON`.",
    "Baseline: `build-main-sm75` (`arch=75`, FORCE_MMQ off).",
    "Same llama-bench flags: `-fa on -ctk q8_0 -ctv q8_0 -ngl 99`.",
    "prod = MMVQ Q4_K/Q6_K ncols1 rows=2.",
    "",
    "| bin | model | label | pp | tg | tok/s | std | vs sm75 |",
    "|---|---|---|---:|---:|---:|---:|---:|",
]
for r in rows:
    ts = f(r["avg_ts"])
    key = (r["model"], r["label"], r["n_prompt"], r["n_gen"])
    b = base.get(key)
    rel = ""
    if ts and b:
        rel = f"{(ts/b - 1)*100:+.1f}%"
    lines.append(
        f"| {r['bin']} | {r['model']} | {r['label']} | {r['n_prompt']} | {r['n_gen']} | "
        f"{r['avg_ts']} | {r['stddev_ts']} | {rel} |"
    )

# decode-only summary
lines += ["", "## Decode (n_gen>0)", ""]
for model in ("minicpm", "lfmvl", "nex"):
    for label in ("def", "prod"):
        s = next((f(r["avg_ts"]) for r in rows if r["bin"]=="sm75" and r["model"]==model and r["label"]==label and int(r["n_gen"] or 0)>0 and r["status"]=="ok"), None)
        p = next((f(r["avg_ts"]) for r in rows if r["bin"]=="pascal" and r["model"]==model and r["label"]==label and int(r["n_gen"] or 0)>0 and r["status"]=="ok"), None)
        if s and p:
            lines.append(f"- {model} {label}: sm75 {s:.2f} vs pascal {p:.2f} ({(p/s-1)*100:+.1f}%)")

(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("wrote", out / "SUMMARY.md")
PY

echo "done $OUT"
