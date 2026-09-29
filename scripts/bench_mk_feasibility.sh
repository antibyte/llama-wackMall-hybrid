#!/usr/bin/env bash
# Measure whether Cohere-style megakernel work is worth it on GTX 1660 Ti:
# CUDA-graph on/off throughput, MMVQ rows, and nsys inter-kernel GPU idle.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/mk-feasibility-${STAMP}}"
mkdir -p "$OUT"

BENCH="${BENCH:-$ROOT/build-main-sm75/bin/llama-bench}"
NSYS="${NSYS:-nsys}"
THREADS="${THREADS:-8}"
REPS="${REPS:-3}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"

MINICPM="${MINICPM:-$HOME/models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf}"
LFM="${LFM:-$HOME/models/lfm2.5-vl-3b/LFM2.5-VL-3B-Q4_K_M.gguf}"
NEX="${NEX:-$HOME/models/nex-n2.5-mini/Nex-N2.5-mini-Q4_K_M.gguf}"

[[ -x "$BENCH" ]] || { echo "missing $BENCH" >&2; exit 1; }

csv="$OUT/runs.csv"
printf '%s\n' 'model,label,n_prompt,n_gen,avg_ts,stddev_ts,fa,ctk,ctv,ubatch,env,status' > "$csv"
echo "results: $OUT"

run_bench() {
    local model_tag="$1" label="$2" model="$3" extra_args="$4" extra_env="$5"
    local jsonl="$OUT/bench-${model_tag}-${label}.jsonl"
    local log="$OUT/bench-${model_tag}-${label}.log"
    printf '%s\n' "$extra_env" > "$OUT/bench-${model_tag}-${label}.env"

    echo "BENCH $model_tag $label"
    # shellcheck disable=SC2086
    if env CUDA_VISIBLE_DEVICES="$CUDA_DEV" $extra_env "$BENCH" \
        -m "$model" -ngl 99 -t "$THREADS" -r "$REPS" \
        -fa on -ctk q8_0 -ctv q8_0 \
        -o jsonl -oe md \
        $extra_args \
        >"$jsonl" 2>"$log"
    then
        python3 - "$csv" "$model_tag" "$label" "$extra_env" "$jsonl" <<'PY'
import json, sys
csv, model, label, env, path = sys.argv[1:6]
env = env.replace(",", ";")
with open(path) as f:
    lines = [ln for ln in f if ln.strip()]
if not lines:
    with open(csv, "a") as o:
        o.write(f"{model},{label},,,,,?,?,?,?,{env},empty\n")
    raise SystemExit(0)
for ln in lines:
    row = json.loads(ln)
    with open(csv, "a") as o:
        o.write(
            f"{model},{label},{row.get('n_prompt',0)},{row.get('n_gen',0)},"
            f"{row.get('avg_ts','')},{row.get('stddev_ts','')},"
            f"{row.get('flash_attn','')},{row.get('type_k','')},{row.get('type_v','')},"
            f"{row.get('n_ubatch','')},{env},ok\n"
        )
PY
    else
        printf '%s,%s,,,,,,%s,fail\n' "$model_tag" "$label" "$extra_env" >> "$csv"
        echo "FAIL BENCH $model_tag $label" >&2
        tail -30 "$log" >&2 || true
    fi
}

run_nsys() {
    local model_tag="$1" label="$2" model="$3" extra_args="$4" extra_env="$5"
    local base="$OUT/nsys-${model_tag}-${label}"
    echo "NSYS $model_tag $label"
    # node-level graph trace is required to see intra-graph kernel gaps
    # shellcheck disable=SC2086
    if env CUDA_VISIBLE_DEVICES="$CUDA_DEV" $extra_env "$NSYS" profile \
        --trace=cuda --cuda-graph-trace=node \
        --sample=none --cpuctxsw=none --backtrace=none \
        --force-overwrite=true --stats=false \
        -o "$base" \
        "$BENCH" -m "$model" -ngl 99 -t "$THREADS" -r 1 --no-warmup \
        -fa on -ctk q8_0 -ctv q8_0 \
        -o jsonl -oe md \
        $extra_args \
        >"${base}.bench.jsonl" 2>"${base}.log"
    then
        local rep
        if [[ -f "${base}.nsys-rep" ]]; then
            rep="${base}.nsys-rep"
        elif [[ -f "${base}.qdrep" ]]; then
            rep="${base}.qdrep"
        else
            echo "no nsys report for $label" >&2
            return 0
        fi
        if ! "$NSYS" export --type sqlite --force-overwrite true -o "${base}.sqlite" "$rep" \
            >"${base}.export.log" 2>&1
        then
            echo "nsys export failed $label" >&2
            tail -20 "${base}.export.log" >&2 || true
        fi
    else
        echo "FAIL NSYS $model_tag $label" >&2
        tail -30 "${base}.log" >&2 || true
    fi
}

PROD_ENV="GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2"
NOGRAPH="GGML_CUDA_DISABLE_GRAPHS=1"
PROD_NOGRAPH="GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2"

# Dense GPU-resident models: graphs vs MMVQ. One load runs pp64 and pp512.
if [[ -f "$MINICPM" ]]; then
    run_bench minicpm graphs-on-def "$MINICPM" "-p 64,512 -n 128" ""
    run_bench minicpm graphs-off-def "$MINICPM" "-p 64,512 -n 128" "$NOGRAPH"
    run_bench minicpm graphs-on-prod "$MINICPM" "-p 64,512 -n 128" "$PROD_ENV"
    run_bench minicpm graphs-off-prod "$MINICPM" "-p 64,512 -n 128" "$PROD_NOGRAPH"
    run_nsys minicpm graphs-on-prod "$MINICPM" "-p 128 -n 64" "$PROD_ENV"
    run_nsys minicpm graphs-off-prod "$MINICPM" "-p 128 -n 64" "$PROD_NOGRAPH"
else
    echo "skip minicpm, missing $MINICPM" >&2
fi

if [[ -f "$LFM" ]]; then
    run_bench lfmvl graphs-on-def "$LFM" "-p 64,512 -n 128" ""
    run_bench lfmvl graphs-off-def "$LFM" "-p 64,512 -n 128" "$NOGRAPH"
    run_bench lfmvl graphs-on-prod "$LFM" "-p 64,512 -n 128" "$PROD_ENV"
    run_bench lfmvl graphs-off-prod "$LFM" "-p 64,512 -n 128" "$PROD_NOGRAPH"
    run_nsys lfmvl graphs-on-prod "$LFM" "-p 128 -n 64" "$PROD_ENV"
    run_nsys lfmvl graphs-off-prod "$LFM" "-p 128 -n 64" "$PROD_NOGRAPH"
else
    echo "skip lfmvl, missing $LFM" >&2
fi

# Hybrid MoE: graphs are often already off; still measure both.
if [[ -f "$NEX" ]]; then
    run_bench nex graphs-on-prod "$NEX" "-p 128 -n 64 -ncmoe 99" "$PROD_ENV"
    run_bench nex graphs-off-prod "$NEX" "-p 128 -n 64 -ncmoe 99" "$PROD_NOGRAPH"
    run_nsys nex graphs-off-prod "$NEX" "-p 64 -n 32 -ncmoe 99" "$PROD_NOGRAPH"
else
    echo "skip nex, missing $NEX" >&2
fi

mapfile -t SQLS < <(ls -1 "$OUT"/nsys-*.sqlite 2>/dev/null || true)
if ((${#SQLS[@]})); then
    python3 "$ROOT/scripts/analyze_nsys_gaps.py" --csv "$OUT/nsys-gaps.csv" "${SQLS[@]}" \
        | tee "$OUT/nsys-gaps.txt"
fi

python3 - "$OUT" <<'PY'
import csv, sys
from collections import defaultdict
from pathlib import Path
out = Path(sys.argv[1])
rows = []
with open(out / "runs.csv", newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

def fnum(x):
    try:
        return float(x)
    except Exception:
        return None

sol = {
    "minicpm": 1.561e9 / 288e9,
    "lfmvl": 1.674e9 / 288e9,
    "nex": None,
}

lines = []
lines.append("# Megakernel feasibility (GTX 1660 Ti)")
lines.append("")
lines.append("Question: after CUDA graphs, how much decode time is still inter-kernel idle")
lines.append("that a Cohere-style persistent megakernel could reclaim?")
lines.append("")
lines.append("## llama-bench tok/s")
lines.append("")
lines.append("| model | label | pp | tg | tok/s | std | vs graphs-on-prod | vs SoL |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

by = defaultdict(list)
for r in rows:
    by[(r["model"], r["n_prompt"], r["n_gen"])].append(r)

base = {}
for r in rows:
    if r["label"] == "graphs-on-prod" and r["status"] == "ok":
        ts = fnum(r["avg_ts"])
        if ts:
            base[(r["model"], r["n_prompt"], r["n_gen"])] = ts

for r in rows:
    ts = fnum(r["avg_ts"])
    key = (r["model"], r["n_prompt"], r["n_gen"])
    b = base.get(key)
    rel = f"{(ts/b - 1)*100:+.1f}%" if ts and b else ""
    s = sol.get(r["model"])
    # n_gen>0 is decode; n_prompt>0 and n_gen==0 is prefill
    n_gen = int(r["n_gen"] or 0)
    soll = ""
    if ts and s and n_gen > 0:
        soll = f"{ts / (1.0/s) * 100:.0f}%"
    lines.append(
        f"| {r['model']} | {r['label']} | {r['n_prompt']} | {r['n_gen']} | "
        f"{r['avg_ts']} | {r['stddev_ts']} | {rel} | {soll} |"
    )

gapf = out / "nsys-gaps.csv"
if gapf.exists():
    lines.append("")
    lines.append("## nsys GPU timeline (cuda-graph-trace=node)")
    lines.append("")
    lines.append("| file | kernels | busy | gap | span_ms | launch_api_ms | graph_api_ms | median_gap_us |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    with open(gapf, newline="") as f:
        for r in csv.DictReader(f):
            busy = fnum(r.get("busy_frac") or "")
            gap = fnum(r.get("gap_frac") or "")
            lines.append(
                f"| {r.get('file','')} | {r.get('n_kernels','')} | "
                f"{'' if busy is None else f'{busy*100:.1f}%'} | "
                f"{'' if gap is None else f'{gap*100:.1f}%'} | "
                f"{r.get('span_ms','')} | {r.get('launch_api_ms','')} | "
                f"{r.get('graph_api_ms','')} | {r.get('median_gap_us','')} |"
            )
    lines.append("")
    lines.append("busy = sum(kernel duration) / (last kernel end - first kernel start).")
    lines.append("gap is complementary idle between consecutive kernels on the GPU timeline.")
    lines.append("SoL for MiniCPM/LFM is file-bytes / 288 GB/s, decode-only; not a roofline.")

(out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", out / "SUMMARY.md")
PY

echo "done $OUT"
