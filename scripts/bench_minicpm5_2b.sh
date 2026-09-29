#!/usr/bin/env bash
# MiniCPM5-2B Q4_K_M kernel/KV sweep on GTX 1660 Ti.
# Only knobs that can fire on a dense Llama (no MoE/MTP/DFlash/DSpark).
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/minicpm5-2b-tune-${STAMP}}"
mkdir -p "$OUT"

BENCH="${BENCH:-$ROOT/build-main-sm75/bin/llama-bench}"
CLI="${CLI:-$ROOT/build-main-sm75/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf}"
REPS="${REPS:-3}"
THREADS="${THREADS:-8}"

[[ -x "$BENCH" ]] || { echo "missing $BENCH" >&2; exit 1; }
[[ -x "$CLI" ]] || { echo "missing $CLI" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL" >&2; exit 1; }

csv="$OUT/runs.csv"
printf '%s\n' 'phase,label,n_prompt,n_gen,avg_ts,stddev_ts,fa,ctk,ctv,ubatch,env,status' > "$csv"

run_bench() {
    local phase="$1" label="$2" extra_args="$3" extra_env="$4"
    local jsonl="$OUT/${phase}-${label}.jsonl"
    local log="$OUT/${phase}-${label}.log"
    local env_file="$OUT/${phase}-${label}.env"
    printf '%s\n' "$extra_env" > "$env_file"

    # shellcheck disable=SC2086
    if env CUDA_VISIBLE_DEVICES=0 $extra_env "$BENCH" \
        -m "$MODEL" -ngl 99 -t "$THREADS" -r "$REPS" \
        -o jsonl -oe md \
        $extra_args \
        >"$jsonl" 2>"$log"
    then
        python3 - "$csv" "$phase" "$label" "$extra_env" "$jsonl" <<'PY'
import json, sys
csv, phase, label, env, path = sys.argv[1:6]
with open(path) as f:
    lines = [ln for ln in f if ln.strip()]
if not lines:
    with open(csv, 'a') as o:
        o.write(f'{phase},{label},,,,,?,?,?,?,{env},empty\n')
    raise SystemExit(0)
for ln in lines:
    row = json.loads(ln)
    n_prompt = row.get('n_prompt', 0)
    n_gen = row.get('n_gen', 0)
    avg = row.get('avg_ts', '')
    std = row.get('stddev_ts', '')
    fa = row.get('flash_attn', '')
    ctk = row.get('type_k', '')
    ctv = row.get('type_v', '')
    ub = row.get('n_ubatch', '')
    with open(csv, 'a') as o:
        o.write(f'{phase},{label},{n_prompt},{n_gen},{avg},{std},{fa},{ctk},{ctv},{ub},{env},ok\n')
PY
    else
        printf '%s,%s,,,,,,%s,fail\n' "$phase" "$label" "$extra_env" >> "$csv"
        echo "FAIL $phase $label" >&2
        tail -40 "$log" >&2 || true
    fi
}

echo "results: $OUT"

# --- llama-bench: FA and KV ---
run_bench fa_kv fa-on-q8 "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128" ""
run_bench fa_kv fa-off-q8 "-fa off -ctk q8_0 -ctv q8_0 -p 512 -n 128" ""
run_bench fa_kv fa-on-f16 "-fa on -ctk f16 -ctv f16 -p 512 -n 128" ""
run_bench fa_kv fa-on-q4 "-fa on -ctk q4_0 -ctv q4_0 -p 512 -n 128" ""

# --- MMVQ Q4_K decode rows (default 1) ---
for rows in 0 1 2 4; do
    run_bench q4k "rows${rows}" "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128" \
        "GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=${rows}"
done

# --- MMVQ Q6_K decode rows ---
for rows in 0 1 2 4; do
    run_bench q6k "n1-rows${rows}" "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128" \
        "GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=${rows}"
done

# --- MMVQ Q6_K ncols3 (prefill/verify) ---
for rows in 0 2 4; do
    run_bench q6k3 "n3-rows${rows}" "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128" \
        "GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=${rows}"
done

# --- scheduler / copy knobs ---
run_bench sched concat-on "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128" \
    "GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1 GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=128"
run_bench sched async-h2d "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128" \
    "GGML_CUDA_ASYNC_HOST_COPY=1"
run_bench sched dedup-sync "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128" \
    "GGML_SCHED_DEDUP_DST_SYNC=1"
run_bench sched register-host "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128" \
    "GGML_CUDA_REGISTER_HOST=1"
run_bench sched graphs-off "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128" \
    "GGML_CUDA_DISABLE_GRAPHS=1"

# --- prefill ubatch ---
for ub in 256 512 1024 2048; do
    run_bench prefill "ub${ub}" "-fa on -ctk q8_0 -ctv q8_0 -p 512,2048 -n 0 -b ${ub} -ub ${ub}" ""
done

# --- combined recipe from Ling/Spark winners ---
run_bench combo spark-ling "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128" \
    "GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=4 GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1 GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=128 GGML_CUDA_ASYNC_HOST_COPY=1 GGML_SCHED_DEDUP_DST_SYNC=1 GGML_CUDA_REGISTER_HOST=1"

# --- llama-cli: turbo4, ngram, kvflash, backend sampling ---
cli_case() {
    local name="$1"
    shift
    local log="$OUT/cli-${name}.log"
    if "$@" >"$log" 2>&1; then
        echo "cli ${name}: ok" | tee -a "$OUT/cli-summary.txt"
        grep -E 'Prompt:|Generation:|error|Error|invalid|garbled|turbo4|ngram|KVFlash|spec' "$log" \
            | tail -20 >> "$OUT/cli-summary.txt" || true
        echo "-----" >> "$OUT/cli-summary.txt"
    else
        echo "cli ${name}: FAIL rc=$?" | tee -a "$OUT/cli-summary.txt"
        tail -30 "$log" >> "$OUT/cli-summary.txt" || true
        echo "-----" >> "$OUT/cli-summary.txt"
    fi
}

cli_case turbo4 env \
    CUDA_VISIBLE_DEVICES=0 \
    LLAMA_TURBO4_V_EXPERIMENTAL=1 \
    "$CLI" -m "$MODEL" -ngl 99 -fa on \
    --cache-type-k turbo4_k --cache-type-v turbo4_k \
    -c 2048 -n 48 -t "$THREADS" --offline --no-display-prompt \
    --single-turn -cnv --jinja \
    -p "What is 2+2? Answer with one number."

cli_case ngram-none env \
    CUDA_VISIBLE_DEVICES=0 \
    "$CLI" -m "$MODEL" -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 2048 -n 96 -t "$THREADS" --offline --no-display-prompt \
    --spec-type none \
    -p "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. Continue:"

cli_case ngram-simple env \
    CUDA_VISIBLE_DEVICES=0 \
    "$CLI" -m "$MODEL" -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 2048 -n 96 -t "$THREADS" --offline --no-display-prompt \
    --spec-type ngram-simple --spec-draft-n-max 8 \
    --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8 \
    -p "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. Continue:"

cli_case kvflash env \
    CUDA_VISIBLE_DEVICES=0 \
    LLAMA_KVFLASH=8192 LLAMA_KVFLASH_MAX_POOL=8192 LLAMA_KVFLASH_POLICY=lru \
    "$CLI" -m "$MODEL" -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 32768 -n 16 -t "$THREADS" --offline --no-display-prompt \
    -p "Say ok."

cli_case backend-samp env \
    CUDA_VISIBLE_DEVICES=0 \
    "$CLI" -m "$MODEL" -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 2048 -n 32 -t "$THREADS" --offline --no-display-prompt --backend-sampling \
    --single-turn -cnv --jinja \
    -p "Reply with the word ping."

python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = list(csv.DictReader((out / 'runs.csv').open()))
ok = [r for r in rows if r.get('status') == 'ok' and r.get('avg_ts')]
md = ['# MiniCPM5-2B Q4_K_M GTX 1660 Ti tune', '',
      'Dense Llama: MoE/MTP/DFlash/DSpark knobs are inert. Sweep is FA, KV, MMVQ Q4_K/Q6_K, scheduler copies, prefill ubatch.', '']
by_phase = {}
for r in ok:
    by_phase.setdefault(r['phase'], []).append(r)
for phase, items in by_phase.items():
    md.append(f'## {phase}')
    md.append('')
    md.append('| label | test | t/s | std | fa | ctk/ctv | ub | env |')
    md.append('|---|---|---:|---:|---|---|---|---|')
    for r in items:
        test = f"pp{r['n_prompt']}" if r['n_gen'] in ('0','') else (f"tg{r['n_gen']}" if r['n_prompt'] in ('0','') else f"pp{r['n_prompt']}/tg{r['n_gen']}")
        md.append(f"| {r['label']} | {test} | {float(r['avg_ts']):.2f} | {float(r['stddev_ts'] or 0):.2f} | {r['fa']} | {r['ctk']}/{r['ctv']} | {r['ubatch']} | `{r['env']}` |")
    md.append('')
# pick best tg128 among fa_kv and kernel phases
tg = [r for r in ok if r.get('n_gen') not in ('0','') and float(r['n_gen']) >= 64]
if tg:
    best = max(tg, key=lambda r: float(r['avg_ts']))
    md.append('## Best decode (tg>=64)')
    md.append('')
    md.append(f"- **{best['phase']}/{best['label']}**: {float(best['avg_ts']):.2f} t/s env=`{best['env']}`")
    md.append('')
pp = [r for r in ok if r.get('n_prompt') not in ('0','') and float(r['n_prompt']) >= 512 and r.get('n_gen') in ('0','')]
if not pp:
    pp = [r for r in ok if r.get('n_prompt') == '512']
if pp:
    bestp = max(pp, key=lambda r: float(r['avg_ts']))
    md.append('## Best prefill')
    md.append('')
    md.append(f"- **{bestp['phase']}/{bestp['label']} pp{bestp['n_prompt']}**: {float(bestp['avg_ts']):.2f} t/s")
    md.append('')
cli = out / 'cli-summary.txt'
if cli.exists():
    md.append('## llama-cli extras')
    md.append('')
    md.append('```')
    md.append(cli.read_text()[:4000])
    md.append('```')
(out / 'SUMMARY.md').write_text('\n'.join(md) + '\n')
print(f'wrote {out / "SUMMARY.md"} ({len(ok)} ok rows)')
PY

echo DONE "$OUT"
