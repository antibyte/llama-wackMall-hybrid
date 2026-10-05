#!/usr/bin/env bash
# FrogNano-4B-2609 Q6_K_L kernel/KV/sampling sweep on GTX 1660 Ti.
# qwen35 hybrid (24 linear + 8 full attention) with one NextN layer.
# MoE knobs are inert. MTP is exercised via llama-cli --spec-type draft-mtp.
# Sweep: sm75 DP4A vs Pascal FORCE_MMQ, FA, KV, Q6_K/Q8/Q4_K MMVQ rows,
# scheduler copies, prefill ubatch, ngram, MTP, Turbo4, KVFlash, sampling.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/frognano-4b-q6kl-tune-${STAMP}}"
mkdir -p "$OUT"

SM75="${SM75:-$ROOT/build-main-sm75/bin/llama-bench}"
PASCAL="${PASCAL:-$ROOT/build-mmq-pascal/bin/llama-bench}"
SM75_CLI="${SM75_CLI:-$ROOT/build-main-sm75/bin/llama-cli}"
PASCAL_CLI="${PASCAL_CLI:-$ROOT/build-mmq-pascal/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/frognano-4b-2609/FrogNano-4B-2609-Q6_K_L.gguf}"
REPS="${REPS:-3}"
THREADS="${THREADS:-8}"

[[ -x "$SM75" ]] || { echo "missing $SM75" >&2; exit 1; }
[[ -x "$PASCAL" ]] || { echo "missing $PASCAL" >&2; exit 1; }
[[ -x "$SM75_CLI" ]] || { echo "missing $SM75_CLI" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL (run ./download-frognano-4b.sh)" >&2; exit 1; }

csv="$OUT/runs.csv"
printf '%s\n' 'phase,bin,label,n_prompt,n_gen,avg_ts,stddev_ts,fa,ctk,ctv,ubatch,env,status' > "$csv"
echo "results: $OUT" | tee "$OUT/progress.log"

if command -v busctl >/dev/null 2>&1; then
    busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Performance >/dev/null 2>&1 || true
fi

dump_gguf() {
    python3 - "$MODEL" "$OUT/gguf-meta.txt" <<'PY'
import struct, sys
path, outp = sys.argv[1], sys.argv[2]
def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')
def skip_value(f, t):
    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t == 8:
        read_str(f)
        return
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        if et == 8:
            for _ in range(n):
                read_str(f)
        else:
            f.read(n * sizes[et])
        return
    f.read(sizes[t])
def read_value(f, t):
    if t == 4: return struct.unpack('<I', f.read(4))[0]
    if t == 5: return struct.unpack('<i', f.read(4))[0]
    if t == 6: return struct.unpack('<f', f.read(4))[0]
    if t == 7: return bool(f.read(1)[0])
    if t == 8: return read_str(f)
    if t == 10: return struct.unpack('<Q', f.read(8))[0]
    skip_value(f, t)
    return '<skip>'
want_suffixes = (
    'architecture', 'name', 'file_type', 'block_count', 'context_length',
    'embedding_length', 'feed_forward_length', 'attention.head_count',
    'attention.head_count_kv', 'attention.key_length', 'attention.value_length',
    'attention.sliding_window', 'expert_count', 'expert_used_count',
)
with open(path, 'rb') as f, open(outp, 'w') as o:
    assert f.read(4) == b'GGUF'
    ver, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))
    o.write(f'version={ver} n_tensors={n_tensors} n_kv={n_kv}\n')
    for _ in range(n_kv):
        key = read_str(f)
        t = struct.unpack('<I', f.read(4))[0]
        if key in ('general.architecture', 'general.name', 'general.file_type',
                   'tokenizer.ggml.pre') or key.endswith(want_suffixes) or 'sliding' in key:
            val = read_value(f, t)
            s = str(val)
            if len(s) > 240:
                s = s[:240] + '...'
            o.write(f'{key}: {s}\n')
        else:
            skip_value(f, t)
print(open(outp).read())
PY
}

append_jsonl() {
    local phase="$1" bin_tag="$2" label="$3" extra_env="$4" jsonl="$5"
    python3 - "$csv" "$phase" "$bin_tag" "$label" "$extra_env" "$jsonl" <<'PY'
import json, sys
csv, phase, btag, label, env, path = sys.argv[1:7]
with open(path) as f:
    lines = [ln for ln in f if ln.strip()]
if not lines:
    with open(csv, 'a') as o:
        o.write(f'{phase},{btag},{label},,,,,?,?,?,?,{env},empty\n')
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
        o.write(f'{phase},{btag},{label},{n_prompt},{n_gen},{avg},{std},{fa},{ctk},{ctv},{ub},{env},ok\n')
PY
}

run_bench() {
    local phase="$1" bin_tag="$2" bin="$3" label="$4" extra_args="$5" extra_env="$6"
    local jsonl="$OUT/${phase}-${bin_tag}-${label}.jsonl"
    local log="$OUT/${phase}-${bin_tag}-${label}.log"
    printf '%s\n' "$extra_env" > "$OUT/${phase}-${bin_tag}-${label}.env"
    echo "BENCH $phase $bin_tag $label" | tee -a "$OUT/progress.log"
    # shellcheck disable=SC2086
    if env CUDA_VISIBLE_DEVICES=0 $extra_env "$bin" \
        -m "$MODEL" -ngl 99 -t "$THREADS" -r "$REPS" \
        -o jsonl -oe md \
        $extra_args \
        >"$jsonl" 2>"$log"
    then
        append_jsonl "$phase" "$bin_tag" "$label" "$extra_env" "$jsonl"
        grep -E 'pp[0-9]|tg[0-9]|error|failed|out of memory' "$log" | tail -8 | tee -a "$OUT/progress.log" || true
    else
        printf '%s,%s,%s,,,,,,%s,fail\n' "$phase" "$bin_tag" "$label" "$extra_env" >> "$csv"
        echo "FAIL $phase $bin_tag $label" | tee -a "$OUT/progress.log" >&2
        tail -30 "$log" | tee -a "$OUT/progress.log" >&2 || true
    fi
}

dump_gguf | tee -a "$OUT/progress.log"

# --- binary A/B: production sm75 DP4A vs Pascal FORCE_MMQ ---
BASE_ARGS="-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512"
run_bench binary sm75 "$SM75" baseline "$BASE_ARGS" ""
run_bench binary pascal "$PASCAL" baseline "$BASE_ARGS" ""

WIN_TAG="sm75"
WIN_BENCH="$SM75"
WIN_CLI="$SM75_CLI"
python3 - "$csv" "$OUT/winner-bin.txt" <<'PY'
import csv, sys
path, outp = sys.argv[1], sys.argv[2]
tg, pp = {}, {}
with open(path) as f:
    for r in csv.DictReader(f):
        if r.get('status') != 'ok' or r.get('phase') != 'binary':
            continue
        tag = r['bin']
        ts = float(r['avg_ts'])
        if r.get('n_gen') not in ('', '0'):
            tg[tag] = max(ts, tg.get(tag, 0.0))
        elif r.get('n_prompt') not in ('', '0'):
            pp[tag] = max(ts, pp.get(tag, 0.0))
if not tg:
    open(outp, 'w').write('sm75\n')
    raise SystemExit(0)
def e2e(tag):
    p = pp.get(tag, 1.0)
    d = tg[tag]
    return 3328.0 / (3200.0 / p + 128.0 / d)
winner = max(tg, key=e2e)
open(outp, 'w').write(winner + '\n')
print('binary tg128:', tg, 'pp512:', pp, 'winner', winner, 'e2e', {t: round(e2e(t), 1) for t in tg})
PY
WIN_TAG="$(tr -d '\n' < "$OUT/winner-bin.txt")"
if [[ "$WIN_TAG" == pascal ]]; then
    WIN_BENCH="$PASCAL"
    WIN_CLI="$PASCAL_CLI"
fi
if [[ ! -x "$WIN_CLI" ]]; then
    WIN_CLI="$SM75_CLI"
    echo "CLI fallback $WIN_CLI (no llama-cli on $WIN_TAG)" | tee -a "$OUT/progress.log"
fi
echo "WIN_BIN=$WIN_TAG" | tee -a "$OUT/progress.log"

# --- FA and KV on the winning binary ---
run_bench fa_kv "$WIN_TAG" "$WIN_BENCH" fa-on-q8 "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" ""
run_bench fa_kv "$WIN_TAG" "$WIN_BENCH" fa-off-q8 "-fa off -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" ""
run_bench fa_kv "$WIN_TAG" "$WIN_BENCH" fa-on-f16 "-fa on -ctk f16 -ctv f16 -p 512 -n 128 -b 2048 -ub 512" ""
run_bench fa_kv "$WIN_TAG" "$WIN_BENCH" fa-on-q4 "-fa on -ctk q4_0 -ctv q4_0 -p 512 -n 128 -b 2048 -ub 512" ""

# --- MMVQ Q6_K (Q6_K_L body is mostly Q6_K; ncols1 = decode) ---
for rows in 0 1 2 4; do
    run_bench q6n1 "$WIN_TAG" "$WIN_BENCH" "rows${rows}" \
        "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" \
        "GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=${rows}"
done
for rows in 0 2 4; do
    run_bench q6n3 "$WIN_TAG" "$WIN_BENCH" "rows${rows}" \
        "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" \
        "GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=${rows}"
done

# --- upgraded tensors in the _L layout (Q8_0 / Q4_K) ---
for rows in 0 4; do
    run_bench q8n1 "$WIN_TAG" "$WIN_BENCH" "rows${rows}" \
        "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" \
        "GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS=${rows}"
done
for rows in 0 2; do
    run_bench q4k "$WIN_TAG" "$WIN_BENCH" "rows${rows}" \
        "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" \
        "GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=${rows}"
done

# --- scheduler / copy knobs ---
run_bench sched "$WIN_TAG" "$WIN_BENCH" concat-on \
    "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" \
    "GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1 GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=128"
run_bench sched "$WIN_TAG" "$WIN_BENCH" async-h2d \
    "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" \
    "GGML_CUDA_ASYNC_HOST_COPY=1"
run_bench sched "$WIN_TAG" "$WIN_BENCH" dedup-sync \
    "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" \
    "GGML_SCHED_DEDUP_DST_SYNC=1"
run_bench sched "$WIN_TAG" "$WIN_BENCH" register-host \
    "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" \
    "GGML_CUDA_REGISTER_HOST=1"
run_bench sched "$WIN_TAG" "$WIN_BENCH" graphs-off \
    "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" \
    "GGML_CUDA_DISABLE_GRAPHS=1"

# --- prefill ubatch ---
for ub in 256 512 1024 2048; do
    run_bench prefill "$WIN_TAG" "$WIN_BENCH" "ub${ub}" \
        "-fa on -ctk q8_0 -ctv q8_0 -p 512,2048 -n 0 -b ${ub} -ub ${ub}" ""
done

# --- combined recipe from Ling/Spark winners, Q8 rows from start1660 ---
COMBO_ENV="GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS=4 GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS=0 GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=4 GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=4 GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1 GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=128 GGML_CUDA_ASYNC_HOST_COPY=1 GGML_SCHED_DEDUP_DST_SYNC=1 GGML_CUDA_REGISTER_HOST=1"
run_bench combo "$WIN_TAG" "$WIN_BENCH" spark-ling \
    "-fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 2048 -ub 512" \
    "$COMBO_ENV"
run_bench combo "$WIN_TAG" "$WIN_BENCH" spark-ling-pp \
    "-fa on -ctk q8_0 -ctv q8_0 -p 512,2048 -n 0 -b 2048 -ub 512" \
    "$COMBO_ENV"

# --- llama-cli extras on the winning binary ---
cli_csv="$OUT/cli.csv"
printf '%s\n' 'label,gen_tps,prompt_tps,status,notes' > "$cli_csv"

run_cli() {
    local label="$1"
    local extra_env="$2"
    shift 2
    local log="$OUT/cli-${label}.log"
    echo "CLI $label" | tee -a "$OUT/progress.log"
    set +e
    # shellcheck disable=SC2086
    timeout 180 env CUDA_VISIBLE_DEVICES=0 $extra_env \
        "$WIN_CLI" -m "$MODEL" -ngl 99 -t "$THREADS" --offline --no-display-prompt --single-turn \
        "$@" \
        >"$log" 2>&1
    local rc=$?
    set -e
    local gen prompt notes=""
    gen=$(grep -oE 'Generation: [0-9]+[.,][0-9]+ t/s' "$log" | tail -1 | grep -oE '[0-9]+[.,][0-9]+' || true)
    prompt=$(grep -oE 'Prompt: [0-9]+[.,][0-9]+ t/s' "$log" | tail -1 | grep -oE '[0-9]+[.,][0-9]+' || true)
    grep -qiE 'garbled|invalid|error:|out of memory' "$log" && notes="see-log"
    if [[ $rc -eq 0 && -n "$gen" ]]; then
        printf '%s,%s,%s,ok,%s\n' "$label" "${gen/,/.}" "${prompt/,/.}" "$notes" | tee -a "$cli_csv" | tee -a "$OUT/progress.log"
    else
        printf '%s,%s,%s,fail,rc=%s\n' "$label" "${gen/,/.}" "${prompt/,/.}" "$rc" | tee -a "$cli_csv" | tee -a "$OUT/progress.log"
        tail -25 "$log" | tee -a "$OUT/progress.log" || true
    fi
}

CARD_SAMP=(--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5)
CHAT_PROMPT='My friend just lost their job and seems really down. What should I say to them?'

run_cli cpu-samp "$COMBO_ENV" -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 2048 -n 96 --jinja --single-turn -cnv \
    "${CARD_SAMP[@]}" \
    -p "$CHAT_PROMPT"

run_cli backend-samp "$COMBO_ENV" -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 2048 -n 96 --jinja --single-turn -cnv --backend-sampling \
    "${CARD_SAMP[@]}" \
    -p "$CHAT_PROMPT"

run_cli ngram-none "$COMBO_ENV" -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 2048 -n 96 --spec-type none \
    -p "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. Continue:"

run_cli ngram-simple "$COMBO_ENV" -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 2048 -n 96 --spec-type ngram-simple --spec-draft-n-max 8 \
    --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8 \
    -p "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. Continue:"

run_cli mtp-n2 "$COMBO_ENV" -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 2048 -n 96 --spec-type draft-mtp --spec-draft-n-max 2 \
    -p "Explain in two sentences why the sky is blue."

run_cli mtp-n4 "$COMBO_ENV" -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 2048 -n 96 --spec-type draft-mtp --spec-draft-n-max 4 \
    -p "Explain in two sentences why the sky is blue."

run_cli turbo4 "LLAMA_TURBO4_V_EXPERIMENTAL=1 $COMBO_ENV" -fa on \
    --cache-type-k turbo4_k --cache-type-v turbo4_k \
    -c 2048 -n 48 --jinja --single-turn -cnv \
    -p "What is 2+2? Answer with one number."

run_cli kvflash "LLAMA_KVFLASH=8192 LLAMA_KVFLASH_MAX_POOL=8192 LLAMA_KVFLASH_POLICY=lru $COMBO_ENV" \
    -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -c 32768 -n 16 \
    -p "Say ok."

python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = list(csv.DictReader((out / 'runs.csv').open()))
ok = [r for r in rows if r.get('status') == 'ok' and r.get('avg_ts')]
md = [
    '# FrogNano-4B-2609 Q6_K_L GTX 1660 Ti tune',
    '',
    'qwen35 hybrid, 32 trunk layers, 1 NextN. MoE knobs are inert.',
    'Sweep: sm75 DP4A vs Pascal FORCE_MMQ, FA, KV, Q6_K/Q8/Q4_K MMVQ rows, scheduler, prefill ubatch, ngram, MTP, Turbo4, KVFlash, sampling.',
    '',
]
meta = out / 'gguf-meta.txt'
if meta.exists():
    md += ['## GGUF', '', '```', meta.read_text().strip(), '```', '']
win = (out / 'winner-bin.txt').read_text().strip() if (out / 'winner-bin.txt').exists() else '?'
md += [f'Winning binary for the rest of the sweep: **{win}**', '']
by_phase = {}
for r in ok:
    by_phase.setdefault(r['phase'], []).append(r)
for phase, items in by_phase.items():
    md.append(f'## {phase}')
    md.append('')
    md.append('| bin | label | test | t/s | std | fa | ctk/ctv | ub | env |')
    md.append('|---|---|---|---:|---:|---|---|---|---|')
    for r in items:
        if r['n_gen'] in ('0', ''):
            test = f"pp{r['n_prompt']}"
        elif r['n_prompt'] in ('0', ''):
            test = f"tg{r['n_gen']}"
        else:
            test = f"pp{r['n_prompt']}/tg{r['n_gen']}"
        md.append(
            f"| {r['bin']} | {r['label']} | {test} | {float(r['avg_ts']):.2f} | {float(r['stddev_ts'] or 0):.2f} | {r['fa']} | {r['ctk']}/{r['ctv']} | {r['ubatch']} | `{r['env']}` |"
        )
    md.append('')
tg = [r for r in ok if r.get('n_gen') not in ('0', '') and float(r['n_gen']) >= 64]
if tg:
    best = max(tg, key=lambda r: float(r['avg_ts']))
    md += [
        '## Best decode (tg>=64)',
        '',
        f"- **{best['phase']}/{best['bin']}/{best['label']}**: {float(best['avg_ts']):.2f} t/s env=`{best['env']}`",
        '',
    ]
pp = [r for r in ok if r.get('n_prompt') not in ('0', '') and float(r['n_prompt']) >= 512 and r.get('n_gen') in ('0', '')]
if not pp:
    pp = [r for r in ok if r.get('n_prompt') == '512']
if pp:
    bestp = max(pp, key=lambda r: float(r['avg_ts']))
    md += [
        '## Best prefill',
        '',
        f"- **{bestp['phase']}/{bestp['bin']}/{bestp['label']} pp{bestp['n_prompt']}**: {float(bestp['avg_ts']):.2f} t/s",
        '',
    ]
cli_path = out / 'cli.csv'
if cli_path.exists():
    md += ['## llama-cli extras', '', '| label | gen t/s | prompt t/s | status | notes |', '|---|---:|---:|---|---|']
    for r in csv.DictReader(cli_path.open()):
        md.append(f"| {r['label']} | {r.get('gen_tps','')} | {r.get('prompt_tps','')} | {r['status']} | {r.get('notes','')} |")
    md.append('')
(out / 'SUMMARY.md').write_text('\n'.join(md) + '\n')
print(f'wrote {out / "SUMMARY.md"} ({len(ok)} ok llama-bench rows)')
PY

echo DONE "$OUT" | tee -a "$OUT/progress.log"
