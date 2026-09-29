#!/usr/bin/env bash
# Nex-N2.5-mini i1-Q4_K_M hybrid sweep on GTX 1660 Ti.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/nex-n25-mini-i1-tune-${STAMP}}"
mkdir -p "$OUT"

BENCH="${BENCH:-$ROOT/build-main-sm75/bin/llama-bench}"
CLI="${CLI:-$ROOT/build-main-sm75/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/nex-n2.5-mini-i1/Nex-N2.5-mini.i1-Q4_K_M.gguf}"
TMPL="${TMPL:-$ROOT/models/templates/nex-N2.5-mini.jinja}"
PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"

[[ -x "$BENCH" ]] || { echo "missing $BENCH" >&2; exit 1; }
[[ -x "$CLI" ]] || { echo "missing $CLI" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL" >&2; exit 1; }

csv="$OUT/runs.csv"
printf '%s\n' 'kind,label,n_prompt,n_gen,avg_ts,stddev_ts,status,notes' > "$csv"
echo "results: $OUT"

python3 - "$MODEL" "$OUT/gguf-meta.txt" <<'PY'
import struct, sys
path, outp = sys.argv[1], sys.argv[2]
def read_str(f):
    n=struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8','replace')
def skip_value(f,t):
    sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t==8: read_str(f); return
    if t==9:
        et=struct.unpack('<I', f.read(4))[0]; n=struct.unpack('<Q', f.read(8))[0]
        if et==8:
            for _ in range(n): read_str(f)
        else:
            f.read(n*sizes[et])
        return
    f.read(sizes[t])
def read_value(f,t):
    if t==4: return struct.unpack('<I', f.read(4))[0]
    if t==5: return struct.unpack('<i', f.read(4))[0]
    if t==6: return struct.unpack('<f', f.read(4))[0]
    if t==7: return bool(f.read(1)[0])
    if t==8: return read_str(f)
    if t==10: return struct.unpack('<Q', f.read(8))[0]
    skip_value(f,t); return '<skip>'
want=('general.architecture','general.name','general.file_type',
      'qwen35moe.block_count','qwen35moe.expert_count','qwen35moe.expert_used_count',
      'qwen35moe.nextn_predict_layers','qwen35moe.context_length','qwen35moe.embedding_length',
      'tokenizer.ggml.pre','tokenizer.ggml.bos_token_id','tokenizer.ggml.eos_token_id')
with open(path,'rb') as f, open(outp,'w') as o:
    assert f.read(4)==b'GGUF'
    ver,n_tensors,n_kv=struct.unpack('<IQQ', f.read(20))
    o.write(f'version={ver} n_tensors={n_tensors} n_kv={n_kv}\n')
    for _ in range(n_kv):
        key=read_str(f); t=struct.unpack('<I', f.read(4))[0]
        if key in want or 'nextn' in key or 'expert' in key or key.endswith('file_type'):
            val=read_value(f,t)
            s=str(val)
            if len(s)>240: s=s[:240]+'...'
            o.write(f'{key}: {s}\n')
        else:
            skip_value(f,t)
print(open(outp).read())
PY

run_cli() {
    local label="$1"; shift
    local extra_env="$1"; shift
    local log="$OUT/cli-${label}.log"
    echo "CLI $label"
    set +e
    # shellcheck disable=SC2086
    timeout 240 env CUDA_VISIBLE_DEVICES=0 LLAMA_ARG_CPU_MOE=1 \
        LLAMA_EXPERT_HOT="$PROFILE" $extra_env \
        "$CLI" -m "$MODEL" -ngl 99 -fa on \
        --chat-template-file "$TMPL" --jinja \
        --chat-template-kwargs '{"reasoning_effort":"none"}' \
        --single-turn -cnv --offline --no-display-prompt \
        --temp 0.7 --top-p 0.95 --top-k 40 \
        -c 4096 -t 8 \
        "$@" \
        >"$log" 2>&1
    local rc=$?
    set -e
    local gen prompt notes=""
    gen=$(grep -oE 'Generation: [0-9]+[.,][0-9]+ t/s' "$log" | tail -1 | grep -oE '[0-9]+[.,][0-9]+' || true)
    prompt=$(grep -oE 'Prompt: [0-9]+[.,][0-9]+ t/s' "$log" | tail -1 | grep -oE '[0-9]+[.,][0-9]+' || true)
    grep -q '<think>' "$log" && notes="${notes}think;"
    if [[ $rc -eq 0 && -n "$gen" ]]; then
        printf 'cli,%s,0,%s,%s,,ok,%s\n' "$label" "${gen/,/.}" "${prompt/,/.}" "$notes" | tee -a "$csv"
    else
        printf 'cli,%s,0,,%s,,fail,rc=%s\n' "$label" "${gen:-}" "$rc" | tee -a "$csv"
        tail -30 "$log" || true
    fi
}

run_bench() {
    local label="$1" extra_args="$2" extra_env="$3"
    local jsonl="$OUT/bench-${label}.jsonl"
    local log="$OUT/bench-${label}.log"
    echo "BENCH $label"
    set +e
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 LLAMA_ARG_CPU_MOE=1 \
        LLAMA_EXPERT_HOT="$PROFILE" $extra_env \
        "$BENCH" -m "$MODEL" -ngl 99 -ncmoe 99 -t 8 -r 3 \
        -fa on \
        -o jsonl -oe md \
        $extra_args \
        >"$jsonl" 2>"$log"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        printf 'bench,%s,,,,,fail,rc=%s\n' "$label" "$rc" | tee -a "$csv"
        tail -25 "$log" || true
        return 0
    fi
    python3 - "$csv" "$label" "$jsonl" <<'PY'
import json, sys
csv, label, path = sys.argv[1:4]
for ln in open(path):
    if not ln.strip():
        continue
    r = json.loads(ln)
    open(csv,"a").write(
        f"bench,{label},{r.get('n_prompt',0)},{r.get('n_gen',0)},"
        f"{r.get('avg_ts','')},{r.get('stddev_ts','')},ok,\n"
    )
PY
}

PROD="GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2"
CONCAT="GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1 GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=128 GGML_CUDA_ASYNC_HOST_COPY=1 GGML_SCHED_DEDUP_DST_SYNC=1"

run_cli smoke "LLAMA_EXPERT_S=20 LLAMA_EXPERT_WARM_SLOTS=0" \
    --cache-type-k q8_0 --cache-type-v q8_0 -n 32 \
    -p "Reply with the single word ping."

run_cli thinkoff "LLAMA_EXPERT_S=20 LLAMA_EXPERT_WARM_SLOTS=0" \
    --cache-type-k q8_0 --cache-type-v q8_0 -n 48 \
    -p "Explain binary search in three short sentences. No preamble."

run_bench s16-q8 "-p 128 -n 64 --cache-type-k q8_0 --cache-type-v q8_0" \
    "LLAMA_EXPERT_S=16 LLAMA_EXPERT_WARM_SLOTS=0"
run_bench s20-q8 "-p 128 -n 64 --cache-type-k q8_0 --cache-type-v q8_0" \
    "LLAMA_EXPERT_S=20 LLAMA_EXPERT_WARM_SLOTS=0"
run_bench s24-q8 "-p 128 -n 64 --cache-type-k q8_0 --cache-type-v q8_0" \
    "LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0"
run_bench s24-turbo4 "-p 128 -n 64 --cache-type-k turbo4_k --cache-type-v turbo4_k" \
    "LLAMA_TURBO4_V_EXPERIMENTAL=1 LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0"
run_bench s24-q8-mmvq2 "-p 128 -n 64 --cache-type-k q8_0 --cache-type-v q8_0" \
    "LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 $PROD"
run_bench s24-combo "-p 64,128 -n 64 --cache-type-k turbo4_k --cache-type-v turbo4_k" \
    "LLAMA_TURBO4_V_EXPERIMENTAL=1 LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 $PROD $CONCAT"
run_bench s20-combo "-p 128 -n 64 --cache-type-k turbo4_k --cache-type-v turbo4_k" \
    "LLAMA_TURBO4_V_EXPERIMENTAL=1 LLAMA_EXPERT_S=20 LLAMA_EXPERT_WARM_SLOTS=0 $PROD $CONCAT"

run_cli ngram-none "LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 $PROD" \
    --cache-type-k q8_0 --cache-type-v q8_0 --spec-type none -n 48 \
    -p "What is 2+2? One number."
run_cli ngram-on "LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 $PROD" \
    --cache-type-k q8_0 --cache-type-v q8_0 --spec-type ngram-simple \
    --spec-draft-n-max 4 --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8 -n 48 \
    -p "What is 2+2? One number."

python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = list(csv.DictReader(open(out/"runs.csv")))
lines = ["# Nex-N2.5-mini i1-Q4_K_M tune", "", "| kind | label | pp | tg | tok/s | std | status | notes |",
         "|---|---|---:|---:|---:|---:|---|---|"]
for r in rows:
    lines.append(
        f"| {r['kind']} | {r['label']} | {r['n_prompt']} | {r['n_gen']} | "
        f"{r['avg_ts']} | {r['stddev_ts']} | {r['status']} | {r['notes']} |"
    )
meta = (out/"gguf-meta.txt").read_text() if (out/"gguf-meta.txt").exists() else ""
lines += ["", "## GGUF", "", "```", meta.strip(), "```", ""]
(out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
print("wrote", out/"SUMMARY.md")
PY

echo DONE "$OUT"
cat "$csv"
