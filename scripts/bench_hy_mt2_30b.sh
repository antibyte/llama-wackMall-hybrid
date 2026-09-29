#!/usr/bin/env bash
# Hy-MT2-30B-A3B Q4_K_M hybrid sweep on GTX 1660 Ti.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/hy-mt2-30b-tune-${STAMP}}"
mkdir -p "$OUT"

BENCH="${BENCH:-$ROOT/build-main-sm75/bin/llama-bench}"
PASCAL="${PASCAL:-$ROOT/build-mmq-pascal/bin/llama-bench}"
CLI="${CLI:-$ROOT/build-main-sm75/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/hy-mt2-30b/Hy-MT2-30B-A3B.Q4_K_M.gguf}"

[[ -x "$BENCH" ]] || { echo "missing $BENCH" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL" >&2; exit 1; }

csv="$OUT/runs.csv"
printf '%s\n' 'kind,bin,label,n_prompt,n_gen,avg_ts,stddev_ts,status,notes' > "$csv"
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
            f.read(n*sizes.get(et,0))
        return
    f.read(sizes.get(t,0))
def read_value(f,t):
    if t==4: return struct.unpack('<I', f.read(4))[0]
    if t==5: return struct.unpack('<i', f.read(4))[0]
    if t==8:
        s=read_str(f)
        return s if len(s)<240 else s[:240]+'...'
    if t==10: return struct.unpack('<Q', f.read(8))[0]
    skip_value(f,t); return None
want=('general.architecture','general.name','general.file_type',
      'hy_v3.block_count','hy_v3.expert_count','hy_v3.expert_used_count',
      'hy_v3.nextn_predict_layers','hy_v3.context_length','hy_v3.embedding_length',
      'tokenizer.ggml.pre','tokenizer.ggml.bos_token_id','tokenizer.ggml.eos_token_id',
      'tokenizer.chat_template')
with open(path,'rb') as f, open(outp,'w') as o:
    assert f.read(4)==b'GGUF'
    ver,n_tensors,n_kv=struct.unpack('<IQQ', f.read(20))
    o.write(f'version={ver} n_tensors={n_tensors} n_kv={n_kv}\n')
    for _ in range(n_kv):
        key=read_str(f); t=struct.unpack('<I', f.read(4))[0]
        if key in want or 'nextn' in key or key.endswith('file_type') or 'expert' in key:
            val=read_value(f,t)
            o.write(f'{key}: {val}\n')
        else:
            skip_value(f,t)
print(open(outp).read())
PY

PROMPT='Translate the following segment into German, without additional explanation: The quick brown fox jumps over the lazy dog.'

run_cli() {
    local label="$1"; shift
    local extra_env="$1"; shift
    local log="$OUT/cli-${label}.log"
    echo "CLI $label"
    set +e
    # shellcheck disable=SC2086
    timeout 240 env CUDA_VISIBLE_DEVICES=0 LLAMA_ARG_CPU_MOE=1 $extra_env \
        "$CLI" -m "$MODEL" -ngl 99 -fa on --jinja --offline \
        --single-turn -cnv --no-display-prompt \
        --temp 0.7 --top-p 1.0 --top-k 0 --repeat-penalty 1.0 \
        -c 4096 -t 8 \
        -p "$PROMPT" \
        "$@" \
        >"$log" 2>&1
    local rc=$?
    set -e
    python3 - "$csv" "$label" "$log" "$rc" <<'PY'
import re, sys
csv, label, log, rc = sys.argv[1:5]
t=open(log,encoding='utf-8',errors='replace').read()
def g(p):
    m=re.search(p,t)
    return m.group(1).replace(',','.') if m else ''
gen=g(r'Generation: ([0-9]+[.,][0-9]+) t/s')
pp=g(r'Prompt: ([0-9]+[.,][0-9]+) t/s')
notes=[]
if '<think>' in t: notes.append('think')
if int(rc)!=0: notes.append('rc='+rc)
status='ok' if rc=='0' and gen else 'fail'
open(csv,'a').write(f'cli,sm75,{label},0,0,{gen},{pp},{status},{";".join(notes)}\n')
print(' ', status, 'gen', gen, 'pp', pp, 'notes', notes)
i=t.rfind('> ')
print((t[i:i+400] if i>=0 else t[-300:]).replace('\n',' | ')[:400])
if status!='ok':
    for ln in t.splitlines():
        if re.search(r'error|ASSERT|failed|unknown architecture', ln, re.I):
            print('  ', ln[:200])
PY
}

run_bench() {
    local bin_tag="$1" bin="$2" label="$3" extra_args="$4" extra_env="$5"
    local jsonl="$OUT/bench-${bin_tag}-${label}.jsonl"
    local log="$OUT/bench-${bin_tag}-${label}.log"
    echo "BENCH $bin_tag $label"
    set +e
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 LLAMA_ARG_CPU_MOE=1 $extra_env \
        "$bin" -m "$MODEL" -ngl 99 -ncmoe 99 -t 8 -r 3 \
        -fa on -o jsonl -oe md \
        $extra_args \
        >"$jsonl" 2>"$log"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "bench,$bin_tag,$label,,,,,fail,rc=$rc" | tee -a "$csv"
        tail -20 "$log" || true
        return 0
    fi
    python3 - "$csv" "$bin_tag" "$label" "$jsonl" <<'PY'
import json, sys
csv, btag, label, path = sys.argv[1:5]
for ln in open(path):
    if not ln.strip():
        continue
    r=json.loads(ln)
    open(csv,'a').write(
        f"bench,{btag},{label},{r.get('n_prompt',0)},{r.get('n_gen',0)},"
        f"{r.get('avg_ts','')},{r.get('stddev_ts','')},ok,\n"
    )
PY
}

PROD="GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2"

run_cli smoke "LLAMA_EXPERT_S=20 LLAMA_EXPERT_WARM_SLOTS=0" \
    --cache-type-k q8_0 --cache-type-v q8_0 -n 64

run_bench sm75 "$BENCH" s16-q8 "-p 128,512 -n 64 --cache-type-k q8_0 --cache-type-v q8_0" \
    "LLAMA_EXPERT_S=16 LLAMA_EXPERT_WARM_SLOTS=0"
run_bench sm75 "$BENCH" s20-q8 "-p 128,512 -n 64 --cache-type-k q8_0 --cache-type-v q8_0" \
    "LLAMA_EXPERT_S=20 LLAMA_EXPERT_WARM_SLOTS=0"
run_bench sm75 "$BENCH" s24-q8 "-p 128,512 -n 64 --cache-type-k q8_0 --cache-type-v q8_0" \
    "LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0"
run_bench sm75 "$BENCH" s24-q8-mmvq2 "-p 128,512 -n 64 --cache-type-k q8_0 --cache-type-v q8_0" \
    "LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 $PROD"
run_bench sm75 "$BENCH" s24-turbo4-mmvq2 "-p 128,512 -n 64 --cache-type-k turbo4_k --cache-type-v turbo4_k" \
    "LLAMA_TURBO4_V_EXPERIMENTAL=1 LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 $PROD"

if [[ -x "$PASCAL" ]]; then
    run_bench pascal "$PASCAL" s24-q8-mmvq2 "-p 128,512 -n 64 --cache-type-k q8_0 --cache-type-v q8_0" \
        "LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 $PROD"
fi

run_cli s24-q8-n64 "LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 $PROD" \
    --cache-type-k q8_0 --cache-type-v q8_0 -n 64
run_cli s24-turbo4-n64 "LLAMA_TURBO4_V_EXPERIMENTAL=1 LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 $PROD" \
    --cache-type-k turbo4_k --cache-type-v turbo4_k -n 64

python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
out=Path(sys.argv[1])
rows=list(csv.DictReader(open(out/"runs.csv")))
lines=["# Hy-MT2-30B-A3B Q4_K_M tune", "",
       "| kind | bin | label | pp | tg | tok/s | std | status | notes |",
       "|---|---|---|---:|---:|---:|---:|---|---|"]
for r in rows:
    lines.append(
        f"| {r['kind']} | {r['bin']} | {r['label']} | {r['n_prompt']} | {r['n_gen']} | "
        f"{r['avg_ts']} | {r['stddev_ts']} | {r['status']} | {r['notes']} |"
    )
meta=(out/"gguf-meta.txt").read_text() if (out/"gguf-meta.txt").exists() else ""
lines += ["", "## GGUF", "", "```", meta.strip(), "```", ""]
(out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
print("wrote", out/"SUMMARY.md")
PY
echo DONE "$OUT"
cat "$csv"
