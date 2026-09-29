#!/usr/bin/env bash
# Extra decode knobs toward tg>30. Prefill already >100 on Pascal MMQ.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/hy-mt2-30b-decode-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
PASCAL="$ROOT/build-mmq-pascal/bin/llama-bench"
CLI="$ROOT/build-main-sm75/bin/llama-cli"
MODEL="$HOME/models/hy-mt2-30b/Hy-MT2-30B-A3B.Q4_K_M.gguf"
csv="$OUT/runs.csv"
echo 'kind,label,n_prompt,n_gen,avg_ts,stddev_ts,status,notes' > "$csv"
echo "results: $OUT"
PROD='GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2'

bench() {
  local label="$1" extra_args="$2" extra_env="$3"
  echo "BENCH $label"
  local jsonl="$OUT/bench-${label}.jsonl" log="$OUT/bench-${label}.log"
  set +e
  # shellcheck disable=SC2086
  env CUDA_VISIBLE_DEVICES=0 LLAMA_ARG_CPU_MOE=1 $PROD $extra_env \
    "$PASCAL" -m "$MODEL" -ngl 99 -ncmoe 99 -fa on -r 3 -o jsonl -oe md \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    $extra_args >"$jsonl" 2>"$log"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "bench,$label,,,,,fail,rc=$rc" | tee -a "$csv"
    grep -E 'error|OOM|failed|out of memory' "$log" | tail -8 || tail -15 "$log"
    return 0
  fi
  python3 - "$csv" "$label" "$jsonl" <<'PY'
import json,sys
csv,label,path=sys.argv[1:4]
for ln in open(path):
    if not ln.strip(): continue
    r=json.loads(ln)
    open(csv,'a').write(f"bench,{label},{r.get('n_prompt',0)},{r.get('n_gen',0)},{r.get('avg_ts','')},{r.get('stddev_ts','')},ok,\n")
    print(f"  pp={r.get('n_prompt')} tg={r.get('n_gen')} ts={r.get('avg_ts'):.2f}")
PY
}

cli() {
  local label="$1" extra_env="$2"; shift 2
  echo "CLI $label"
  local log="$OUT/cli-${label}.log"
  set +e
  # shellcheck disable=SC2086
  timeout 240 env CUDA_VISIBLE_DEVICES=0 LLAMA_ARG_CPU_MOE=1 $PROD $extra_env \
    "$CLI" -m "$MODEL" -ngl 99 -fa on --jinja --offline --single-turn -cnv --no-display-prompt \
    --temp 0.7 --top-p 1.0 --top-k 0 -c 2048 -t 8 \
    -p 'Translate into German, without additional explanation: Hello world. How are you today?' \
    "$@" >"$log" 2>&1
  local rc=$?
  set -e
  python3 - "$csv" "$label" "$log" "$rc" <<'PY'
import re,sys
csv,label,log,rc=sys.argv[1:5]
t=open(log,encoding='utf-8',errors='replace').read()
def g(p):
    m=re.search(p,t)
    return m.group(1).replace(',','.') if m else ''
gen=g(r'Generation: ([0-9]+[.,][0-9]+) t/s')
acc=g(r'draft acceptance = ([0-9.]+)')
status='ok' if rc=='0' and gen else 'fail'
open(csv,'a').write(f'cli,{label},0,0,{gen},,{status},acc={acc} rc={rc}\n')
print(' ', status, 'gen', gen, 'acc', acc or '-')
if status!='ok':
    for ln in t.splitlines():
        if re.search(r'error|ASSERT|OOM|failed', ln, re.I):
            print('  ', ln[:180])
PY
}

# decode-focused + one pp512 confirm
bench s24-t8-base '-p 64,512 -n 128 -t 8' 'LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0'
bench s28-t8 '-p 64 -n 128 -t 8' 'LLAMA_EXPERT_S=28 LLAMA_EXPERT_WARM_SLOTS=0'
bench s32-t8 '-p 64 -n 128 -t 8' 'LLAMA_EXPERT_S=32 LLAMA_EXPERT_WARM_SLOTS=0'
bench s24-t12 '-p 64 -n 128 -t 12' 'LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0'
bench s24-t16 '-p 64 -n 128 -t 16' 'LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0'
bench s24-chunk32 '-p 64 -n 128 -t 8' 'LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 LLAMA_EXPERT_CPU_CHUNK=32'
bench s24-chunk128 '-p 64 -n 128 -t 8' 'LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 LLAMA_EXPERT_CPU_CHUNK=128'
bench s24-async '-p 64 -n 128 -t 8' 'LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 LLAMA_EXPERT_CPU_ASYNC=1'
bench s24-w8 '-p 64 -n 128 -t 8' 'LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=8'

cli ngram-mod-pxa 'LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0' \
  --spec-type ngram-mod --spec-ngram-mod-n-min 2 --spec-ngram-mod-n-max 4 --spec-ngram-mod-n-match 24 \
  --spec-draft-n-max 4 --cache-type-k q8_0 --cache-type-v q8_0 -n 64
cli ngram-simple 'LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0' \
  --spec-type ngram-simple --spec-draft-n-max 4 --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8 \
  --cache-type-k q8_0 --cache-type-v q8_0 -n 64

python3 - "$OUT" <<'PY'
import csv,sys
from pathlib import Path
out=Path(sys.argv[1])
rows=list(csv.DictReader(open(out/'runs.csv')))
lines=['# Hy-MT2 extra decode knobs','','| kind | label | pp | tg | tok/s | std | status | notes |','|---|---|---:|---:|---:|---:|---|---|']
for r in rows:
    lines.append(f"| {r['kind']} | {r['label']} | {r['n_prompt']} | {r['n_gen']} | {r['avg_ts']} | {r['stddev_ts']} | {r['status']} | {r['notes']} |")
(out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
print('wrote', out/'SUMMARY.md')
PY
echo DONE "$OUT"
cat "$csv"
