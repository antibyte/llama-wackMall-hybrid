#!/usr/bin/env bash
# Nex-N2.5-mini Q4_K_M hybrid sweep on GTX 1660 Ti (qwen35moe, CPU MoE).
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/nex-n25-mini-tune-${STAMP}}"
mkdir -p "$OUT"

CLI="${CLI:-$ROOT/build-main-sm75/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/nex-n2.5-mini/Nex-N2.5-mini-Q4_K_M.gguf}"
TMPL="${TMPL:-$ROOT/models/templates/nex-N2.5-mini.jinja}"
PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"
NTOK="${NTOK:-48}"

[[ -x "$CLI" ]] || { echo "missing $CLI" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL" >&2; exit 1; }

csv="$OUT/runs.csv"
printf '%s\n' 'label,gen_tps,prompt_tps,status,notes' > "$csv"

dump_gguf() {
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
want=('general.architecture','general.name','qwen35moe.block_count','qwen35moe.expert_count',
      'qwen35moe.expert_used_count','qwen35moe.nextn_predict_layers','qwen35moe.context_length',
      'tokenizer.ggml.pre','tokenizer.chat_template')
with open(path,'rb') as f, open(outp,'w') as o:
    assert f.read(4)==b'GGUF'
    ver,n_tensors,n_kv=struct.unpack('<IQQ', f.read(20))
    o.write(f'version={ver} n_tensors={n_tensors} n_kv={n_kv}\n')
    for _ in range(n_kv):
        key=read_str(f); t=struct.unpack('<I', f.read(4))[0]
        if key in want or 'nextn' in key or key.endswith('expert_count'):
            val=read_value(f,t)
            s=str(val)
            if len(s)>200: s=s[:200]+'...'
            o.write(f'{key}: {s}\n')
        else:
            skip_value(f,t)
print(open(outp).read())
PY
}

run_cli() {
    local label="$1"; shift
    local log="$OUT/${label}.log"
    local extra_env="$1"; shift
    echo "=== $label ==="
    set +e
    # shellcheck disable=SC2086
    timeout 180 env CUDA_VISIBLE_DEVICES=0 LLAMA_ARG_CPU_MOE=1 $extra_env \
        "$CLI" -m "$MODEL" -ngl 99 -fa on \
        --chat-template-file "$TMPL" --jinja \
        --single-turn -cnv --offline --no-display-prompt \
        --temp 0.7 --top-p 0.95 --top-k 40 \
        -c 4096 -n "$NTOK" -t 8 \
        "$@" \
        >"$log" 2>&1
    local rc=$?
    set -e
    local gen prompt notes
    gen=$(grep -oE 'Generation: [0-9]+[.,][0-9]+ t/s' "$log" | tail -1 | grep -oE '[0-9]+[.,][0-9]+' || true)
    prompt=$(grep -oE 'Prompt: [0-9]+[.,][0-9]+ t/s' "$log" | tail -1 | grep -oE '[0-9]+[.,][0-9]+' || true)
    notes=""
    grep -q 'garbled\|invalid\|error:' "$log" && notes="see-log"
    if [[ $rc -eq 0 && -n "$gen" ]]; then
        printf '%s,%s,%s,ok,%s\n' "$label" "${gen/,/.}" "${prompt/,/.}" "$notes" | tee -a "$csv"
    else
        printf '%s,%s,%s,fail,rc=%s\n' "$label" "${gen:-}" "${prompt:-}" "$rc" | tee -a "$csv"
        tail -25 "$log" || true
    fi
}

dump_gguf

base_env="LLAMA_EXPERT_HOT=$PROFILE LLAMA_EXPERT_S=20 LLAMA_EXPERT_WARM_SLOTS=0"

run_cli smoke "$base_env" --cache-type-k q8_0 --cache-type-v q8_0 -p "Reply with the word ping."

run_cli kv-q8 "$base_env" --cache-type-k q8_0 --cache-type-v q8_0 -p "What is 2+2? One number."
run_cli kv-q4 "$base_env" --cache-type-k q4_0 --cache-type-v q4_0 -p "What is 2+2? One number."
run_cli kv-turbo4 "LLAMA_TURBO4_V_EXPERIMENTAL=1 $base_env" \
    --cache-type-k turbo4_k --cache-type-v turbo4_k -p "What is 2+2? One number."

run_cli s16 "LLAMA_EXPERT_HOT=$PROFILE LLAMA_EXPERT_S=16 LLAMA_EXPERT_WARM_SLOTS=0" \
    --cache-type-k q8_0 --cache-type-v q8_0 -p "What is 2+2? One number."
run_cli s20 "$base_env" --cache-type-k q8_0 --cache-type-v q8_0 -p "What is 2+2? One number."
run_cli s24 "LLAMA_EXPERT_HOT=$PROFILE LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0" \
    --cache-type-k q8_0 --cache-type-v q8_0 -p "What is 2+2? One number."

run_cli q4k-r0 "GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=0 $base_env" \
    --cache-type-k q8_0 --cache-type-v q8_0 -p "What is 2+2? One number."
run_cli q4k-r2 "GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 $base_env" \
    --cache-type-k q8_0 --cache-type-v q8_0 -p "What is 2+2? One number."

run_cli w8 "LLAMA_EXPERT_HOT=$PROFILE LLAMA_EXPERT_S=20 LLAMA_EXPERT_WARM_SLOTS=8 LLAMA_EXPERT_WARM_ADMISSION=frequency LLAMA_EXPERT_WARM_ADMISSION_WINDOW=200 LLAMA_EXPERT_WARM_REPLACE_RATIO=2.5 LLAMA_EXPERT_WARM_PREFETCH=1" \
    --cache-type-k q8_0 --cache-type-v q8_0 -p "What is 2+2? One number."

run_cli ngram-none "$base_env" --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type none -p "What is 2+2? One number."
run_cli ngram-on "$base_env" --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type ngram-simple --spec-draft-n-max 4 \
    --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8 \
    -p "What is 2+2? One number."

run_cli combo "GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2 GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1 GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=128 GGML_CUDA_ASYNC_HOST_COPY=1 GGML_SCHED_DEDUP_DST_SYNC=1 $base_env" \
    --cache-type-k q8_0 --cache-type-v q8_0 -p "What is 2+2? One number."

echo DONE "$OUT"
cat "$csv"
