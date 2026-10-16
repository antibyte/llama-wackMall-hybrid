#!/usr/bin/env bash
# MiniCPM-V-4.7-35B-A3B AD-Q4_K-IQ4_XS on the GTX 1660 Ti.
# Routed gate/up are IQ4_XS, routed down is Q4_K, attention and the shared expert are Q8_0.
# CPU fused gate-up and multi-row only accept Q4_K gate and Q5_K down, so those flags are measured, not assumed.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/minicpm-v47-35b-${STAMP}}"
mkdir -p "$OUT"

BENCH="${BENCH:-$ROOT/build-main-sm75/bin/llama-bench}"
CLI="${CLI:-$ROOT/build-main-sm75/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/minicpm-v-4.7-35b-a3b/MiniCPM-V-4.7-35B-A3B-AD-Q4_K-IQ4_XS.gguf}"

[[ -x "$BENCH" ]] || { echo "missing $BENCH" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL" >&2; exit 1; }

csv="$OUT/runs.csv"
printf '%s\n' 'kind,label,n_prompt,n_gen,avg_ts,stddev_ts,status,notes' > "$csv"
echo "results: $OUT"

python3 - "$MODEL" "$OUT/gguf-meta.txt" <<'PY'
import struct, sys
from collections import Counter
path, outp = sys.argv[1], sys.argv[2]
names={12:"q4_k",23:"iq4_xs",8:"q8_0",0:"f32",1:"f16",30:"bf16"}
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
    skip_value(f,t); return None
with open(path,'rb') as f, open(outp,'w') as o:
    assert f.read(4)==b'GGUF'
    ver,n_tensors,n_kv=struct.unpack('<IQQ', f.read(20))
    o.write(f'version={ver} n_tensors={n_tensors} n_kv={n_kv}\n')
    for _ in range(n_kv):
        key=read_str(f); t=struct.unpack('<I', f.read(4))[0]
        if 'chat_template' in key or 'tokenizer.ggml.tokens' in key or 'merges' in key:
            skip_value(f,t); continue
        if any(s in key for s in ('architecture','expert','block_count','embedding','context','file_type','feed_forward','head_count','ssm','nextn','name')):
            o.write(f'{key}: {read_value(f,t)}\n')
        else:
            skip_value(f,t)
    c=Counter()
    for _ in range(n_tensors):
        name=read_str(f)
        nd=struct.unpack('<I', f.read(4))[0]
        f.read(8*nd)
        typ=struct.unpack('<I', f.read(4))[0]
        f.read(8)
        kind='other'
        for prefix in ('ffn_down_exps','ffn_gate_exps','ffn_up_exps','ffn_down_shexp','ffn_gate_shexp','ffn_up_shexp','attn_','ssm_','token_embd','output.weight'):
            if prefix in name:
                kind=prefix
                break
        c[(kind, names.get(typ, str(typ)))] += 1
    o.write('tensors:\n')
    for (kind, tn), n in sorted(c.items()):
        o.write(f'  {n:4d} {kind} {tn}\n')
print(open(outp).read())
PY

# Plain CPU-MoE. Callers set S and ADAPT. Graphs stay off so warm and adapt runs match.
BASE="LLAMA_EXPERT_WARM_SLOTS=0 LLAMA_EXPERT_LOOKAHEAD=0 LLAMA_EXPERT_CPU_FUSED_GATE_UP=0 LLAMA_EXPERT_CPU_MULTI_ROW=0 LLAMA_EXPERT_CPU_REUSE_ROWS=0 LLAMA_EXPERT_CPU_ASYNC=0 LLAMA_EXPERT_CPU_ACT_PARALLEL=0 LLAMA_EXPERT_CPU_DOWN_PREFETCH=0 LLAMA_EXPERT_CPU_CHUNK=64 GGML_CUDA_DISABLE_GRAPHS=1"
OFF="LLAMA_EXPERT_ADAPT=0 LLAMA_EXPERT_S=0"

run_bench() {
    local label="$1" extra_env="$2"
    local jsonl="$OUT/bench-${label}.jsonl"
    local log="$OUT/bench-${label}.log"
    echo "BENCH $label"
    set +e
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 LLAMA_ARG_CPU_MOE=1 $BASE $extra_env \
        "$BENCH" -m "$MODEL" -ngl 99 -ncmoe 99 -t 8 -r 1 \
        -fa on -ctk q8_0 -ctv q8_0 -p 512 -n 64 -b 2048 -ub 512 \
        -o jsonl \
        >"$jsonl" 2>"$log"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "bench,$label,,,,,fail,rc=$rc" | tee -a "$csv"
        tail -30 "$log" || true
        return 0
    fi
    python3 - "$csv" "$label" "$jsonl" "$log" <<'PY'
import json, sys
csv, label, path, log = sys.argv[1:5]
text=open(log, encoding='utf-8', errors='replace').read()
notes=[]
for needle in (
    "clamping to", "auto-fit", "rows/block override", "fused",
    "warm", "lookahead", "CUDA graph", "error", "failed"):
    if needle.lower() in text.lower():
        notes.append(needle)
note=";".join(dict.fromkeys(notes))
for ln in open(path):
    if not ln.strip():
        continue
    r=json.loads(ln)
    open(csv,'a').write(
        f"bench,{label},{r.get('n_prompt',0)},{r.get('n_gen',0)},"
        f"{r.get('avg_ts','')},{r.get('stddev_ts','')},ok,{note}\n"
    )
    print(f"  pp={r.get('n_prompt')} tg={r.get('n_gen')} {r.get('avg_ts')} t/s")
PY
}

run_bench base "$OFF"
run_bench cpu-pack "$OFF LLAMA_EXPERT_CPU_FUSED_GATE_UP=1 LLAMA_EXPERT_CPU_MULTI_ROW=1 LLAMA_EXPERT_CPU_REUSE_ROWS=1 LLAMA_EXPERT_CPU_DOWN_PREFETCH=1"
run_bench async "$OFF LLAMA_EXPERT_CPU_ASYNC=1"
run_bench s16 "LLAMA_EXPERT_ADAPT=0 LLAMA_EXPERT_S=16"
run_bench s24 "LLAMA_EXPERT_ADAPT=0 LLAMA_EXPERT_S=24"
run_bench s32 "LLAMA_EXPERT_ADAPT=0 LLAMA_EXPERT_S=32"
run_bench autofit "LLAMA_EXPERT_ADAPT=1"
run_bench s16-mmvq "LLAMA_EXPERT_ADAPT=0 LLAMA_EXPERT_S=16 GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS=4 GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=4"
run_bench s16-warm "LLAMA_EXPERT_ADAPT=0 LLAMA_EXPERT_S=16 LLAMA_EXPERT_WARM_SLOTS=8 LLAMA_EXPERT_WARM_ADMISSION=frequency LLAMA_EXPERT_WARM_ADMISSION_WINDOW=200 LLAMA_EXPERT_WARM_REPLACE_RATIO=2.5 LLAMA_EXPERT_WARM_PREFETCH=1"
run_bench s16-lookahead "LLAMA_EXPERT_ADAPT=0 LLAMA_EXPERT_S=16 LLAMA_EXPERT_LOOKAHEAD=1 LLAMA_EXPERT_LOOKAHEAD_DISTANCE=1 LLAMA_EXPERT_LOOKAHEAD_TOP_M=12 LLAMA_EXPERT_LOOKAHEAD_POINT=post-moe"
run_bench s16-moe-rows4 "LLAMA_EXPERT_ADAPT=0 LLAMA_EXPERT_S=16 GGML_CUDA_MMVQ_MOE_PLAIN_ROWS=4 GGML_CUDA_MMVQ_MOE_FUSED_ROWS=4"

python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
out=Path(sys.argv[1])
rows=list(csv.DictReader(open(out/"runs.csv")))
lines=["# MiniCPM-V-4.7-35B-A3B AD-Q4_K-IQ4_XS", "",
       "| label | pp | tg | tok/s | std | status | notes |",
       "|---|---:|---:|---:|---:|---|---|"]
for r in rows:
    lines.append(
        f"| {r['label']} | {r['n_prompt']} | {r['n_gen']} | {r['avg_ts']} | {r['stddev_ts']} | {r['status']} | {r['notes']} |"
    )
meta=(out/"gguf-meta.txt").read_text() if (out/"gguf-meta.txt").exists() else ""
lines += ["", "## GGUF", "", "```", meta.strip(), "```", ""]
(out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
print("wrote", out/"SUMMARY.md")
PY
echo DONE "$OUT"
cat "$csv"
