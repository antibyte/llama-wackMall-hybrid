#!/usr/bin/env bash
# Cross-model test: Qwen3.6-35B-A3B DFlash/JetSpec as drafter for Nex-N2.5-mini i1.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/nex-i1-qwen-dflash-${STAMP}}"
mkdir -p "$OUT"

CLI="${CLI:-$ROOT/build-main-sm75/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/nex-n2.5-mini-i1/Nex-N2.5-mini.i1-Q4_K_M.gguf}"
TMPL="${TMPL:-$ROOT/models/templates/nex-N2.5-mini.jinja}"
PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"
DFLASH="${DFLASH:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf}"
JETSPEC="${JETSPEC:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-JetSpec-Q4_K_M.gguf}"

PROMPT_CODE='Write a Python binary_search(arr, target) with a short docstring and a walk-through for searching 7 in [1, 3, 5, 7, 9]. No preamble.'
PROMPT_PROSE='Explain how a hash table handles collisions, with a small Python example. Three short paragraphs. No preamble.'

[[ -x "$CLI" ]] || { echo "missing $CLI" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL" >&2; exit 1; }
[[ -f "$DFLASH" ]] || { echo "missing $DFLASH" >&2; exit 1; }

csv="$OUT/runs.csv"
printf '%s\n' 'label,n,prompt_kind,gen_tps,prompt_tps,draft_accept,draft_acc_n,draft_tot,mean_len,think,status,notes' > "$csv"
echo "results: $OUT"

BASE_ENV="CUDA_VISIBLE_DEVICES=0 LLAMA_ARG_CPU_MOE=1 LLAMA_EXPERT_HOT=$PROFILE LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2 GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1 GGML_CUDA_ASYNC_HOST_COPY=1 GGML_SCHED_DEDUP_DST_SYNC=1"

run_one() {
    local label="$1" extra_env="$2" ntok="$3" pkind="$4" prompt="$5"
    shift 5
    local log="$OUT/${label}-${pkind}-n${ntok}.log"
    echo "=== $label $pkind n=$ntok ==="
    set +e
    # shellcheck disable=SC2086
    timeout 300 env $BASE_ENV $extra_env \
        "$CLI" -m "$MODEL" -ngl 99 -fa on \
        --chat-template-file "$TMPL" --jinja \
        --chat-template-kwargs '{"reasoning_effort":"none"}' \
        --single-turn -cnv --offline --no-display-prompt \
        --temp 0.7 --top-p 0.95 --top-k 40 \
        -c 4096 -t 8 -n "$ntok" \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        -p "$prompt" \
        "$@" \
        >"$log" 2>&1
    local rc=$?
    set -e

    python3 - "$csv" "$label" "$ntok" "$pkind" "$log" "$rc" <<'PY'
import re, sys
csv, label, ntok, pkind, log, rc = sys.argv[1:7]
text = open(log, encoding="utf-8", errors="replace").read()
def g(pat):
    m = re.search(pat, text)
    return m.group(1) if m else ""
gen = g(r"Generation: ([0-9]+[.,][0-9]+) t/s").replace(",", ".")
prompt_tps = g(r"Prompt: ([0-9]+[.,][0-9]+) t/s").replace(",", ".")
# also slot eval line
if not gen:
    m = re.search(r"eval time =.*?([0-9]+[.,][0-9]+) tokens per second", text)
    gen = m.group(1).replace(",", ".") if m else ""
acc = g(r"draft acceptance = ([0-9.]+)")
acc_n = g(r"draft acceptance = [0-9.]+ \(\s*([0-9]+) accepted")
tot = g(r"accepted / \s*([0-9]+) generated")
mean = g(r"mean len = \s*([0-9.]+)")
think = str(text.count("<think>"))
notes = []
if "error:" in text.lower() or "GGML_ASSERT" in text or "runtime_error" in text:
    notes.append("error")
if "out of memory" in text.lower() or "failed to allocate" in text.lower():
    notes.append("oom")
if int(rc) == 124:
    notes.append("timeout")
status = "ok" if rc == "0" and gen else "fail"
open(csv, "a").write(
    f"{label},{ntok},{pkind},{gen},{prompt_tps},{acc},{acc_n},{tot},{mean},{think},{status},{';'.join(notes)} rc={rc}\n"
)
print(f"  gen={gen or '-'} accept={acc or '-'} mean_len={mean or '-'} think={think} status={status} rc={rc}")
if status != "ok":
    # last error-ish lines
    lines = [ln for ln in text.splitlines() if re.search(r"error|Error|ASSERT|abort|except|OOM|failed", ln)]
    for ln in lines[-12:]:
        print("   ", ln[:200])
PY
}

# --- load/smoke: DFlash must construct ---
run_one load-dflash "LLAMA_DFLASH_COMBINED=1" 32 code "$PROMPT_CODE" \
    --spec-type draft-dflash --spec-draft-model "$DFLASH" \
    --spec-draft-ngl 99 --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0

# If load failed, still try JetSpec and baseline for the table.
run_one baseline "" 128 code "$PROMPT_CODE"
run_one baseline "" 128 prose "$PROMPT_PROSE"

run_one dflash-n4-p75 "LLAMA_DFLASH_COMBINED=1" 128 code "$PROMPT_CODE" \
    --spec-type draft-dflash --spec-draft-model "$DFLASH" \
    --spec-draft-ngl 99 --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0
run_one dflash-n4-p75 "LLAMA_DFLASH_COMBINED=1" 128 prose "$PROMPT_PROSE" \
    --spec-type draft-dflash --spec-draft-model "$DFLASH" \
    --spec-draft-ngl 99 --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0

run_one dflash-n4-p0 "LLAMA_DFLASH_COMBINED=1" 128 code "$PROMPT_CODE" \
    --spec-type draft-dflash --spec-draft-model "$DFLASH" \
    --spec-draft-ngl 99 --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.0 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0

run_one dflash-n2-p75 "LLAMA_DFLASH_COMBINED=1" 128 code "$PROMPT_CODE" \
    --spec-type draft-dflash --spec-draft-model "$DFLASH" \
    --spec-draft-ngl 99 --spec-draft-n-max 2 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0

run_one dflash-n8-p75 "LLAMA_DFLASH_COMBINED=1" 128 code "$PROMPT_CODE" \
    --spec-type draft-dflash --spec-draft-model "$DFLASH" \
    --spec-draft-ngl 99 --spec-draft-n-max 8 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0

if [[ -f "$JETSPEC" ]]; then
    run_one jetspec-n4-p75 "LLAMA_DFLASH_COMBINED=1" 128 code "$PROMPT_CODE" \
        --spec-type draft-dflash --spec-draft-model "$JETSPEC" \
        --spec-draft-ngl 99 --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
        --spec-draft-type-k q8_0 --spec-draft-type-v q8_0
    run_one jetspec-n4-p0 "LLAMA_DFLASH_COMBINED=1" 128 code "$PROMPT_CODE" \
        --spec-type draft-dflash --spec-draft-model "$JETSPEC" \
        --spec-draft-ngl 99 --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.0 \
        --spec-draft-type-k q8_0 --spec-draft-type-v q8_0
fi

# turbo4 + dflash (production KV) only if q8 dflash loaded
run_one dflash-n4-p75-turbo4 "LLAMA_DFLASH_COMBINED=1 LLAMA_TURBO4_V_EXPERIMENTAL=1 LLAMA_TURBO4_DFLASH_EXPERIMENTAL=1 LLAMA_TURBO4_DRAFT_EXPERIMENTAL=1" 128 code "$PROMPT_CODE" \
    --spec-type draft-dflash --spec-draft-model "$DFLASH" \
    --spec-draft-ngl 99 --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
    --cache-type-k turbo4_k --cache-type-v turbo4_k \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0

# longer decode for the production-like DFlash recipe vs baseline
run_one baseline "" 192 code "$PROMPT_CODE"
run_one dflash-n4-p75 "LLAMA_DFLASH_COMBINED=1" 192 code "$PROMPT_CODE" \
    --spec-type draft-dflash --spec-draft-model "$DFLASH" \
    --spec-draft-ngl 99 --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0

python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = list(csv.DictReader(open(out/"runs.csv")))
lines = [
    "# Nex-i1 + Qwen3.6 DFlash/JetSpec drafter",
    "",
    "Target: Nex-N2.5-mini i1-Q4_K_M. Draft: Qwen3.6-35B-A3B DFlash Q4_K_M (and JetSpec).",
    "Same hybrid recipe: CPU MoE S=24, FA, q8 KV unless noted, MMVQ Q4_K=2.",
    "",
    "| label | n | prompt | tg/s | pp/s | accept | acc/tot | mean_len | think | status |",
    "|---|---:|---|---:|---:|---:|---|---:|---:|---|",
]
for r in rows:
    lines.append(
        f"| {r['label']} | {r['n']} | {r['prompt_kind']} | {r['gen_tps']} | {r['prompt_tps']} | "
        f"{r['draft_accept']} | {r['draft_acc_n']}/{r['draft_tot']} | {r['mean_len']} | "
        f"{r['think']} | {r['status']} |"
    )
(out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
print("wrote", out/"SUMMARY.md")
PY
echo DONE "$OUT"
cat "$csv"
