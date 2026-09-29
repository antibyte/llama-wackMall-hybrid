#!/usr/bin/env bash
# A/B: ngram-mod vs DFlash vs none vs ngram-simple on Nex-i1 coding.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/nex-i1-ngram-mod-${STAMP}}"
mkdir -p "$OUT"

SRV="${SRV:-$ROOT/build-main-sm75/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/nex-n2.5-mini-i1/Nex-N2.5-mini.i1-Q4_K_M.gguf}"
TMPL="${TMPL:-$ROOT/models/templates/nex-N2.5-mini.jinja}"
PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"
DFLASH="${DFLASH:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf}"
PORT="${PORT:-18090}"

CODE1='Write a Python binary_search(arr, target) with a short docstring and a walk-through for searching 7 in [1, 3, 5, 7, 9]. No preamble.'
CODE2='Write a Python function merge_sort(arr) with a docstring and sort [5, 1, 4, 2, 8] step by step. No preamble.'

ENVBASE="CUDA_VISIBLE_DEVICES=0 LLAMA_ARG_CPU_MOE=1 LLAMA_EXPERT_HOT=$PROFILE LLAMA_EXPERT_S=24 LLAMA_EXPERT_WARM_SLOTS=0 GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2 GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=1 GGML_CUDA_ASYNC_HOST_COPY=1 GGML_SCHED_DEDUP_DST_SYNC=1 LLAMA_DFLASH_COMBINED=1"

COMMON=(
  -m "$MODEL" --host 127.0.0.1 --port "$PORT"
  --offline --no-ui --cpu-moe
  -ngl 99 -fa on -c 4096 -np 1 --fit on --fit-target 80
  --jinja --chat-template-file "$TMPL"
  --chat-template-kwargs '{"reasoning_effort":"none"}'
  --cache-type-k q8_0 --cache-type-v q8_0
  --temp 0.7 --top-p 0.95 --top-k 40
  --alias nex-n2.5-mini-i1
)

wait_up() {
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

stop_srv() {
  if [[ -f "$OUT/srv.pid" ]]; then
    kill "$(cat "$OUT/srv.pid")" 2>/dev/null || true
    sleep 2
    rm -f "$OUT/srv.pid"
  fi
}

ask() {
  python3 - "$1" "$2" "$3" "$OUT" <<'PY'
import json, sys, urllib.request
label, prompt, n, out = sys.argv[1:5]
body = {
    "model": "nex-n2.5-mini-i1",
    "max_tokens": int(n),
    "temperature": 0.7,
    "top_p": 0.95,
    "messages": [{"role": "user", "content": prompt}],
}
req = urllib.request.Request(
    "http://127.0.0.1:18090/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=180) as r:
    data = json.load(r)
open(f"{out}/api-{label}.json", "w").write(json.dumps(data, indent=2)[:12000])
t = data.get("timings") or {}
txt = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
dn = t.get("draft_n") or 0
da = t.get("draft_n_accepted") or 0
print(json.dumps({
    "label": label,
    "pred_n": t.get("predicted_n"),
    "tg": round(t.get("predicted_per_second") or 0, 2),
    "pp": round(t.get("prompt_per_second") or 0, 2),
    "draft_n": dn,
    "draft_n_accepted": da,
    "accept": None if not dn else round(da / dn, 4),
    "chars": len(txt),
    "think": txt.count("<think>"),
    "head": txt[:100].replace("\n", " | "),
}, ensure_ascii=False))
PY
}

start_one() {
  local name="$1"
  shift
  echo "SERVER $name" >&2
  env $ENVBASE "$SRV" "${COMMON[@]}" "$@" >"$OUT/srv-${name}.log" 2>&1 &
  echo $! >"$OUT/srv.pid"
  if ! wait_up; then
    echo "FAIL $name" >&2
    tail -25 "$OUT/srv-${name}.log" >&2 || true
    return 1
  fi
}

trap stop_srv EXIT
jsonl="$OUT/api-results.jsonl"
: >"$jsonl"

run_arm() {
  local name="$1"
  shift
  start_one "$name" "$@"
  ask "${name}-code1" "$CODE1" 128 | tee -a "$jsonl"
  ask "${name}-code2" "$CODE2" 128 | tee -a "$jsonl"
  stop_srv
}

{
  echo "=== none ==="
  run_arm none --spec-type none

  echo "=== dflash n4 p0 ==="
  run_arm dflash-n4-p0 \
    --spec-type draft-dflash --spec-draft-model "$DFLASH" \
    --spec-draft-ngl 99 --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.0 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 --spec-draft-backend-sampling

  echo "=== ngram-mod pxa n_min=2 n_max=4 ==="
  run_arm ngram-mod-pxa \
    --spec-type ngram-mod \
    --spec-ngram-mod-n-min 2 --spec-ngram-mod-n-max 4 --spec-ngram-mod-n-match 24 \
    --spec-draft-n-max 4

  echo "=== ngram-mod llama default n_min=48 n_max=64 ==="
  run_arm ngram-mod-def \
    --spec-type ngram-mod \
    --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 --spec-ngram-mod-n-match 24 \
    --spec-draft-n-max 64

  echo "=== ngram-simple n_max=4 ==="
  run_arm ngram-simple \
    --spec-type ngram-simple \
    --spec-draft-n-max 4 --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8
} 

python3 - "$OUT" <<'PY'
import json, csv
from pathlib import Path
out = Path(__import__("sys").argv[1])
rows = []
for ln in (out / "api-results.jsonl").read_text().splitlines():
    ln = ln.strip()
    if ln.startswith("{"):
        rows.append(json.loads(ln))
lines = [
    "# Nex-i1 coding spec A/B: none / DFlash / ngram-mod / ngram-simple",
    "",
    "Target i1-Q4_K_M, CPU MoE S=24, FA, q8 KV, MMVQ Q4_K=2, think off, temp 0.7.",
    "code1 = binary_search; code2 = merge_sort on the same server (ngram warmup).",
    "",
    "| label | pred_n | tg/s | pp/s | accept | acc/draft | chars | think |",
    "|---|---:|---:|---:|---:|---|---:|---:|",
]
with open(out / "runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label","pred_n","tg","pp","draft_n","draft_n_accepted","accept","chars","think"])
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in w.fieldnames})
        acc = r.get("accept")
        acc_s = "" if acc is None else f"{acc:.3f}"
        dn, da = r.get("draft_n") or 0, r.get("draft_n_accepted") or 0
        lines.append(
            f"| {r['label']} | {r.get('pred_n')} | {r.get('tg')} | {r.get('pp')} | "
            f"{acc_s} | {da}/{dn} | {r.get('chars')} | {r.get('think')} |"
        )
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("wrote", out / "SUMMARY.md")
PY
echo DONE "$OUT"
cat "$jsonl"
