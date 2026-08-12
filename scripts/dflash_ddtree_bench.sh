#!/usr/bin/env bash
# Overnight DDTree A/B for llama-wackMall-hybrid (GTX 1660 Ti).
#
# Compares:
#   1) baseline chain  (p_min=0.75, classic sampler)
#   2) full-chain      (p_min=0, classic sampler, full n_max block)
#   3) ddtree full-chain (LLAMA_DFLASH_DDTREE=1, top-K+tree, emit top-1 chain)
#
# Note: this tree path still uses *chain verify* (server single-seq). True
# tree-attention verify (Lucebox AL~8) is not in llama.cpp yet; the bench
# measures (a) tree-build cost, (b) whether full-block draft helps AL/TPS.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-$ROOT/benchmark-results/dflash-ddtree-$STAMP}"
NMAX_LIST="${NMAX_LIST:-3 4 6 8 12 15}"
N_PREDICT="${N_PREDICT:-256}"
REPEATS="${REPEATS:-1}"
WARMUP_TOKENS="${WARMUP_TOKENS:-32}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-5}"
PORT_BASE="${PORT_BASE:-18210}"

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
SUMMARY_MD="$OUT_ROOT/SUMMARY.md"
RESULTS_MD="$OUT_ROOT/RESULTS.md"

echo "out=$OUT_ROOT n_max=[$NMAX_LIST] n_predict=$N_PREDICT" | tee "$OUT_ROOT/run.log"

export SPEC_MODE=dflash
export DFLASH_COMBINED=1
export SPEC_DRAFT_N_MIN=0
export SPEC_DRAFT_BACKEND_SAMPLING=1
export SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf}"
export MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
export PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"
export CONTEXT="${CONTEXT:-8192}"
export CMOE_BATCH="${CMOE_BATCH:-64}"
export CMOE_UBATCH="${CMOE_UBATCH:-64}"
export CMOE_PREFILL_BATCH="${CMOE_PREFILL_BATCH:-768}"
export CMOE_PREFILL_UBATCH="${CMOE_PREFILL_UBATCH:-768}"
export TARGET_TYPE_K="${TARGET_TYPE_K:-q4_0}"
export TARGET_TYPE_V="${TARGET_TYPE_V:-q4_0}"
export DRAFT_TYPE_K="${DRAFT_TYPE_K:-q4_0}"
export DRAFT_TYPE_V="${DRAFT_TYPE_V:-q4_0}"
export STATIC_FIXED_S="${STATIC_FIXED_S:-33}"
export SKIP_SENTINEL="${SKIP_SENTINEL:-1}"
export MOE_MULTI_FUSION="${MOE_MULTI_FUSION:-1}"
export CUDA_ASYNC_HOST_COPY="${CUDA_ASYNC_HOST_COPY:-1}"
export CONCAT_NONCONT_FLAT_DIM0="${CONCAT_NONCONT_FLAT_DIM0:-1}"
export TARGET_BACKEND_SAMPLING="${TARGET_BACKEND_SAMPLING:-1}"
export REASONING_BUDGET="${REASONING_BUDGET:-1024}"
export LOAD_MODE="${LOAD_MODE:-mmap}"
export N_PREDICT REPEATS WARMUP_TOKENS COOLDOWN_SECONDS
export CASES=SC
export SERVER="${SERVER:-$ROOT/build-main-sm75/bin/llama-server}"

# mode configs: name|p_min|ddtree|budget|k|full_chain
MODES=(
  "chain_pmin|0.75|0|22|8|0"
  "full_chain|0.0|0|22|8|0"
  "ddtree_full|0.0|1|22|8|1"
  "ddtree_b16|0.0|1|16|8|1"
  "ddtree_b12|0.0|1|12|8|1"
)

printf '%s\n' \
  'mode,n_max,decode_tps,spec_acceptance,mean_accepted_length,draft_ms,inject_ms,verify_ms,draft_decode_ms,draft_sample_ms,n_draft_calls,n_inject_calls,n_verify_calls,draft_ms_per_call,inject_ms_per_call,verify_ms_per_call,step_ms_est,tokens_per_step_est,n_process_combined,n_process_standalone,predicted_n,ddtree_builds,results_dir' \
  > "$SUMMARY_CSV"

port_i=0
for mode_spec in "${MODES[@]}"; do
  IFS='|' read -r mode_name p_min ddtree budget k full_chain <<< "$mode_spec"
  for n_max in $NMAX_LIST; do
    port_i=$((port_i + 1))
    port=$((PORT_BASE + port_i))
    res="$OUT_ROOT/${mode_name}_n${n_max}"
    echo "=== mode=$mode_name n_max=$n_max p_min=$p_min ddtree=$ddtree budget=$budget port=$port ===" | tee -a "$OUT_ROOT/run.log"

    export SPEC_DRAFT_N_MAX="$n_max"
    export SPEC_DRAFT_P_MIN="$p_min"
    export LLAMA_DFLASH_DDTREE="$ddtree"
    export LLAMA_DFLASH_DDTREE_BUDGET="$budget"
    export LLAMA_DFLASH_DDTREE_K="$k"
    export LLAMA_DFLASH_DDTREE_FULL_CHAIN="$full_chain"
    export LLAMA_DFLASH_DDTREE_CHAIN_SEED=1
    export LLAMA_DFLASH_DDTREE_TEMP=1.0

    RESULTS_DIR="$res" \
    PORT="$port" \
    bash "$ROOT/scripts/bench_hybrid.sh" SC "$REPEATS" "$N_PREDICT" "$WARMUP_TOKENS" "$res" \
      2>&1 | tee -a "$OUT_ROOT/run.log" || {
        echo "bench failed for mode=$mode_name n_max=$n_max" | tee -a "$OUT_ROOT/run.log"
        continue
      }

    python3 - "$mode_name" "$n_max" "$res" "$SUMMARY_CSV" <<'PY'
import json, re, sys
from pathlib import Path

mode = sys.argv[1]
n_max = int(sys.argv[2])
res = Path(sys.argv[3])
summary = Path(sys.argv[4])

cands = sorted(res.glob("*-run*.response.json"))
if not cands:
    print(f"no response for {mode} n_max={n_max}", file=sys.stderr)
    raise SystemExit(0)
resp = json.loads(cands[0].read_text())
t = resp.get("timings") or {}
log_text = ""
logs = sorted(res.glob("*-run*.log"))
if logs:
    log_text = logs[-1].read_text(errors="replace")

def last(pat, default=""):
    m = re.findall(pat, log_text)
    return m[-1] if m else default

draft_n = float(t.get("draft_n") or 0)
draft_acc = float(t.get("draft_n_accepted") or 0)
acc = draft_acc / draft_n if draft_n else 0.0
mean_len = last(r"mean len\s*=\s*([0-9.]+)", "0")
try:
    mean_len_f = float(mean_len)
except ValueError:
    mean_len_f = 0.0

draft_ms = float(t.get("draft_ms") or 0)
inject_ms = float(t.get("inject_ms") or 0)
verify_ms = float(t.get("verify_ms") or 0)
dec_ms = float(t.get("draft_decode_ms") or 0)
smp_ms = float(t.get("draft_sample_ms") or 0)
n_d = int(t.get("n_draft_calls") or 0)
n_i = int(t.get("n_inject_calls") or 0)
n_v = int(t.get("n_verify_calls") or 0)
d_pc = draft_ms / n_d if n_d else 0.0
i_pc = inject_ms / n_i if n_i else 0.0
v_pc = verify_ms / n_v if n_v else 0.0
n_step = n_d if n_d else max(n_v, 1)
step_ms = (draft_ms + verify_ms + inject_ms) / n_step if n_step else 0.0
tok_step = mean_len_f if mean_len_f > 0 else (draft_acc / n_step if n_step and draft_acc else 0.0)
tps = float(t.get("predicted_per_second") or 0)
pred_n = t.get("predicted_n") or resp.get("tokens_predicted") or 0
ddtree_builds = last(r"dflash ddtree: builds=(\d+)", "0")

row = [
    mode, n_max, f"{tps:.4f}", f"{acc:.5f}", f"{mean_len_f:.3f}",
    f"{draft_ms:.2f}", f"{inject_ms:.2f}", f"{verify_ms:.2f}",
    f"{dec_ms:.2f}", f"{smp_ms:.2f}",
    n_d, n_i, n_v,
    f"{d_pc:.3f}", f"{i_pc:.3f}", f"{v_pc:.3f}",
    f"{step_ms:.3f}", f"{tok_step:.3f}",
    int(t.get("n_process_combined") or 0),
    int(t.get("n_process_standalone") or 0),
    pred_n, ddtree_builds, str(res),
]
with summary.open("a") as f:
    f.write(",".join(map(str, row)) + "\n")
print("summary:", ",".join(map(str, row)))
PY

    sleep "$COOLDOWN_SECONDS" || true
  done
done

python3 - "$SUMMARY_CSV" "$SUMMARY_MD" "$RESULTS_MD" <<'PY'
import csv
from pathlib import Path
import sys
from collections import defaultdict

csv_path, md_path, results_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = list(csv.DictReader(csv_path.open()))

lines = [
    "# DFlash DDTree overnight bench",
    "",
    "Chain verify only (llama.cpp has no tree-attention verify yet).",
    "DDTree path = top-K extract + best-first tree build + emit top-1 chain.",
    "",
    "| mode | n_max | tok/s | acc | mean_len | draft ms/call | verify ms/call | step ms | tokens/step | ddtree builds |",
    "|------|------:|------:|----:|---------:|--------------:|---------------:|--------:|------------:|--------------:|",
]
by_mode = defaultdict(list)
for r in rows:
    by_mode[r["mode"]].append(r)
    lines.append(
        f"| {r['mode']} | {r['n_max']} | {float(r['decode_tps']):.2f} | {float(r['spec_acceptance']):.3f} | "
        f"{float(r['mean_accepted_length']):.2f} | {float(r['draft_ms_per_call']):.2f} | "
        f"{float(r['verify_ms_per_call']):.2f} | {float(r['step_ms_est']):.2f} | "
        f"{float(r['tokens_per_step_est']):.2f} | {r.get('ddtree_builds','')} |"
    )

# Pick best per mode
best_lines = ["", "## Best tok/s per mode", "", "| mode | best n_max | tok/s | mean_len | acc |",
              "|------|----------:|------:|---------:|----:|"]
for mode, rs in by_mode.items():
    best = max(rs, key=lambda x: float(x["decode_tps"]))
    best_lines.append(
        f"| {mode} | {best['n_max']} | {float(best['decode_tps']):.2f} | "
        f"{float(best['mean_accepted_length']):.2f} | {float(best['spec_acceptance']):.3f} |"
    )

# Baseline comparison
base = None
for r in rows:
    if r["mode"] == "chain_pmin" and r["n_max"] == "3":
        base = r
        break
if base is None and rows:
    base = rows[0]

cmp = ["", "## vs baseline (chain_pmin n_max=3 if present)", ""]
if base:
    bt = float(base["decode_tps"])
    cmp.append(f"Baseline: **{base['mode']} n_max={base['n_max']}** @ **{bt:.2f} tok/s**, mean_len={float(base['mean_accepted_length']):.2f}")
    cmp.append("")
    cmp.append("| mode | n_max | Δ tok/s | Δ% | mean_len |")
    cmp.append("|------|------:|--------:|---:|---------:|")
    for r in rows:
        t = float(r["decode_tps"])
        d = t - bt
        pct = 100.0 * d / bt if bt else 0.0
        cmp.append(f"| {r['mode']} | {r['n_max']} | {d:+.2f} | {pct:+.1f}% | {float(r['mean_accepted_length']):.2f} |")

notes = [
    "",
    "## Interpretation",
    "",
    "- **chain_pmin**: production baseline (p_min early-exit).",
    "- **full_chain**: always draft n_max tokens (p_min=0); chain verify.",
    "- **ddtree_***: build DDTree from draft top-K, still *chain-verify* the top-1 path.",
    "- Lucebox gains (~+0.6 AL, +15 tok/s on 3090) come from **tree-attention verify + fast rollback**,",
    "  not from the tree builder alone. Without that, ddtree_full ≈ full_chain (+ extract/build cost).",
    "- If full_chain mean_len does not grow with n_max, draft confidence collapses past the first few tokens",
    "  and longer blocks only add verify cost.",
    "",
    f"Raw CSV: `{csv_path}`",
    "",
]

md = "\n".join(lines + best_lines + cmp + notes) + "\n"
md_path.write_text(md)
results_path.write_text(md)
print(md)
PY

echo "DONE summary=$SUMMARY_CSV md=$SUMMARY_MD" | tee -a "$OUT_ROOT/run.log"
