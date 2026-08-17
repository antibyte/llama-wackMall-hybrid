#!/usr/bin/env bash
# Phase-matched 3x512 A/B: TARGET KV q4_0/q4_0 vs turbo4_k/turbo4_k.
# Everything else mirrors the measured start.sh production path (SC: static S,
# no adapt, static_no_sync, MTP-2, specialist profile).
#
# Usage:
#   ./scripts/ab_turbo4_vs_q4_512.sh
#   N_PREDICT=256 REPEATS=1 ./scripts/ab_turbo4_vs_q4_512.sh   # smoke
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PARENT="${RESULTS_PARENT:-$ROOT/benchmark-results/ab-turbo4-vs-q4-512-${STAMP}}"
mkdir -p "$PARENT"

export SERVER="${SERVER:-$ROOT/build-main-sm75/bin/llama-server}"
export CLIENT="${CLIENT:-$ROOT/tools/bench_hybrid_client.py}"
export MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
export PROFILE="${PROFILE:-$ROOT/profiles/specialist-benchprompt.csv}"
export PORT="${PORT:-18081}"

# start.sh-aligned production knobs (only TARGET_TYPE_* varies between arms)
export REPEATS="${REPEATS:-3}"
export N_PREDICT="${N_PREDICT:-512}"
export WARMUP_TOKENS="${WARMUP_TOKENS:-64}"
export COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-8}"
export CONTEXT="${CONTEXT:-24576}"
export STATIC_FIXED_S="${STATIC_FIXED_S:-34}"
export HYBRID_FIXED_S="${HYBRID_FIXED_S:-34}"
export CPU_THREADS="${CPU_THREADS:-8}"
export DRAFT_THREADS="${DRAFT_THREADS:-8}"
export CMOE_BATCH="${CMOE_BATCH:-64}"
export CMOE_UBATCH="${CMOE_UBATCH:-64}"
export CMOE_PREFILL_BATCH="${CMOE_PREFILL_BATCH:-768}"
export CMOE_PREFILL_UBATCH="${CMOE_PREFILL_UBATCH:-768}"
export DRAFT_TYPE_K="${DRAFT_TYPE_K:-q4_0}"
export DRAFT_TYPE_V="${DRAFT_TYPE_V:-q4_0}"
export KV_Q4_SCALE="${KV_Q4_SCALE:-weighted-k}"
export MTP_OVERRIDE="${MTP_OVERRIDE:-2}"
export MTP_REQUANTIZE_OUTPUT="${MTP_REQUANTIZE_OUTPUT:-none}"
export TARGET_BACKEND_SAMPLING="${TARGET_BACKEND_SAMPLING:-1}"
export DRAFT_BACKEND_SAMPLING="${DRAFT_BACKEND_SAMPLING:-1}"
export REASONING_BUDGET="${REASONING_BUDGET:-1024}"
export VRAM_RESERVE_MIB="${VRAM_RESERVE_MIB:-400}"
export LOAD_MODE="${LOAD_MODE:-mmap}"
export SHARED_HOT_IDS="${SHARED_HOT_IDS:-1}"
export SKIP_SENTINEL="${SKIP_SENTINEL:-1}"
export MOE_MULTI_FUSION="${MOE_MULTI_FUSION:-1}"
export MOE_COMBINE_FUSION="${MOE_COMBINE_FUSION:-1}"
export MMVQ_Q8_NCOLS3_ROWS="${MMVQ_Q8_NCOLS3_ROWS:-4}"
export MMVQ_Q6_K_NCOLS1_ROWS="${MMVQ_Q6_K_NCOLS1_ROWS:-2}"
export CUDA_ASYNC_HOST_COPY="${CUDA_ASYNC_HOST_COPY:-1}"
export CONCAT_NONCONT_BLOCK_SIZE="${CONCAT_NONCONT_BLOCK_SIZE:-128}"
export CONCAT_NONCONT_FLAT_DIM0="${CONCAT_NONCONT_FLAT_DIM0:-1}"
export TURBO4_MTP_EXPERIMENTAL="${TURBO4_MTP_EXPERIMENTAL:-1}"
export TURBO4_V_EXPERIMENTAL="${TURBO4_V_EXPERIMENTAL:-1}"
export TURBO4_DRAFT_EXPERIMENTAL="${TURBO4_DRAFT_EXPERIMENTAL:-0}"
export TURBO4_F16_PREFILL_MIN_BATCH="${TURBO4_F16_PREFILL_MIN_BATCH:-2}"
export TURBO4_FAST_F16_CONVERT="${TURBO4_FAST_F16_CONVERT:-0}"
export TURBO4_WHT_SHUFFLE="${TURBO4_WHT_SHUFFLE:-0}"
export CASES="${CASES:-SC}"   # static S, adapt off, static_no_sync

# Prefer the same bench prompt style if present
if [[ -z "${PROMPT_SOURCE:-}" ]]; then
  for cand in \
    "$ROOT/profiles/specialist-benchprompt.txt" \
    "$ROOT/ROUTER_LOOKAHEAD_ANALYSIS.md" \
    "$ROOT/HYBRID_ANALYSIS.md"; do
    if [[ -f "$cand" ]]; then
      export PROMPT_SOURCE="$cand"
      break
    fi
  done
fi

run_arm() {
  local name="$1" tk="$2" tv="$3"
  local out="$PARENT/${name}"
  echo ""
  echo "############################################################"
  echo "# ARM $name  target_kv=$tk/$tv  n_predict=$N_PREDICT x$REPEATS"
  echo "############################################################"
  RESULTS_DIR="$out" \
  TARGET_TYPE_K="$tk" \
  TARGET_TYPE_V="$tv" \
  "$ROOT/scripts/bench_hybrid.sh" "$CASES" "$REPEATS" "$N_PREDICT" "$WARMUP_TOKENS" "$out"
}

# Interleave would need one process; sequential arms keep cold starts fair
# (fresh server per rep inside bench_hybrid).
run_arm "q4_0" "q4_0" "q4_0"
run_arm "turbo4_k" "turbo4_k" "turbo4_k"

# Aggregate summary
python3 - "$PARENT" <<'PY'
import csv, statistics, sys
from pathlib import Path

parent = Path(sys.argv[1])
rows = []
for arm in ("q4_0", "turbo4_k"):
    p = parent / arm / "runs.csv"
    if not p.exists():
        print(f"missing {p}", file=sys.stderr)
        continue
    with p.open() as f:
        for r in csv.DictReader(f):
            r["arm"] = arm
            rows.append(r)

out = parent / "summary.md"
lines = [
    "# A/B: turbo4_k vs q4_0 (3×512 phase-matched)",
    "",
    f"Parent: `{parent}`",
    "",
    "Arms differ **only** in `TARGET_TYPE_K/V`. Production knobs from `start.sh` (SC, MTP-2, S=34, batch 64, prefill 768, draft q4_0, skip-sentinel, async H2D, …).",
    "",
    "| Arm | Runs | Decode TPS (median) | Sustained TPS (median) | MTP accept (median) | Mean accepted | VRAM peak MiB | Token hashes identical |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]

def fnum(r, k):
    try:
        return float(r.get(k) or "nan")
    except ValueError:
        return float("nan")

by = {}
for r in rows:
    by.setdefault(r["arm"], []).append(r)

for arm in ("q4_0", "turbo4_k"):
    rs = by.get(arm, [])
    if not rs:
        lines.append(f"| {arm} | 0 | — | — | — | — | — | — |")
        continue
    dec = sorted(fnum(r, "decode_tps") for r in rs)
    sus = sorted(fnum(r, "sustained_decode_tps") for r in rs)
    acc = sorted(fnum(r, "mtp_acceptance") for r in rs)
    mal = sorted(fnum(r, "mean_accepted_length") for r in rs)
    vram = sorted(fnum(r, "vram_peak_mib") for r in rs)
    hashes = [r.get("token_sha256", "") for r in rs]
    same = "yes" if len(set(hashes)) == 1 else f"no ({len(set(hashes))} unique)"
    def med(xs):
        xs = [x for x in xs if x == x]
        return statistics.median(xs) if xs else float("nan")
    lines.append(
        f"| {arm} | {len(rs)} | {med(dec):.3f} | {med(sus):.3f} | {med(acc):.4f} | {med(mal):.3f} | {med(vram):.0f} | {same} |"
    )

if "q4_0" in by and "turbo4_k" in by and by["q4_0"] and by["turbo4_k"]:
    def med_field(arm, k):
        xs = [fnum(r, k) for r in by[arm] if fnum(r, k) == fnum(r, k)]
        return statistics.median(xs) if xs else float("nan")
    q = med_field("q4_0", "decode_tps")
    t = med_field("turbo4_k", "decode_tps")
    if q == q and t == t and q > 0:
        delta = (t - q) / q * 100.0
        lines += [
            "",
            f"**Decode TPS delta (turbo4 − q4) / q4 = {delta:+.2f}%**",
            "",
            "Positive ⇒ turbo4 faster; negative ⇒ q4 faster.",
        ]

lines += [
    "",
    "## Per-run detail",
    "",
    "| Arm | Rep | decode_tps | sustained | mtp_accept | mean_acc_len | vram | token_sha256 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for r in rows:
    lines.append(
        f"| {r['arm']} | {r.get('rep')} | {r.get('decode_tps')} | {r.get('sustained_decode_tps')} | "
        f"{r.get('mtp_acceptance')} | {r.get('mean_accepted_length')} | {r.get('vram_peak_mib')} | "
        f"`{(r.get('token_sha256') or '')[:12]}` |"
    )

out.write_text("\n".join(lines) + "\n")
print(out.read_text())
print(f"\nWrote {out}")
PY

echo "All results under: $PARENT"
