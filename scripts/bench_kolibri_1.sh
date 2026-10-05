#!/usr/bin/env bash
# Kolibri-1 Q4_K_M prefill/decode on the GTX 1660 Ti.
# Routed experts stay on the CPU mmap (-ncmoe) and fault from the SSD.
# Readahead stays on; the CPU MoE kernel faults several experts at once.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR:-$ROOT/benchmark-results/kolibri-1-q4-${STAMP}}"
mkdir -p "$OUT"

BENCH="${BENCH:-$ROOT/build-main-sm75/bin/llama-bench}"
MODEL="${MODEL:-$HOME/models/kolibri-1/Kolibri-1-Q4_K_M.gguf}"
REPS="${REPS:-1}"
THREADS="${THREADS:-8}"
EXPECT=47454113472

[[ -x "$BENCH" ]] || { echo "missing $BENCH" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL (run ./download-kolibri-1.sh)" >&2; exit 1; }
[[ "$(stat -c%s "$MODEL")" == "$EXPECT" ]] || { echo "incomplete model: $MODEL" >&2; exit 1; }

export LLAMA_MMAP_PREFETCH=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if command -v busctl >/dev/null 2>&1; then
    busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Performance >/dev/null 2>&1 || true
fi

echo "results: $OUT"
# -ncmoe 99 keeps every routed expert tensor in the CPU mmap.
# -ngl 99 still offloads attention, norms, the router and the shared expert.
# -nopo 1 computes the experts on the CPU instead of copying them over PCIe.
"$BENCH" \
    -m "$MODEL" \
    -ngl 99 \
    -ncmoe 99 \
    -p 512 \
    -n 128 \
    -b 2048 \
    -ub 2048 \
    -nopo 1 \
    -t "$THREADS" \
    -r "$REPS" \
    -fa on \
    -ctk q8_0 \
    -ctv q8_0 \
    --progress \
    -o md \
    2>&1 | tee "$OUT/llama-bench.md"
