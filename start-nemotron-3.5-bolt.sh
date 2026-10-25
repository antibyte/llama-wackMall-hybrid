#!/usr/bin/env bash
#
# GTX 1660 Ti launcher for NVIDIA-Nemotron-3.5-Bolt-E4B Q4_K_M (gemma4 E4B).
#
# Weights:
#   ~/models/nemotron-3.5-bolt-e4b/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf
#   companion DFlash BF16 is in the same directory and is not loaded.
#
# llama-bench 2026-10-24, build-main-sm75, FA on, f16 KV, 8 threads, ubatch 512.
# Decode winner is three MMVQ overrides together:
#   Q4_K ncols1 rows=2, Q6_K ncols1 rows=2, Q8_0 ncols1 rows=4.
#   tg128 65.98, pp512 ~830, pp2048 800, tg256 66.0
#   depth 8192: pp512 610, tg64 61.4
#   server chat, 256 tokens: 64.0 t/s
# Q4 rows=2 alone is tg128 63.1. Q8 rows=2 is tg128 65.5.
# Dropping any one of the three lands at tg128 62.4-64.9.
# Host register, async copy, dedup, concat 128, flat dim0, graph opt,
# graphs off, fusion off, reuse off, poll 0, offload-min 1, turbo4_k,
# Pascal MMQ, and token_embd on CUDA were slower or flat.
# ctx 65536, decision-seqs 0: 4342 MiB of 5748.
# DFlash n_max 7 (LLAMA_DFLASH_COMBINED=1) lost: German prose 30.5 t/s
# (accept 0.20, mean len 2.40) vs 61.3 plain. Inject was about 45 ms/step.
# Code was 56.4 vs 61.0. Sampling from the GGUF card: temp 1.0, top_p 0.95, top_k 64.
#
# agollm recipe: config/llama/nemotron-3.5-bolt.yaml (same knobs, ctx 65536).
# This script binds port 8080. Do not run it while agollm is up.
#
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SERVER="$PROJECT_ROOT/build-main-sm75/bin/llama-server"
MODEL="${HOME}/models/nemotron-3.5-bolt-e4b/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"

if [[ $# -ne 0 ]]; then
    printf 'start-nemotron-3.5-bolt.sh: no arguments; edit the script.\n' >&2
    exit 1
fi
[[ -x "$SERVER" ]] || { printf 'llama-server missing: %s\n' "$SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { printf 'model missing: %s\n' "$MODEL" >&2; exit 1; }

export GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2
export GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2
export GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS=4
export CUDA_VISIBLE_DEVICES=0
export LLAMA_ARG_POWER_BUSY_CMD="busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Performance"
export LLAMA_ARG_POWER_IDLE_CMD="busctl call com.system76.PowerDaemon /com/system76/PowerDaemon com.system76.PowerDaemon Battery"
export LLAMA_ARG_POWER_IDLE_DELAY=2000

exec "$SERVER" \
    -m "$MODEL" \
    --alias nemotron-bolt \
    --host 0.0.0.0 \
    --port 8080 \
    --jinja \
    -ngl 99 \
    -fa on \
    -t 8 \
    -c 65536 \
    -b 2048 \
    -ub 512 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --decision-seqs 0 \
    --temp 1.0 \
    --top-k 64 \
    --top-p 0.95 \
    --spec-type none
