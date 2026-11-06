#!/usr/bin/env bash
# bench_gpu.sh LABEL MODEL [llama-bench args...]
#
# Messlauf fuer die GPU-Seite: ein llama-bench-Lauf mit denselben Parametern
# wie der Vergleich (Prefill -p, Decode -n), dazu optional ein nsys-Profil
# desselben Laufs, um die Kernel-Verteilung zu sehen.
#
# Aufruf:
#   tools/bench_gpu.sh label /pfad/modell.gguf -p 4096 -n 64 -r 3 -t 8 -ngl 99 -fa on
#   PROFILE=1 tools/bench_gpu.sh label /pfad/modell.gguf -p 4096 ...
#
# Ergebnis in benchmark-results/gpu-bench/<label>/
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="benchmark-results/gpu-bench"
mkdir -p "$OUT"

label=$1; shift
model=$1; shift
dir="$OUT/$label"
mkdir -p "$dir"
BIN="${LLAMA_BIN:-build-main-sm75/bin/llama-bench}"

echo "== $label =="
echo "Modell: $model"
echo "Argumente: $*"

if [[ "${PROFILE:-0}" == "1" ]]; then
    echo "-- nsys-Profil --"
    nsys profile --force-overwrite=true --stats=true -o "$dir/nsys" \
        "$BIN" -m "$model" "$@" 2>&1 | tee "$dir/nsys.log"
    echo "Profil: $dir/nsys.nsys-rep"
else
    echo "-- llama-bench --"
    "$BIN" -m "$model" "$@" 2>&1 | tee "$dir/bench.log"
fi
