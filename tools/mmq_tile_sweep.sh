#!/usr/bin/env bash
# mmq_tile_sweep.sh LABEL SPEC [SPEC ...] -- MODEL [MODEL ...]
#
# Compile-Sweep der MMQ-Kachel auf Turing (sm_75, DP4A-Layout). SPEC ist
# I:J:SK, also Zeilen pro Kachel, groesstes Spalten-Kachelmass und
# Stream-K (0/1). Die Werte stehen als Default in
# ggml/src/ggml-cuda/mmq-config-turing.cuh; das Skript schreibt sie um, baut
# neu und misst mit llama-bench. Die Originalwerte werden am Ende wiederhergestellt.
#
# Aufruf:
#   tools/mmq_tile_sweep.sh mmq 64:64:0 128:64:0 -- <modell> <modell>
#
# Ergebnis: benchmark-results/mmq-tile-sweep/<LABEL>/
set -uo pipefail
cd "$(dirname "$0")/.."

label=$1; shift
specs=()
while [[ $# -gt 0 && "$1" != "--" ]]; do specs+=("$1"); shift; done
shift
models=("$@")

CFG=ggml/src/ggml-cuda/mmq-config-turing.cuh
OUT=benchmark-results/mmq-tile-sweep/$label
mkdir -p "$OUT"
BACKUP=$(mktemp)
cp "$CFG" "$BACKUP"
restore() { cp "$BACKUP" "$CFG"; rm -f "$BACKUP"; }
trap restore EXIT

set_macro() {
    local name=$1 val=$2
    sed -i -E "s|^#define (${name}) .*$|#define \1 ${val}|" "$CFG"
}

for spec in "${specs[@]}"; do
    IFS=: read -r i j sk <<< "$spec"
    set_macro GGML_CUDA_TURING_MMQ_I "$i"
    set_macro GGML_CUDA_TURING_MMQ_J_MAX "$j"
    if [[ "$sk" == "1" ]]; then
        set_macro GGML_CUDA_TURING_MMQ_STREAM_K true
    else
        set_macro GGML_CUDA_TURING_MMQ_STREAM_K false
    fi

    echo "== $label I=$i J=$j stream_k=$sk =="
    grep -n "^#define GGML_CUDA_TURING_MMQ" "$CFG"

    echo "-- build --"
    if ! cmake --build build-main-sm75 -j16 2>&1 | tail -3; then
        echo "BUILD FEHLGESCHLAGEN"
        exit 1
    fi

    echo "-- messen --"
    for model in "${models[@]}"; do
        name=$(basename "$model" .gguf)
        build-main-sm75/bin/llama-bench -m "$model" -p 4096 -n 64 -r 3 -t 8 -ngl 99 -fa on \
            2>&1 | tee "$OUT/bench-$name-I${i}J${j}SK${sk}.log" | grep -E "pp4096|tg64"
    done
done
echo "Ergebnis: $OUT"
