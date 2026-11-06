#!/usr/bin/env bash
# fa_tile_sweep.sh LABEL NTHREADS128 NBATCH_FA128 NBATCH_K128 NTHREADS256 NBATCH_FA256 NBATCH_K256
#
# Compile-Sweep der FA-Tile-Konfiguration (NVIDIA fp16, Zeile ncols=32, also der
# Prefill-Pfad) fuer die Kopfgroessen 128 und 256. Die Werte stehen als
# Default in ggml/src/ggml-cuda/fattn-tile.cuh; das Skript schreibt sie um,
# baut neu und misst mit llama-bench. Die Originalwerte werden am Ende
# wiederhergestellt.
#
# Aufruf:
#   tools/fa_tile_sweep.sh d128t128 128 64 64 256 64 64
#
# Ergebnis: benchmark-results/fa-tile-sweep/<LABEL>/
set -uo pipefail
cd "$(dirname "$0")/.."

label=$1; shift
v1=$1; v2=$2; v3=$3; v4=$4; v5=$5; v6=$6
shift 6

TILE=ggml/src/ggml-cuda/fattn-tile.cuh
OUT=benchmark-results/fa-tile-sweep/$label
mkdir -p "$OUT"
BACKUP=$(mktemp)
cp "$TILE" "$BACKUP"
restore() { cp "$BACKUP" "$TILE"; rm -f "$BACKUP"; }
trap restore EXIT

set_macro() {
    local name=$1 val=$2
    sed -i -E "s|^#define (${name}) [0-9]+$|#define \1 ${val}|" "$TILE"
}
set_macro GGML_CUDA_FATTN_TILE_D128_NTHREADS  "$v1"
set_macro GGML_CUDA_FATTN_TILE_D128_NBATCH_FA "$v2"
set_macro GGML_CUDA_FATTN_TILE_D128_NBATCH_K  "$v3"
set_macro GGML_CUDA_FATTN_TILE_D256_NTHREADS  "$v4"
set_macro GGML_CUDA_FATTN_TILE_D256_NBATCH_FA "$v5"
set_macro GGML_CUDA_FATTN_TILE_D256_NBATCH_K  "$v6"

echo "== $label =="
grep -n "GGML_CUDA_FATTN_TILE_D1\|GGML_CUDA_FATTN_TILE_D2" "$TILE" | grep define

echo "-- build --"
if ! cmake --build build-main-sm75 -j16 2>&1 | tail -5; then
    echo "BUILD FEHLGESCHLAGEN"
    exit 1
fi

echo "-- messen --"
for model in "$@"; do
    name=$(basename "$model" .gguf)
    build-main-sm75/bin/llama-bench -m "$model" -p 4096 -n 64 -r 3 -t 8 -ngl 99 -fa on \
        2>&1 | tee "$OUT/bench-$name.log" | grep -E "pp4096|tg64"
done
echo "Ergebnis: $OUT"
