#!/usr/bin/env bash
# Build Ling-3.0-flash with Q3_K routed experts: $HOME/models/ling-3-flash/Ling-3.0-flash-Q4_K_M-expq3k.gguf
#
# The routed experts (123 tensors, 68 of the 72.5 GiB) come from bartowski's Q3_K_S, everything
# else (attention, KDA, routers, shared experts, dense layers, embeddings) stays Q4_K_M.
# 52.7 GiB instead of 72.5: fewer expert bytes to stream from the SSD and more of them in RAM.
# 2026-11-14: decode 14.3 vs 7.5-8.4 t/s, KLD 0.095 against Q4_K_M (pure Q3_K_S: 0.129).
# Needs the Q4_K_M splits (./download-ling-3-flash.sh) and a Python with numpy (LING3_PYTHON).
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEST="${LING3_MODEL_DIR:-$HOME/models/ling-3-flash}"
PYTHON="${LING3_PYTHON:-python3}"
BASE='https://huggingface.co/bartowski/Ling-3.0-flash-GGUF/resolve/main/Ling-3.0-flash-Q3_K_S'
OUT="$DEST/Ling-3.0-flash-Q4_K_M-expq3k.gguf"

for f in 00001 00002; do
    [[ -f "$DEST/Ling-3.0-flash-Q4_K_M-$f-of-00002.gguf" ]] || { echo "missing Q4_K_M split $f (run ./download-ling-3-flash.sh)" >&2; exit 1; }
done

download() {
    local name="$1"
    local expect="$2"
    local sha="$3"
    local out="$DEST/$name"
    if [[ -f "$out" ]] && [[ "$(stat -c%s "$out")" == "$expect" ]]; then
        echo "ok $out"
    else
        wget -c --retry-connrefused --tries=20 --timeout=60 --waitretry=5 \
            --progress=dot:giga -O "$out" "$BASE/$name?download=true"
        [[ "$(stat -c%s "$out")" == "$expect" ]] || {
            echo "size mismatch: $out" >&2
            exit 1
        }
    fi
    if [[ "${LING3_VERIFY:-1}" == 1 ]]; then
        echo "$sha  $out" | sha256sum -c -
    fi
}

download Ling-3.0-flash-Q3_K_S-00001-of-00002.gguf 39754557248 ffa6b15a48d424c60e5e2258dc7a4360700ee87fb977f36ea5c3dd338a13b159
download Ling-3.0-flash-Q3_K_S-00002-of-00002.gguf 16533050880 3b8139d6422e7f09e0af2440c148236c3a9285b08817aa71fe72c3fd3ecc01cd

"$PYTHON" "$PROJECT_ROOT/tools/gguf_mix_experts.py" \
    --base "$DEST/Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf" "$DEST/Ling-3.0-flash-Q4_K_M-00002-of-00002.gguf" \
    --donor "$DEST/Ling-3.0-flash-Q3_K_S-00001-of-00002.gguf" "$DEST/Ling-3.0-flash-Q3_K_S-00002-of-00002.gguf" \
    --out "$OUT.tmp"
mv "$OUT.tmp" "$OUT"

echo "model ready: $OUT (the Q3_K_S splits are no longer needed)"
