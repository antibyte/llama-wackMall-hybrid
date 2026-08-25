#!/usr/bin/env bash
# Fetch LFM2.5-8B-A1B APEX I-Compact + DSpark Q8_0 into $HOME/models/lfm2.5-8b-a1b
set -euo pipefail

DEST="${LFM25_MODEL_DIR:-$HOME/models/lfm2.5-8b-a1b}"
mkdir -p "$DEST"

download() {
    local url="$1"
    local out="$2"
    local expect="$3"
    if [[ -f "$out" ]] && [[ "$(stat -c%s "$out")" == "$expect" ]]; then
        echo "ok $out"
        return 0
    fi
    wget -c --retry-connrefused --tries=20 --timeout=60 --waitretry=5 \
        --progress=dot:giga -O "$out" "$url"
    [[ "$(stat -c%s "$out")" == "$expect" ]] || {
        echo "size mismatch: $out" >&2
        exit 1
    }
}

download \
    'https://huggingface.co/mudler/LFM2.5-8B-A1B-APEX-GGUF/resolve/main/LFM2.5-8B-A1B-APEX-I-Compact.gguf?download=true' \
    "$DEST/LFM2.5-8B-A1B-APEX-I-Compact.gguf" \
    4212288992

download \
    'https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF/resolve/main/LFM2.5-8B-A1B-DSpark-Q8_0.gguf?download=true' \
    "$DEST/LFM2.5-8B-A1B-DSpark-Q8_0.gguf" \
    356491104

echo "models ready in $DEST"
