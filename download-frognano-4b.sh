#!/usr/bin/env bash
# Fetch FrogNano-4B-2609 Q6_K_L plus the F16 mmproj into $HOME/models/frognano-4b-2609
#
# Source: bartowski/FrogNano-4B-2609-GGUF (microsoft/FrogNano-4B-2609, qwen35).
# Q6_K_L is 3.92 GB. The F16 projector is 0.64 GB and is only needed for images.
set -euo pipefail

DEST="${FROGNANO_MODEL_DIR:-$HOME/models/frognano-4b-2609}"
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
    'https://huggingface.co/bartowski/FrogNano-4B-2609-GGUF/resolve/main/FrogNano-4B-2609-Q6_K_L.gguf?download=true' \
    "$DEST/FrogNano-4B-2609-Q6_K_L.gguf" \
    3918386976

download \
    'https://huggingface.co/bartowski/FrogNano-4B-2609-GGUF/resolve/main/mmproj-FrogNano-4B-2609-f16.gguf?download=true' \
    "$DEST/mmproj-FrogNano-4B-2609-f16.gguf" \
    672423104

echo "models ready in $DEST"
