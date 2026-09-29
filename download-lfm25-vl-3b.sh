#!/usr/bin/env bash
# Fetch Unsloth LFM2.5-VL-3B Q4_K_M + F16 mmproj into $HOME/models/lfm2.5-vl-3b
#
# Source: unsloth/LFM2.5-VL-3B-GGUF
#   LFM2.5-VL-3B-Q4_K_M.gguf  1.67 GB  (language backbone)
#   mmproj-F16.gguf           0.85 GB  (SigLIP2 NaFlex projector)
set -euo pipefail

DEST="${LFM25_VL_MODEL_DIR:-$HOME/models/lfm2.5-vl-3b}"
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
    'https://huggingface.co/unsloth/LFM2.5-VL-3B-GGUF/resolve/main/LFM2.5-VL-3B-Q4_K_M.gguf?download=true' \
    "$DEST/LFM2.5-VL-3B-Q4_K_M.gguf" \
    1674455424

download \
    'https://huggingface.co/unsloth/LFM2.5-VL-3B-GGUF/resolve/main/mmproj-F16.gguf?download=true' \
    "$DEST/mmproj-F16.gguf" \
    853994080

echo "models ready in $DEST"
