#!/usr/bin/env bash
# Fetch Lythri-4B-A2B Q8_0 GGUF into $HOME/models/lythri-4b-a2b
#
# Source: Lythri/Lythri-4B-A2B-GGUF (Gemma 4 E2B companion fine-tune).
# Q8_0 is 4.95 GB. Q4_K_M is listed in the comments if VRAM is tighter.
set -euo pipefail

DEST="${LYTHRI4B_MODEL_DIR:-$HOME/models/lythri-4b-a2b}"
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
    'https://huggingface.co/Lythri/Lythri-4B-A2B-GGUF/resolve/main/Lythri-4B-A2B-Q8_0.gguf?download=true' \
    "$DEST/Lythri-4B-A2B-Q8_0.gguf" \
    4947415072

# Optional lower-VRAM quant:
# download \
#     'https://huggingface.co/Lythri/Lythri-4B-A2B-GGUF/resolve/main/Lythri-4B-A2B-Q4_K_M.gguf?download=true' \
#     "$DEST/Lythri-4B-A2B-Q4_K_M.gguf" \
#     3416120352

echo "models ready in $DEST"
