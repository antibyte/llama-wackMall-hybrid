#!/usr/bin/env bash
# Fetch Nex-N2.5-mini Q4_K_M GGUF into $HOME/models/nex-n2.5-mini
#
# Source: abenzerps/Nex-N2.5-mini-GGUF (llama.cpp convert of nex-agi/Nex-N2.5-mini).
# Optional mmproj is commented; text-only hybrid does not need it.
set -euo pipefail

DEST="${NEX_N25_MODEL_DIR:-$HOME/models/nex-n2.5-mini}"
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
    'https://huggingface.co/abenzerps/Nex-N2.5-mini-GGUF/resolve/main/Nex-N2.5-mini-Q4_K_M.gguf?download=true' \
    "$DEST/Nex-N2.5-mini-Q4_K_M.gguf" \
    21166757664

# Optional vision projector (~899 MB):
# download \
#     'https://huggingface.co/abenzerps/Nex-N2.5-mini-GGUF/resolve/main/mmproj-Nex-N2.5-mini-F16.gguf?download=true' \
#     "$DEST/mmproj-Nex-N2.5-mini-F16.gguf" \
#     899282976

echo "models ready in $DEST"
