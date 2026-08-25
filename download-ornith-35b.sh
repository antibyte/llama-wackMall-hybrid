#!/usr/bin/env bash
# Fetch Ornith-1.5-35B-A3B AD-Q4_K-IQ4_XS into $HOME/models/ornith-1.5-35b-a3b
set -euo pipefail

DEST="${ORNITH_MODEL_DIR:-$HOME/models/ornith-1.5-35b-a3b}"
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
    'https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf?download=true' \
    "$DEST/Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf" \
    20125923520

echo "models ready in $DEST"
