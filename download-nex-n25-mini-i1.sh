#!/usr/bin/env bash
# Fetch mradermacher Nex-N2.5-mini imatrix Q4_K_M into $HOME/models/nex-n2.5-mini-i1
#
# Source: mradermacher/Nex-N2.5-mini-i1-GGUF
#   Nex-N2.5-mini.i1-Q4_K_M.gguf  21.17 GB  (weighted/imatrix Q4_K_M)
# mmproj lives in the static repo (mradermacher/Nex-N2.5-mini-GGUF), not here.
set -euo pipefail

DEST="${NEX_N25_I1_MODEL_DIR:-$HOME/models/nex-n2.5-mini-i1}"
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
    'https://huggingface.co/mradermacher/Nex-N2.5-mini-i1-GGUF/resolve/main/Nex-N2.5-mini.i1-Q4_K_M.gguf?download=true' \
    "$DEST/Nex-N2.5-mini.i1-Q4_K_M.gguf" \
    21166758304

echo "models ready in $DEST"
