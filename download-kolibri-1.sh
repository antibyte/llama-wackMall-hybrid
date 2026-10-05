#!/usr/bin/env bash
# Fetch Kolibri-1 Q4_K_M into $HOME/models/kolibri-1
#
# Source: Hob-forge/Kolibri-1-GGUF (Aleph-Alpha/Kolibri-1).
# Q4_K_M is about 44.2 GiB. Routed experts stay in the mmap and fault from disk.
set -euo pipefail

DEST="${KOLIBRI_MODEL_DIR:-$HOME/models/kolibri-1}"
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
    'https://huggingface.co/Hob-forge/Kolibri-1-GGUF/resolve/main/Kolibri-1-Q4_K_M.gguf?download=true' \
    "$DEST/Kolibri-1-Q4_K_M.gguf" \
    47454113472

echo "model ready in $DEST"
