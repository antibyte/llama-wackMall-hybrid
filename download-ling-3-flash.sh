#!/usr/bin/env bash
# Fetch Ling-3.0-flash Q4_K_M into $HOME/models/ling-3-flash
#
# Source: bartowski/Ling-3.0-flash-GGUF (inclusionAI/Ling-3.0-flash, 127B, 512 experts top-8,
# KDA + MLA hybrid, one MTP layer). Q4_K_M is 72.5 GiB in two splits, larger than RAM:
# routed experts stay in the mmap and fault from disk.
set -euo pipefail

DEST="${LING3_MODEL_DIR:-$HOME/models/ling-3-flash}"
BASE='https://huggingface.co/bartowski/Ling-3.0-flash-GGUF/resolve/main/Ling-3.0-flash-Q4_K_M'
mkdir -p "$DEST"

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

download Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf 39587396416 ef62f8506c994a90cf64238c28b2bd0e02fbf421fa4b42a353bc1004ac0cfdfd
download Ling-3.0-flash-Q4_K_M-00002-of-00002.gguf 38217593728 8f5f18ab076ce4989ae06fed44fc0bc9e00137299b0b6c9f3619a8d33e857011

echo "model ready in $DEST"
