#!/usr/bin/env bash
# Fetch SenseNova-Loop-9B Q3_K_M into $HOME/models/sensenova-loop-9b
#
# Source: guideboardlabs/SenseNova-Loop-9B-GGUF, a text-only quant of
# sensenova/SenseNova-Loop-9B (the release that replaces the Preview).
# Arch sensenova_loop: needs this fork (ggml_mhc_route / ggml_mhc_merge).
# Same stored shape as the Preview (layers 4..27 run twice, two mHC streams).
# The release GGUF carries YaRN (factor 2, original context 262144, trained
# context 524288) and no MTP head. The official repo is gated and its card
# says draft weights are still to come, so this script does not graft one.
set -euo pipefail

DEST="${SENSENOVA_MODEL_DIR:-$HOME/models/sensenova-loop-9b}"
BASE='https://huggingface.co/guideboardlabs/SenseNova-Loop-9B-GGUF/resolve/main'
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
    if [[ "${SENSENOVA_VERIFY:-1}" == 1 ]]; then
        echo "$sha  $out" | sha256sum -c -
    fi
}

download SenseNova-Loop-9B-Q3_K_M.gguf 4642406752 7ca9d982004734502d36a2ae06bf714ae31ae24262806a27b80497573d11c869

echo "model ready in $DEST"
