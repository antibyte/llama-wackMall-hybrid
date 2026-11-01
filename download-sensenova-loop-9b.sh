#!/usr/bin/env bash
# Fetch SenseNova-Loop-9B-Preview Q3_K_M into $HOME/models/sensenova-loop-9b
#
# Source: guideboardlabs/SenseNova-Loop-9B-Preview-GGUF (sensenova/SenseNova-Loop-9B-Preview,
# Qwen3.5 9B hybrid trunk, layers 4..27 run twice with two mHC residual streams; text only).
# Arch sensenova_loop: needs this fork (ggml_mhc_route / ggml_mhc_merge).
#
# That GGUF has no MTP head. The trained head ships separately in the HF repo
# (mtp.safetensors); scripts/sensenova_loop_add_mtp.py appends it as block 56 and
# writes ...-Q3_K_M-mtp-<type>.gguf next to the base file (needs Python with numpy).
#   SENSENOVA_MTP=0         skip the MTP head
#   SENSENOVA_MTP_TYPE=q4_0 type of the head weights (q4_0, q5_0, q8_0, f16)
#   SENSENOVA_PYTHON=python3
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEST="${SENSENOVA_MODEL_DIR:-$HOME/models/sensenova-loop-9b}"
BASE='https://huggingface.co/guideboardlabs/SenseNova-Loop-9B-Preview-GGUF/resolve/main'
BASE_HF='https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/resolve/main'
MTP="${SENSENOVA_MTP:-1}"
MTP_TYPE="${SENSENOVA_MTP_TYPE:-q4_0}"
PYTHON="${SENSENOVA_PYTHON:-python3}"
mkdir -p "$DEST"

download() {
    local base="$1"
    local name="$2"
    local expect="$3"
    local sha="$4"
    local out="$DEST/$name"
    if [[ -f "$out" ]] && [[ "$(stat -c%s "$out")" == "$expect" ]]; then
        echo "ok $out"
    else
        wget -c --retry-connrefused --tries=20 --timeout=60 --waitretry=5 \
            --progress=dot:giga -O "$out" "$base/$name?download=true"
        [[ "$(stat -c%s "$out")" == "$expect" ]] || {
            echo "size mismatch: $out" >&2
            exit 1
        }
    fi
    if [[ "${SENSENOVA_VERIFY:-1}" == 1 ]]; then
        echo "$sha  $out" | sha256sum -c -
    fi
}

download "$BASE" SenseNova-Loop-9B-Preview-Q3_K_M.gguf 4642406560 02fa9b5c21ca4c60a8fac3fcbcfccbf2c87d5c6928be98de2666234cc251ba9e

if [[ "$MTP" == 1 ]]; then
    download "$BASE_HF" mtp.safetensors 486582880 fc870cc5953e18abce3664f29bd47e174656fcd3dbf50226998a64f79c83a178
    out="$DEST/SenseNova-Loop-9B-Preview-Q3_K_M-mtp-$MTP_TYPE.gguf"
    if [[ -f "$out" ]]; then
        echo "ok $out"
    else
        "$PYTHON" -c 'import numpy' 2>/dev/null || {
            echo "$PYTHON has no numpy; set SENSENOVA_PYTHON to a Python with numpy" >&2
            exit 1
        }
        "$PYTHON" "$ROOT/scripts/sensenova_loop_add_mtp.py" \
            "$DEST/SenseNova-Loop-9B-Preview-Q3_K_M.gguf" "$DEST/mtp.safetensors" "$out.tmp" --type "$MTP_TYPE"
        mv "$out.tmp" "$out"
    fi
fi

echo "model ready in $DEST"
