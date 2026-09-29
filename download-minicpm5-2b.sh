#!/usr/bin/env bash
# Fetch official MiniCPM5-2B Q4_K_M GGUF into $HOME/models/minicpm5-2b
#
# Source: openbmb/MiniCPM5-2B-GGUF (not a community re-quant).
# Q4_K_M is 1.56 GB. Optional F16 / Q8_0 are listed in the comments.
set -euo pipefail

DEST="${MINICPM5_MODEL_DIR:-$HOME/models/minicpm5-2b}"
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
    'https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/resolve/main/MiniCPM5-2B-Q4_K_M.gguf?download=true' \
    "$DEST/MiniCPM5-2B-Q4_K_M.gguf" \
    1561318368

# Optional higher-fidelity quants (uncomment if needed):
# download \
#     'https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/resolve/main/MiniCPM5-2B-Q8_0.gguf?download=true' \
#     "$DEST/MiniCPM5-2B-Q8_0.gguf" \
#     2679710688
# download \
#     'https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/resolve/main/MiniCPM5-2B-F16.gguf?download=true' \
#     "$DEST/MiniCPM5-2B-F16.gguf" \
#     5039006688

echo "models ready in $DEST"
