#!/usr/bin/env bash
# Fetch mradermacher Hy-MT2-30B-A3B Q4_K_M into $HOME/models/hy-mt2-30b
set -euo pipefail

DEST="${HY_MT2_MODEL_DIR:-$HOME/models/hy-mt2-30b}"
mkdir -p "$DEST"

out="$DEST/Hy-MT2-30B-A3B.Q4_K_M.gguf"
expect=18236702976
url='https://huggingface.co/mradermacher/Hy-MT2-30B-A3B-GGUF/resolve/main/Hy-MT2-30B-A3B.Q4_K_M.gguf?download=true'

if [[ -f "$out" ]] && [[ "$(stat -c%s "$out")" == "$expect" ]]; then
    echo "ok $out"
    exit 0
fi

wget -c --retry-connrefused --tries=20 --timeout=60 --waitretry=5 \
    --progress=dot:giga -O "$out" "$url"
[[ "$(stat -c%s "$out")" == "$expect" ]] || {
    echo "size mismatch: $out" >&2
    exit 1
}
echo "models ready in $DEST"
