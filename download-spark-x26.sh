#!/usr/bin/env bash
# Fetch mradermacher/Spark-X2.6-4B-GGUF Q6_K into $HOME/models/spark-x2.6-4b.
set -euo pipefail

DEST="${SPARK26_MODEL_DIR:-$HOME/models/spark-x2.6-4b}"
mkdir -p "$DEST"

out="$DEST/Spark-X2.6-4B.Q6_K.gguf"
expect=3379173408
url='https://huggingface.co/mradermacher/Spark-X2.6-4B-GGUF/resolve/main/Spark-X2.6-4B.Q6_K.gguf?download=true'

if [[ -f "$out" ]] && [[ "$(stat -c%s "$out")" == "$expect" ]]; then
    echo "ok $out"
    exit 0
fi

export HF_HUB_DISABLE_XET=1

wget -c --retry-connrefused --tries=20 --timeout=60 --waitretry=5 \
    --progress=dot:giga -O "$out" "$url"
[[ "$(stat -c%s "$out")" == "$expect" ]] || {
    echo "size mismatch: $out" >&2
    exit 1
}
echo "model ready in $DEST"
