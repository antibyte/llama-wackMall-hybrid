#!/usr/bin/env bash
# Fetch unsloth/flint-GGUF UD-Q4_K_XL into $HOME/models/flint.
# Revision 20468edfc38ab43d6914cc88b08cbecb05d0fc62.
set -euo pipefail

DEST="${FLINT_MODEL_DIR:-$HOME/models/flint}"
mkdir -p "$DEST"

out="$DEST/flint-UD-Q4_K_XL.gguf"
expect=25054688704
url='https://huggingface.co/unsloth/flint-GGUF/resolve/main/flint-UD-Q4_K_XL.gguf?download=true'

if [[ -f "$out" ]] && [[ "$(stat -c%s "$out")" == "$expect" ]]; then
    echo "ok $out"
    exit 0
fi

# The Xet CDN stalls on this host. The resolve URL still resumes with wget.
export HF_HUB_DISABLE_XET=1

wget -c --retry-connrefused --tries=20 --timeout=60 --waitretry=5 \
    --progress=dot:giga -O "$out" "$url"
[[ "$(stat -c%s "$out")" == "$expect" ]] || {
    echo "size mismatch: $out" >&2
    exit 1
}
echo "model ready in $DEST"
