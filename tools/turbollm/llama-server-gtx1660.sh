#!/usr/bin/env bash
# Pin this engine to the GTX 1660 Ti stack regardless of the installed GPU.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export HYBRID_GPU=1660
exec "$HERE/llama-server-wrapper.sh" "$@"
