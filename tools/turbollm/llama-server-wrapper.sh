#!/usr/bin/env bash
# TurboLLM engine entrypoint. Selects the GTX 1660 Ti or GTX 1080 hybrid stack.
# Register llama-server-gtx1660.sh / llama-server-gtx1080.sh as engine binPath
# (or this file with HYBRID_GPU=auto).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=/dev/null
source "$HERE/common.sh"

export HYBRID_ROOT
HYBRID_ROOT="$(hybrid_root)"

gpu="${HYBRID_GPU:-auto}"
case "$(basename -- "$0")" in
    *1660*) gpu=1660 ;;
    *1080*) gpu=1080 ;;
esac
if [[ "$gpu" == auto ]]; then
    gpu="$(detect_gpu)"
fi
case "$gpu" in
    1660|1080) ;;
    *)
        echo "hybrid wrapper: cannot detect GPU (set HYBRID_GPU=1660 or 1080)." >&2
        exit 64
        ;;
esac

# Profile env must not inherit leftover knobs from the other GPU.
unset "${!LLAMA_@}" 2>/dev/null || true
unset "${!GGML_CUDA_@}" 2>/dev/null || true
unset "${!GGML_SCHED_@}" 2>/dev/null || true

ENV_FILE="${HYBRID_TURBOLLM_ENV:-$HERE/env/gtx${gpu}.env}"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "hybrid wrapper: env file missing: $ENV_FILE" >&2
    exit 66
fi

if hot="$(resolve_expert_hot "$gpu")"; then
    export HYBRID_EXPERT_HOT="$hot"
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -n "${LLAMA_EXPERT_S:-}" ]]; then
    unset LLAMA_EXPERT_PLACEMENT || true
fi
if [[ "$gpu" == 1080 ]]; then
    unset LLAMA_KVFLASH LLAMA_KVFLASH_MAX_POOL LLAMA_KVFLASH_TAU LLAMA_KVFLASH_POLICY LLAMA_KVFLASH_STATS || true
fi

REAL="${HYBRID_LLAMA_SERVER:-}"
if [[ -z "$REAL" ]]; then
    REAL="$(resolve_server "$gpu" || true)"
fi
if [[ -z "$REAL" || ! -x "$REAL" ]]; then
    echo "hybrid wrapper: llama-server for GPU $gpu not found or not executable." >&2
    echo "hybrid wrapper: expected sm75 build-main-sm75 (1660) or sm61 build-turbo-opt-sm61 (1080)." >&2
    exit 127
fi

if ! is_probe_invocation "$@" && [[ "${HYBRID_ALLOW_SHARED:-0}" != 1 ]]; then
    used="$(gpu_vram_used_mib || true)"
    if llama_server_running || { [[ "${used:-0}" =~ ^[0-9]+$ ]] && (( used > 1500 )); }; then
        echo "hybrid wrapper: GPU already in use (llama-server running or ${used:-?} MiB VRAM)." >&2
        echo "hybrid wrapper: stop start1660.sh / start1080.sh / the existing server first." >&2
        echo "hybrid wrapper: TurboLLM and the standalone launcher cannot share one GPU." >&2
        exit 78
    fi
fi

echo "hybrid wrapper: gpu=$gpu binary=$REAL env=$ENV_FILE" >&2
exec "$REAL" "$@"
