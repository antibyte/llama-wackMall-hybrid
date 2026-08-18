# Shared path/GPU helpers for tools/turbollm/*.sh
# shellcheck shell=bash

hybrid_root() {
    local here
    here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    cd -- "$here/../.." && pwd -P
}

# Prints 1660 | 1080 | unknown
detect_gpu() {
    local override name cap
    override="${HYBRID_GPU:-auto}"
    case "$override" in
        1660|gtx1660|sm75|7.5) echo 1660; return 0 ;;
        1080|gtx1080|sm61|6.1) echo 1080; return 0 ;;
        auto|"") ;;
        *) echo unknown; return 0 ;;
    esac
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo unknown
        return 0
    fi
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr '[:upper:]' '[:lower:]')"
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    case "$name" in
        *1660*) echo 1660; return 0 ;;
        *1080*) echo 1080; return 0 ;;
    esac
    case "$cap" in
        7.5) echo 1660; return 0 ;;
        6.1) echo 1080; return 0 ;;
    esac
    echo unknown
}

first_existing() {
    local p
    for p in "$@"; do
        [[ -n "$p" && -e "$p" ]] && { echo "$p"; return 0; }
    done
    return 1
}

resolve_server() {
    local gpu="${1:?}" root
    root="$(hybrid_root)"
    case "$gpu" in
        1660)
            first_existing \
                "${HYBRID_LLAMA_SERVER_1660:-}" \
                "${HYBRID_LLAMA_SERVER:-}" \
                "$root/build-main-sm75/bin/llama-server"
            ;;
        1080)
            first_existing \
                "${HYBRID_LLAMA_SERVER_1080:-}" \
                "$root/build-turbo-opt-sm61/bin/llama-server" \
                "$root/build-lookahead-sm61-gcc12/bin/llama-server" \
                "$root/build-lookahead-sm61/bin/llama-server"
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_model() {
    first_existing \
        "${HYBRID_MODEL:-}" \
        "${HOME}/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
        /root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf \
        /root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
}

resolve_dflash() {
    first_existing \
        "${HYBRID_DFLASH:-}" \
        "${HOME}/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf" \
        /root/atomic-nextn-good/models/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf
}

resolve_expert_hot() {
    local gpu="${1:?}" root
    root="$(hybrid_root)"
    case "$gpu" in
        1660)
            first_existing \
                "${HYBRID_EXPERT_HOT_1660:-}" \
                "$root/profiles/specialist-benchprompt.csv"
            ;;
        1080)
            first_existing \
                "${HYBRID_EXPERT_HOT_1080:-}" \
                "$root/1" \
                "$root/profiles/specialist-benchprompt.csv"
            ;;
        *)
            return 1
            ;;
    esac
}

engine_name() {
    case "${1:?}" in
        1660) echo "Hybrid GTX 1660 Ti" ;;
        1080) echo "Hybrid GTX 1080" ;;
        *) return 1 ;;
    esac
}

is_probe_invocation() {
    local a
    for a in "$@"; do
        case "$a" in
            -h|--help|--version|--list-devices) return 0 ;;
        esac
    done
    return 1
}

llama_server_running() {
    # Match the executed binary name only. `pgrep -f llama-server` also hits
    # this wrapper and unrelated shell lines that mention the path.
    pgrep -x llama-server >/dev/null 2>&1
}

gpu_vram_used_mib() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' '
}
