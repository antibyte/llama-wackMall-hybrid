#!/usr/bin/env bash
#
# llama-wackMall-hybrid GTX 1080 router: Qwen and Ling on demand.
#
# One model in VRAM at a time (--models-max 1). OpenWebUI / API pick the
# model name; the first request loads it, a switch unloads the other.
#
# Names:
#   qwen3.6-35b-a3b-hybrid-gtx1080  (start1080.sh params, DFlash n_max=8, S=58)
#   ling-tiny                       (start-ling-tiny-1080.sh params)
#
# No extra CLI args. Edit the block below, then ./router1080.sh
#
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# ============================================================================
# EDITABLE CONFIGURATION
# ============================================================================

SERVER="$PROJECT_ROOT/build-turbo-opt-sm61/bin/llama-server"

QWEN_MODEL="/root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf"
QWEN_DRAFT="/root/atomic-nextn-good/models/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf"
QWEN_NAME="qwen3.6-35b-a3b-hybrid-gtx1080"
PROFILE="$PROJECT_ROOT/1"

LING_MODEL="${LING_MODEL:-$HOME/models/ling-3.0-tiny/Ling-3.0-tiny-Q4_K_M.gguf}"
LING_NAME="ling-tiny"

HOST="0.0.0.0"
PORT="8080"
CORS_ORIGINS="*"
API_KEY=""
API_KEY_FILE=""
CUDA_VISIBLE_DEVICES_VALUE="0"

# ============================================================================
# End of editable configuration
# ============================================================================

die() {
    printf 'router1080.sh: %s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 0 ]]; then
    die "Keine Kommandozeilenparameter: Einstellungen oben in router1080.sh aendern."
fi

[[ -x "$SERVER" ]] || die "llama-server nicht ausfuehrbar: $SERVER"
[[ -f "$QWEN_MODEL" ]] || die "Qwen-Modell nicht gefunden: $QWEN_MODEL"
[[ -f "$QWEN_DRAFT" ]] || die "DFlash-Modell nicht gefunden: $QWEN_DRAFT"
[[ -f "$LING_MODEL" ]] || die "Ling-Modell nicht gefunden: $LING_MODEL"
[[ -f "$PROFILE" ]] || die "Expert-Profil nicht gefunden: $PROFILE"
[[ -f "$PROJECT_ROOT/models/templates/Qwen-Fixed-v22.3.jinja" ]] || die "Qwen chat template nicht gefunden: $PROJECT_ROOT/models/templates/Qwen-Fixed-v22.3.jinja"
if [[ -n "$API_KEY_FILE" && ! -f "$API_KEY_FILE" ]]; then
    die "API-Key-Datei nicht gefunden: $API_KEY_FILE"
fi

RUNTIME="$PROJECT_ROOT/.router-runtime/1080"
mkdir -p "$RUNTIME/empty-cache"
INI="$RUNTIME/models.ini"
QWEN_ENV="$RUNTIME/qwen.env"
LING_ENV="$RUNTIME/ling.env"

UNSET_SHARED=(
    LLAMA_ARG_CPU_MOE
    LLAMA_EXPERT_S
    LLAMA_EXPERT_HOT
    LLAMA_EXPERT_PLACEMENT
    LLAMA_ARG_OVERRIDE_TENSOR
    LLAMA_ARG_SPEC_DRAFT_MODEL
    LLAMA_ARG_SPEC_DRAFT_N_MIN
    LLAMA_ARG_SPEC_DRAFT_P_MIN
    LLAMA_DFLASH_COMBINED
    LLAMA_DFLASH_DDTREE
    LLAMA_DFLASH_TREE_VERIFY
    LLAMA_DFLASH_TREE_DIRECT_COMMIT
    LLAMA_DFLASH_DDTREE_K
    LLAMA_DFLASH_DDTREE_BUDGET
    LLAMA_DFLASH_DDTREE_TEMP
    LLAMA_DFLASH_DDTREE_CHAIN_SEED
    LLAMA_DFLASH_DDTREE_FULL_CHAIN
    LLAMA_TURBO4_V_EXPERIMENTAL
    LLAMA_TURBO4_MTP_EXPERIMENTAL
    LLAMA_TURBO4_DFLASH_EXPERIMENTAL
    LLAMA_TURBO4_DRAFT_EXPERIMENTAL
    LLAMA_TURBO4_Q8_FALLBACK_LAYERS
    LLAMA_KV_Q4_SCALE
    LLAMA_CMOE_PREFILL_BATCH
    LLAMA_CMOE_PREFILL_UBATCH
    LLAMA_CMOE_DECODE_BATCH
    LLAMA_CMOE_DECODE_UBATCH
    LLAMA_CMOE_BATCH
    LLAMA_CMOE_UBATCH
    LLAMA_KVFLASH
    LLAMA_KVFLASH_MAX_POOL
    GGML_CUDA_MOE_MULTI_FUSION
    GGML_CUDA_MOE_COMBINE_FUSION
    GGML_CUDA_REGISTER_HOST
    GGML_SCHED_PREFETCH_EXPERTS
    LLAMA_ARG_CHAT_TEMPLATE_FILE
    LLAMA_ARG_THINK
)

{
    printf 'version = 1\n\n'
    printf '[*]\n'
    printf 'offline = true\n'
    printf 'no-ui = true\n'
    printf 'jinja = true\n'
    printf 'kv-unified = true\n'
    printf 'cache-prompt = true\n'
    printf 'cache-idle-slots = true\n'
    printf 'cont-batching = true\n'
    printf 'parallel = 1\n'
    printf 'load-on-startup = false\n\n'

    printf '[%s]\n' "$QWEN_NAME"
    printf 'model = %s\n' "$QWEN_MODEL"
    printf 'spec-draft-model = %s\n' "$QWEN_DRAFT"
    printf 'spec-type = draft-dflash\n'
    printf 'spec-draft-n-max = 8\n'
    printf 'spec-draft-n-min = 0\n'
    printf 'spec-draft-p-min = 0.75\n'
    printf 'spec-draft-backend-sampling = true\n'
    printf 'n-gpu-layers-draft = 99\n'
    printf 'override-tensor = ^blk[.]40[.]=CPU\n'
    printf 'cpu-moe = true\n'
    printf 'ctx-size = 32768\n'
    printf 'n-predict = 20000\n'
    printf 'batch-size = 128\n'
    printf 'ubatch-size = 128\n'
    printf 'threads = 4\n'
    printf 'threads-batch = 4\n'
    printf 'spec-draft-threads = 4\n'
    printf 'spec-draft-threads-batch = 4\n'
    printf 'cache-type-k = turbo4_k\n'
    printf 'cache-type-v = turbo4_k\n'
    printf 'spec-draft-type-k = turbo4_k\n'
    printf 'spec-draft-type-v = turbo4_k\n'
    printf 'flash-attn = on\n'
    printf 'kv-offload = true\n'
    printf 'load-mode = none\n'
    printf 'backend-sampling = true\n'
    printf 'reasoning = auto\n'
    printf 'reasoning-budget = 512\n'
    printf 'reasoning-preserve = true\n'
    printf 'reasoning-format = deepseek\n'
    printf 'chat-template-file = %s\n' "$PROJECT_ROOT/models/templates/Qwen-Fixed-v22.3.jinja"
    printf 'ctx-checkpoints = 0\n'
    printf 'cache-ram = 1024\n'
    printf 'cache-reuse = 16\n'
    printf 'mmproj-auto = false\n'
    printf 'child-env-file = %s\n' "$QWEN_ENV"
    printf 'stop-timeout = 120\n\n'

    printf '[%s]\n' "$LING_NAME"
    printf 'model = %s\n' "$LING_MODEL"
    printf 'spec-type = ngram-simple\n'
    printf 'spec-draft-n-max = 4\n'
    printf 'n-gpu-layers = 99\n'
    printf 'no-cpu-moe = true\n'
    printf 'flash-attn = on\n'
    printf 'cache-type-k = q8_0\n'
    printf 'cache-type-v = q8_0\n'
    printf 'fit = on\n'
    printf 'fit-target = 80\n'
    printf 'fit-ctx = 2048\n'
    printf 'ctx-size = 131072\n'
    printf 'n-predict = 8192\n'
    printf 'batch-size = 128\n'
    printf 'ubatch-size = 128\n'
    printf 'threads = 4\n'
    printf 'threads-batch = 4\n'
    printf 'backend-sampling = true\n'
    printf 'reasoning = on\n'
    printf 'reasoning-budget = 1000\n'
    printf 'reasoning-preserve = true\n'
    printf 'ctx-checkpoints = 4\n'
    printf 'cache-ram = 2048\n'
    printf 'cache-reuse = 16\n'
    printf 'op-offload = true\n'
    printf 'load-mode = mmap\n'
    printf 'child-env-file = %s\n' "$LING_ENV"
    printf 'stop-timeout = 30\n'
} > "$INI"

{
    printf '# Qwen3.6-35B-A3B hybrid, start1080.sh getenv knobs\n'
    printf 'LLAMA_CMOE_BATCH=128\n'
    printf 'LLAMA_CMOE_UBATCH=128\n'
    printf 'LLAMA_CMOE_PREFILL_BATCH=1024\n'
    printf 'LLAMA_CMOE_PREFILL_UBATCH=1024\n'
    printf 'LLAMA_CMOE_DECODE_BATCH=128\n'
    printf 'LLAMA_CMOE_DECODE_UBATCH=128\n'
    printf 'LLAMA_KV_Q4_SCALE=legacy\n'
    printf 'LLAMA_TURBO4_V_EXPERIMENTAL=1\n'
    printf 'LLAMA_TURBO4_MTP_EXPERIMENTAL=1\n'
    printf 'LLAMA_TURBO4_DFLASH_EXPERIMENTAL=1\n'
    printf 'LLAMA_TURBO4_DRAFT_EXPERIMENTAL=1\n'
    printf 'GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH=2\n'
    printf 'GGML_CUDA_TURBO4_FAST_F16_CONVERT=0\n'
    printf 'GGML_CUDA_TURBO4_WHT_SHUFFLE=0\n'
    printf 'LLAMA_MTP_REQUANTIZE_OUTPUT=none\n'
    printf 'LLAMA_MTP_HEAD_TRACE=0\n'
    printf 'LLAMA_DFLASH_COMBINED=1\n'
    printf 'LLAMA_DFLASH_DDTREE=1\n'
    printf 'LLAMA_DFLASH_TREE_VERIFY=1\n'
    printf 'LLAMA_DFLASH_TREE_DIRECT_COMMIT=1\n'
    printf 'LLAMA_DFLASH_DDTREE_K=2\n'
    printf 'LLAMA_DFLASH_DDTREE_BUDGET=4\n'
    printf 'LLAMA_DFLASH_DDTREE_TEMP=1.0\n'
    printf 'LLAMA_DFLASH_DDTREE_CHAIN_SEED=1\n'
    printf 'LLAMA_DFLASH_DDTREE_FULL_CHAIN=1\n'
    printf 'LLAMA_EXPERT_HOT=%s\n' "$PROFILE"
    printf 'LLAMA_EXPERT_S=58\n'
    printf 'LLAMA_EXPERT_TMAX=32\n'
    printf 'LLAMA_EXPERT_STATS=0\n'
    printf 'LLAMA_EXPERT_ADAPT=0\n'
    printf 'LLAMA_EXPERT_DECAY=1.0\n'
    printf 'LLAMA_EXPERT_USAGE=0\n'
    printf 'LLAMA_EXPERT_ADAPT_INTERVAL=request\n'
    printf 'LLAMA_EXPERT_ADAPT_CUDA_GRAPHS=0\n'
    printf 'LLAMA_EXPERT_STATS_JSON=0\n'
    printf 'LLAMA_EXPERT_TIMING=0\n'
    printf 'LLAMA_EXPERT_CPU_CHUNK=64\n'
    printf 'LLAMA_EXPERT_CPU_ACT_PARALLEL=0\n'
    printf 'LLAMA_EXPERT_CPU_ASYNC=0\n'
    printf 'LLAMA_EXPERT_CPU_DOWN_PREFETCH=0\n'
    printf 'LLAMA_EXPERT_CPU_REUSE_ROWS=1\n'
    printf 'LLAMA_EXPERT_CPU_MULTI_ROW=1\n'
    printf 'LLAMA_EXPERT_CPU_FUSED_GATE_UP=0\n'
    printf 'LLAMA_EXPERT_WARM_SLOTS=0\n'
    printf 'LLAMA_EXPERT_WARM_AUTO_MAX=8\n'
    printf 'LLAMA_EXPERT_WARM_PREFETCH=0\n'
    printf 'LLAMA_EXPERT_VRAM_RESERVE_MIB=640\n'
    printf 'LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL=0\n'
    printf 'LLAMA_EXPERT_STATIC_NO_SYNC=1\n'
    printf 'LLAMA_EXPERT_LOOKAHEAD=0\n'
    printf 'LLAMA_EXPERT_SHARED_HOT_IDS=1\n'
    printf 'LLAMA_EXPERT_SKIP_SENTINEL=1\n'
    printf 'GGML_CUDA_MOE_MULTI_FUSION=0\n'
    printf 'GGML_CUDA_MOE_COMBINE_FUSION=0\n'
    printf 'GGML_CUDA_ASYNC_HOST_COPY=1\n'
    printf 'GGML_SCHED_ASYNC_D2H_COPY=0\n'
    printf 'GGML_SCHED_DEDUP_DST_SYNC=1\n'
    printf 'GGML_CUDA_REGISTER_HOST=0\n'
    printf 'GGML_SCHED_PREFETCH_EXPERTS=0\n'
} > "$QWEN_ENV"

{
    printf '# Ling-3.0-tiny, start-ling-tiny-1080.sh getenv knobs\n'
    for key in "${UNSET_SHARED[@]}"; do
        printf -- '-u %s\n' "$key"
    done
    printf -- '-u LLAMA_EXPERT_TMAX\n'
    printf -- '-u LLAMA_EXPERT_SKIP_SENTINEL\n'
    printf -- '-u LLAMA_EXPERT_SHARED_HOT_IDS\n'
    printf -- '-u LLAMA_EXPERT_WARM_SLOTS\n'
    printf -- '-u LLAMA_EXPERT_CPU_REUSE_ROWS\n'
    printf -- '-u LLAMA_EXPERT_CPU_MULTI_ROW\n'
    printf 'LLAMA_ARG_NO_CPU_MOE=1\n'
    printf 'GGML_CUDA_MOE_MULTI_FUSION=0\n'
    printf 'GGML_CUDA_MOE_COMBINE_FUSION=0\n'
    printf 'GGML_CUDA_ASYNC_HOST_COPY=1\n'
    printf 'GGML_SCHED_DEDUP_DST_SYNC=1\n'
    printf 'LLAMA_CMOE_PREFILL_BATCH=2560\n'
    printf 'LLAMA_CMOE_PREFILL_UBATCH=2560\n'
    printf 'LLAMA_CMOE_DECODE_BATCH=128\n'
    printf 'LLAMA_CMOE_DECODE_UBATCH=128\n'
    printf 'LLAMA_KVFLASH=12288\n'
    printf 'LLAMA_KVFLASH_MAX_POOL=12288\n'
    printf 'LLAMA_KVFLASH_POLICY=lru\n'
} > "$LING_ENV"

if [[ -z "$API_KEY" && -z "$API_KEY_FILE" ]]; then
    printf 'WARNUNG: API_KEY/API_KEY_FILE ist leer; der Dienst ist ohne Authentifizierung im LAN erreichbar.\n' >&2
fi

cat <<EOF
llama-wackMall-hybrid router 1080
  server:   $SERVER
  listen:   $HOST:$PORT
  max:      1 loaded model (LRU unload on switch)
  qwen:     $QWEN_NAME
            $QWEN_MODEL
  ling:     $LING_NAME
            $LING_MODEL
  preset:   $INI
  OpenWebUI model names: $QWEN_NAME  |  $LING_NAME
EOF

unset_args=(
    -u GGML_CUDA_DISABLE_GRAPHS
    -u LLAMA_ARG_MODEL
    -u LLAMA_ARG_MODELS_DIR
    -u LLAMA_TURBOQUANT_LIVE_MASK_LAYER
    -u LLAMA_TURBOQUANT_CAPTURE_VALUES
)
for key in "${UNSET_SHARED[@]}"; do
    unset_args+=(-u "$key")
done

server_args=(
    --models-preset "$INI"
    --models-max 1
    --host "$HOST"
    --port "$PORT"
    --cors-origins "$CORS_ORIGINS"
    --offline
    --no-ui
)
[[ -n "$API_KEY" ]] && server_args+=(--api-key "$API_KEY")
[[ -n "$API_KEY_FILE" ]] && server_args+=(--api-key-file "$API_KEY_FILE")

exec env "${unset_args[@]}" \
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE" \
    "LLAMA_CACHE=$RUNTIME/empty-cache" \
    "$SERVER" "${server_args[@]}"
