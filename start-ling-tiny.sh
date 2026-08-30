#!/usr/bin/env bash
#
# llama-wackMall-hybrid GTX 1660 Ti LAN/OpenWebUI launcher for Ling-3.0-tiny
#
# Same knob surface as start1660.sh. Edit only the block below, then rerun
# ./start-ling-tiny.sh. No CLI args and no silent shell overrides.
#
# Ling-3.0-tiny (bailingmoe3) on GTX 1660 Ti (sm_75, 6 GiB):
#   ngram-simple n=12 m=4 (m<n disables drafts), q8_0 KV, no-cpu-moe, --fit on,
#   ctx 131072, KVFlash 8192/8192, prefill 2048, decode 64, Q8 ncols1=4, Q4_K ncols1=2.
# Turbo4 is incompatible with MLA k=576. This GGUF has no MTP/DFlash sidecar.
# Hybrid S/W, DFlash, and the Qwen chat template stay in the block but off.
# Qwen production stack: start1660.sh. 1080 Ling recipe: start-ling-tiny-1080.sh.
#
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"  # directory containing this script

# ============================================================================
# EDITABLE CONFIGURATION -- only edit values in this section
# ============================================================================

# Paths
SERVER="$PROJECT_ROOT/build-main-sm75/bin/llama-server"  # current Turbo4-enabled native sm_75 Release executable
MODEL="$HOME/models/ling-3.0-tiny/Ling-3.0-tiny-Q4_K_M.gguf"  # target GGUF
SPEC_MODE="ngram"  # none | mtp | dflash | ngram; Ling winner is ngram-simple
SPEC_DRAFT_MODEL=""  # unused for ngram; set a sidecar only after switching SPEC_MODE to dflash
DFLASH_TARGET_TENSOR_OVERRIDE=""  # Qwen NextN override; empty for Ling
# Expert heat profiles (LLAMA_EXPERT_HOT). Unused while CPU_MOE=0.
#   general    = portable multi-domain corpus (~51.6% top-33 seed coverage)
#   specialist = phase-1 bench prompt match (~72.9% coverage; Qwen-overfit)
# Leave PROFILE_KIND empty for all-GPU Ling. Set general|specialist only with CPU_MOE=1.
PROFILE_GENERAL="$PROJECT_ROOT/benchmark-results/profile-corpus-train8-512-20260802T124500Z/general-profile.csv"
PROFILE_SPECIALIST="$PROJECT_ROOT/profiles/specialist-benchprompt.csv"
PROFILE_KIND=""  # empty | general | specialist -- empty skips expert hot seed
# Resolve immediately so LLAMA_EXPERT_HOT below sees the real path.
case "$PROFILE_KIND" in
    general)    PROFILE="$PROFILE_GENERAL" ;;
    specialist) PROFILE="$PROFILE_SPECIALIST" ;;
    *)          PROFILE="" ;;  # validated after die() is defined
esac
PLACEMENT=""  # leave empty for uniform S; variable placement not promoted for production

# Network / OpenWebUI
HOST="0.0.0.0"  # listen address; 0.0.0.0 exposes the API on all interfaces
PORT="8080"  # TCP port used by OpenWebUI and API clients
CORS_ORIGINS="*"  # allowed browser origins; restrict this for a non-local deployment
API_KEY=""  # inline API key; leave empty only on a trusted network
API_KEY_FILE=""  # optional file containing one or more API keys, one per line
MODEL_ALIAS="ling-tiny"  # model name advertised by the OpenAI-compatible API
CUDA_VISIBLE_DEVICES_VALUE="0"  # CUDA device list, e.g. 0 or 0,1
N_PARALLEL="1"  # number of simultaneous server slots; 1 keeps the tested path isolated
CPU_MOE="0"  # Ling fits on GPU; 0 sends --no-cpu-moe. 1 enables hybrid expert tiering.
N_GPU_LAYERS="99"  # all layers on GPU; unused auto-fit path is "auto"
MMPROJ_AUTO="0"  # do not load an optional multimodal projector for this text model
UI="0"  # 0 = --no-ui
CONT_BATCHING="1"  # continuous batching
OP_OFFLOAD="1"  # offload host tensor ops to the device
FIT="on"  # VRAM fit; Ling measured winner with FIT_TARGET=80
FIT_TARGET="80"  # MiB headroom per device for --fit
FIT_CTX="2048"  # minimum ctx --fit is allowed to choose

# Model, context, KV, and server behavior
CONTEXT="131072"  # maximum context tokens; raises KV memory when increased
N_PREDICT="32768"  # CLI/server generation limit; API requests may impose a lower limit
# Direkt editierbare KV-Presets (jeweils TARGET_TYPE_K und TARGET_TYPE_V):
#   q8_0/q8_0         = Ling measured path (MLA k=576; Turbo4 incompatible)
#   q4_0/q4_0         = lower VRAM
#   turbo4_k/turbo4_k = Qwen/DFlash path; keep off for Ling MLA
# Nur die beiden folgenden Zeilen aendern; Turbo4-Guards nur bei turbo4_k auf 1.
TARGET_TYPE_K="q8_0"  # Ling target K cache
TARGET_TYPE_V="q8_0"  # Ling target V cache
DRAFT_TYPE_K="q8_0"  # unused while SPEC_MODE=ngram
DRAFT_TYPE_V="q8_0"  # unused while SPEC_MODE=ngram
LLAMA_KV_Q4_SCALE="legacy"  # Q4 scale policy; inert with q8_0 KV
LLAMA_KVFLASH="8192"  # resident target-KV tokens; this is a token count, not a boolean switch
LLAMA_KVFLASH_MAX_POOL="8192"  # measured Ling pool; keep equal to resident
LLAMA_KVFLASH_TAU="64"  # scorer reselection interval; inert while the current LRU policy has no scorer
LLAMA_KVFLASH_POLICY="lru"  # only supported KVFlash replacement policy
LLAMA_KVFLASH_STATS="0"  # production default; set to 1 to report paging and host-memory counters
LLAMA_TURBO4_V_EXPERIMENTAL="0"  # required only for Turbo4 target V
LLAMA_TURBO4_MTP_EXPERIMENTAL="0"  # required only for target Turbo4 with draft-mtp
LLAMA_TURBO4_DFLASH_EXPERIMENTAL="0"  # required only for target Turbo4 with draft-dflash
LLAMA_TURBO4_DRAFT_EXPERIMENTAL="0"  # required only for Turbo4 in a draft context
LLAMA_TURBO4_Q8_FALLBACK_LAYERS=""  # optional comma/range list of target K layers kept at Q8, e.g. 23,35,27
GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH="2"  # tested MTP-2 crossover; inert without Turbo4
GGML_CUDA_TURBO4_FAST_F16_CONVERT="0"  # generic converter is the SM75 winner; 1/2 were slower experiments
GGML_CUDA_TURBO4_WHT_SHUFFLE="0"  # original WHT is the SM75 winner; shuffle-first was neutral/slower
MTP_N="2"  # inactive unless SPEC_MODE=mtp
LLAMA_MTP_REQUANTIZE_OUTPUT="none"  # none is the 6-GiB winner; unused without MTP
LLAMA_MTP_HEAD_TRACE="0"  # 1 prints the selected MTP-head source/type once; diagnostic only
SPEC_DRAFT_N_MAX="8"  # DFlash/MTP draft cap only; ngram-simple uses NGRAM_SIMPLE_SIZE_M
SPEC_DRAFT_N_MIN="1"  # DFlash-only minimum accepted draft length
SPEC_DRAFT_P_MIN="0.75"  # DFlash-only confidence threshold in [0, 1]
SPEC_DRAFT_BACKEND_SAMPLING="1"  # DFlash-only backend sampling switch
DFLASH_COMBINED="1"  # fuse target feature projection and draft KV injection; inert for ngram
DRAFT_NGL="all"  # DFlash/MTP draft layers on GPU; unused for ngram
NGRAM_SIMPLE_SIZE_N="12"  # lookup length (server default). ngram-simple returns no draft when size_m < size_n
NGRAM_SIMPLE_SIZE_M="4"  # with n=12 this disables ngram-simple and avoids 12-48 token verifies off the MMVQ/graph path
NGRAM_SIMPLE_MIN_HITS=""  # empty = server default
REASONING="1"  # let the model template select reasoning mode
REASONING_BUDGET="8000"  # Ling measured; start1660 Qwen uses 6000
REASONING_PRESERVE="1"  # preserve reasoning in history: 1 enabled, 0 disabled
LOAD_MODE="none"  # measured loading/runtime winner
OFFLINE="1"  # prevent network model/template downloads when set to 1
JINJA="1"  # use the model's Jinja chat template when set to 1
CHAT_TEMPLATE_FILE=""  # empty = GGUF metadata; do not point at the Qwen froggeric file
REASONING_FORMAT="auto"  # none | deepseek | deepseek-legacy | auto
FLASH_ATTN="on"  # Flash Attention mode: on, off, or auto
KV_OFFLOAD="1"  # place KV cache on the GPU when set to 1
THREADS="8"  # target generation CPU workers
THREADS_BATCH="8"  # CPU workers for prompt/batch processing
DRAFT_THREADS="8"  # CPU workers for a speculative draft context
DRAFT_THREADS_BATCH="8"  # draft workers for draft prompt/batch processing
THREADS_HTTP=""  # HTTP worker count; empty uses the server default
TARGET_BACKEND_SAMPLING="1"  # target sampling on CUDA; 0 provides an A/B control
DRAFT_BACKEND_SAMPLING="1"  # MTP/DFlash draft sampling on the backend
TEMP="1.0"  # Ling sampling; start1660 leaves server defaults
TOP_K="20"
TOP_P="0.95"
MIN_P="0"

# Prompt/decode batching. GPU_ARCH accepts auto, 6.1 (GTX1080), or 7.5
# (GTX1660Ti). CMOE_BATCH/CMOE_UBATCH accept a number or auto. The four phase
# values are empty by default; setting them enables the experimental prefill /
# decode switch implemented in common/arg.cpp.
GPU_ARCH="auto"  # architecture used to choose an auto batch default
CMOE_BATCH="64"  # base/decode fallback; phase decode below overrides when set
CMOE_UBATCH="64"  # physical base ubatch; keep equal to CMOE_BATCH
# Phase batching: large prefill for PP; small decode so graph peak can shrink
# after set_runtime_ubatch.
CMOE_PREFILL_BATCH="2048"  # Ling measured winner; 4096 only reserved VRAM, runtime clamped to 2048
CMOE_PREFILL_UBATCH="2048"  # physical target prompt ubatch
CMOE_DECODE_BATCH="64"  # logical batch during generation
CMOE_DECODE_UBATCH="64"  # physical decode ubatch; phase switch frees prefill peak

# Prompt-cache controls
CTX_CHECKPOINTS="8"  # think/tool/role/turn-end anchors; 4 evicted too quickly
CACHE_RAM="4096"  # host-side full-state prompt-cache budget (MiB), including KVFlash host pages
CACHE_PROMPT="1"  # enable prompt-prefix reuse when set to 1
CACHE_REUSE="0"  # keep 0 unless the architecture allows chunk KV shift
KV_UNIFIED="1"  # share one unified KV buffer between slots when set to 1
CACHE_IDLE_SLOTS="1"  # retain idle slots through the full-state RAM prompt cache

# Expert tier. These names are the actual LLAMA_EXPERT_* runtime variables.
# With CPU_MOE=0 the runtime ignores S/W; the knobs stay here to match start1660.
LLAMA_EXPERT_HOT="$PROFILE"  # ranking/usage CSV used to select fixed hot experts
LLAMA_EXPERT_S=""  # empty = unset; start1660 uses 20 with CPU_MOE=1
LLAMA_EXPERT_PLACEMENT="$PLACEMENT"  # validated per-layer slot manifest; mutually exclusive with S
LLAMA_EXPERT_TMAX="32"  # maximum token count for the tiered single-row path
LLAMA_EXPERT_STATS="0"  # 0 disables stats, 1 prints to stderr, a path writes a file
LLAMA_EXPERT_ADAPT="0"  # online repinning: 0 immutable tier, 1 learn and repin
LLAMA_EXPERT_DECAY="1.0"  # score decay per update; 1.0 is cumulative, lower forgets history
LLAMA_EXPERT_USAGE="0"  # production off; prevents request logging from creating a file named "1"
LLAMA_EXPERT_USAGE_MODE="session"  # cumulative includes the seed; session records this run only
LLAMA_EXPERT_USAGE_CHECKPOINT="request"  # write usage after request or only at exit
LLAMA_EXPERT_ADAPT_INTERVAL="request"  # publish repins at request or graph boundaries
LLAMA_EXPERT_ADAPT_CUDA_GRAPHS="0"  # unsafe diagnostic CUDA-graph reuse during repinning
LLAMA_EXPERT_STATS_JSON="0"  # 0 off, otherwise path for machine-readable statistics
LLAMA_EXPERT_TIMING="0"  # collect detailed CPU expert timing; adds measurement overhead
LLAMA_EXPERT_CPU_CHUNK="64"  # cold-row chunk size; power of two from 16 through 256
LLAMA_EXPERT_CPU_ACT_PARALLEL="0"  # experimental path did not establish a sustained winner
LLAMA_EXPERT_CPU_ASYNC="0"  # bridge/async path regressed on this GPU; keep synchronous fallback
LLAMA_EXPERT_CPU_DOWN_PREFETCH="1"  # no measured sustained gain on the local CPU
LLAMA_EXPERT_CPU_REUSE_ROWS="1"  # conservative winner
LLAMA_EXPERT_CPU_MULTI_ROW="1"  # AVX2 multi-row
LLAMA_EXPERT_CPU_FUSED_GATE_UP="1"  # exact AVX2 dual-dot

# Warmcache. Off while CPU_MOE=0. start1660 live: S=20 W=8 frequency/200 ratio=2.5.
LLAMA_EXPERT_BW_PROFILE="$PROJECT_ROOT/profiles/gtx1660-expert-bw.json"
LLAMA_EXPERT_WARM_SLOTS="0"  # extra LRU slots per layer; 0 while all experts stay on GPU
LLAMA_EXPERT_WARM_AUTO_MAX="8"  # cap for W=auto; raise only after VRAM measurements
LLAMA_EXPERT_WARM_POLICY="lru"  # replacement policy; currently only lru is supported
LLAMA_EXPERT_WARM_RESET="request"  # LRU ages reset per request; long think is one request
LLAMA_EXPERT_WARM_ADMISSION="frequency"  # live trial on Qwen; immediate thrashed on this GPU
LLAMA_EXPERT_WARM_ADMISSION_WINDOW="200"  # frequency half-life in graphs (~tokens in decode)
LLAMA_EXPERT_WARM_REPLACE_RATIO="2.5"  # candidate must beat occupant by 2.5x frequency
LLAMA_EXPERT_WARM_PREFETCH="0"  # async H2D; required only when W>0
LLAMA_EXPERT_PREFETCH_STREAMS="1"  # number of prefetch CUDA streams; current implementation requires 1
LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT="4"  # maximum simultaneous expert copies
LLAMA_EXPERT_VRAM_RESERVE_MIB="250"  # runtime auto-fit reserve; forced S remains capped if necessary
LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL="0"  # keep 0 if switching SPEC_MODE=mtp
LLAMA_EXPERT_STATIC_NO_SYNC="1"  # rejected automatically while W>0; harmless leftover

# Lookahead/bridge. These values enable diagnostics/experiments and are not a
# production recommendation until the Oracle, quality, and MTP gates pass.
LLAMA_EXPERT_LOOKAHEAD="0"  # predictor/bridge remains experimental and lost on the GTX 1660 Ti
LLAMA_EXPERT_LOOKAHEAD_TRACE="0"  # collect routing-only predictor traces when set to 1
LLAMA_EXPERT_LOOKAHEAD_TRACE_JSON="0"  # output JSON path required when TRACE=1
LLAMA_EXPERT_LOOKAHEAD_DISTANCE="1"  # diagnostic winner; inactive while LOOKAHEAD=0
LLAMA_EXPERT_LOOKAHEAD_TOP_M="12"  # best ranking pool; inactive while LOOKAHEAD=0
LLAMA_EXPERT_LOOKAHEAD_POINT="post-moe"  # predictor point: post-attn or post-moe
LLAMA_EXPERT_LOOKAHEAD_NORM="target"  # normalization source: target or source layer
LLAMA_EXPERT_LOOKAHEAD_PREFETCH="0"  # no productive lookahead transfers in the stable path
LLAMA_EXPERT_LOOKAHEAD_STREAMS="1"  # conservative diagnostic default
LLAMA_EXPERT_LOOKAHEAD_MAX_INFLIGHT="2"  # conservative diagnostic default
LLAMA_EXPERT_LOOKAHEAD_LAYER_MIN="0"  # inactive while LOOKAHEAD=0
LLAMA_EXPERT_LOOKAHEAD_LAYER_MAX="-1"  # last eligible layer; -1 means all layers
LLAMA_EXPERT_LOOKAHEAD_MTP_EXPERIMENTAL="0"  # retain the MTP lookahead guard
# Bridge controls. The fast static path leaves the layer lists empty because
# listing layers allocates bridge scratch and starts workers even with consume=0.
GGML_CUDA_EXPERT_BRIDGE_LAYERS=""  # empty disables bridge registration entirely
GGML_CUDA_EXPERT_BRIDGE_K="2"  # candidate limit used only after explicitly enabling bridge layers
GGML_CUDA_EXPERT_BRIDGE_RECURRENCE="0"  # keep recurrence limited to the explicit layer list below
GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_LAYERS=""  # optional recurrence layer list; empty in the production path
GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_K="1"  # candidate limit used only for enabled recurrence layers
GGML_CUDA_EXPERT_BRIDGE_CONSUME="0"  # GTX 1660 Ti bridge median regressed; production off
GGML_CUDA_EXPERT_BRIDGE_VERIFY="0"  # diagnostic recomputation; 1 is correctness-test-only and slower
GGML_CUDA_EXPERT_BRIDGE_JSON="0"  # 0 keeps bridge off; set an absolute JSON path to opt in
GGML_CUDA_EXPERT_BRIDGE_TELEMETRY="0"  # CUDA-event timing for bridge jobs; only meaningful with CONSUME=1
GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT="0"  # quantize streamed activations on CUDA; experimental and bridge-only
GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT_RECURRENCE="0"  # separately allow device quantization for recurrence jobs
GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY="0"  # execute only predicted actual hits; experimental correctness/performance probe
GGML_CUDA_EXPERT_BRIDGE_CACHE_LAYERS=""  # optional comma/range list using transient bridge cache slots
GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS="2"  # 2..16 and at least K; inactive while CACHE_LAYERS is empty

# TriAttention capture-only diagnostics are intentionally not server options.
# Use build-turbo-capture-sm75/bin/llama-turboquant-capture with
# --attention-live-mask, --attention-live-mask-continue,
# --dump-logits-window, and --logits-window-start. The launcher explicitly
# removes the two internal callback variables so they cannot leak into serving.

# Exact kernel controls. start1660 sm_75 winners plus Ling Q8 ncols1=4 / Q4_K ncols1=2.
LLAMA_EXPERT_SHARED_HOT_IDS="1"  # share one mapped hot-ID tensor across Gate, Up, and Down
LLAMA_EXPERT_SKIP_SENTINEL="1"  # 3x2K winner on Qwen hybrid; inert with no-cpu-moe
GGML_CUDA_MOE_MULTI_FUSION="1"  # fuse Gate+Up+GLU for 2-4 target tokens on sm_75+
GGML_CUDA_MOE_COMBINE_FUSION="1"  # exact combine fusion
GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS="4"  # start1660 / Ling
GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS="4"  # Ling measured; start1660 uses 0=auto
GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS="0"  # A/B candidate; 0=auto
GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS="2"  # Ling measured; start1660 does not set this
GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS="2"  # Q6_K output projection, one-token graph
GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARPS="0"  # 1/4-warp screens lost; 0 keeps the architecture default of 2
GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARP_ROWS="0"  # independent warp-row kernels did not beat the winner
GGML_CUDA_MMVQ_Q6_K_NCOLS1_REUSE_Y="0"  # shared Q8_1 unpack was exact but slightly slower on sm_75
GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS="4"  # Q6_K output projection, multi-token verify graph
GGML_CUDA_MMVQ_MOE_FUSED_ROWS="0"  # fused MoE rows/block sweep was neutral or slower
GGML_CUDA_MMVQ_MOE_PLAIN_ROWS="0"  # plain MoE rows/block sweep was neutral or slower
GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE="128"  # start1660 / Ling
GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0="1"  # start1660 / Ling
GGML_CUDA_ASYNC_HOST_COPY="1"  # pinned split H2D
GGML_SCHED_ASYNC_D2H_COPY="0"  # exact but below promotion gate
GGML_SCHED_DEDUP_DST_SYNC="1"  # exact 3x2K winner
GGML_CUDA_REGISTER_HOST="1"  # mmap pin; 21 GiB cudaHostRegister failed on 1660 Ti (host RAM, not VRAM)
GGML_SCHED_PREFETCH_EXPERTS="2"  # persistent staging; n-cpu-moe overlap. Inert without -ncmoe.

# ============================================================================
# End of editable configuration
# ============================================================================

die() {
    printf 'start-ling-tiny.sh: %s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 0 ]]; then
    die "Keine Kommandozeilenparameter: Einstellungen oben in start-ling-tiny.sh aendern."
fi

case "$PROFILE_KIND" in
    "")         PROFILE="" ;;
    general)    PROFILE="$PROFILE_GENERAL" ;;
    specialist) PROFILE="$PROFILE_SPECIALIST" ;;
    *)          die "PROFILE_KIND muss leer, general oder specialist sein (ist: $PROFILE_KIND)." ;;
esac
LLAMA_EXPERT_HOT="$PROFILE"

[[ -x "$SERVER" ]] || die "llama-server nicht ausfuehrbar: $SERVER"
[[ -f "$MODEL" ]] || die "Modell nicht gefunden: $MODEL"
if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
    [[ -f "$CHAT_TEMPLATE_FILE" ]] || die "Chat-Template nicht gefunden: $CHAT_TEMPLATE_FILE"
    [[ "$JINJA" == 1 ]] || die "CHAT_TEMPLATE_FILE benoetigt JINJA=1."
fi
case "$REASONING_FORMAT" in none|deepseek|deepseek-legacy|auto) ;; *) die "REASONING_FORMAT muss none, deepseek, deepseek-legacy oder auto sein." ;; esac
if [[ -n "$PROFILE_KIND" ]]; then
    [[ -f "$PROFILE" ]] || die "Expert-Profil nicht gefunden: $PROFILE (PROFILE_KIND=$PROFILE_KIND)"
fi
if [[ -n "$LLAMA_EXPERT_HOT" && ! -f "$LLAMA_EXPERT_HOT" ]]; then
    die "Expert-Profil nicht gefunden: $LLAMA_EXPERT_HOT"
fi
if [[ -n "$LLAMA_EXPERT_PLACEMENT" && ! -f "$LLAMA_EXPERT_PLACEMENT" ]]; then
    die "Placement-Manifest nicht gefunden: $LLAMA_EXPERT_PLACEMENT"
fi
if [[ -n "$API_KEY_FILE" && ! -f "$API_KEY_FILE" ]]; then
    die "API-Key-Datei nicht gefunden: $API_KEY_FILE"
fi
if [[ "$CPU_MOE" == 1 && -n "$LLAMA_EXPERT_BW_PROFILE" && ! -f "$LLAMA_EXPERT_BW_PROFILE" ]]; then
    die "Bandwidth-Profil nicht gefunden: $LLAMA_EXPERT_BW_PROFILE"
fi

case "$SPEC_MODE" in none|mtp|dflash|ngram|ngram-simple) ;; *) die "SPEC_MODE muss none, mtp, dflash oder ngram sein." ;; esac
case "$MTP_N" in 0|1|2|3) ;; *) die "MTP_N muss 0, 1, 2 oder 3 sein." ;; esac
[[ "$SPEC_DRAFT_N_MAX" =~ ^[1-9][0-9]*$ ]] || die "SPEC_DRAFT_N_MAX muss eine positive Ganzzahl sein."
[[ "$SPEC_DRAFT_N_MIN" =~ ^[0-9]+$ ]] || die "SPEC_DRAFT_N_MIN muss eine nichtnegative Ganzzahl sein."
(( SPEC_DRAFT_N_MIN <= SPEC_DRAFT_N_MAX )) || die "SPEC_DRAFT_N_MIN darf SPEC_DRAFT_N_MAX nicht ueberschreiten."
[[ "$SPEC_DRAFT_P_MIN" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || die "SPEC_DRAFT_P_MIN muss zwischen 0 und 1 liegen."
case "$SPEC_DRAFT_BACKEND_SAMPLING" in 0|1) ;; *) die "SPEC_DRAFT_BACKEND_SAMPLING muss 0 oder 1 sein." ;; esac
case "$DFLASH_COMBINED" in 0|1) ;; *) die "DFLASH_COMBINED muss 0 oder 1 sein." ;; esac
case "$TARGET_BACKEND_SAMPLING" in 0|1) ;; *) die "TARGET_BACKEND_SAMPLING muss 0 oder 1 sein." ;; esac
case "$DRAFT_BACKEND_SAMPLING" in 0|1) ;; *) die "DRAFT_BACKEND_SAMPLING muss 0 oder 1 sein." ;; esac
case "$REASONING_PRESERVE" in 0|1) ;; *) die "REASONING_PRESERVE muss 0 oder 1 sein." ;; esac
case "$LLAMA_TURBO4_V_EXPERIMENTAL" in 0|1) ;; *) die "LLAMA_TURBO4_V_EXPERIMENTAL muss 0 oder 1 sein." ;; esac
case "$LLAMA_TURBO4_MTP_EXPERIMENTAL" in 0|1) ;; *) die "LLAMA_TURBO4_MTP_EXPERIMENTAL muss 0 oder 1 sein." ;; esac
case "$LLAMA_TURBO4_DFLASH_EXPERIMENTAL" in 0|1) ;; *) die "LLAMA_TURBO4_DFLASH_EXPERIMENTAL muss 0 oder 1 sein." ;; esac
case "$LLAMA_TURBO4_DRAFT_EXPERIMENTAL" in 0|1) ;; *) die "LLAMA_TURBO4_DRAFT_EXPERIMENTAL muss 0 oder 1 sein." ;; esac
case "$LLAMA_KV_Q4_SCALE" in legacy|weighted|weighted-k|weighted-v) ;; *) die "LLAMA_KV_Q4_SCALE muss legacy, weighted, weighted-k oder weighted-v sein." ;; esac
[[ "$LLAMA_KVFLASH" == 0 || "$LLAMA_KVFLASH" == auto || "$LLAMA_KVFLASH" =~ ^[1-9][0-9]*$ ]] || \
    die "LLAMA_KVFLASH muss 0, auto oder eine positive Ganzzahl sein."
if [[ "$LLAMA_KVFLASH" =~ ^[1-9][0-9]*$ ]] && (( LLAMA_KVFLASH < 512 )); then
    die "LLAMA_KVFLASH ist eine Tokenanzahl (mindestens 512), kein Boolean."
fi
[[ "$LLAMA_KVFLASH_MAX_POOL" =~ ^[1-9][0-9]*$ ]] || die "LLAMA_KVFLASH_MAX_POOL muss eine positive Ganzzahl sein."
[[ "$LLAMA_KVFLASH_TAU" =~ ^[1-9][0-9]*$ ]] || die "LLAMA_KVFLASH_TAU muss eine positive Ganzzahl sein."
case "$LLAMA_KVFLASH_POLICY" in lru) ;; *) die "LLAMA_KVFLASH_POLICY muss lru sein." ;; esac
case "$LLAMA_KVFLASH_STATS" in 0|1) ;; *) die "LLAMA_KVFLASH_STATS muss 0 oder 1 sein." ;; esac
[[ "$CTX_CHECKPOINTS" =~ ^[0-9]+$ ]] || die "CTX_CHECKPOINTS muss eine nichtnegative Ganzzahl sein."
[[ "$CACHE_RAM" =~ ^(-1|[0-9]+)$ ]] || die "CACHE_RAM muss -1 oder eine nichtnegative Ganzzahl sein."
case "$CACHE_PROMPT" in 0|1) ;; *) die "CACHE_PROMPT muss 0 oder 1 sein." ;; esac
[[ "$CACHE_REUSE" =~ ^[0-9]+$ ]] || die "CACHE_REUSE muss eine nichtnegative Ganzzahl sein."
case "$KV_UNIFIED" in 0|1) ;; *) die "KV_UNIFIED muss 0 oder 1 sein." ;; esac
case "$CACHE_IDLE_SLOTS" in 0|1) ;; *) die "CACHE_IDLE_SLOTS muss 0 oder 1 sein." ;; esac
if [[ "$CACHE_IDLE_SLOTS" == 1 && "$CACHE_RAM" == 0 ]]; then
    die "CACHE_IDLE_SLOTS=1 benoetigt CACHE_RAM ungleich 0."
fi
if [[ "$CACHE_REUSE" != 0 && "$CACHE_PROMPT" != 1 ]]; then
    die "CACHE_REUSE groesser 0 benoetigt CACHE_PROMPT=1."
fi
case "$LLAMA_MTP_REQUANTIZE_OUTPUT" in none|q4_K|q4_k|q5_K|q5_k) ;; *) die "LLAMA_MTP_REQUANTIZE_OUTPUT muss none, q4_K oder q5_K sein." ;; esac
case "$LLAMA_MTP_HEAD_TRACE" in 0|1) ;; *) die "LLAMA_MTP_HEAD_TRACE muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_TURBO4_FAST_F16_CONVERT" in 0|1|2) ;; *) die "GGML_CUDA_TURBO4_FAST_F16_CONVERT muss 0, 1 oder 2 sein." ;; esac
case "$GGML_CUDA_TURBO4_WHT_SHUFFLE" in 0|1) ;; *) die "GGML_CUDA_TURBO4_WHT_SHUFFLE muss 0 oder 1 sein." ;; esac
[[ "$GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH" =~ ^[1-9][0-9]*$ ]] || \
    die "GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH muss eine positive Ganzzahl sein."
case "$LLAMA_EXPERT_CPU_FUSED_GATE_UP" in 0|1) ;; *) die "LLAMA_EXPERT_CPU_FUSED_GATE_UP muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_EXPERT_BRIDGE_TELEMETRY" in 0|1) ;; *) die "GGML_CUDA_EXPERT_BRIDGE_TELEMETRY muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT" in 0|1) ;; *) die "GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT_RECURRENCE" in 0|1) ;; *) die "GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT_RECURRENCE muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY" in 0|1) ;; *) die "GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY muss 0 oder 1 sein." ;; esac
[[ "$GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS" =~ ^([2-9]|1[0-6])$ ]] || \
    die "GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS muss zwischen 2 und 16 liegen."
case "$LLAMA_EXPERT_SHARED_HOT_IDS" in 0|1) ;; *) die "LLAMA_EXPERT_SHARED_HOT_IDS muss 0 oder 1 sein." ;; esac
case "$LLAMA_EXPERT_SKIP_SENTINEL" in 0|1) ;; *) die "LLAMA_EXPERT_SKIP_SENTINEL muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_MOE_MULTI_FUSION" in 0|1) ;; *) die "GGML_CUDA_MOE_MULTI_FUSION muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_MOE_COMBINE_FUSION" in 0|1) ;; *) die "GGML_CUDA_MOE_COMBINE_FUSION muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS muss 0, 1, 2 oder 4 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS muss 0, 1, 2 oder 4 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS muss 0, 1, 2 oder 4 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS muss 0, 1, 2 oder 4 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS muss 0, 1, 2 oder 4 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARPS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARPS muss 0, 1, 2 oder 4 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARP_ROWS" in 0|2|4|8|16) ;; *) die "GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARP_ROWS muss 0, 2, 4, 8 oder 16 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q6_K_NCOLS1_REUSE_Y" in 0|1) ;; *) die "GGML_CUDA_MMVQ_Q6_K_NCOLS1_REUSE_Y muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS muss 0, 1, 2 oder 4 sein." ;; esac
case "$GGML_CUDA_MMVQ_MOE_FUSED_ROWS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_MOE_FUSED_ROWS muss 0, 1, 2 oder 4 sein." ;; esac
case "$GGML_CUDA_MMVQ_MOE_PLAIN_ROWS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_MOE_PLAIN_ROWS muss 0, 1, 2 oder 4 sein." ;; esac
case "$GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE" in 0|32|64|128|256) ;; *) die "GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE muss 0, 32, 64, 128 oder 256 sein." ;; esac
case "$GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0" in 0|1) ;; *) die "GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0 muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_ASYNC_HOST_COPY" in 0|1) ;; *) die "GGML_CUDA_ASYNC_HOST_COPY muss 0 oder 1 sein." ;; esac
case "$GGML_SCHED_ASYNC_D2H_COPY" in 0|1) ;; *) die "GGML_SCHED_ASYNC_D2H_COPY muss 0 oder 1 sein." ;; esac
case "$GGML_SCHED_DEDUP_DST_SYNC" in 0|1) ;; *) die "GGML_SCHED_DEDUP_DST_SYNC muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_REGISTER_HOST" in 0|1) ;; *) die "GGML_CUDA_REGISTER_HOST muss 0 oder 1 sein." ;; esac
case "$GGML_SCHED_PREFETCH_EXPERTS" in 0|1|2|3|4|5|6|7|8) ;; *) die "GGML_SCHED_PREFETCH_EXPERTS muss 0 oder 1..8 sein." ;; esac
case "$CPU_MOE" in 0|1) ;; *) die "CPU_MOE muss 0 oder 1 sein." ;; esac
case "$UI" in 0|1) ;; *) die "UI muss 0 oder 1 sein." ;; esac
case "$CONT_BATCHING" in 0|1) ;; *) die "CONT_BATCHING muss 0 oder 1 sein." ;; esac
case "$OP_OFFLOAD" in 0|1) ;; *) die "OP_OFFLOAD muss 0 oder 1 sein." ;; esac
case "$FIT" in on|off|1|0|true|false) ;; *) die "FIT muss on oder off sein." ;; esac
[[ "$FIT_TARGET" =~ ^[0-9]+$ ]] || die "FIT_TARGET muss eine nichtnegative Ganzzahl sein."
[[ "$FIT_CTX" =~ ^[1-9][0-9]*$ ]] || die "FIT_CTX muss eine positive Ganzzahl sein."
[[ -z "$NGRAM_SIMPLE_SIZE_N" || "$NGRAM_SIMPLE_SIZE_N" =~ ^[1-9][0-9]*$ ]] || die "NGRAM_SIMPLE_SIZE_N muss leer oder eine positive Ganzzahl sein."
[[ -z "$NGRAM_SIMPLE_SIZE_M" || "$NGRAM_SIMPLE_SIZE_M" =~ ^[1-9][0-9]*$ ]] || die "NGRAM_SIMPLE_SIZE_M muss leer oder eine positive Ganzzahl sein."
[[ -z "$NGRAM_SIMPLE_MIN_HITS" || "$NGRAM_SIMPLE_MIN_HITS" =~ ^[1-9][0-9]*$ ]] || die "NGRAM_SIMPLE_MIN_HITS muss leer oder eine positive Ganzzahl sein."
[[ -z "$LLAMA_EXPERT_S" || -z "$LLAMA_EXPERT_PLACEMENT" ]] || \
    die "LLAMA_EXPERT_S und LLAMA_EXPERT_PLACEMENT sind gegenseitig exklusiv."
if [[ "$LLAMA_EXPERT_LOOKAHEAD_TRACE" != 0 && "$LLAMA_EXPERT_LOOKAHEAD_TRACE_JSON" == 0 ]]; then
    die "LLAMA_EXPERT_LOOKAHEAD_TRACE=1 benoetigt einen JSON-Pfad."
fi
if [[ "$TARGET_TYPE_V" == turbo4_k && "$LLAMA_TURBO4_V_EXPERIMENTAL" != 1 ]]; then
    die "TARGET_TYPE_V=turbo4_k benoetigt LLAMA_TURBO4_V_EXPERIMENTAL=1."
fi
if [[ ( "$TARGET_TYPE_K" == turbo4_k || "$TARGET_TYPE_V" == turbo4_k ) && "$FLASH_ATTN" != on ]]; then
    die "Turbo4-KV benoetigt FLASH_ATTN=on."
fi
if [[ "$LLAMA_KVFLASH" != 0 && "$FLASH_ATTN" != on ]]; then
    die "KVFlash benoetigt FLASH_ATTN=on."
fi
if [[ "$LLAMA_KVFLASH" != 0 && "$N_PARALLEL" != 1 ]]; then
    die "KVFlash benoetigt N_PARALLEL=1."
fi
if [[ "$SPEC_MODE" == mtp && "$MTP_N" != 0 && ( "$TARGET_TYPE_K" == turbo4_k || "$TARGET_TYPE_V" == turbo4_k ) && "$LLAMA_TURBO4_MTP_EXPERIMENTAL" != 1 ]]; then
    die "Target-Turbo4 mit MTP benoetigt LLAMA_TURBO4_MTP_EXPERIMENTAL=1."
fi
if [[ "$SPEC_MODE" == dflash && ( "$TARGET_TYPE_K" == turbo4_k || "$TARGET_TYPE_V" == turbo4_k ) && "$LLAMA_TURBO4_DFLASH_EXPERIMENTAL" != 1 ]]; then
    die "Target-Turbo4 mit DFlash benoetigt LLAMA_TURBO4_DFLASH_EXPERIMENTAL=1."
fi
if [[ ( "$DRAFT_TYPE_K" == turbo4_k || "$DRAFT_TYPE_V" == turbo4_k ) && "$LLAMA_TURBO4_DRAFT_EXPERIMENTAL" != 1 ]]; then
    die "Draft-Turbo4 (MTP oder DFlash) benoetigt LLAMA_TURBO4_DRAFT_EXPERIMENTAL=1."
fi
if [[ -n "$LLAMA_TURBO4_Q8_FALLBACK_LAYERS" && "$TARGET_TYPE_K" != turbo4_k ]]; then
    die "LLAMA_TURBO4_Q8_FALLBACK_LAYERS benoetigt TARGET_TYPE_K=turbo4_k."
fi

# Resolve only the two values explicitly marked auto in the file.
if [[ "$GPU_ARCH" == auto ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 { print; exit }' || true)"
    fi
fi
if [[ "$CMOE_BATCH" == auto ]]; then
    case "$GPU_ARCH" in
        6.1) CMOE_BATCH="128" ;;
        7.5) CMOE_BATCH="376" ;;
        *)   CMOE_BATCH="256" ;;
    esac
fi
if [[ "$CMOE_UBATCH" == auto ]]; then
    CMOE_UBATCH="$CMOE_BATCH"
fi
[[ "$CMOE_BATCH" =~ ^[1-9][0-9]*$ ]] || die "CMOE_BATCH muss eine positive Ganzzahl oder auto sein."
[[ "$CMOE_UBATCH" =~ ^[1-9][0-9]*$ ]] || die "CMOE_UBATCH muss eine positive Ganzzahl oder auto sein."
(( CMOE_UBATCH <= CMOE_BATCH )) || die "CMOE_UBATCH darf CMOE_BATCH nicht ueberschreiten."

for phase_value in "$CMOE_PREFILL_BATCH" "$CMOE_PREFILL_UBATCH" "$CMOE_DECODE_BATCH" "$CMOE_DECODE_UBATCH"; do
    [[ -z "$phase_value" || "$phase_value" =~ ^[1-9][0-9]*$ ]] || die "Phase-Batchwerte muessen leer oder positive Ganzzahlen sein."
done

case "$SPEC_MODE" in
    none)
        SPEC_TYPE="none"
        EFFECTIVE_SPEC_DRAFT_N_MAX="0"
        EFFECTIVE_SPEC_DRAFT_BACKEND_SAMPLING="$DRAFT_BACKEND_SAMPLING"
        ;;
    mtp)
        if [[ "$MTP_N" == 0 ]]; then
            SPEC_TYPE="none"
        else
            SPEC_TYPE="draft-mtp"
        fi
        EFFECTIVE_SPEC_DRAFT_N_MAX="$MTP_N"
        EFFECTIVE_SPEC_DRAFT_BACKEND_SAMPLING="$DRAFT_BACKEND_SAMPLING"
        ;;
    dflash)
        [[ -n "$SPEC_DRAFT_MODEL" ]] || die "SPEC_DRAFT_MODEL darf fuer SPEC_MODE=dflash nicht leer sein."
        [[ -f "$SPEC_DRAFT_MODEL" ]] || die "DFlash-Modell nicht gefunden: $SPEC_DRAFT_MODEL"
        SPEC_TYPE="draft-dflash"
        EFFECTIVE_SPEC_DRAFT_N_MAX="$SPEC_DRAFT_N_MAX"
        EFFECTIVE_SPEC_DRAFT_BACKEND_SAMPLING="$SPEC_DRAFT_BACKEND_SAMPLING"
        ;;
    ngram|ngram-simple)
        SPEC_TYPE="ngram-simple"
        EFFECTIVE_SPEC_DRAFT_N_MAX="$SPEC_DRAFT_N_MAX"
        EFFECTIVE_SPEC_DRAFT_BACKEND_SAMPLING="$DRAFT_BACKEND_SAMPLING"
        ;;
esac

# Explicit environment passed to the server. Optional variables are removed
# first so a caller's shell environment cannot silently change this file's
# configuration.
env_args=(
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE"
    "LLAMA_CMOE_BATCH=$CMOE_BATCH"
    "LLAMA_CMOE_UBATCH=$CMOE_UBATCH"
    "LLAMA_ARG_CTX_SIZE=$CONTEXT"
    "LLAMA_ARG_N_PREDICT=$N_PREDICT"
    "LLAMA_ARG_BATCH=$CMOE_BATCH"
    "LLAMA_ARG_UBATCH=$CMOE_UBATCH"
    "LLAMA_ARG_THREADS=$THREADS"
    "LLAMA_ARG_THREADS_BATCH=$THREADS_BATCH"
    "LLAMA_ARG_N_PARALLEL=$N_PARALLEL"
    "LLAMA_ARG_N_GPU_LAYERS=$N_GPU_LAYERS"
    "LLAMA_ARG_MMPROJ_AUTO=$MMPROJ_AUTO"
    "LLAMA_ARG_LOAD_MODE=$LOAD_MODE"
    "LLAMA_ARG_CACHE_TYPE_K=$TARGET_TYPE_K"
    "LLAMA_ARG_CACHE_TYPE_V=$TARGET_TYPE_V"
    "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K=$DRAFT_TYPE_K"
    "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V=$DRAFT_TYPE_V"
    "LLAMA_KV_Q4_SCALE=$LLAMA_KV_Q4_SCALE"
    "LLAMA_KVFLASH=$LLAMA_KVFLASH"
    "LLAMA_KVFLASH_MAX_POOL=$LLAMA_KVFLASH_MAX_POOL"
    "LLAMA_KVFLASH_TAU=$LLAMA_KVFLASH_TAU"
    "LLAMA_KVFLASH_POLICY=$LLAMA_KVFLASH_POLICY"
    "LLAMA_KVFLASH_STATS=$LLAMA_KVFLASH_STATS"
    "LLAMA_TURBO4_V_EXPERIMENTAL=$LLAMA_TURBO4_V_EXPERIMENTAL"
    "LLAMA_TURBO4_MTP_EXPERIMENTAL=$LLAMA_TURBO4_MTP_EXPERIMENTAL"
    "LLAMA_TURBO4_DFLASH_EXPERIMENTAL=$LLAMA_TURBO4_DFLASH_EXPERIMENTAL"
    "LLAMA_TURBO4_DRAFT_EXPERIMENTAL=$LLAMA_TURBO4_DRAFT_EXPERIMENTAL"
    "GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH=$GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH"
    "GGML_CUDA_TURBO4_FAST_F16_CONVERT=$GGML_CUDA_TURBO4_FAST_F16_CONVERT"
    "GGML_CUDA_TURBO4_WHT_SHUFFLE=$GGML_CUDA_TURBO4_WHT_SHUFFLE"
    "LLAMA_ARG_SPEC_TYPE=$SPEC_TYPE"
    "LLAMA_ARG_SPEC_DRAFT_N_MAX=$EFFECTIVE_SPEC_DRAFT_N_MAX"
    "LLAMA_MTP_REQUANTIZE_OUTPUT=$LLAMA_MTP_REQUANTIZE_OUTPUT"
    "LLAMA_MTP_HEAD_TRACE=$LLAMA_MTP_HEAD_TRACE"
    "LLAMA_ARG_N_GPU_LAYERS_DRAFT=$DRAFT_NGL"
    "LLAMA_ARG_SPEC_DRAFT_THREADS=$DRAFT_THREADS"
    "LLAMA_ARG_SPEC_DRAFT_THREADS_BATCH=$DRAFT_THREADS_BATCH"
    "LLAMA_ARG_BACKEND_SAMPLING=$TARGET_BACKEND_SAMPLING"
    "LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING=$EFFECTIVE_SPEC_DRAFT_BACKEND_SAMPLING"
    "LLAMA_ARG_REASONING=$REASONING"
    "LLAMA_ARG_THINK_BUDGET=$REASONING_BUDGET"
    "LLAMA_ARG_REASONING_PRESERVE=$REASONING_PRESERVE"
    "LLAMA_ARG_THINK=$REASONING_FORMAT"
    "LLAMA_ARG_FLASH_ATTN=$FLASH_ATTN"
    "LLAMA_ARG_KV_OFFLOAD=$KV_OFFLOAD"
    "LLAMA_ARG_OFFLINE=$OFFLINE"
    "LLAMA_ARG_JINJA=$JINJA"
    "LLAMA_ARG_CTX_CHECKPOINTS=$CTX_CHECKPOINTS"
    "LLAMA_ARG_CACHE_RAM=$CACHE_RAM"
    "LLAMA_ARG_CACHE_PROMPT=$CACHE_PROMPT"
    "LLAMA_ARG_CACHE_REUSE=$CACHE_REUSE"
    "LLAMA_ARG_KV_UNIFIED=$KV_UNIFIED"
    "LLAMA_ARG_CACHE_IDLE_SLOTS=$CACHE_IDLE_SLOTS"
    "LLAMA_ARG_ALIAS=$MODEL_ALIAS"
    "LLAMA_ARG_HOST=$HOST"
    "LLAMA_ARG_PORT=$PORT"
    "LLAMA_ARG_CORS_ORIGINS=$CORS_ORIGINS"
    "LLAMA_ARG_UI=$UI"
    "LLAMA_ARG_CONT_BATCHING=$CONT_BATCHING"
    "LLAMA_ARG_FIT=$FIT"
    "LLAMA_ARG_FIT_TARGET=$FIT_TARGET"
    "LLAMA_ARG_FIT_CTX=$FIT_CTX"
    "LLAMA_ARG_TOP_K=$TOP_K"
    "LLAMA_EXPERT_HOT=$LLAMA_EXPERT_HOT"
    "LLAMA_EXPERT_PLACEMENT=$LLAMA_EXPERT_PLACEMENT"
    "LLAMA_EXPERT_TMAX=$LLAMA_EXPERT_TMAX"
    "LLAMA_EXPERT_STATS=$LLAMA_EXPERT_STATS"
    "LLAMA_EXPERT_ADAPT=$LLAMA_EXPERT_ADAPT"
    "LLAMA_EXPERT_DECAY=$LLAMA_EXPERT_DECAY"
    "LLAMA_EXPERT_USAGE=$LLAMA_EXPERT_USAGE"
    "LLAMA_EXPERT_ADAPT_INTERVAL=$LLAMA_EXPERT_ADAPT_INTERVAL"
    "LLAMA_EXPERT_ADAPT_CUDA_GRAPHS=$LLAMA_EXPERT_ADAPT_CUDA_GRAPHS"
    "LLAMA_EXPERT_STATS_JSON=$LLAMA_EXPERT_STATS_JSON"
    "LLAMA_EXPERT_TIMING=$LLAMA_EXPERT_TIMING"
    "LLAMA_EXPERT_CPU_CHUNK=$LLAMA_EXPERT_CPU_CHUNK"
    "LLAMA_EXPERT_CPU_ACT_PARALLEL=$LLAMA_EXPERT_CPU_ACT_PARALLEL"
    "LLAMA_EXPERT_CPU_ASYNC=$LLAMA_EXPERT_CPU_ASYNC"
    "LLAMA_EXPERT_CPU_DOWN_PREFETCH=$LLAMA_EXPERT_CPU_DOWN_PREFETCH"
    "LLAMA_EXPERT_CPU_REUSE_ROWS=$LLAMA_EXPERT_CPU_REUSE_ROWS"
    "LLAMA_EXPERT_CPU_MULTI_ROW=$LLAMA_EXPERT_CPU_MULTI_ROW"
    "LLAMA_EXPERT_CPU_FUSED_GATE_UP=$LLAMA_EXPERT_CPU_FUSED_GATE_UP"
    "LLAMA_EXPERT_WARM_SLOTS=$LLAMA_EXPERT_WARM_SLOTS"
    "LLAMA_EXPERT_BW_PROFILE=$LLAMA_EXPERT_BW_PROFILE"
    "LLAMA_EXPERT_WARM_AUTO_MAX=$LLAMA_EXPERT_WARM_AUTO_MAX"
    "LLAMA_EXPERT_WARM_PREFETCH=$LLAMA_EXPERT_WARM_PREFETCH"
    "LLAMA_EXPERT_VRAM_RESERVE_MIB=$LLAMA_EXPERT_VRAM_RESERVE_MIB"
    "LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL=$LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL"
    "LLAMA_EXPERT_STATIC_NO_SYNC=$LLAMA_EXPERT_STATIC_NO_SYNC"
    "LLAMA_EXPERT_LOOKAHEAD=$LLAMA_EXPERT_LOOKAHEAD"
    "LLAMA_EXPERT_LOOKAHEAD_TRACE=$LLAMA_EXPERT_LOOKAHEAD_TRACE"
    "LLAMA_EXPERT_LOOKAHEAD_TRACE_JSON=$LLAMA_EXPERT_LOOKAHEAD_TRACE_JSON"
    "LLAMA_EXPERT_LOOKAHEAD_DISTANCE=$LLAMA_EXPERT_LOOKAHEAD_DISTANCE"
    "LLAMA_EXPERT_LOOKAHEAD_TOP_M=$LLAMA_EXPERT_LOOKAHEAD_TOP_M"
    "LLAMA_EXPERT_LOOKAHEAD_POINT=$LLAMA_EXPERT_LOOKAHEAD_POINT"
    "LLAMA_EXPERT_LOOKAHEAD_NORM=$LLAMA_EXPERT_LOOKAHEAD_NORM"
    "LLAMA_EXPERT_LOOKAHEAD_PREFETCH=$LLAMA_EXPERT_LOOKAHEAD_PREFETCH"
    "LLAMA_EXPERT_LOOKAHEAD_STREAMS=$LLAMA_EXPERT_LOOKAHEAD_STREAMS"
    "LLAMA_EXPERT_LOOKAHEAD_MAX_INFLIGHT=$LLAMA_EXPERT_LOOKAHEAD_MAX_INFLIGHT"
    "LLAMA_EXPERT_LOOKAHEAD_LAYER_MIN=$LLAMA_EXPERT_LOOKAHEAD_LAYER_MIN"
    "LLAMA_EXPERT_LOOKAHEAD_LAYER_MAX=$LLAMA_EXPERT_LOOKAHEAD_LAYER_MAX"
    "LLAMA_EXPERT_LOOKAHEAD_MTP_EXPERIMENTAL=$LLAMA_EXPERT_LOOKAHEAD_MTP_EXPERIMENTAL"
    "GGML_CUDA_EXPERT_BRIDGE_LAYERS=$GGML_CUDA_EXPERT_BRIDGE_LAYERS"
    "GGML_CUDA_EXPERT_BRIDGE_K=$GGML_CUDA_EXPERT_BRIDGE_K"
    "GGML_CUDA_EXPERT_BRIDGE_JSON=$GGML_CUDA_EXPERT_BRIDGE_JSON"
    "GGML_CUDA_EXPERT_BRIDGE_RECURRENCE=$GGML_CUDA_EXPERT_BRIDGE_RECURRENCE"
    "GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_LAYERS=$GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_LAYERS"
    "GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_K=$GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_K"
    "GGML_CUDA_EXPERT_BRIDGE_CONSUME=$GGML_CUDA_EXPERT_BRIDGE_CONSUME"
    "GGML_CUDA_EXPERT_BRIDGE_VERIFY=$GGML_CUDA_EXPERT_BRIDGE_VERIFY"
    "GGML_CUDA_EXPERT_BRIDGE_TELEMETRY=$GGML_CUDA_EXPERT_BRIDGE_TELEMETRY"
    "GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT=$GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT"
    "GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT_RECURRENCE=$GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT_RECURRENCE"
    "GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY=$GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY"
    "GGML_CUDA_EXPERT_BRIDGE_CACHE_LAYERS=$GGML_CUDA_EXPERT_BRIDGE_CACHE_LAYERS"
    "GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS=$GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS"
    "LLAMA_EXPERT_SHARED_HOT_IDS=$LLAMA_EXPERT_SHARED_HOT_IDS"
    "LLAMA_EXPERT_SKIP_SENTINEL=$LLAMA_EXPERT_SKIP_SENTINEL"
    "GGML_CUDA_MOE_MULTI_FUSION=$GGML_CUDA_MOE_MULTI_FUSION"
    "GGML_CUDA_MOE_COMBINE_FUSION=$GGML_CUDA_MOE_COMBINE_FUSION"
    "GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS"
    "GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS"
    "GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS"
    "GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARPS=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARPS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARP_ROWS=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARP_ROWS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS1_REUSE_Y=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_REUSE_Y"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=$GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS"
    "GGML_CUDA_MMVQ_MOE_FUSED_ROWS=$GGML_CUDA_MMVQ_MOE_FUSED_ROWS"
    "GGML_CUDA_MMVQ_MOE_PLAIN_ROWS=$GGML_CUDA_MMVQ_MOE_PLAIN_ROWS"
    "GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE=$GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE"
    "GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0=$GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0"
    "GGML_CUDA_ASYNC_HOST_COPY=$GGML_CUDA_ASYNC_HOST_COPY"
    "GGML_SCHED_ASYNC_D2H_COPY=$GGML_SCHED_ASYNC_D2H_COPY"
    "GGML_SCHED_DEDUP_DST_SYNC=$GGML_SCHED_DEDUP_DST_SYNC"
    "GGML_CUDA_REGISTER_HOST=$GGML_CUDA_REGISTER_HOST"
    "GGML_SCHED_PREFETCH_EXPERTS=$GGML_SCHED_PREFETCH_EXPERTS"
    "LLAMA_DFLASH_COMBINED=$DFLASH_COMBINED"
)

if [[ "$CPU_MOE" == 1 ]]; then
    env_args+=("LLAMA_ARG_CPU_MOE=1")
else
    env_args+=("LLAMA_ARG_NO_CPU_MOE=1")
fi

if [[ "$SPEC_MODE" == dflash ]]; then
    env_args+=(
        "LLAMA_ARG_SPEC_DRAFT_MODEL=$SPEC_DRAFT_MODEL"
        "LLAMA_ARG_SPEC_DRAFT_N_MIN=$SPEC_DRAFT_N_MIN"
        "LLAMA_ARG_SPEC_DRAFT_P_MIN=$SPEC_DRAFT_P_MIN"
    )
    [[ -n "$DFLASH_TARGET_TENSOR_OVERRIDE" ]] && env_args+=("LLAMA_ARG_OVERRIDE_TENSOR=$DFLASH_TARGET_TENSOR_OVERRIDE")
fi

# Phase-specific batching is opt-in by editing the four values above.
[[ -n "$CMOE_PREFILL_BATCH" ]] && env_args+=("LLAMA_CMOE_PREFILL_BATCH=$CMOE_PREFILL_BATCH")
[[ -n "$CMOE_PREFILL_UBATCH" ]] && env_args+=("LLAMA_CMOE_PREFILL_UBATCH=$CMOE_PREFILL_UBATCH")
[[ -n "$CMOE_DECODE_BATCH" ]] && env_args+=("LLAMA_CMOE_DECODE_BATCH=$CMOE_DECODE_BATCH")
[[ -n "$CMOE_DECODE_UBATCH" ]] && env_args+=("LLAMA_CMOE_DECODE_UBATCH=$CMOE_DECODE_UBATCH")
[[ -n "$THREADS_HTTP" ]] && env_args+=("LLAMA_ARG_THREADS_HTTP=$THREADS_HTTP")
[[ -n "$LLAMA_TURBO4_Q8_FALLBACK_LAYERS" ]] && env_args+=("LLAMA_TURBO4_Q8_FALLBACK_LAYERS=$LLAMA_TURBO4_Q8_FALLBACK_LAYERS")
[[ -n "$CHAT_TEMPLATE_FILE" ]] && env_args+=("LLAMA_ARG_CHAT_TEMPLATE_FILE=$CHAT_TEMPLATE_FILE")

if [[ -n "$LLAMA_EXPERT_S" ]]; then
    env_args+=("LLAMA_EXPERT_S=$LLAMA_EXPERT_S")
fi
if [[ "$LLAMA_EXPERT_USAGE" != 0 ]]; then
    env_args+=(
        "LLAMA_EXPERT_USAGE_MODE=$LLAMA_EXPERT_USAGE_MODE"
        "LLAMA_EXPERT_USAGE_CHECKPOINT=$LLAMA_EXPERT_USAGE_CHECKPOINT"
    )
fi
if [[ "$LLAMA_EXPERT_WARM_SLOTS" != 0 ]]; then
    env_args+=(
        "LLAMA_EXPERT_WARM_POLICY=$LLAMA_EXPERT_WARM_POLICY"
        "LLAMA_EXPERT_WARM_RESET=$LLAMA_EXPERT_WARM_RESET"
        "LLAMA_EXPERT_WARM_ADMISSION=$LLAMA_EXPERT_WARM_ADMISSION"
        "LLAMA_EXPERT_WARM_ADMISSION_WINDOW=$LLAMA_EXPERT_WARM_ADMISSION_WINDOW"
        "LLAMA_EXPERT_WARM_REPLACE_RATIO=$LLAMA_EXPERT_WARM_REPLACE_RATIO"
    )
    if [[ "$LLAMA_EXPERT_WARM_PREFETCH" == 1 ]]; then
        env_args+=(
            "LLAMA_EXPERT_PREFETCH_STREAMS=$LLAMA_EXPERT_PREFETCH_STREAMS"
            "LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT=$LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT"
        )
    fi
fi

server_args=(
    -m "$MODEL"
    --temp "$TEMP"
    --top-p "$TOP_P"
    --min-p "$MIN_P"
)
[[ "$OP_OFFLOAD" == 1 ]] && server_args+=(--op-offload)
[[ -n "$NGRAM_SIMPLE_SIZE_N" ]] && server_args+=(--spec-ngram-simple-size-n "$NGRAM_SIMPLE_SIZE_N")
[[ -n "$NGRAM_SIMPLE_SIZE_M" ]] && server_args+=(--spec-ngram-simple-size-m "$NGRAM_SIMPLE_SIZE_M")
[[ -n "$NGRAM_SIMPLE_MIN_HITS" ]] && server_args+=(--spec-ngram-simple-min-hits "$NGRAM_SIMPLE_MIN_HITS")
[[ -n "$API_KEY" ]] && server_args+=(--api-key "$API_KEY")
[[ -n "$API_KEY_FILE" ]] && server_args+=(--api-key-file "$API_KEY_FILE")

if [[ -z "$API_KEY" && -z "$API_KEY_FILE" ]]; then
    printf 'WARNUNG: API_KEY/API_KEY_FILE ist leer; der Dienst ist ohne Authentifizierung im LAN erreichbar.\n' >&2
fi
if [[ "$CPU_MOE" == 1 && -z "$LLAMA_EXPERT_HOT" ]]; then
    printf 'WARNUNG: LLAMA_EXPERT_HOT ist leer; es wird ohne gelerntes Profil gestartet.\n' >&2
fi

cat <<EOF
llama-wackMall-hybrid start (ling-tiny)
  project:      $PROJECT_ROOT
  server:       $SERVER
  model:        $MODEL
  profile:      ${LLAMA_EXPERT_HOT:-<kein Profil>}
  profile kind: ${PROFILE_KIND:-<none>}
  listen:       $HOST:$PORT
  GPU:          $CUDA_VISIBLE_DEVICES_VALUE${GPU_ARCH:+ (compute $GPU_ARCH)}
  context:      $CONTEXT
  spec mode:    $SPEC_MODE ($SPEC_TYPE)
  MTP fallback: $MTP_N
  DFlash:       model=${SPEC_DRAFT_MODEL:-<none>} n=$SPEC_DRAFT_N_MIN..$SPEC_DRAFT_N_MAX p-min=$SPEC_DRAFT_P_MIN backend-sampling=$SPEC_DRAFT_BACKEND_SAMPLING combined=$DFLASH_COMBINED
  DFlash target override: ${DFLASH_TARGET_TENSOR_OVERRIDE:-none}
  KV target:    $TARGET_TYPE_K/$TARGET_TYPE_V
  KV draft:     $DRAFT_TYPE_K/$DRAFT_TYPE_V
  Q4 scale:     $LLAMA_KV_Q4_SCALE
  KVFlash:      $LLAMA_KVFLASH (max=$LLAMA_KVFLASH_MAX_POOL policy=$LLAMA_KVFLASH_POLICY tau=$LLAMA_KVFLASH_TAU stats=$LLAMA_KVFLASH_STATS)
  prompt cache: ram=$CACHE_RAM MiB prompt=$CACHE_PROMPT reuse=$CACHE_REUSE idle=$CACHE_IDLE_SLOTS checkpoints=$CTX_CHECKPOINTS
  Turbo4:       V-guard=$LLAMA_TURBO4_V_EXPERIMENTAL MTP-guard=$LLAMA_TURBO4_MTP_EXPERIMENTAL DFlash-guard=$LLAMA_TURBO4_DFLASH_EXPERIMENTAL draft-guard=$LLAMA_TURBO4_DRAFT_EXPERIMENTAL FP16-threshold=$GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH convert=$GGML_CUDA_TURBO4_FAST_F16_CONVERT wht-shuffle=$GGML_CUDA_TURBO4_WHT_SHUFFLE Q8-layers=${LLAMA_TURBO4_Q8_FALLBACK_LAYERS:-none}
  MTP head:     $LLAMA_MTP_REQUANTIZE_OUTPUT (trace=$LLAMA_MTP_HEAD_TRACE)
  CMoE batch:   $CMOE_BATCH/$CMOE_UBATCH
  phase batch:  prefill=${CMOE_PREFILL_BATCH:-base}/${CMOE_PREFILL_UBATCH:-base} decode=${CMOE_DECODE_BATCH:-base}/${CMOE_DECODE_UBATCH:-base}
  cpu-moe:      $CPU_MOE  ngl=$N_GPU_LAYERS  fit=$FIT/$FIT_TARGET/$FIT_CTX
  fixed S:      ${LLAMA_EXPERT_S:-auto-fit}, warm W: $LLAMA_EXPERT_WARM_SLOTS (admission=$LLAMA_EXPERT_WARM_ADMISSION window=$LLAMA_EXPERT_WARM_ADMISSION_WINDOW prefetch=$LLAMA_EXPERT_WARM_PREFETCH)
  CPU threads:  $THREADS/$THREADS_BATCH (draft $DRAFT_THREADS/$DRAFT_THREADS_BATCH)
  backend samp: target=$TARGET_BACKEND_SAMPLING draft=$EFFECTIVE_SPEC_DRAFT_BACKEND_SAMPLING
  sampling:     temp=$TEMP top-k=$TOP_K top-p=$TOP_P min-p=$MIN_P
  CUDA rows:    Q8n1=$GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS Q8n2=$GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS Q8n3=$GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS Q4n1=$GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS Q6n1=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS Q6n3=$GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS
  Q6 probes:    warps=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARPS warp-rows=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARP_ROWS reuse-y=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_REUSE_Y
  CUDA concat:  flat-dim0=$GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0 block=$GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE
  split copies: async-h2d=$GGML_CUDA_ASYNC_HOST_COPY async-d2h=$GGML_SCHED_ASYNC_D2H_COPY dedup-dst-sync=$GGML_SCHED_DEDUP_DST_SYNC
  n-cpu-moe:    register-host=$GGML_CUDA_REGISTER_HOST prefetch-experts=$GGML_SCHED_PREFETCH_EXPERTS
  skip sentinel: $LLAMA_EXPERT_SKIP_SENTINEL (1=MMVQ early-exit on cold sentinel slots)
  shared hot IDs: $LLAMA_EXPERT_SHARED_HOT_IDS
  CPU fused GU: $LLAMA_EXPERT_CPU_FUSED_GATE_UP
  bridge:       consume=$GGML_CUDA_EXPERT_BRIDGE_CONSUME device-quant=$GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT hit-only=$GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY cache=${GGML_CUDA_EXPERT_BRIDGE_CACHE_LAYERS:-off}/$GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS
  reasoning:    $REASONING_BUDGET format=$REASONING_FORMAT
  chat template: ${CHAT_TEMPLATE_FILE:-<model metadata>}
EOF

# Empty values for S/placement and inactive optional controls must not inherit
# from the shell that launched this file.
unset_args=(
    -u GGML_CUDA_DISABLE_GRAPHS
    -u LLAMA_EXPERT_S
    -u LLAMA_EXPERT_PLACEMENT
    -u LLAMA_ARG_OVERRIDE_TENSOR
    -u GGML_CUDA_EXPERT_BRIDGE_LAYER
    -u LLAMA_TURBOQUANT_LIVE_MASK_LAYER
    -u LLAMA_TURBOQUANT_CAPTURE_VALUES
)
if [[ "$CPU_MOE" == 1 ]]; then
    unset_args+=(-u LLAMA_ARG_NO_CPU_MOE)
else
    unset_args+=(-u LLAMA_ARG_CPU_MOE)
fi
if [[ "$SPEC_MODE" != dflash ]]; then
    unset_args+=(
        -u LLAMA_ARG_SPEC_DRAFT_MODEL
        -u LLAMA_ARG_SPEC_DRAFT_N_MIN
        -u LLAMA_ARG_SPEC_DRAFT_P_MIN
    )
fi
[[ -n "$THREADS_HTTP" ]] || unset_args+=(-u LLAMA_ARG_THREADS_HTTP)
[[ -n "$CMOE_PREFILL_BATCH" ]] || unset_args+=(-u LLAMA_CMOE_PREFILL_BATCH)
[[ -n "$CMOE_PREFILL_UBATCH" ]] || unset_args+=(-u LLAMA_CMOE_PREFILL_UBATCH)
[[ -n "$CMOE_DECODE_BATCH" ]] || unset_args+=(-u LLAMA_CMOE_DECODE_BATCH)
[[ -n "$CMOE_DECODE_UBATCH" ]] || unset_args+=(-u LLAMA_CMOE_DECODE_UBATCH)
[[ -n "$LLAMA_TURBO4_Q8_FALLBACK_LAYERS" ]] || unset_args+=(-u LLAMA_TURBO4_Q8_FALLBACK_LAYERS)
if [[ "$LLAMA_EXPERT_USAGE" == 0 ]]; then
    unset_args+=(-u LLAMA_EXPERT_USAGE_MODE -u LLAMA_EXPERT_USAGE_CHECKPOINT)
fi
if [[ "$LLAMA_EXPERT_WARM_SLOTS" == 0 || "$LLAMA_EXPERT_WARM_PREFETCH" != 1 ]]; then
    unset_args+=(-u LLAMA_EXPERT_PREFETCH_STREAMS -u LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT)
fi

exec env "${unset_args[@]}" "${env_args[@]}" "$SERVER" "${server_args[@]}"
