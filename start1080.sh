#!/usr/bin/env bash
#
# llama-wackMall-hybrid GTX 1080 LAN/OpenWebUI launcher
#
# Alle Einstellungen werden in diesem Block geändert. Das Script nimmt bewusst
# keine Kommandozeilenparameter und keine externen Overrides an. Nach einer
# Änderung einfach erneut ./start1080.sh ausführen.
#
# Bekannte Referenzpunkte (die unten eingetragenen Werte sind die maßgebliche
# Konfiguration und können bewusst davon abweichen):
#   GTX 1080 (sm_61): 64K, q8_0/q8_0, MTP-2, 128/128, vier CPU-Kerne,
#   angefordert S=70 (Runtime darf wegen VRAM sicher auf etwa S=64-66 klemmen).
#
# Diese Datei ist eine Test-Ausgangsbasis. Die sm_75-MMVQ-Werte werden nicht
# blind übernommen. Zuerst automatic/1/2/4 auf der GTX 1080 gegeneinander
# messen und dabei Hashes, effektives S und VRAM kontrollieren.
# Der unten gewaehlte Pascal-Build aktiviert den aus Upstream PR #25479
# portierten, sm_61-spezifischen Einspalten-MMVQ-Kandidaten. Das ist ein
# Build-Schalter und noch kein gemessener GTX-1080-Winner. Fuer die Kontrolle
# denselben Lauf mit build-transient-sm61 wiederholen.
#
# Der Expert-S-Wert bleibt standardmäßig leer: der Runtime-Autofit nutzt das
# jeweils freie VRAM. Für eine reproduzierbare 1660-Ti-Messung S=33 eintragen;
# auf der GTX1080 nach Messung z. B. S=66 oder S=70 eintragen.

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"  # directory containing this script

# ============================================================================
# EDITABLE CONFIGURATION -- only edit values in this section
# ============================================================================

# Paths
SERVER="$PROJECT_ROOT/build-pascal-tuned-sm61/bin/llama-server"  # experimental sm_61 build with GGML_CUDA_PASCAL_MMVQ_TUNING=ON
MODEL="$HOME/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf"  # GTX1080 target GGUF
PROFILE=""  # set this to the absolute validated GTX1080 learned-profile CSV before production use
PLACEMENT=""  # optional validated per-layer placement manifest; empty = uniform S/auto-fit

# Network / OpenWebUI
HOST="0.0.0.0"  # listen address; 0.0.0.0 exposes the API on all interfaces
PORT="8080"  # TCP port used by OpenWebUI and API clients
CORS_ORIGINS="*"  # allowed browser origins; restrict this for a non-local deployment
API_KEY=""  # inline API key; leave empty only on a trusted network
API_KEY_FILE=""  # optional file containing one or more API keys, one per line
MODEL_ALIAS="qwen3.6-35b-a3b-hybrid-gtx1080"  # model name advertised by the OpenAI-compatible API
CUDA_VISIBLE_DEVICES_VALUE="0"  # CUDA device list, e.g. 0 or 0,1
N_PARALLEL="1"  # number of simultaneous server slots; 1 keeps the tested path isolated
CPU_MOE="1"  # keep the hybrid CPU/GPU MoE tier enabled (0 disables it)
MMPROJ_AUTO="0"  # do not load an optional multimodal projector for this text model

# Model, context, KV, and server behavior
CONTEXT="65536"  # tested GTX1080 context; use 32768 for the lower-VRAM comparison
N_PREDICT="4096"  # CLI/server generation limit; API requests may impose a lower limit
TARGET_TYPE_K="q8_0"  # measured quality winner: avoids the q4_0 reasoning/self-doubt regression
TARGET_TYPE_V="q8_0"  # quality-oriented GTX1080 target V-cache
DRAFT_TYPE_K="q8_0"  # measured GTX1080 draft K-cache baseline
DRAFT_TYPE_V="q4_0"  # MTP draft V-cache type; affects draft memory and speed
MTP_N="2"  # draft tokens per MTP step; valid values are 0 (off), 1, 2, or 3
DRAFT_NGL="auto"  # draft GPU layers; auto fits the draft context to available VRAM
REASONING="auto"  # let the model template select reasoning mode
REASONING_BUDGET="256"  # measured quality/latency compromise; clients may request another value
REASONING_PRESERVE="1"  # preserve reasoning in history: 1 enabled, 0 disabled
LOAD_MODE="none"  # model loading mode: mmap, mlock, mmap+mlock, dio, or none
OFFLINE="1"  # prevent network model/template downloads when set to 1
JINJA="1"  # use the model's Jinja chat template when set to 1
FLASH_ATTN="on"  # Flash Attention mode: on, off, or auto
KV_OFFLOAD="1"  # place KV cache on the GPU when set to 1
THREADS="4"  # measured winner on i7-4770: 25.09 tok/s versus 22.26 with 8 SMT workers
THREADS_BATCH="4"  # match the four physical cores for prompt processing
DRAFT_THREADS="4"  # measured GTX1080 draft-context baseline
DRAFT_THREADS_BATCH="4"  # avoid SMT oversubscription during draft prefill
THREADS_HTTP=""  # HTTP worker count; empty uses the server default
TARGET_BACKEND_SAMPLING="1"  # target sampling on CUDA; 0 provides an A/B control
DRAFT_BACKEND_SAMPLING="1"  # draft sampling on the backend; 0 forces CPU draft sampling

# Prompt/decode batching. GPU_ARCH accepts auto, 6.1 (GTX1080), or 7.5
# (GTX1660Ti). CMOE_BATCH/CMOE_UBATCH accept a number or auto. The four phase
# values are empty by default; setting them enables the experimental prefill /
# decode switch implemented in common/arg.cpp.
GPU_ARCH="6.1"  # GTX 1080 Pascal compute capability
CMOE_BATCH="128"  # user-selected long-prompt baseline; 63% higher measured prompt throughput than 32
CMOE_UBATCH="128"  # fixed 128 decode beat the experimental 128->32 phase switch end-to-end
CMOE_PREFILL_BATCH=""  # optional logical batch used only during prompt processing
CMOE_PREFILL_UBATCH=""  # optional physical ubatch used only during prompt processing
CMOE_DECODE_BATCH=""  # optional logical batch used only during token generation
CMOE_DECODE_UBATCH=""  # optional physical ubatch used only during token generation

# Prompt-cache controls
CTX_CHECKPOINTS="0"  # disabled: avoids background checkpoint work after a completed response
CACHE_RAM="0"  # disabled until checkpoint/cache latency is benchmarked independently
CACHE_PROMPT="1"  # enable prompt-prefix reuse when set to 1
CACHE_REUSE="16"  # minimum reusable prompt tokens/threshold for server cache logic
KV_UNIFIED="1"  # share one unified KV buffer between slots when set to 1
CACHE_IDLE_SLOTS="1"  # retain idle slots in the prompt cache when set to 1

# Expert tier. These names are the actual LLAMA_EXPERT_* runtime variables.
LLAMA_EXPERT_HOT="$PROFILE"  # ranking/usage CSV used to select fixed hot experts
LLAMA_EXPERT_S="70"  # requested GTX1080 tier; runtime safely clamps to the available 64-66 slots at 64K q8
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
LLAMA_EXPERT_CPU_DOWN_PREFETCH="0"  # no measured sustained gain on the local CPU
LLAMA_EXPERT_CPU_REUSE_ROWS="1"  # reuse quantized cold rows for repeated MTP selections
LLAMA_EXPERT_CPU_MULTI_ROW="1"  # promising on the older AVX2 quad-core; keep as an explicit A/B candidate

# Warmcache. Fixed hot slots are never evicted; W=0 is the measured MTP path.
LLAMA_EXPERT_WARM_SLOTS="0"  # extra LRU slots per layer; integer, auto, or 0 to disable
LLAMA_EXPERT_WARM_AUTO_MAX="8"  # cap for W=auto; raise only after VRAM measurements
LLAMA_EXPERT_WARM_POLICY="lru"  # replacement policy; currently only lru is supported
LLAMA_EXPERT_WARM_RESET="request"  # age reset policy: request or persistent
LLAMA_EXPERT_WARM_ADMISSION="immediate"  # admission: immediate, second-hit, or frequency
LLAMA_EXPERT_WARM_ADMISSION_WINDOW="8"  # window/half-life used by frequency admission
LLAMA_EXPERT_WARM_PREFETCH="0"  # W=0 production winner; do not create copy traffic
LLAMA_EXPERT_PREFETCH_STREAMS="1"  # number of prefetch CUDA streams; current implementation requires 1
LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT="2"  # maximum simultaneous expert copies
LLAMA_EXPERT_VRAM_RESERVE_MIB="512"  # safety headroom retained after model/runtime allocation
LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL="0"  # retain the MTP warmcache correctness guard
LLAMA_EXPERT_STATIC_NO_SYNC="1"  # safe here: adaptation, stats, timing, usage, and W are disabled

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
# Bridge winner defaults (GTX 1080 phase-matched test): layer 1 uses lookahead
# K=2, while layer 2 is the recurrence fallback with K=1.  These values are
# also safe on larger GPUs because the implementation validates layer/count
# limits at runtime; they do not hard-code a model size or VRAM capacity.
GGML_CUDA_EXPERT_BRIDGE_LAYERS="1,2"  # selected layers; the first is lookahead, later layers are recurrence
GGML_CUDA_EXPERT_BRIDGE_K="2"  # candidate limit for the first bridge layer (winner value)
GGML_CUDA_EXPERT_BRIDGE_RECURRENCE="0"  # keep recurrence limited to the explicit layer list below
GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_LAYERS="2"  # recurrence layer(s); winner uses layer 2 only
GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_K="1"  # candidate limit for recurrence layers (winner value)
GGML_CUDA_EXPERT_BRIDGE_CONSUME="0"  # GTX 1660 Ti bridge median regressed; production off
GGML_CUDA_EXPERT_BRIDGE_VERIFY="0"  # diagnostic recomputation; 1 is correctness-test-only and slower
GGML_CUDA_EXPERT_BRIDGE_JSON="0"  # 0 keeps bridge off; set an absolute JSON path to opt in

# Exact kernel controls. Shared IDs are architecture-independent and are a
# plausible starting point. Multi-token fusion is guarded off below sm_75.
# MMVQ geometry starts at automatic because launch balance differs on Pascal.
# GGML_CUDA_PASCAL_MMVQ_TUNING is a CMake-only switch; it cannot be changed by
# this launcher. Compare this binary against build-transient-sm61 before
# promoting it, because only compile/link correctness has been established.
LLAMA_EXPERT_SHARED_HOT_IDS="1"  # first GTX1080 A/B candidate; set 0 for the exact old baseline
LLAMA_EXPERT_SKIP_SENTINEL="1"  # sm_75 3x2K winner (+3.90%); still needs phase-matched sm_61 A/B
GGML_CUDA_MOE_MULTI_FUSION="0"  # sm_75+ path; disabled on Pascal
GGML_CUDA_MOE_COMBINE_FUSION="0"  # exact combine fusion measured neutral (+0.07%)
GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS="0"  # automatic baseline; test 1,2,4 independently on sm_61
GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS="0"  # automatic baseline; likely candidates are 1,2,4
GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS="0"  # automatic baseline; likely candidates are 1,2,4
GGML_CUDA_ASYNC_HOST_COPY="1"  # architecture-neutral pinned H2D candidate; +1.90% on sm_75, validate on sm_61
GGML_SCHED_ASYNC_D2H_COPY="0"  # below the sm_75 promotion gate; enable only for a controlled sm_61 A/B

# ============================================================================
# End of editable configuration
# ============================================================================

die() {
    printf 'start1080.sh: %s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 0 ]]; then
    die "Keine Kommandozeilenparameter: Einstellungen oben in start1080.sh ändern."
fi

[[ -x "$SERVER" ]] || die "llama-server nicht ausführbar: $SERVER"
[[ -f "$MODEL" ]] || die "Modell nicht gefunden: $MODEL"
if [[ -n "$LLAMA_EXPERT_HOT" && ! -f "$LLAMA_EXPERT_HOT" ]]; then
    die "Expert-Profil nicht gefunden: $LLAMA_EXPERT_HOT"
fi
if [[ -n "$LLAMA_EXPERT_PLACEMENT" && ! -f "$LLAMA_EXPERT_PLACEMENT" ]]; then
    die "Placement-Manifest nicht gefunden: $LLAMA_EXPERT_PLACEMENT"
fi
if [[ -n "$API_KEY_FILE" && ! -f "$API_KEY_FILE" ]]; then
    die "API-Key-Datei nicht gefunden: $API_KEY_FILE"
fi

case "$MTP_N" in 0|1|2|3) ;; *) die "MTP_N muss 0, 1, 2 oder 3 sein." ;; esac
case "$TARGET_BACKEND_SAMPLING" in 0|1) ;; *) die "TARGET_BACKEND_SAMPLING muss 0 oder 1 sein." ;; esac
case "$DRAFT_BACKEND_SAMPLING" in 0|1) ;; *) die "DRAFT_BACKEND_SAMPLING muss 0 oder 1 sein." ;; esac
case "$REASONING_PRESERVE" in 0|1) ;; *) die "REASONING_PRESERVE muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_ASYNC_HOST_COPY" in 0|1) ;; *) die "GGML_CUDA_ASYNC_HOST_COPY muss 0 oder 1 sein." ;; esac
case "$GGML_SCHED_ASYNC_D2H_COPY" in 0|1) ;; *) die "GGML_SCHED_ASYNC_D2H_COPY muss 0 oder 1 sein." ;; esac
[[ -z "$LLAMA_EXPERT_S" || -z "$LLAMA_EXPERT_PLACEMENT" ]] || \
    die "LLAMA_EXPERT_S und LLAMA_EXPERT_PLACEMENT sind gegenseitig exklusiv."
if [[ "$LLAMA_EXPERT_LOOKAHEAD_TRACE" != 0 && "$LLAMA_EXPERT_LOOKAHEAD_TRACE_JSON" == 0 ]]; then
    die "LLAMA_EXPERT_LOOKAHEAD_TRACE=1 benötigt einen JSON-Pfad."
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

if [[ "$MTP_N" != 0 ]]; then
    SPEC_TYPE="draft-mtp"
else
    SPEC_TYPE="none"
fi

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
    "LLAMA_ARG_CPU_MOE=$CPU_MOE"
    "LLAMA_ARG_MMPROJ_AUTO=$MMPROJ_AUTO"
    "LLAMA_ARG_LOAD_MODE=$LOAD_MODE"
    "LLAMA_ARG_CACHE_TYPE_K=$TARGET_TYPE_K"
    "LLAMA_ARG_CACHE_TYPE_V=$TARGET_TYPE_V"
    "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K=$DRAFT_TYPE_K"
    "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V=$DRAFT_TYPE_V"
    "LLAMA_ARG_SPEC_TYPE=$SPEC_TYPE"
    "LLAMA_ARG_SPEC_DRAFT_N_MAX=$MTP_N"
    "LLAMA_ARG_N_GPU_LAYERS_DRAFT=$DRAFT_NGL"
    "LLAMA_ARG_SPEC_DRAFT_THREADS=$DRAFT_THREADS"
    "LLAMA_ARG_SPEC_DRAFT_THREADS_BATCH=$DRAFT_THREADS_BATCH"
    "LLAMA_ARG_BACKEND_SAMPLING=$TARGET_BACKEND_SAMPLING"
    "LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING=$DRAFT_BACKEND_SAMPLING"
    "LLAMA_ARG_REASONING=$REASONING"
    "LLAMA_ARG_THINK_BUDGET=$REASONING_BUDGET"
    "LLAMA_ARG_REASONING_PRESERVE=$REASONING_PRESERVE"
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
    "LLAMA_EXPERT_WARM_SLOTS=$LLAMA_EXPERT_WARM_SLOTS"
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
    "LLAMA_EXPERT_SHARED_HOT_IDS=$LLAMA_EXPERT_SHARED_HOT_IDS"
    "LLAMA_EXPERT_SKIP_SENTINEL=$LLAMA_EXPERT_SKIP_SENTINEL"
    "GGML_CUDA_MOE_MULTI_FUSION=$GGML_CUDA_MOE_MULTI_FUSION"
    "GGML_CUDA_MOE_COMBINE_FUSION=$GGML_CUDA_MOE_COMBINE_FUSION"
    "GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS"
    "GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS=$GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS"
    "GGML_CUDA_ASYNC_HOST_COPY=$GGML_CUDA_ASYNC_HOST_COPY"
    "GGML_SCHED_ASYNC_D2H_COPY=$GGML_SCHED_ASYNC_D2H_COPY"
)

# Phase-specific batching is opt-in by editing the four values above.
[[ -n "$CMOE_PREFILL_BATCH" ]] && env_args+=("LLAMA_CMOE_PREFILL_BATCH=$CMOE_PREFILL_BATCH")
[[ -n "$CMOE_PREFILL_UBATCH" ]] && env_args+=("LLAMA_CMOE_PREFILL_UBATCH=$CMOE_PREFILL_UBATCH")
[[ -n "$CMOE_DECODE_BATCH" ]] && env_args+=("LLAMA_CMOE_DECODE_BATCH=$CMOE_DECODE_BATCH")
[[ -n "$CMOE_DECODE_UBATCH" ]] && env_args+=("LLAMA_CMOE_DECODE_UBATCH=$CMOE_DECODE_UBATCH")
[[ -n "$THREADS_HTTP" ]] && env_args+=("LLAMA_ARG_THREADS_HTTP=$THREADS_HTTP")

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
)
[[ -n "$API_KEY" ]] && server_args+=(--api-key "$API_KEY")
[[ -n "$API_KEY_FILE" ]] && server_args+=(--api-key-file "$API_KEY_FILE")

if [[ -z "$API_KEY" && -z "$API_KEY_FILE" ]]; then
    printf 'WARNUNG: API_KEY/API_KEY_FILE ist leer; der Dienst ist ohne Authentifizierung im LAN erreichbar.\n' >&2
fi
if [[ -z "$LLAMA_EXPERT_HOT" ]]; then
    printf 'WARNUNG: LLAMA_EXPERT_HOT ist leer; es wird ohne gelerntes Profil gestartet.\n' >&2
fi

cat <<EOF
llama-wackMall-hybrid GTX1080 start
  project:      $PROJECT_ROOT
  server:       $SERVER
  model:        $MODEL
  profile:      ${LLAMA_EXPERT_HOT:-<kein Profil>}
  listen:       $HOST:$PORT
  GPU:          $CUDA_VISIBLE_DEVICES_VALUE${GPU_ARCH:+ (compute $GPU_ARCH)}
  context:      $CONTEXT
  MTP:          $MTP_N
  KV target:    $TARGET_TYPE_K/$TARGET_TYPE_V
  KV draft:     $DRAFT_TYPE_K/$DRAFT_TYPE_V
  CMoE batch:   $CMOE_BATCH/$CMOE_UBATCH
  phase batch:  prefill=${CMOE_PREFILL_BATCH:-base}/${CMOE_PREFILL_UBATCH:-base} decode=${CMOE_DECODE_BATCH:-base}/${CMOE_DECODE_UBATCH:-base}
  fixed S:      ${LLAMA_EXPERT_S:-auto-fit}, warm W: $LLAMA_EXPERT_WARM_SLOTS
  CPU threads:  $THREADS/$THREADS_BATCH (draft $DRAFT_THREADS/$DRAFT_THREADS_BATCH)
  backend samp: target=$TARGET_BACKEND_SAMPLING draft=$DRAFT_BACKEND_SAMPLING
  CUDA rows:    Q8n3=$GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS Q6n1=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS Q6n3=$GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS
  reasoning:    $REASONING_BUDGET
EOF

# Empty values for S/placement and inactive optional controls must not inherit
# from the shell that launched this file.
unset_args=(-u LLAMA_EXPERT_S -u LLAMA_EXPERT_PLACEMENT)
[[ -n "$THREADS_HTTP" ]] || unset_args+=(-u LLAMA_ARG_THREADS_HTTP)
[[ -n "$CMOE_PREFILL_BATCH" ]] || unset_args+=(-u LLAMA_CMOE_PREFILL_BATCH)
[[ -n "$CMOE_PREFILL_UBATCH" ]] || unset_args+=(-u LLAMA_CMOE_PREFILL_UBATCH)
[[ -n "$CMOE_DECODE_BATCH" ]] || unset_args+=(-u LLAMA_CMOE_DECODE_BATCH)
[[ -n "$CMOE_DECODE_UBATCH" ]] || unset_args+=(-u LLAMA_CMOE_DECODE_UBATCH)
if [[ "$LLAMA_EXPERT_USAGE" == 0 ]]; then
    unset_args+=(-u LLAMA_EXPERT_USAGE_MODE -u LLAMA_EXPERT_USAGE_CHECKPOINT)
fi
if [[ "$LLAMA_EXPERT_WARM_SLOTS" == 0 || "$LLAMA_EXPERT_WARM_PREFETCH" != 1 ]]; then
    unset_args+=(-u LLAMA_EXPERT_PREFETCH_STREAMS -u LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT)
fi

exec env "${unset_args[@]}" "${env_args[@]}" "$SERVER" "${server_args[@]}"
