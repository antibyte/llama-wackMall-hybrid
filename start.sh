#!/usr/bin/env bash
#
# llama-wackMall-hybrid LAN/OpenWebUI launcher
#
# Alle Einstellungen werden in diesem Block geändert. Das Script nimmt bewusst
# keine Kommandozeilenparameter und keine externen Overrides an. Nach einer
# Änderung einfach erneut ./start.sh ausführen.
#
# Die Werte unten kombinieren die schnellen lokalen GTX-1660-Ti-Kernelwerte
# (CMOE 376/376, MTP-2, S=33) mit dem experimentellen Turbo4/Turbo4-Target-KV.
# Turbo4/Turbo4 spart deutlich KV-Speicher, kostet gegenueber Q8/Q8 aber
# messbar Qualitaet. Fuer maximale Qualitaet TARGET_TYPE_K/V auf q8_0 setzen.
# Der MTP-Draft-Cache bleibt q4_0/q4_0; Turbo4 wird dort nicht unterstuetzt.
# Bekannter weiterer Referenzpunkt:
#   GTX 1080    (sm_61): siehe start1080.sh; eigene Kernel-Sweeps erforderlich
#
# Der Expert-S-Wert ist fuer den reproduzierbaren 1660-Ti-Pfad fest auf 33
# gesetzt. Fuer andere GPUs start1080.sh verwenden oder S nach einem VRAM-Sweep
# anpassen; der Runtime-Autofit begrenzt einen zu hohen Wert weiterhin sicher.

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"  # directory containing this script

# ============================================================================
# EDITABLE CONFIGURATION -- only edit values in this section
# ============================================================================

# Paths
SERVER="$PROJECT_ROOT/build-turbo-fastconvert-sm75/bin/llama-server"  # current Turbo4-enabled native sm_75 Release executable
MODEL="$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"  # target GGUF
PROFILE="$HOME/src/llama-wackMall-hybrid/benchmark-results/profile-corpus-train8-512-20260802T124500Z/general-profile.csv"  # measured general heat/ranking CSV; edit after cloning elsewhere
PLACEMENT=""  # optional validated per-layer placement manifest; empty = uniform S/auto-fit

# Network / OpenWebUI
HOST="0.0.0.0"  # listen address; 0.0.0.0 exposes the API on all interfaces
PORT="8080"  # TCP port used by OpenWebUI and API clients
CORS_ORIGINS="*"  # allowed browser origins; restrict this for a non-local deployment
API_KEY=""  # inline API key; leave empty only on a trusted network
API_KEY_FILE=""  # optional file containing one or more API keys, one per line
MODEL_ALIAS="qwen3.6-35b-a3b-hybrid"  # model name advertised by the OpenAI-compatible API
CUDA_VISIBLE_DEVICES_VALUE="0"  # CUDA device list, e.g. 0 or 0,1
N_PARALLEL="1"  # number of simultaneous server slots; 1 keeps the tested path isolated
CPU_MOE="1"  # keep the hybrid CPU/GPU MoE tier enabled (0 disables it)
MMPROJ_AUTO="0"  # do not load an optional multimodal projector for this text model

# Model, context, KV, and server behavior
CONTEXT="32768"  # maximum context tokens; raises KV memory when increased
N_PREDICT="4096"  # CLI/server generation limit; API requests may impose a lower limit
# Direkt editierbare KV-Presets (jeweils TARGET_TYPE_K und TARGET_TYPE_V):
#   q4_0/q4_0       = gemessener 32K-TPS-Sieger (~46.5 sustained TPS), geringste Reasoning-Qualitaet
#   turbo4_k/turbo4_k = aktuell aktiv; guter Kapazitaets-/64K-Kandidat, experimentell
#   q8_0/q8_0       = Qualitaetskandidat mit hoeherem VRAM-Bedarf und etwas weniger TPS
# Nur die beiden folgenden Zeilen aendern; die passenden Turbo4-Guards duerfen auf 1 bleiben.
TARGET_TYPE_K="turbo4_k"  # experimental 4-bit randomized-WHT target K cache
TARGET_TYPE_V="turbo4_k"  # experimental symmetric Turbo4 V cache; saves the most target KV memory
DRAFT_TYPE_K="q4_0"  # tested Turbo4+MTP draft K cache; draft Turbo4 is unsupported
DRAFT_TYPE_V="q4_0"  # tested Turbo4+MTP draft V cache
LLAMA_KV_Q4_SCALE="legacy"  # Q4-only scale policy: legacy winner for speed; weighted-k is the quality A/B candidate
LLAMA_TURBO4_V_EXPERIMENTAL="1"  # required guard for Turbo4 target V; 0 rejects this configuration
LLAMA_TURBO4_MTP_EXPERIMENTAL="1"  # required guard for target Turbo4 with draft-mtp
LLAMA_TURBO4_Q8_FALLBACK_LAYERS=""  # optional comma/range list of target K layers kept at Q8, e.g. 23,35,27
GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH="2"  # tested MTP-2 crossover: multi-token verify/prefill uses the FP16 FA path
GGML_CUDA_TURBO4_FAST_F16_CONVERT="0"  # generic converter is the SM75 winner; 1/2 were slower experiments
GGML_CUDA_TURBO4_WHT_SHUFFLE="0"  # original WHT is the SM75 winner; shuffle-first was neutral/slower
MTP_N="2"  # draft tokens per MTP step; valid values are 0 (off), 1, 2, or 3
LLAMA_MTP_REQUANTIZE_OUTPUT="none"  # none is the 6-GiB winner; q4_K/q5_K add a draft-only LM head but consume 273/334 MiB VRAM
LLAMA_MTP_HEAD_TRACE="0"  # 1 prints the selected MTP-head source/type once; diagnostic only
DRAFT_NGL="auto"  # draft GPU layers; auto fits the draft context to available VRAM
REASONING="auto"  # let the model template select reasoning mode
REASONING_BUDGET="1024"  # measured quality/latency compromise; clients may request another value
REASONING_PRESERVE="1"  # preserve reasoning in history: 1 enabled, 0 disabled
LOAD_MODE="mmap"  # measured loading/runtime winner; "none" was 1.87% slower and reduced S to 32
OFFLINE="1"  # prevent network model/template downloads when set to 1
JINJA="1"  # use the model's Jinja chat template when set to 1
FLASH_ATTN="on"  # Flash Attention mode: on, off, or auto
KV_OFFLOAD="1"  # place KV cache on the GPU when set to 1
THREADS="8"  # target generation CPU workers; test physical-core/SMT alternatives
THREADS_BATCH="8"  # CPU workers for prompt/batch processing
DRAFT_THREADS="8"  # CPU workers for the MTP draft context
DRAFT_THREADS_BATCH="8"  # draft workers for draft prompt/batch processing
THREADS_HTTP=""  # HTTP worker count; empty uses the server default
TARGET_BACKEND_SAMPLING="1"  # target sampling on CUDA; 0 provides an A/B control
DRAFT_BACKEND_SAMPLING="1"  # draft sampling on the backend; 0 forces CPU draft sampling

# Prompt/decode batching. GPU_ARCH accepts auto, 6.1 (GTX1080), or 7.5
# (GTX1660Ti). CMOE_BATCH/CMOE_UBATCH accept a number or auto. The four phase
# values are empty by default; setting them enables the experimental prefill /
# decode switch implemented in common/arg.cpp.
GPU_ARCH="auto"  # architecture used to choose an auto batch default
CMOE_BATCH="376"  # measured 1660-Ti maximum-batch winner retaining all 33 fixed slots
CMOE_UBATCH="376"  # measured physical-ubatch winner with the default VRAM reserve
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
LLAMA_EXPERT_S="33"  # measured fixed-tier winner on the 6-GiB GTX 1660 Ti at 32K
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
LLAMA_EXPERT_CPU_REUSE_ROWS="0"  # conservative winner; row reuse is not part of the 49.134-TPS promoted Q4 path
LLAMA_EXPERT_CPU_MULTI_ROW="0"  # AVX2 multi-row reduced CPU time but was neutral in 2K decode
LLAMA_EXPERT_CPU_FUSED_GATE_UP="0"  # exact AVX2 dual-dot; +0.21% median was below promotion threshold on Ryzen 4800H

# Warmcache. Fixed hot slots are never evicted; W=0 is the measured MTP path.
LLAMA_EXPERT_WARM_SLOTS="0"  # extra LRU slots per layer; W=0 is the measured 6-GiB winner
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

# Exact kernel controls. These are the measured GTX 1660 Ti MTP-2 winners.
# They remain explicit environment flags so start1080.sh can use independent
# Pascal values after its own 1/2/4 geometry sweep.
LLAMA_EXPERT_SHARED_HOT_IDS="1"  # share one mapped hot-ID tensor across Gate, Up, and Down
GGML_CUDA_MOE_MULTI_FUSION="1"  # fuse Gate+Up+GLU for 2-4 target tokens on sm_75+
GGML_CUDA_MOE_COMBINE_FUSION="0"  # exact combine fusion measured neutral (+0.07%)
GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS="4"  # 3x2K winner: 46.831 -> 47.436 token/s
GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS="0"  # one-column grouping was neutral; 0 keeps automatic geometry
GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS="2"  # Q6_K output projection, one-token graph
GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARPS="0"  # 1/4-warp screens lost; 0 keeps the architecture default of 2
GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARP_ROWS="0"  # independent warp-row kernels did not beat the winner
GGML_CUDA_MMVQ_Q6_K_NCOLS1_REUSE_Y="0"  # shared Q8_1 unpack was exact but slightly slower on sm_75
GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS="4"  # Q6_K output projection, MTP-2 verify graph
GGML_CUDA_MMVQ_MOE_FUSED_ROWS="0"  # fused MoE rows/block sweep was neutral or slower
GGML_CUDA_MMVQ_MOE_PLAIN_ROWS="0"  # plain MoE rows/block sweep was neutral or slower
GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE="256"  # active together with flat dim-0 in the confirmed 3x2K MTP-2 winner
GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0="1"  # 3x2K winner: 47.794 -> 48.599 token/s
GGML_CUDA_ASYNC_HOST_COPY="1"  # 3x2K winner: pinned split H2D, 47.435 -> 48.335 token/s
GGML_SCHED_ASYNC_D2H_COPY="0"  # exact but below promotion gate: +0.48% q4 and +0.24% q8 screens

# ============================================================================
# End of editable configuration
# ============================================================================

die() {
    printf 'start.sh: %s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 0 ]]; then
    die "Keine Kommandozeilenparameter: Einstellungen oben in start.sh ändern."
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
case "$LLAMA_TURBO4_V_EXPERIMENTAL" in 0|1) ;; *) die "LLAMA_TURBO4_V_EXPERIMENTAL muss 0 oder 1 sein." ;; esac
case "$LLAMA_TURBO4_MTP_EXPERIMENTAL" in 0|1) ;; *) die "LLAMA_TURBO4_MTP_EXPERIMENTAL muss 0 oder 1 sein." ;; esac
case "$LLAMA_KV_Q4_SCALE" in legacy|weighted|weighted-k|weighted-v) ;; *) die "LLAMA_KV_Q4_SCALE muss legacy, weighted, weighted-k oder weighted-v sein." ;; esac
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
case "$GGML_CUDA_MOE_MULTI_FUSION" in 0|1) ;; *) die "GGML_CUDA_MOE_MULTI_FUSION muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_MOE_COMBINE_FUSION" in 0|1) ;; *) die "GGML_CUDA_MOE_COMBINE_FUSION muss 0 oder 1 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS muss 0, 1, 2 oder 4 sein." ;; esac
case "$GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS" in 0|1|2|4) ;; *) die "GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS muss 0, 1, 2 oder 4 sein." ;; esac
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
[[ -z "$LLAMA_EXPERT_S" || -z "$LLAMA_EXPERT_PLACEMENT" ]] || \
    die "LLAMA_EXPERT_S und LLAMA_EXPERT_PLACEMENT sind gegenseitig exklusiv."
if [[ "$LLAMA_EXPERT_LOOKAHEAD_TRACE" != 0 && "$LLAMA_EXPERT_LOOKAHEAD_TRACE_JSON" == 0 ]]; then
    die "LLAMA_EXPERT_LOOKAHEAD_TRACE=1 benötigt einen JSON-Pfad."
fi
if [[ "$TARGET_TYPE_V" == turbo4_k && "$LLAMA_TURBO4_V_EXPERIMENTAL" != 1 ]]; then
    die "TARGET_TYPE_V=turbo4_k benötigt LLAMA_TURBO4_V_EXPERIMENTAL=1."
fi
if [[ ( "$TARGET_TYPE_K" == turbo4_k || "$TARGET_TYPE_V" == turbo4_k ) && "$FLASH_ATTN" != on ]]; then
    die "Turbo4-KV benötigt FLASH_ATTN=on."
fi
if [[ "$MTP_N" != 0 && ( "$TARGET_TYPE_K" == turbo4_k || "$TARGET_TYPE_V" == turbo4_k ) && "$LLAMA_TURBO4_MTP_EXPERIMENTAL" != 1 ]]; then
    die "Target-Turbo4 mit MTP benötigt LLAMA_TURBO4_MTP_EXPERIMENTAL=1."
fi
if [[ "$DRAFT_TYPE_K" == turbo4_k || "$DRAFT_TYPE_V" == turbo4_k ]]; then
    die "Turbo4 wird im MTP-Draft-Cache nicht unterstuetzt."
fi
if [[ -n "$LLAMA_TURBO4_Q8_FALLBACK_LAYERS" && "$TARGET_TYPE_K" != turbo4_k ]]; then
    die "LLAMA_TURBO4_Q8_FALLBACK_LAYERS benötigt TARGET_TYPE_K=turbo4_k."
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
    "LLAMA_KV_Q4_SCALE=$LLAMA_KV_Q4_SCALE"
    "LLAMA_TURBO4_V_EXPERIMENTAL=$LLAMA_TURBO4_V_EXPERIMENTAL"
    "LLAMA_TURBO4_MTP_EXPERIMENTAL=$LLAMA_TURBO4_MTP_EXPERIMENTAL"
    "GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH=$GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH"
    "GGML_CUDA_TURBO4_FAST_F16_CONVERT=$GGML_CUDA_TURBO4_FAST_F16_CONVERT"
    "GGML_CUDA_TURBO4_WHT_SHUFFLE=$GGML_CUDA_TURBO4_WHT_SHUFFLE"
    "LLAMA_ARG_SPEC_TYPE=$SPEC_TYPE"
    "LLAMA_ARG_SPEC_DRAFT_N_MAX=$MTP_N"
    "LLAMA_MTP_REQUANTIZE_OUTPUT=$LLAMA_MTP_REQUANTIZE_OUTPUT"
    "LLAMA_MTP_HEAD_TRACE=$LLAMA_MTP_HEAD_TRACE"
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
    "LLAMA_EXPERT_CPU_FUSED_GATE_UP=$LLAMA_EXPERT_CPU_FUSED_GATE_UP"
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
    "GGML_CUDA_EXPERT_BRIDGE_TELEMETRY=$GGML_CUDA_EXPERT_BRIDGE_TELEMETRY"
    "GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT=$GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT"
    "GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT_RECURRENCE=$GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT_RECURRENCE"
    "GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY=$GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY"
    "GGML_CUDA_EXPERT_BRIDGE_CACHE_LAYERS=$GGML_CUDA_EXPERT_BRIDGE_CACHE_LAYERS"
    "GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS=$GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS"
    "LLAMA_EXPERT_SHARED_HOT_IDS=$LLAMA_EXPERT_SHARED_HOT_IDS"
    "GGML_CUDA_MOE_MULTI_FUSION=$GGML_CUDA_MOE_MULTI_FUSION"
    "GGML_CUDA_MOE_COMBINE_FUSION=$GGML_CUDA_MOE_COMBINE_FUSION"
    "GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS"
    "GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS=$GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS"
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
)

# Phase-specific batching is opt-in by editing the four values above.
[[ -n "$CMOE_PREFILL_BATCH" ]] && env_args+=("LLAMA_CMOE_PREFILL_BATCH=$CMOE_PREFILL_BATCH")
[[ -n "$CMOE_PREFILL_UBATCH" ]] && env_args+=("LLAMA_CMOE_PREFILL_UBATCH=$CMOE_PREFILL_UBATCH")
[[ -n "$CMOE_DECODE_BATCH" ]] && env_args+=("LLAMA_CMOE_DECODE_BATCH=$CMOE_DECODE_BATCH")
[[ -n "$CMOE_DECODE_UBATCH" ]] && env_args+=("LLAMA_CMOE_DECODE_UBATCH=$CMOE_DECODE_UBATCH")
[[ -n "$THREADS_HTTP" ]] && env_args+=("LLAMA_ARG_THREADS_HTTP=$THREADS_HTTP")
[[ -n "$LLAMA_TURBO4_Q8_FALLBACK_LAYERS" ]] && env_args+=("LLAMA_TURBO4_Q8_FALLBACK_LAYERS=$LLAMA_TURBO4_Q8_FALLBACK_LAYERS")

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
llama-wackMall-hybrid start
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
  Q4 scale:     $LLAMA_KV_Q4_SCALE
  Turbo4:       V-guard=$LLAMA_TURBO4_V_EXPERIMENTAL MTP-guard=$LLAMA_TURBO4_MTP_EXPERIMENTAL FP16-threshold=$GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH convert=$GGML_CUDA_TURBO4_FAST_F16_CONVERT wht-shuffle=$GGML_CUDA_TURBO4_WHT_SHUFFLE Q8-layers=${LLAMA_TURBO4_Q8_FALLBACK_LAYERS:-none}
  MTP head:     $LLAMA_MTP_REQUANTIZE_OUTPUT (trace=$LLAMA_MTP_HEAD_TRACE)
  CMoE batch:   $CMOE_BATCH/$CMOE_UBATCH
  phase batch:  prefill=${CMOE_PREFILL_BATCH:-base}/${CMOE_PREFILL_UBATCH:-base} decode=${CMOE_DECODE_BATCH:-base}/${CMOE_DECODE_UBATCH:-base}
  fixed S:      ${LLAMA_EXPERT_S:-auto-fit}, warm W: $LLAMA_EXPERT_WARM_SLOTS
  CPU threads:  $THREADS/$THREADS_BATCH (draft $DRAFT_THREADS/$DRAFT_THREADS_BATCH)
  backend samp: target=$TARGET_BACKEND_SAMPLING draft=$DRAFT_BACKEND_SAMPLING
  CUDA rows:    Q8n1=$GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS Q8n3=$GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS Q6n1=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS Q6n3=$GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS
  Q6 probes:    warps=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARPS warp-rows=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARP_ROWS reuse-y=$GGML_CUDA_MMVQ_Q6_K_NCOLS1_REUSE_Y
  CUDA concat:  flat-dim0=$GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0 block=$GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE
  CPU fused GU: $LLAMA_EXPERT_CPU_FUSED_GATE_UP
  bridge:       consume=$GGML_CUDA_EXPERT_BRIDGE_CONSUME device-quant=$GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT hit-only=$GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY cache=${GGML_CUDA_EXPERT_BRIDGE_CACHE_LAYERS:-off}/$GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS
  reasoning:    $REASONING_BUDGET
EOF

# Empty values for S/placement and inactive optional controls must not inherit
# from the shell that launched this file.
unset_args=(
    -u LLAMA_EXPERT_S
    -u LLAMA_EXPERT_PLACEMENT
    -u GGML_CUDA_EXPERT_BRIDGE_LAYER
    -u LLAMA_TURBOQUANT_LIVE_MASK_LAYER
    -u LLAMA_TURBOQUANT_CAPTURE_VALUES
)
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
