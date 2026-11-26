#!/usr/bin/env bash
#
# llama-wackMall-hybrid launcher for Mage-VL-Cowork 4B Q4_K_M (screenshot agent).
#
# Alle Einstellungen stehen in diesem Block. Keine Kommandozeilenparameter.
#
# Model: Hansda/Mage-VL-Cowork-GGUF (Q4_K_M decoder + Q8_0 mmproj, projector magevl).
# The magevl projector was ported from ch686 e07bdc8 (upstream PR 26796).
# Prompting: system prompt = MAGE_GUI_SYSTEM_PROMPT from the model card's
# osworld_eval/mm_agents/mage_gui_agent.py, user turn = screenshot + "Task: ...".
# The hybrid prompt (with a screenshot action) makes it answer "screenshot".
# Bench 2026-11-02 (benchmark-results/mage-vl-cowork-20261102), image time =
# encode + prefill of the image tokens, system prompt cached:
#   GTX 1660 Ti: 1920x1080 5.97 s before, 5.16 s full resolution after,
#   2.24 s with IMAGE_MAX_TOKENS 1024, 1.55 s with 768. Decode 74 t/s.
#   Vega 7 Vulkan: 1920x1080 37.7 s before, 32.0 s full, 12.6 s at 1024,
#   8.5 s at 768, 5.3 s at 512. Decode 14-16 t/s.
# Click and reading tests were equally weak at every resolution (2-4 of 14
# buttons, same coordinates as the unmodified reference runtime), so 1024
# tokens (~1 MP) is the default.
#
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# ============================================================================
# EDITABLE CONFIGURATION -- only edit values in this section
# ============================================================================

BACKEND="cuda"   # cuda = GTX 1660 Ti (build-main-sm75), vulkan = Vega iGPU (build-magevl-vulkan)
MODEL="$HOME/models/mage-vl-cowork/hansda/Mage-VL-Cowork-Q4_K_M.gguf"
MMPROJ="$HOME/models/mage-vl-cowork/hansda/mmproj-Mage-VL-Cowork-Q8_0.gguf"  # F16: 35-39 % slower encode on CUDA, +300 MiB

HOST="0.0.0.0"
PORT="8080"
ALIAS="mage-vl-cowork"

CTX_SIZE="12288"          # 3 screenshots of ~1000-2000 tokens + prompt; f16 KV peak 5.3 GiB on CUDA
BATCH="2048"
UBATCH_CUDA="2048"        # image prefill 5.66 -> 5.57 s vs 1024 at 1080p
UBATCH_VULKAN="1024"      # 512/1024/2048 equal on Vega
THREADS="8"
CACHE_TYPE_K="f16"        # q8_0 at 16k decodes 15 % slower on the 1660 Ti
CACHE_TYPE_V="f16"
FLASH_ATTN="on"           # Vega: prefill 135 vs 155 t/s without, decode 11.6 vs 10.5 with: kept on

IMAGE_MAX_TOKENS="1024"   # 0 = model limit (4 MP, ~3900 tokens); 768/512 are faster still
MTMD_FA_PREC="f16"        # LLAMA_MTMD_FA_PREC: f16 = hfma2 KQ sums in the vision attention (CUDA -10 % encode)
SPEC_TYPE="ngram-mod"     # no drafts on a first answer, repeats Action/tool_call formats in a running loop

TEMP="0.7"                # model card
TOP_P="0.8"
TOP_K="20"

# ============================================================================

case "$BACKEND" in
  cuda)
    SERVER="$PROJECT_ROOT/build-main-sm75/bin/llama-server"
    UBATCH="$UBATCH_CUDA"
    export CUDA_VISIBLE_DEVICES="0"
    unset GGML_VK_VISIBLE_DEVICES VK_ICD_FILENAMES MESA_VK_DEVICE_SELECT GGML_CUDA_DISABLE
    ;;
  vulkan)
    SERVER="$PROJECT_ROOT/build-magevl-vulkan/bin/llama-server"
    UBATCH="$UBATCH_VULKAN"
    export GGML_VK_VISIBLE_DEVICES="0"
    export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/radeon_icd.json"
    export MESA_VK_DEVICE_SELECT="1002:1636"
    export GGML_CUDA_DISABLE="1"
    ;;
  *) echo "BACKEND must be cuda or vulkan" >&2; exit 2 ;;
esac

[[ -x "$SERVER" ]] || { echo "missing $SERVER" >&2; exit 1; }
[[ -f "$MODEL" && -f "$MMPROJ" ]] || { echo "missing model or mmproj" >&2; exit 1; }

export LLAMA_ARG_DECISION_SEQS="0"   # the 8-sequence decision context OOMs next to the projector
export LLAMA_MTMD_FA_PREC="$MTMD_FA_PREC"
export LLAMA_ARG_SPEC_TYPE="$SPEC_TYPE"
export LLAMA_CMOE_PREFILL_STEPDOWN="1"
export LLAMA_KVFLASH="0"             # image tokens share one batch; KVFlash rejects it

ARGS=(
  --host "$HOST" --port "$PORT" --alias "$ALIAS"
  -m "$MODEL" --mmproj "$MMPROJ"
  -ngl 99 -fa "$FLASH_ATTN" -c "$CTX_SIZE" -np 1 -b "$BATCH" -ub "$UBATCH" -t "$THREADS"
  -ctk "$CACHE_TYPE_K" -ctv "$CACHE_TYPE_V"
  --jinja --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --min-p 0
)
if [[ "$IMAGE_MAX_TOKENS" != "0" ]]; then
  ARGS+=(--image-max-tokens "$IMAGE_MAX_TOKENS")
fi

echo "Mage-VL-Cowork on $BACKEND: ctx $CTX_SIZE, image tokens <= ${IMAGE_MAX_TOKENS/#0/model limit}, mtmd FA prec $MTMD_FA_PREC"
exec "$SERVER" "${ARGS[@]}"
