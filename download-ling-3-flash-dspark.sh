#!/usr/bin/env bash
# Fetch the Ling-3.0-flash DSpark draft (inclusionAI/Ling-3.0-flash-dspark, 5 full-attention
# layers, block 8, vanilla Markov head) and convert it to
# $HOME/models/ling-3-flash/Ling-3.0-flash-DSpark-Q8_0.gguf.
#
# The checkpoint carries its own embed_tokens and lm_head (1.6 GiB in BF16). They are dropped:
# the draft uses the target's at runtime, which keeps the VRAM for the expert hot set.
# The tokenizer is copied from the Ling GGUF (run ./download-ling-3-flash.sh first).
# Needs a Python with torch, safetensors and numpy (LING3_PYTHON).
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEST="${LING3_MODEL_DIR:-$HOME/models/ling-3-flash}"
TARGET="${LING3_MODEL:-$DEST/Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf}"
PYTHON="${LING3_PYTHON:-python3}"
QUANTIZE="${LING3_QUANTIZE:-$PROJECT_ROOT/build-main-sm75/bin/llama-quantize}"
REPO='https://huggingface.co/inclusionAI/Ling-3.0-flash-dspark/resolve/d60b0e1291dacb256194c6d48154398730cf8986'
SHA256='0885659804b3e4460f4bcd50c292923ea4f60e5a83745f96ff3d0736a5e5ddb5'

[[ -f "$TARGET" ]] || { echo "target GGUF not found: $TARGET (run ./download-ling-3-flash.sh)" >&2; exit 1; }
[[ -x "$QUANTIZE" ]] || { echo "llama-quantize not found: $QUANTIZE" >&2; exit 1; }

mkdir -p "$DEST/dspark" "$DEST/dspark-src" "$DEST/dspark-target-meta"

for f in README.md config.json; do
    [[ -f "$DEST/dspark/$f" ]] || wget -q -O "$DEST/dspark/$f" "$REPO/$f?download=true"
done
if ! echo "$SHA256  $DEST/dspark/model.safetensors" | sha256sum -c --status 2>/dev/null; then
    wget -c --retry-connrefused --tries=20 --timeout=60 --waitretry=5 \
        --progress=dot:giga -O "$DEST/dspark/model.safetensors" "$REPO/model.safetensors?download=true"
    echo "$SHA256  $DEST/dspark/model.safetensors" | sha256sum -c
fi

# The converter validates the draft against the target's HF config; only these keys are read.
cat > "$DEST/dspark-target-meta/config.json" <<'EOF'
{
  "architectures": ["Qwen3ForCausalLM"],
  "hidden_size": 2560,
  "vocab_size": 157184,
  "num_hidden_layers": 42
}
EOF

# Flat LingDSparkModel config -> DFlash config. target_layer_ids [1, 11, 23, 29, 35] already
# use the z-lab convention (k = output of layer k); the converter adds 1.
# Full attention without a sliding window, so no causal head.
cat > "$DEST/dspark-src/config.json" <<'EOF'
{
  "architectures": ["Qwen3DSparkModel"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "block_size": 8,
  "dflash_config": {
    "confidence_head_with_markov": true,
    "enable_confidence_head": true,
    "markov_head_type": "vanilla",
    "markov_rank": 256,
    "mask_token_id": 156892,
    "target_layer_ids": [1, 11, 23, 29, 35]
  },
  "draft_vocab_size": 157184,
  "dtype": "bfloat16",
  "eos_token_id": 156895,
  "pad_token_id": 156892,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 6144,
  "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"],
  "max_position_embeddings": 262144,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 32,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 6000000.0, "rope_type": "default"},
  "rope_theta": 6000000,
  "tie_word_embeddings": false,
  "use_cache": true,
  "vocab_size": 157184,
  "markov_rank": 256,
  "enable_confidence_head": true,
  "confidence_head_with_markov": true
}
EOF

if [[ ! -f "$DEST/dspark-src/model.safetensors" ]]; then
    "$PYTHON" - "$DEST/dspark/model.safetensors" "$DEST/dspark-src/model.safetensors" <<'EOF'
import sys
from safetensors.torch import load_file, save_file

tensors = load_file(sys.argv[1])
for name in ("embed_tokens.weight", "lm_head.weight"):
    del tensors[name]
save_file(tensors, sys.argv[2], metadata={"format": "pt"})
EOF
fi

LLAMA_DSPARK_VOCAB_GGUF="$TARGET" "$PYTHON" "$PROJECT_ROOT/convert_hf_to_gguf.py" "$DEST/dspark-src" \
    --target-model-dir "$DEST/dspark-target-meta" --outtype bf16 --outfile "$DEST/Ling-3.0-flash-DSpark-BF16.gguf"
"$QUANTIZE" "$DEST/Ling-3.0-flash-DSpark-BF16.gguf" "$DEST/Ling-3.0-flash-DSpark-Q8_0.gguf" Q8_0

echo "draft ready: $DEST/Ling-3.0-flash-DSpark-Q8_0.gguf"
