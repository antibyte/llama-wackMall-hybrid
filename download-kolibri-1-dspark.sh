#!/usr/bin/env bash
# Fetch the Kolibri-1 DSpark draft (intuit/Kolibri-1-dspark, speculators 0.8 format)
# and convert it to $HOME/models/kolibri-1/Kolibri-1-DSpark-Q8_0.gguf.
#
# The checkpoint has no embeddings or lm_head; the draft uses Kolibri's own at runtime.
# The tokenizer is copied from the Kolibri GGUF (run ./download-kolibri-1.sh first).
# Needs a Python with torch, safetensors and numpy (KOLIBRI_PYTHON).
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEST="${KOLIBRI_MODEL_DIR:-$HOME/models/kolibri-1}"
TARGET="${KOLIBRI_MODEL:-$DEST/Kolibri-1-Q4_K_M.gguf}"
PYTHON="${KOLIBRI_PYTHON:-python3}"
QUANTIZE="${KOLIBRI_QUANTIZE:-$PROJECT_ROOT/build-main-sm75/bin/llama-quantize}"
REPO='https://huggingface.co/intuit/Kolibri-1-dspark/resolve/a59d5884479d5391db9d71a62391bf0f85b2d15f'
SHA256='c787c98d5811d6a9b2fb1391ca062888fe50c4953aaddf775028af7fe762adf7'

[[ -f "$TARGET" ]] || { echo "target GGUF not found: $TARGET (run ./download-kolibri-1.sh)" >&2; exit 1; }
[[ -x "$QUANTIZE" ]] || { echo "llama-quantize not found: $QUANTIZE" >&2; exit 1; }

mkdir -p "$DEST/dspark" "$DEST/dspark-src" "$DEST/dspark-target-meta"

for f in README.md config.json config.py; do
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
  "vocab_size": 128000,
  "num_hidden_layers": 50
}
EOF

# speculators -> DFlash config:
# - aux_hidden_state_layer_ids [4,14,24,34,44] index HF hidden_states (k = input of layer k).
#   dflash_config.target_layer_ids use the z-lab convention (k = output of layer k), hence -1.
# - sliding_window with sliding_window_non_causal=false makes the block attention causal.
# - sample_from_anchor=true is the converter default (anchor-first, 8 draft slots).
cat > "$DEST/dspark-src/config.json" <<'EOF'
{
  "architectures": ["Qwen3DSparkModel"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "block_size": 8,
  "dflash_config": {
    "causal_head": true,
    "confidence_head_with_markov": true,
    "enable_confidence_head": true,
    "markov_head_type": "vanilla",
    "markov_rank": 256,
    "mask_token_id": 127998,
    "target_layer_ids": [3, 13, 23, 33, 43]
  },
  "draft_vocab_size": 128000,
  "dtype": "bfloat16",
  "eos_token_id": 127906,
  "pad_token_id": 127901,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 6144,
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
  "max_position_embeddings": 262144,
  "model_type": "qwen3",
  "num_attention_heads": 20,
  "num_hidden_layers": 5,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
  "sliding_window": 2048,
  "use_sliding_window": true,
  "tie_word_embeddings": false,
  "use_cache": true,
  "vocab_size": 128000,
  "markov_rank": 256,
  "enable_confidence_head": true,
  "confidence_head_with_markov": true
}
EOF
ln -sf ../dspark/model.safetensors "$DEST/dspark-src/model.safetensors"

LLAMA_DSPARK_VOCAB_GGUF="$TARGET" "$PYTHON" "$PROJECT_ROOT/convert_hf_to_gguf.py" "$DEST/dspark-src" \
    --target-model-dir "$DEST/dspark-target-meta" --outtype bf16 --outfile "$DEST/Kolibri-1-DSpark-BF16.gguf"
"$QUANTIZE" "$DEST/Kolibri-1-DSpark-BF16.gguf" "$DEST/Kolibri-1-DSpark-Q8_0.gguf" Q8_0

echo "draft ready: $DEST/Kolibri-1-DSpark-Q8_0.gguf"
