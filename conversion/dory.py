from __future__ import annotations

import math

from .base import ModelBase, TextModel, gguf


@ModelBase.register("DoryForCausalLM")
class DoryModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DORY

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if n_dims == 1 and any(self.match_model_tensor_name(new_name, key, bid) for key in (
            gguf.MODEL_TENSOR.RESIDUAL_ALPHA,
            gguf.MODEL_TENSOR.ATTN_QK_SCALE,
            gguf.MODEL_TENSOR.ATTN_GATE_SCALE,
            gguf.MODEL_TENSOR.FFN_GATE_SCALE,
            gguf.MODEL_TENSOR.FFN_UP_SCALE,
            gguf.MODEL_TENSOR.OUTPUT_SCALE,
        )):
            return {
                gguf.LlamaFileType.MOSTLY_BF16: gguf.GGMLQuantizationType.BF16,
                gguf.LlamaFileType.MOSTLY_F16: gguf.GGMLQuantizationType.F16,
            }.get(self.ftype, gguf.GGMLQuantizationType.F32)
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def set_vocab(self):
        # Transformers rejects layer type "mlp" when it opens config.json for the tokenizer.
        # The tokenizer files do not depend on that list, so load them from a copy without it.
        import json
        import os
        import shutil
        import tempfile
        from pathlib import Path

        src = Path(self.dir_model)
        td = Path(tempfile.mkdtemp(prefix="dory-tok-"))
        try:
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                target = src / name
                if target.is_file():
                    os.symlink(target, td / name)
            cfg = json.loads((src / "config.json").read_text())
            cfg.pop("layer_types", None)
            cfg.pop("architectures", None)
            cfg.pop("model_type", None)
            (td / "config.json").write_text(json.dumps(cfg))
            saved = self.dir_model
            self.dir_model = td
            try:
                tokens, toktypes, tokpre = self.get_vocab_base()
            finally:
                self.dir_model = saved
        finally:
            shutil.rmtree(td)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        gguf.SpecialVocab(src, load_merges=True).add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        h = self.hparams
        n_in = int(h["n_input_layers"])
        n_re = int(h["n_recurrent_layers"])
        n_out = int(h["n_output_layers"])
        n = int(h["num_hidden_layers"])
        if n_in + n_re + n_out != n:
            raise ValueError("Dory layer counts do not add up")
        types = h["layer_types"]
        if len(types) != n:
            raise ValueError("Dory layer_types length does not match num_hidden_layers")
        kind = {"sliding_attention": "S", "full_attention": "*", "mlp": "-"}
        if any(t not in kind for t in types):
            raise ValueError("unsupported Dory layer type")
        pattern = "".join(kind[t] for t in types)
        # Profile 1 is the long RoPE base, profile 2 is the sliding-window base.
        rope = h["rope_parameters"]
        theta_full = float(rope["full_attention"]["rope_theta"])
        theta_swa = float(rope["sliding_attention"]["rope_theta"])
        profile = [2 if t == "sliding_attention" else 1 for t in types]
        if h.get("tie_word_embeddings", False):
            raise ValueError("Dory requires untied embeddings")
        super().set_gguf_parameters()
        head = int(h["head_dim"])
        gate = h.get("attention_output_gate_logit_scale_init")
        if gate is None:
            gate = math.sqrt(float(h["hidden_size"]))
        w = self.gguf_writer
        w.add_key_length(head)
        w.add_value_length(head)
        w.add_rope_dimension_count(head)
        w.add_rope_freq_base(theta_full)
        w.add_rope_freq_base_2(theta_swa)
        w.add_sliding_window(int(h["sliding_window"]))
        w.add_input_layer_count(n_in)
        w.add_recurrent_layer_count(n_re)
        w.add_output_layer_count(n_out)
        w.add_num_loops(int(h["n_recurrent_loops"]))
        w.add_layer_pattern(pattern)
        w.add_recurrent_kv_cache_mode(h.get("recurrent_kv_cache_mode", "per_loop"))
        w.add_rope_profile(profile)
        w.add_init_std(float(h["initializer_range"]))
        w.add_alpha_init(float(h["ngpt_alpha_init"]))
        w.add_gate_init(float(gate))

    def modify_tensors(self, data_torch, name, bid):
        mapped = self.map_tensor_name(name)
        if mapped is not None and not mapped.endswith(".weight"):
            mapped += ".weight"
        yield mapped, data_torch
