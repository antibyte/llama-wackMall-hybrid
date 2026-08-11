#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conversion.qwen import DFlashModel, Qwen3Model  # noqa: E402


class DFlashConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.target_model_dir = Path(self.temp.name)
        self.target_config = {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "vocab_size": 248320,
        }
        self.dflash_config = {
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
        }

    def tearDown(self):
        self.temp.cleanup()

    def make_model(self) -> DFlashModel:
        (self.target_model_dir / "config.json").write_text(json.dumps(self.target_config), encoding="utf-8")
        model = DFlashModel.__new__(DFlashModel)
        model.hparams = {
            "dflash_config": self.dflash_config,
            "hidden_size": 2048,
            "num_hidden_layers": 6,
            "vocab_size": 248320,
        }
        model.target_model_dir = self.target_model_dir
        model.gguf_writer = Mock()
        return model

    def test_writes_nested_block_size_and_shifted_target_layers(self):
        model = self.make_model()
        model.hparams["block_size"] = 99

        with patch.object(Qwen3Model, "set_gguf_parameters"):
            model.set_gguf_parameters()

        model.gguf_writer.add_block_size.assert_called_once_with(16)
        model.gguf_writer.add_target_layers.assert_called_once_with([2, 7, 12, 17, 23, 28, 33, 38])

    def test_requires_load_bearing_dflash_metadata(self):
        for key in ("block_size", "mask_token_id", "target_layer_ids"):
            with self.subTest(key=key):
                model = self.make_model()
                del self.dflash_config[key]
                with self.assertRaisesRegex(ValueError, key):
                    model._validate_config()
                self.dflash_config = {
                    "block_size": 16,
                    "mask_token_id": 248077,
                    "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
                }

        model = self.make_model()
        self.dflash_config["block_size"] = 1
        with self.assertRaisesRegex(ValueError, "block_size"):
            model._validate_config()

    def test_rejects_duplicate_and_out_of_range_target_layers(self):
        model = self.make_model()
        self.dflash_config["target_layer_ids"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            model._validate_config()

        self.dflash_config["target_layer_ids"] = [39]
        with self.assertRaisesRegex(ValueError, "block count"):
            model._validate_config()

    def test_rejects_incompatible_target_shape_and_vocabulary(self):
        model = self.make_model()
        self.target_config["hidden_size"] = 4096
        (self.target_model_dir / "config.json").write_text(json.dumps(self.target_config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hidden size"):
            model._validate_config()

        self.target_config["hidden_size"] = 2048
        self.target_config["vocab_size"] = 248319
        (self.target_model_dir / "config.json").write_text(json.dumps(self.target_config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "vocabulary size"):
            model._validate_config()

    def test_rejects_mask_token_outside_target_vocabulary(self):
        model = self.make_model()
        self.dflash_config["mask_token_id"] = 248320
        with self.assertRaisesRegex(ValueError, "mask token ID"):
            model._validate_config()

    def test_rejects_invalid_sliding_window_pattern(self):
        model = self.make_model()
        model.hparams.update({
            "use_sliding_window": True,
            "sliding_window": 4096,
            "layer_types": ["sliding_attention"] * 5,
        })
        with self.assertRaisesRegex(ValueError, "6 entries"):
            model._validate_config()

    def test_accepts_nested_target_text_config(self):
        self.target_config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": self.target_config,
        }
        model = self.make_model()
        dflash_config, target_hparams = model._validate_config()
        self.assertEqual(dflash_config["block_size"], 16)
        self.assertEqual(target_hparams["num_hidden_layers"], 40)

    def test_validates_feature_fusion_weight_shape(self):
        model = self.make_model()
        tensor = SimpleNamespace(shape=(2048, 16384))
        with patch.object(Qwen3Model, "modify_tensors", return_value=[("fc.weight", tensor)]):
            self.assertEqual(list(model.modify_tensors(tensor, "model.fc.weight", None)), [("fc.weight", tensor)])

        tensor = SimpleNamespace(shape=(2048, 14336))
        with self.assertRaisesRegex(ValueError, "Unexpected DFlash FC shape"):
            list(model.modify_tensors(tensor, "model.fc.weight", None))


if __name__ == "__main__":
    unittest.main()
