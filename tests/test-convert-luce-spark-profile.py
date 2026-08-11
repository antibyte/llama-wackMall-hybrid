#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "convert_luce_spark_profile.py"
SPEC = importlib.util.spec_from_file_location("convert_luce_spark_profile", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_value(value):
    if isinstance(value, str):
        return converter.GGUF_TYPE_STRING, _gguf_string(value)
    if isinstance(value, int):
        return converter.GGUF_TYPE_UINT32, struct.pack("<I", value)
    raise TypeError(value)


def write_test_gguf(path: Path, metadata: dict[str, object]) -> None:
    payload = bytearray(b"GGUF")
    payload += struct.pack("<IQQ", 3, 0, len(metadata))
    for key, value in metadata.items():
        value_type, encoded = _gguf_value(value)
        payload += _gguf_string(key)
        payload += struct.pack("<I", value_type)
        payload += encoded
    path.write_bytes(payload)


def write_spark(path: Path, rows: list[list[int]], used: int = 2) -> None:
    path.write_text(
        f"# hotness table: n_layer={len(rows)} n_expert={len(rows[0])} n_expert_used={used}\n"
        "# format: one row per layer, columns are expert activation counts (expert 0..N-1)\n"
        + "".join(",".join(str(value) for value in row) + "\n" for row in rows),
        encoding="utf-8",
    )


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.model = self.dir / "model.gguf"
        write_test_gguf(
            self.model,
            {
                "general.architecture": "testmoe",
                "general.name": "test-model",
                "general.file_type": 15,
                "general.quantization_version": 2,
                "testmoe.block_count": 2,
                "testmoe.expert_count": 4,
                "testmoe.expert_used_count": 2,
                "tokenizer.ggml.tokens": "skipped-value",
            },
        )
        self.spark = self.dir / "input.spark.csv"
        write_spark(self.spark, [[1, 0, 3, 4], [5, 6, 0, 8]])

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_conversion_is_complete_and_exact(self):
        profile = converter.parse_spark_profile(self.spark)
        metadata = converter.read_gguf_metadata(self.model)
        converter.validate_dimensions(profile, metadata)
        output = self.dir / "output.csv"
        converter.write_wack_profile(profile, output)
        values = converter.validate_wack_profile(
            output,
            profile.dimensions,
            expected=profile,
            require_complete=True,
        )
        self.assertEqual(len(values), 8)
        self.assertEqual(values[(0, 2)], 3)
        self.assertEqual(values[(1, 2)], 0)

    def test_drop_zero_is_sparse_but_exact(self):
        profile = converter.parse_spark_profile(self.spark)
        output = self.dir / "output.csv"
        converter.write_wack_profile(profile, output, drop_zero=True)
        values = converter.validate_wack_profile(
            output, profile.dimensions, expected=profile
        )
        self.assertNotIn((0, 1), values)
        self.assertEqual(len(values), 6)

    def test_placement_seed_preserves_top_order_and_bounds_scores(self):
        profile = converter.parse_spark_profile(self.spark)
        placement = converter.make_placement_seed(profile, 2)
        self.assertEqual(placement.counts[0], [0, 0, 1, 2])
        self.assertEqual(placement.counts[1], [0, 1, 0, 2])
        self.assertEqual(max(max(row) for row in placement.counts), 2)
        self.assertEqual(sum(value > 0 for row in placement.counts for value in row), 4)

    def test_rejects_invalid_placement_slot_count(self):
        profile = converter.parse_spark_profile(self.spark)
        with self.assertRaisesRegex(converter.ProfileError, "placement slots"):
            converter.make_placement_seed(profile, 5)

    def test_rejects_wrong_column_count(self):
        write_spark(self.spark, [[1, 2, 3, 4], [5, 6, 7]])
        with self.assertRaisesRegex(converter.ProfileError, "expert columns"):
            converter.parse_spark_profile(self.spark)

    def test_rejects_negative_and_float_counts(self):
        self.spark.write_text(
            "# hotness table: n_layer=1 n_expert=2 n_expert_used=1\n"
            "# format: one row per layer, columns are expert activation counts (expert 0..N-1)\n"
            "-1,2.0\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(converter.ProfileError, "unsigned decimal"):
            converter.parse_spark_profile(self.spark)

    def test_rejects_count_too_large_for_wackmall(self):
        self.spark.write_text(
            "# hotness table: n_layer=1 n_expert=1 n_expert_used=1\n"
            "# format: one row per layer, columns are expert activation counts (expert 0..N-1)\n"
            f"{1 << 63}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(converter.ProfileError, "supported maximum"):
            converter.parse_spark_profile(self.spark)

    def test_rejects_dimension_mismatch(self):
        write_spark(self.spark, [[1, 2, 3], [4, 5, 6]], used=2)
        profile = converter.parse_spark_profile(self.spark)
        metadata = converter.read_gguf_metadata(self.model)
        with self.assertRaisesRegex(converter.ProfileError, "do not match"):
            converter.validate_dimensions(profile, metadata)

    def test_rejects_existing_output(self):
        profile = converter.parse_spark_profile(self.spark)
        output = self.dir / "output.csv"
        output.write_text("keep me", encoding="utf-8")
        with self.assertRaisesRegex(converter.ProfileError, "refusing to overwrite"):
            converter.write_wack_profile(profile, output)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

    def test_rejects_out_of_range_wack_ids(self):
        output = self.dir / "bad.csv"
        output.write_text("layer,expert,count\n0,4,1\n", encoding="utf-8")
        dims = converter.ProfileDimensions(2, 4, 2)
        with self.assertRaisesRegex(converter.ProfileError, "outside"):
            converter.validate_wack_profile(output, dims)

    def test_rejects_duplicate_wack_ids(self):
        output = self.dir / "bad.csv"
        output.write_text("layer,expert,count\n0,1,1\n0,1,2\n", encoding="utf-8")
        dims = converter.ProfileDimensions(2, 4, 2)
        with self.assertRaisesRegex(converter.ProfileError, "duplicate"):
            converter.validate_wack_profile(output, dims)

    def test_rejects_missing_required_gguf_metadata(self):
        bad = self.dir / "bad.gguf"
        write_test_gguf(bad, {"general.architecture": "testmoe"})
        with self.assertRaisesRegex(converter.GGUFError, "block_count"):
            converter.read_gguf_metadata(bad)

    def test_declared_nextn_layers_are_excluded_from_profile_layers(self):
        mtp = self.dir / "mtp.gguf"
        write_test_gguf(
            mtp,
            {
                "general.architecture": "testmoe",
                "testmoe.block_count": 3,
                "testmoe.nextn_predict_layers": 1,
                "testmoe.expert_count": 4,
                "testmoe.expert_used_count": 2,
            },
        )
        metadata = converter.read_gguf_metadata(mtp)
        self.assertEqual(metadata.block_count_total, 3)
        self.assertEqual(metadata.nextn_predict_layers, 1)
        self.assertEqual(metadata.dimensions.n_layer, 2)
        converter.validate_dimensions(converter.parse_spark_profile(self.spark), metadata)

    def test_rejects_source_target_architecture_mismatch(self):
        source = converter.read_gguf_metadata(self.model)
        other_path = self.dir / "other.gguf"
        write_test_gguf(
            other_path,
            {
                "general.architecture": "othermoe",
                "othermoe.block_count": 2,
                "othermoe.expert_count": 4,
                "othermoe.expert_used_count": 2,
            },
        )
        other = converter.read_gguf_metadata(other_path)
        with self.assertRaisesRegex(converter.ProfileError, "architecture"):
            converter.validate_models_compatible(source, other)

    def test_rejects_source_target_file_type_mismatch(self):
        source = converter.read_gguf_metadata(self.model)
        other_path = self.dir / "other.gguf"
        write_test_gguf(
            other_path,
            {
                "general.architecture": "testmoe",
                "general.file_type": 14,
                "general.quantization_version": 2,
                "testmoe.block_count": 2,
                "testmoe.expert_count": 4,
                "testmoe.expert_used_count": 2,
            },
        )
        other = converter.read_gguf_metadata(other_path)
        with self.assertRaisesRegex(converter.ProfileError, "file type"):
            converter.validate_models_compatible(source, other)

    def test_rejects_source_target_quantization_version_mismatch(self):
        source = converter.read_gguf_metadata(self.model)
        other_path = self.dir / "other.gguf"
        write_test_gguf(
            other_path,
            {
                "general.architecture": "testmoe",
                "general.file_type": 15,
                "general.quantization_version": 3,
                "testmoe.block_count": 2,
                "testmoe.expert_count": 4,
                "testmoe.expert_used_count": 2,
            },
        )
        other = converter.read_gguf_metadata(other_path)
        with self.assertRaisesRegex(converter.ProfileError, "quantization version"):
            converter.validate_models_compatible(source, other)


if __name__ == "__main__":
    unittest.main()
