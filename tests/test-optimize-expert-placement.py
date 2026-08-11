#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "optimize_expert_placement", TOOLS / "optimize_expert_placement.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OptimizeExpertPlacementTests(unittest.TestCase):
    def test_quantized_tensor_sizes(self) -> None:
        dims = (2048, 512, 256)
        self.assertEqual(MODULE.tensor_nbytes(dims, 12), 150_994_944)
        down = (512, 2048, 256)
        self.assertEqual(MODULE.tensor_nbytes(down, 13), 184_549_376)
        self.assertEqual(MODULE.tensor_nbytes(down, 14), 220_200_960)

    def test_unsupported_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(MODULE.GGUFError, "unsupported ggml tensor type"):
            MODULE.tensor_nbytes((32, 1), 999)

    def test_expert_slot_bytes_ignore_appended_mtp_layer(self) -> None:
        tensor = MODULE.TensorInfo
        tensors = [
            tensor("blk.0.ffn_gate_exps.weight", (8, 2, 4), 0, 0, 256),
            tensor("blk.0.ffn_down_exps.weight", (2, 8, 4), 0, 256, 256),
            tensor("blk.1.ffn_gate_up_exps.weight", (8, 4, 4), 0, 512, 512),
            tensor("blk.2.ffn_gate_exps.weight", (8, 2, 4), 0, 1024, 256),
        ]
        self.assertEqual(MODULE.expert_slot_bytes(tensors, 2, 4), (128, 128))

    def test_counts_per_byte_prefers_cheap_layer(self) -> None:
        result = MODULE.optimize(
            [[100, 90, 1], [100, 99, 98]],
            (1, 2),
            (Fraction(1), Fraction(1)),
            fixed_budget_bytes=5,
            min_slots=1,
            max_slots=3,
            objective="counts-per-byte",
        )
        self.assertEqual(result.slots, (3, 1))
        self.assertEqual(result.fixed_bytes_used, 5)

    def test_cost_objective_can_prefer_expensive_layer(self) -> None:
        result = MODULE.optimize(
            [[100, 90, 1], [100, 99, 98]],
            (1, 2),
            (Fraction(1), Fraction(1)),
            fixed_budget_bytes=5,
            min_slots=1,
            max_slots=3,
            objective="counts",
        )
        self.assertEqual(result.slots, (1, 2))

    def test_layer_cost_multiplier_is_applied(self) -> None:
        result = MODULE.optimize(
            [[100, 90], [100, 60]],
            (1, 1),
            (Fraction(1), Fraction(2)),
            fixed_budget_bytes=3,
            min_slots=1,
            max_slots=2,
            objective="counts-per-byte",
        )
        self.assertEqual(result.slots, (1, 2))

    def test_timing_json_becomes_per_selection_cost(self) -> None:
        data = {
            "timing_enabled": True,
            "layers": [
                {"layer": 0, "cold_hits": 10, "cpu_cold_ms": 2.5},
                {"layer": 1, "cold_hits": 4, "cpu_cold_ms": 3.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            costs = MODULE.read_layer_costs_json(path, 2)
        self.assertEqual(costs, (Fraction(1, 4), Fraction(3, 4)))

    def test_floor_must_fit(self) -> None:
        with self.assertRaisesRegex(MODULE.ProfileError, "cannot hold min_slots"):
            MODULE.optimize(
                [[1, 0], [1, 0]],
                (10, 10),
                (Fraction(1), Fraction(1)),
                fixed_budget_bytes=19,
                min_slots=1,
                max_slots=2,
                objective="counts-per-byte",
            )

    def test_manifest_refuses_overwrite(self) -> None:
        result = MODULE.PlacementResult((1,), ((0,),), 10, 10, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "placement.csv"
            kwargs = dict(
                model_architecture="test",
                model_file_size=123,
                n_expert=2,
                profile_sha256="0" * 64,
                objective="counts-per-byte",
                min_slots=1,
                max_slots=2,
                slot_bytes=(10,),
                result=result,
            )
            MODULE.write_manifest(output, **kwargs)
            with self.assertRaisesRegex(MODULE.ProfileError, "refusing to overwrite"):
                MODULE.write_manifest(output, **kwargs)


if __name__ == "__main__":
    unittest.main()
