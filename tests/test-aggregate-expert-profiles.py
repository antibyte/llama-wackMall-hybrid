#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "aggregate_expert_profiles", TOOLS / "aggregate_expert_profiles.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AggregateExpertProfilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dims = MODULE.ProfileDimensions(2, 3, 1)

    def test_apportion_is_exact_and_deterministic(self) -> None:
        values = [Fraction(10, 3), Fraction(10, 3), Fraction(10, 3)]
        self.assertEqual(MODULE.apportion(values, 10), [4, 3, 3])

    def test_per_layer_normalization_balances_prompt_lengths(self) -> None:
        short = [[9, 1, 0], [0, 5, 5]]
        long = [[10, 90, 0], [100, 0, 0]]
        result = MODULE.aggregate_per_layer(
            [short, long], [Fraction(1), Fraction(1)], self.dims, 1000
        )
        self.assertEqual(result[0], [500, 500, 0])
        self.assertEqual(result[1], [500, 250, 250])
        self.assertEqual([sum(row) for row in result], [1000, 1000])

    def test_weights_are_explicit(self) -> None:
        first = [[10, 0, 0], [10, 0, 0]]
        second = [[0, 10, 0], [0, 10, 0]]
        result = MODULE.aggregate_per_layer(
            [first, second], [Fraction(3), Fraction(1)], self.dims, 100
        )
        self.assertEqual(result, [[75, 25, 0], [75, 25, 0]])

    def test_empty_layer_is_rejected(self) -> None:
        with self.assertRaisesRegex(MODULE.ProfileError, "no selections for layer 1"):
            MODULE.aggregate_per_layer(
                [[[1, 0, 0], [0, 0, 0]]], [Fraction(1)], self.dims, 100
            )

    def test_global_normalization_has_one_exact_budget(self) -> None:
        result = MODULE.aggregate_global(
            [[[5, 5, 0], [10, 0, 0]]], [Fraction(1)], self.dims, 100
        )
        self.assertEqual(sum(sum(row) for row in result), 100)
        self.assertEqual(result, [[25, 25, 0], [50, 0, 0]])

    def test_top_k_coverage(self) -> None:
        coverage = MODULE.top_k_coverage([[7, 2, 1], [4, 3, 3]], 1)
        self.assertEqual(coverage, Fraction(11, 20))


if __name__ == "__main__":
    unittest.main()
