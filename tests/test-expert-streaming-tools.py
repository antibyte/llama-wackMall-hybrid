#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bench_expert_compute import aggregate as aggregate_compute_runs  # noqa: E402
from bench_expert_transport import aggregate_runs  # noqa: E402
from simulate_expert_streaming import Timing, rank_precision, simulate  # noqa: E402


def transport_run(device_ms: float) -> dict[str, object]:
    return {
        "schema": "llama-wackmall-expert-transport-v1",
        "device": {"name": "test"},
        "config": {"repeats": 3},
        "segments": [
            {
                "bytes": 160,
                "results": [
                    {
                        "mode": "pinned_h2d",
                        "device_ms": {"median": device_ms},
                        "wall_ms": {"median": device_ms + 0.01},
                        "stage_ms": {"median": 0.0},
                        "compute_ms": {"median": 0.0},
                        "exposed_copy_ms": {"median": 0.0},
                        "hidden_copy_ratio": {"median": 0.0},
                        "device_gib_per_s_median": 1.0,
                    }
                ],
            }
        ],
    }


def compute_run(layer: int, latency_ms: float, queued_ms: float) -> dict[str, object]:
    return {
        "schema": "llama-wackmall-expert-compute-v1",
        "model": "/models/test.gguf",
        "device": "test",
        "config": {"layer": layer},
        "results": [
            {
                "mode": "full",
                "weight_bytes": 160,
                "latency_ms": {"median": latency_ms},
                "queued_ms": {"median": queued_ms},
            }
        ],
    }


class TestTransportAggregation(unittest.TestCase):
    def test_median_of_process_medians(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, value in enumerate((0.3, 0.1, 0.2)):
                path = Path(directory) / f"run{index}.json"
                path.write_text(json.dumps(transport_run(value)), encoding="utf-8")
                paths.append(path)
            summary = aggregate_runs(paths)
        self.assertEqual(summary["run_count"], 3)
        self.assertAlmostEqual(
            summary["results"][0]["device_ms_median_of_medians"], 0.2
        )


class TestComputeAggregation(unittest.TestCase):
    def test_process_and_layer_medians_are_kept_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, (layer, latency, queued) in enumerate(
                ((0, 0.4, 0.3), (0, 0.2, 0.1), (1, 0.6, 0.5))
            ):
                path = Path(directory) / f"run{index}.json"
                path.write_text(
                    json.dumps(compute_run(layer, latency, queued)),
                    encoding="utf-8",
                )
                paths.append(path)
            summary = aggregate_compute_runs(paths)
        self.assertEqual(summary["schema"], "llama-wackmall-expert-compute-summary-v1")
        self.assertEqual(summary["results"][0]["runs"], 2)
        self.assertAlmostEqual(
            summary["results"][0]["latency_ms_median_of_medians"], 0.3
        )
        self.assertAlmostEqual(summary["by_size_mode"][0]["latency_ms_median"], 0.45)


class TestStreamingSimulation(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "request_index": 0,
                "token_index": token,
                "target_layer": 0,
                "actual": [7],
                "actual_fixed": [False],
                "predicted": [7, 8],
                "predicted_fixed": [False, False],
            }
            for token in range(2)
        ]
        self.timing = {0: Timing(1.0, 0.6, 0.4)}
        self.layout = {0: {"gate_up": 96, "down": 64, "full": 160}}
        self.transport = {(160, "pinned_h2d"): 0.2}
        self.gpu = {(160, "full"): 0.25}
        self.precision = rank_precision(self.records, 1, 2)

    def test_rank_precision_is_a_probability(self) -> None:
        self.assertEqual(self.precision[0], [1.0, 0.0])

    def run_simulation(self, lead_ms: float | None) -> dict[str, object]:
        return simulate(
            self.records,
            1,
            self.timing,
            self.layout,
            self.transport,
            self.gpu,
            self.precision,
            policy="predictor",
            mode="full",
            arena_kind="global",
            arena_slots=1,
            top_m=2,
            max_copies_layer=1,
            max_bytes_token=1024,
            lead_ms=lead_ms,
        )

    def test_ready_copy_is_reused(self) -> None:
        result = self.run_simulation(None)
        self.assertEqual(result["copies"], 1)
        self.assertEqual(result["useful_copies"], 1)
        self.assertEqual(result["ready_hits"], 2)
        self.assertAlmostEqual(result["theoretical_net_saving_ms"], 1.5)

    def test_late_copy_falls_back_then_becomes_resident(self) -> None:
        result = self.run_simulation(0.1)
        self.assertEqual(result["late_hits"], 1)
        self.assertEqual(result["ready_hits"], 1)
        self.assertAlmostEqual(result["exposed_copy_ms"], 0.1)
        self.assertAlmostEqual(result["theoretical_net_saving_ms"], 0.65)

    def test_non_positive_gpu_benefit_schedules_nothing(self) -> None:
        self.gpu[(160, "full")] = 1.1
        result = self.run_simulation(None)
        self.assertEqual(result["copies"], 0)
        self.assertEqual(result["ready_hits"], 0)


if __name__ == "__main__":
    unittest.main()
