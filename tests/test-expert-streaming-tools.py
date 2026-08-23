#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bench_expert_bw import profile_from_summary, recommend  # noqa: E402
from bench_expert_compute import aggregate as aggregate_compute_runs  # noqa: E402
from bench_expert_bridge_ab import (  # noqa: E402
    BridgeBenchError,
    bootstrap_median_ci,
    validate_pair,
)
from bench_expert_transport import aggregate_runs  # noqa: E402
from simulate_expert_bridge_cache import simulate_lru  # noqa: E402
from simulate_expert_decode_cache import (  # noqa: E402
    admit_budget,
    admit_from_counts,
    attach_costs,
    build_parser as build_decode_cache_parser,
    exposed_ms,
    run as run_decode_cache,
    simulate_fixed_overflow,
    simulate_lru as simulate_decode_lru,
    simulate_static,
)
from simulate_expert_streaming import Timing, rank_precision, read_traces, simulate  # noqa: E402


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


class TestBridgeAB(unittest.TestCase):
    def test_bootstrap_constant_delta(self) -> None:
        low, high = bootstrap_median_ci([1.25] * 7, samples=100, seed=7)
        self.assertEqual((low, high), (1.25, 1.25))

    def test_pair_requires_identical_tokens(self) -> None:
        baseline = {
            "output_sha256": "output",
            "token_sha256": "baseline",
            "predicted_tokens": "256",
            "mtp_acceptance": "0.75",
            "mean_accepted_length": "2.5",
        }
        bridge = dict(baseline, token_sha256="bridge")
        with self.assertRaisesRegex(BridgeBenchError, "token_sha256"):
            validate_pair(baseline, bridge)

    def test_pair_accepts_identical_results(self) -> None:
        result = {
            "output_sha256": "output",
            "token_sha256": "tokens",
            "predicted_tokens": "256",
            "mtp_acceptance": "0.75",
            "mean_accepted_length": "2.5",
        }
        validate_pair(result, dict(result))


class TestBridgeCacheSimulation(unittest.TestCase):
    def test_lru_reuses_candidates_until_eviction(self) -> None:
        steps = [([1, 2], {1}), ([1, 3], {1, 3}), ([2], {2})]
        result = simulate_lru(steps, slots=2, expert_bytes=100)
        self.assertEqual(result["candidates"], 5)
        self.assertEqual(result["copies"], 4)
        self.assertEqual(result["avoided_copies"], 1)
        self.assertEqual(result["useful_cache_hits"], 1)
        self.assertEqual(result["avoided_bytes"], 100)


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


class TestDecodeCacheSimulation(unittest.TestCase):
    def setUp(self) -> None:
        # One active layer alternates two experts. Layer 1 is idle, so a global
        # pool of 2 slots can hold both experts while per-layer S=1 cannot.
        self.records = [
            {
                "target_layer": 0,
                "actual": [1 if token % 2 == 0 else 2],
                "actual_fixed": [token % 2 == 0],
            }
            for token in range(4)
        ]

    def test_static_counts_only_fixed_hits(self) -> None:
        result = simulate_static(self.records)
        self.assertEqual(result["selected"], 4)
        self.assertEqual(result["hits"], 2)
        self.assertAlmostEqual(result["miss_rate"], 0.5)

    def test_qstar_admit_does_not_cache_unfilled_misses(self) -> None:
        result = simulate_decode_lru(
            self.records,
            global_slots=2,
            slots_per_layer=None,
            n_layer=2,
            admit_fraction=0.0,
            min_fill=False,
        )
        self.assertEqual(result["hits"], 0)
        self.assertEqual(result["copy_steps"], [0, 0, 0, 0])

    def test_per_layer_lru_cannot_share_slots(self) -> None:
        result = simulate_decode_lru(
            self.records, global_slots=None, slots_per_layer=1, n_layer=2
        )
        self.assertEqual(result["policy"], "per-layer-lru")
        self.assertEqual(result["hits"], 0)
        self.assertEqual(result["misses"], 4)

    def test_qstar_overlap_uses_the_slower_branch(self) -> None:
        self.assertAlmostEqual(exposed_ms(4, 1.0, 8.0, 0.25, False), 8.0)
        self.assertAlmostEqual(exposed_ms(4, 1.0, 0.2, 0.25, False), 3.0)
        self.assertAlmostEqual(exposed_ms(1, 1.0, 8.0, 0.1, False), 1.0)
        self.assertAlmostEqual(exposed_ms(1, 1.0, 8.0, 0.1, True), 8.0)

    def test_attach_costs_keeps_cpu_only_cheaper_than_fill(self) -> None:
        sim = simulate_decode_lru(
            self.records, global_slots=2, slots_per_layer=None, n_layer=2
        )
        priced = attach_costs(sim, cpu_ms=1.0, copy_ms=8.0, q_star=0.1)
        self.assertNotIn("miss_steps", priced)
        self.assertLess(priced["cpu_only_ms"], priced["fill_all_ms"])

    def test_global_lru_can_steal_idle_layer_slots(self) -> None:
        result = simulate_decode_lru(
            self.records, global_slots=2, slots_per_layer=None, n_layer=2
        )
        self.assertEqual(result["policy"], "global-lru")
        self.assertEqual(result["hits"], 2)
        self.assertEqual(result["misses"], 2)

    def test_overflow_w0_matches_static(self) -> None:
        static = simulate_static(self.records)
        overflow = simulate_fixed_overflow(
            self.records, warm_slots=0, n_layer=2, global_warm=False
        )
        self.assertEqual(overflow["policy"], "fixed-overflow")
        self.assertEqual(overflow["hits"], static["hits"])
        self.assertEqual(overflow["misses"], static["misses"])
        self.assertEqual(overflow["copy_steps"], [0, 0, 0, 0])

    def test_overflow_keeps_fixed_hits_and_caches_cold(self) -> None:
        result = simulate_fixed_overflow(
            self.records, warm_slots=1, n_layer=2, global_warm=False
        )
        self.assertEqual(result["hits"], 3)
        self.assertEqual(result["misses"], 1)
        self.assertEqual(result["copy_steps"], [0, 1, 0, 0])

    def test_admit_budget_matches_runtime(self) -> None:
        self.assertEqual(admit_budget(0, 4, 1.0), 0)
        self.assertEqual(admit_budget(6, 0, 1.0), 0)
        self.assertEqual(admit_budget(6, 4, 0.0), 0)
        self.assertEqual(admit_budget(6, 4, 1.0), 4)
        self.assertEqual(admit_budget(1, 4, 0.1029), 0)
        self.assertEqual(admit_budget(5, 4, 0.1029), 1)
        self.assertEqual(admit_budget(8, 4, 0.5), 4)

    def test_prefill_counts_do_not_admit(self) -> None:
        self.assertTrue(admit_from_counts(1, 32))
        self.assertTrue(admit_from_counts(2, 32))
        self.assertFalse(admit_from_counts(1856, 32))
        self.assertFalse(admit_from_counts(0, 32))
        self.assertFalse(admit_from_counts(8, 0))

    def test_overflow_caps_copies_at_w(self) -> None:
        records = [
            {
                "target_layer": 0,
                "actual": [1, 2, 3],
                "actual_fixed": [False, False, False],
            }
        ]
        result = simulate_fixed_overflow(records, warm_slots=1, n_layer=1)
        self.assertEqual(result["misses"], 3)
        self.assertEqual(result["copy_steps"], [1])
        self.assertEqual(result["hits"], 0)

    def test_overflow_qstar_zero_does_not_fill(self) -> None:
        static = simulate_static(self.records)
        result = simulate_fixed_overflow(
            self.records,
            warm_slots=1,
            n_layer=2,
            admit_fraction=0.0,
        )
        self.assertEqual(result["policy"], "fixed-overflow-q*")
        self.assertEqual(result["hits"], static["hits"])
        self.assertEqual(result["copy_steps"], [0, 0, 0, 0])


class TestDecodeCacheFixture(unittest.TestCase):
    fixture = ROOT / "tests" / "fixtures" / "expert-decode-cache-mini.json"

    def test_fixture_overflow_hits_repeated_cold_expert(self) -> None:
        records, config = read_traces([self.fixture])
        self.assertEqual(config["n_layer"], 2)
        self.assertEqual(len(records), 6)
        static = simulate_static(records)
        overflow = simulate_fixed_overflow(records, warm_slots=1, n_layer=2)
        self.assertEqual(static["selected"], 6)
        self.assertEqual(static["misses"], 4)
        self.assertEqual(overflow["misses"], 2)
        self.assertEqual(overflow["hits"], 4)

    def test_fixture_cli_writes_overflow_policies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.json"
            args = build_decode_cache_parser().parse_args(
                [
                    "--trace",
                    str(self.fixture),
                    "--output",
                    str(output),
                    "--slots-per-layer",
                    "1",
                    "--warm-slots",
                    "1",
                    "--cpu-ms",
                    "0.0456",
                    "--copy-ms",
                    "0.287",
                    "--q-star",
                    "0.1029",
                ]
            )
            run_decode_cache(args)
            data = json.loads(output.read_text(encoding="utf-8"))
        policies = [item["policy"] for item in data["simulations"]]
        self.assertIn("static", policies)
        self.assertIn("fixed-overflow", policies)
        self.assertIn("fixed-overflow-global", policies)
        self.assertIn("fixed-overflow-q*", policies)


class TestBwProfile(unittest.TestCase):
    def test_cpu_heavy_keeps_q_star_small(self) -> None:
        self.assertEqual(recommend(40.0, 10.0, 2.0), "cpu-heavy")
        self.assertEqual(recommend(10.0, 40.0, 2.0), "pcie-heavy")
        self.assertEqual(recommend(20.0, 18.0, 2.0), "hybrid")

    def test_cpu_timing_bounds_select_full_expert_bytes(self) -> None:
        summary = {
            "schema": "llama-wackmall-expert-transport-summary-v1",
            "results": [
                {
                    "mode": "pinned_h2d",
                    "bytes": 100,
                    "device_ms_median_of_medians": 0.01,
                    "compute_ms_median_of_medians": 0.0,
                },
                {
                    "mode": "pinned_h2d",
                    "bytes": 1900544,
                    "device_ms_median_of_medians": 0.287,
                    "compute_ms_median_of_medians": 0.0,
                },
            ],
            "optimistic_transport_lower_bounds": [
                {
                    "layer": 0,
                    "full": {
                        "bytes": 1900544,
                        "cpu_ms_per_cold_selection": 0.0456,
                        "pinned_h2d_ms": 0.287,
                        "copy_to_cpu_ratio": 6.3,
                    }
                }
            ],
        }
        profile = profile_from_summary(summary, threshold=2.0, expert_bytes=None)
        self.assertEqual(profile["expert_bytes"], 1900544)
        self.assertEqual(profile["recommended"], "cpu-heavy")
        self.assertLess(profile["q_star"], 0.25)

    def test_overlap_sets_fetch_fraction(self) -> None:
        summary = {
            "schema": "llama-wackmall-expert-transport-summary-v1",
            "results": [
                {
                    "mode": "pinned_h2d",
                    "bytes": 1900544,
                    "device_ms_median_of_medians": 0.4,
                    "compute_ms_median_of_medians": 0.0,
                },
                {
                    "mode": "pinned_h2d_cpu_overlap",
                    "bytes": 1900544,
                    "device_ms_median_of_medians": 0.5,
                    "compute_ms_median_of_medians": 0.125,
                },
            ],
        }
        profile = profile_from_summary(summary, threshold=2.0, expert_bytes=1900544)
        self.assertEqual(profile["recommended"], "cpu-heavy")
        self.assertAlmostEqual(profile["q_star"], 0.25, places=3)
        self.assertEqual(profile["source"], "pinned_h2d_cpu_overlap")


if __name__ == "__main__":
    unittest.main()
