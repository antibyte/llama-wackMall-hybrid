#!/usr/bin/env python3
"""Run counterbalanced baseline/bridge pairs and verify deterministic output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


class BridgeBenchError(RuntimeError):
    pass


BRIDGE_ENV_PREFIX = "GGML_CUDA_EXPERT_BRIDGE_"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_median_ci(
    values: list[float], samples: int = 10000, seed: int = 0
) -> tuple[float, float]:
    if not values or samples < 1:
        raise BridgeBenchError("bootstrap requires values and a positive sample count")
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(samples)
    )
    low = medians[int(0.025 * (samples - 1))]
    high = medians[int(0.975 * (samples - 1))]
    return low, high


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise BridgeBenchError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def numeric_samples(path: Path, field: str) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        try:
            result.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            pass
    return result


def sample_summary(path: Path) -> dict[str, object]:
    def values(field: str) -> list[float]:
        return numeric_samples(path, field)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pstates = [row.get("pstate", "") for row in rows if row.get("pstate")]
    gpu = values("gpu_util_pct")
    clock = values("sm_clock_mhz")
    power = values("power_w")
    temperature = values("temp_c")
    pcie_gen = values("pcie_gen")
    pcie_width = values("pcie_width")
    return {
        "samples": len(rows),
        "gpu_util_avg": statistics.fmean(gpu) if gpu else None,
        "sm_clock_mhz_median": statistics.median(clock) if clock else None,
        "power_w_avg": statistics.fmean(power) if power else None,
        "temp_c_max": max(temperature) if temperature else None,
        "pcie_gen_min": min(pcie_gen) if pcie_gen else None,
        "pcie_gen_max": max(pcie_gen) if pcie_gen else None,
        "pcie_width_min": min(pcie_width) if pcie_width else None,
        "pstate_mode": Counter(pstates).most_common(1)[0][0] if pstates else None,
    }


def validate_pair(baseline: dict[str, str], bridge: dict[str, str]) -> None:
    for field in ("output_sha256", "token_sha256", "predicted_tokens"):
        if baseline.get(field) != bridge.get(field):
            raise BridgeBenchError(
                f"pair mismatch for {field}: {baseline.get(field)!r} != {bridge.get(field)!r}"
            )
    for field in ("mtp_acceptance", "mean_accepted_length"):
        if abs(float(baseline.get(field, 0)) - float(bridge.get(field, 0))) > 1e-12:
            raise BridgeBenchError(
                f"pair mismatch for {field}: {baseline.get(field)!r} != {bridge.get(field)!r}"
            )


def bridge_environment(
    args: argparse.Namespace, run_dir: Path, enabled: bool, device_quant: bool,
    hit_only: bool, cache_layers: str, cache_slots: int
) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith(BRIDGE_ENV_PREFIX)
    }
    environment.update(
        {
            "ROOT": str(args.root),
            "SERVER": str(args.server),
            "CLIENT": str(args.client),
            "MODEL": str(args.model),
            "PROFILE": str(args.profile),
            "RESULTS_DIR": str(run_dir),
            "PROMPT_SOURCE": str(args.current_prompt),
            "CASES": "L0",
            "REPEATS": "1",
            "N_PREDICT": str(args.n_predict),
            "WARMUP_TOKENS": str(args.warmup_tokens),
            "COOLDOWN_SECONDS": str(args.cooldown_seconds),
            "STATIC_FIXED_S": str(args.fixed_s),
            "CPU_THREADS": str(args.cpu_threads),
            "DRAFT_THREADS": str(args.draft_threads),
            "CMOE_BATCH": str(args.cmoe_batch),
            "CMOE_UBATCH": str(args.cmoe_ubatch),
            "CONTEXT": str(args.context),
            "TARGET_TYPE_K": args.target_type_k,
            "TARGET_TYPE_V": args.target_type_v,
            "DRAFT_TYPE_K": args.draft_type_k,
            "DRAFT_TYPE_V": args.draft_type_v,
            "DRAFT_BACKEND_SAMPLING": "0",
            "MTP_OVERRIDE": str(args.mtp),
            "PORT": str(args.port),
        }
    )
    if enabled:
        environment.update(
            {
                "GGML_CUDA_EXPERT_BRIDGE_LAYERS": args.layers,
                "GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_LAYERS": args.recurrence_layers,
                "GGML_CUDA_EXPERT_BRIDGE_K": str(args.k),
                "GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_K": str(args.recurrence_k),
                "GGML_CUDA_EXPERT_BRIDGE_CONSUME": "1",
                "GGML_CUDA_EXPERT_BRIDGE_JSON": str(run_dir / "bridge.json"),
            }
        )
        if args.telemetry:
            environment["GGML_CUDA_EXPERT_BRIDGE_TELEMETRY"] = "1"
        if device_quant:
            environment["GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT"] = "1"
        if hit_only:
            environment["GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY"] = "1"
        if cache_layers:
            environment["GGML_CUDA_EXPERT_BRIDGE_CACHE_LAYERS"] = cache_layers
            environment["GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS"] = str(cache_slots)
    return environment


def execute_case(
    args: argparse.Namespace, run_dir: Path, enabled: bool, device_quant: bool,
    hit_only: bool, cache_layers: str, cache_slots: int
) -> tuple[dict[str, str], dict[str, object]]:
    log_path = args.output_dir / f"{run_dir.name}.driver.log"
    environment = bridge_environment(
        args, run_dir, enabled, device_quant, hit_only, cache_layers, cache_slots
    )
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(args.runner)],
            cwd=args.root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    row = read_single_row(run_dir / "runs.csv")
    samples = sample_summary(run_dir / "L0-run1.samples.csv")
    if enabled:
        bridge_path = run_dir / "bridge.json"
        if not bridge_path.is_file():
            raise BridgeBenchError(f"bridge did not write {bridge_path}")
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        if bridge.get("schema") not in {
            "llama-wackmall-cuda-transfer-bridge-v5",
            "llama-wackmall-cuda-transfer-bridge-v6",
        }:
            raise BridgeBenchError(f"unexpected bridge schema in {bridge_path}")
    return row, samples


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise BridgeBenchError(f"refusing to reuse output directory: {args.output_dir}")
    if not args.output_dir.parent.is_dir():
        raise BridgeBenchError(f"output parent does not exist: {args.output_dir.parent}")
    for path in (args.runner, args.server, args.client, args.model, args.profile, *args.prompts):
        if not path.is_file():
            raise BridgeBenchError(f"required file does not exist: {path}")
    args.output_dir.mkdir()
    records: list[dict[str, object]] = []
    pair_results: list[dict[str, object]] = []
    prompt_metadata = []
    for prompt_index, prompt in enumerate(args.prompts, 1):
        args.current_prompt = prompt
        prompt_metadata.append({"path": str(prompt), "sha256": sha256_file(prompt)})
        prompt_name = f"p{prompt_index:02d}-{prompt.stem}"
        for pair_index in range(1, args.pairs + 1):
            order = (False, True) if pair_index % 2 else (True, False)
            control_label = "control" if args.control_bridge else "baseline"
            order_name = "variant-first" if order[0] else f"{control_label}-first"
            case_rows: dict[bool, dict[str, str]] = {}
            case_config = {
                False: (
                    args.control_bridge,
                    args.control_device_quant,
                    args.control_hit_only,
                    args.control_cache_layers,
                    args.control_cache_slots,
                ),
                True: (
                    True,
                    args.device_quant,
                    args.hit_only,
                    args.cache_layers,
                    args.cache_slots,
                ),
            }
            for enabled in order:
                bridge_enabled, device_quant, hit_only, cache_layers, cache_slots = case_config[enabled]
                mode = "variant" if enabled else ("control" if bridge_enabled else "baseline")
                run_dir = args.output_dir / f"{prompt_name}-pair{pair_index:02d}-{mode}"
                row, samples = execute_case(
                    args, run_dir, bridge_enabled, device_quant, hit_only,
                    cache_layers, cache_slots
                )
                case_rows[enabled] = row
                records.append(
                    {
                        "prompt": str(prompt),
                        "prompt_sha256": prompt_metadata[-1]["sha256"],
                        "pair": pair_index,
                        "order": order_name,
                        "mode": mode,
                        "run_dir": str(run_dir),
                        "decode_tps": float(row["decode_tps"]),
                        "mtp_acceptance": float(row["mtp_acceptance"]),
                        "mean_accepted_length": float(row["mean_accepted_length"]),
                        "output_sha256": row["output_sha256"],
                        "token_sha256": row["token_sha256"],
                        **samples,
                    }
                )
            validate_pair(case_rows[False], case_rows[True])
            baseline_tps = float(case_rows[False]["decode_tps"])
            bridge_tps = float(case_rows[True]["decode_tps"])
            pair_results.append(
                {
                    "prompt": str(prompt),
                    "pair": pair_index,
                    "order": order_name,
                    "control_label": control_label,
                    "variant_label": "variant",
                    "control_tps": baseline_tps,
                    "variant_tps": bridge_tps,
                    "baseline_tps": baseline_tps,
                    "bridge_tps": bridge_tps,
                    "delta_percent": 100.0 * (bridge_tps / baseline_tps - 1.0),
                }
            )

    fields = list(records[0])
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    deltas = [float(item["delta_percent"]) for item in pair_results]
    low, high = bootstrap_median_ci(deltas, args.bootstrap_samples)
    summary = {
        "schema": "llama-wackmall-expert-bridge-ab-v1",
        "config": {
            "model": str(args.model),
            "profile": str(args.profile),
            "profile_sha256": sha256_file(args.profile),
            "prompts": prompt_metadata,
            "pairs_per_prompt": args.pairs,
            "n_predict": args.n_predict,
            "warmup_tokens": args.warmup_tokens,
            "context": args.context,
            "mtp": args.mtp,
            "layers": args.layers,
            "recurrence_layers": args.recurrence_layers,
            "k": args.k,
            "recurrence_k": args.recurrence_k,
            "telemetry": args.telemetry,
            "control_bridge": args.control_bridge,
            "control_device_quant": args.control_device_quant,
            "control_hit_only": args.control_hit_only,
            "control_cache_layers": args.control_cache_layers,
            "control_cache_slots": args.control_cache_slots,
            "device_quant": args.device_quant,
            "hit_only": args.hit_only,
            "cache_layers": args.cache_layers,
            "cache_slots": args.cache_slots,
        },
        "pairs": pair_results,
        "overall": {
            "pair_count": len(pair_results),
            "median_delta_percent": statistics.median(deltas),
            "mean_delta_percent": statistics.fmean(deltas),
            "bootstrap_median_ci95_percent": [low, high],
            "all_hashes_identical_within_pairs": True,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"pairs={len(pair_results)}")
    print(f"median_delta_percent={summary['overall']['median_delta_percent']:.6f}")
    print(f"bootstrap_ci95_percent={low:.6f},{high:.6f}")
    print(f"summary={summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--prompt", dest="prompts", type=Path, action="append", required=True)
    parser.add_argument("--runner", type=Path, default=Path("scripts/bench_hybrid.sh"))
    parser.add_argument("--server", type=Path, default=Path("build-transient-sm61/bin/llama-server"))
    parser.add_argument("--client", type=Path, default=Path("tools/bench_hybrid_client.py"))
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--n-predict", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--fixed-s", type=int, default=70)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--draft-threads", type=int, default=4)
    parser.add_argument("--cmoe-batch", type=int, default=32)
    parser.add_argument("--cmoe-ubatch", type=int, default=32)
    parser.add_argument("--target-type-k", default="q8_0")
    parser.add_argument("--target-type-v", default="q8_0")
    parser.add_argument("--draft-type-k", default="q8_0")
    parser.add_argument("--draft-type-v", default="q4_0")
    parser.add_argument("--mtp", type=int, default=2)
    parser.add_argument("--layers", default="1,2")
    parser.add_argument("--recurrence-layers", default="2")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--recurrence-k", type=int, default=1)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--control-bridge", action="store_true")
    parser.add_argument("--control-device-quant", action="store_true")
    parser.add_argument("--control-hit-only", action="store_true")
    parser.add_argument("--control-cache-layers", default="")
    parser.add_argument("--control-cache-slots", type=int, default=8)
    parser.add_argument("--device-quant", action="store_true")
    parser.add_argument("--hit-only", action="store_true")
    parser.add_argument("--cache-layers", default="")
    parser.add_argument("--cache-slots", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.pairs < 1 or args.bootstrap_samples < 1:
            raise BridgeBenchError("pairs and bootstrap samples must be positive")
        if args.n_predict < 1 or args.warmup_tokens < 0 or args.cooldown_seconds < 0:
            raise BridgeBenchError("token counts and cooldown are invalid")
        if args.mtp < 1 or args.k not in (1, 2, 3) or args.recurrence_k not in (1, 2, 3):
            raise BridgeBenchError("MTP and candidate counts are invalid")
        if (args.control_device_quant or args.control_hit_only) and not args.control_bridge:
            raise BridgeBenchError("control bridge features require --control-bridge")
        if args.control_cache_layers and not args.control_bridge:
            raise BridgeBenchError("control cache requires --control-bridge")
        if not 1 <= args.cache_slots <= 16 or not 1 <= args.control_cache_slots <= 16:
            raise BridgeBenchError("cache slots must be in [1, 16]")
        args.root = args.runner.resolve().parents[1]
        args.runner = args.runner.resolve()
        args.server = args.server.resolve()
        args.client = args.client.resolve()
        args.model = args.model.resolve()
        args.profile = args.profile.resolve()
        args.prompts = [path.resolve() for path in args.prompts]
        args.output_dir = args.output_dir.resolve(strict=False)
        run(args)
    except (BridgeBenchError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
