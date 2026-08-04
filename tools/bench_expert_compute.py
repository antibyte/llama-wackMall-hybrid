#!/usr/bin/env python3
"""Run and aggregate resident expert GPU compute measurements."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from bench_expert_transport import model_layout
from convert_luce_spark_profile import GGUFError, ProfileError


def parse_layers(text: str | None, n_layer: int) -> list[int]:
    if text is None:
        return sorted({0, n_layer // 2, n_layer - 1})
    try:
        layers = [int(item) for item in text.split(",")]
    except ValueError as error:
        raise ProfileError(f"invalid layer list: {text!r}") from error
    if not layers or any(layer < 0 or layer >= n_layer for layer in layers):
        raise ProfileError(f"layer list is outside [0, {n_layer - 1}]")
    return list(dict.fromkeys(layers))


def aggregate(paths: list[Path]) -> dict[str, object]:
    collected: dict[tuple[int, int, str], list[dict[str, object]]] = {}
    device: str | None = None
    model: str | None = None
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "llama-wackmall-expert-compute-v1":
            raise ProfileError(f"unexpected compute schema in {path}")
        if device is None:
            device = data.get("device")
            model = data.get("model")
        elif device != data.get("device") or model != data.get("model"):
            raise ProfileError("compute runs use incompatible model or device")
        layer = data["config"]["layer"]
        for item in data.get("results", []):
            key = (layer, int(item["weight_bytes"]), str(item["mode"]))
            collected.setdefault(key, []).append(item)

    results: list[dict[str, object]] = []
    by_size: dict[tuple[int, str], list[dict[str, object]]] = {}
    for (layer, size, mode), values in sorted(collected.items()):
        result = {
            "layer": layer,
            "weight_bytes": size,
            "mode": mode,
            "runs": len(values),
            "latency_ms_median_of_medians": statistics.median(
                float(item["latency_ms"]["median"]) for item in values
            ),
            "queued_ms_median_of_medians": statistics.median(
                float(item["queued_ms"]["median"]) for item in values
            ),
        }
        results.append(result)
        by_size.setdefault((size, mode), []).append(result)
    size_results = [
        {
            "weight_bytes": size,
            "mode": mode,
            "layers": [item["layer"] for item in values],
            "latency_ms_median": statistics.median(
                item["latency_ms_median_of_medians"] for item in values
            ),
            "queued_ms_median": statistics.median(
                item["queued_ms_median_of_medians"] for item in values
            ),
        }
        for (size, mode), values in sorted(by_size.items())
    ]
    return {
        "schema": "llama-wackmall-expert-compute-summary-v1",
        "model": model,
        "device": device,
        "run_files": [str(path) for path in paths],
        "results": results,
        "by_size_mode": size_results,
    }


def run(args: argparse.Namespace) -> None:
    model_path = args.model.resolve()
    binary_path = args.binary.resolve()
    output_dir = args.output_dir.resolve(strict=False)
    if not model_path.is_file() or not binary_path.is_file():
        raise ProfileError("model and benchmark binary must exist")
    if output_dir.exists():
        raise ProfileError(f"refusing to reuse existing output directory: {output_dir}")
    if not output_dir.parent.is_dir():
        raise ProfileError(f"output parent does not exist: {output_dir.parent}")
    model, _, _ = model_layout(model_path)
    layers = parse_layers(args.layers, model.dimensions.n_layer)
    if args.expert < 0 or args.expert >= model.dimensions.n_expert:
        raise ProfileError(
            f"expert must be in [0, {model.dimensions.n_expert - 1}]"
        )
    output_dir.mkdir()
    paths: list[Path] = []
    for layer in layers:
        for run_index in range(1, args.runs + 1):
            path = output_dir / f"layer{layer}-expert{args.expert}-run{run_index}.json"
            command = [
                str(binary_path),
                "--model",
                str(model_path),
                "--layer",
                str(layer),
                "--expert",
                str(args.expert),
                "--device",
                str(args.device),
                "--repeats",
                str(args.repeats),
                "--warmups",
                str(args.warmups),
                "--queued-iterations",
                str(args.queued_iterations),
                "--json",
                str(path),
            ]
            log_path = path.with_suffix(".log")
            with log_path.open("wb") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            paths.append(path)
    summary = aggregate(paths)
    summary["config"] = {
        "layers": layers,
        "expert": args.expert,
        "runs": args.runs,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "queued_iterations": args.queued_iterations,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"layers={','.join(str(layer) for layer in layers)}")
    print(f"summary={summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build-hybrid/bin/llama-expert-compute-bench"),
    )
    parser.add_argument("--layers", help="comma-separated base layers; default early,middle,last")
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--queued-iterations", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.runs < 1 or args.repeats < 1 or args.warmups < 0:
            raise ProfileError("runs/repeats must be positive and warmups non-negative")
        if args.queued_iterations < 1 or args.device < 0:
            raise ProfileError("queued-iterations and device are invalid")
        run(args)
    except (GGUFError, ProfileError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
