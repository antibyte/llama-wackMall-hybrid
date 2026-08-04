#!/usr/bin/env python3
"""Run and aggregate model-derived expert transport measurements."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from convert_luce_spark_profile import GGUFError, ProfileError, read_gguf_metadata
from optimize_expert_placement import EXPERT_TENSOR_RE, read_gguf_tensors


def model_layout(model_path: Path) -> tuple[object, list[dict[str, object]], list[int]]:
    model = read_gguf_metadata(model_path)
    tensors = read_gguf_tensors(model_path)
    layers: list[dict[str, object]] = []
    sizes: set[int] = set()
    for layer in range(model.dimensions.n_layer):
        parts: dict[str, int] = {}
        for tensor in tensors:
            match = EXPERT_TENSOR_RE.fullmatch(tensor.name)
            if not match or int(match.group(1)) != layer:
                continue
            if tensor.dimensions[-1] != model.dimensions.n_expert:
                raise ProfileError(
                    f"tensor {tensor.name!r} expert dimension does not match model metadata"
                )
            kind = tensor.name.split(".ffn_", 1)[1].split("_exps.weight", 1)[0]
            parts[kind] = tensor.nbytes // model.dimensions.n_expert
        if not parts:
            raise ProfileError(f"no routed expert tensors found for base layer {layer}")
        full = sum(parts.values())
        gate_up = parts.get("gate_up", parts.get("gate", 0) + parts.get("up", 0))
        sizes.update(parts.values())
        if gate_up:
            sizes.add(gate_up)
        sizes.add(full)
        layers.append(
            {
                "layer": layer,
                "parts": dict(sorted(parts.items())),
                "gate_up_bytes": gate_up,
                "full_bytes": full,
            }
        )
    return model, layers, sorted(sizes)


def aggregate_runs(run_paths: list[Path]) -> dict[str, object]:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_paths]
    if not runs:
        raise ProfileError("no transport runs were produced")
    schema = runs[0].get("schema")
    if schema != "llama-wackmall-expert-transport-v1":
        raise ProfileError(f"unexpected transport schema {schema!r}")
    device = runs[0].get("device")
    config = runs[0].get("config")
    collected: dict[tuple[int, str], list[dict[str, object]]] = {}
    for run in runs:
        if run.get("schema") != schema or run.get("device") != device or run.get("config") != config:
            raise ProfileError("transport runs have incompatible schema, device, or configuration")
        for segment in run.get("segments", []):
            size = segment.get("bytes")
            for result in segment.get("results", []):
                mode = result.get("mode")
                if not isinstance(size, int) or not isinstance(mode, str):
                    raise ProfileError("invalid segment result in transport JSON")
                collected.setdefault((size, mode), []).append(result)

    summaries: list[dict[str, object]] = []
    for (size, mode), values in sorted(collected.items()):
        if len(values) != len(runs):
            raise ProfileError(f"transport mode {mode} size {size} is missing from a run")

        def median_of(field: str, member: str = "median") -> float:
            numbers = [value[field][member] for value in values]
            if not all(isinstance(number, (int, float)) for number in numbers):
                raise ProfileError(f"invalid {field}.{member} for {mode} size {size}")
            return statistics.median(float(number) for number in numbers)

        summaries.append(
            {
                "bytes": size,
                "mode": mode,
                "runs": len(values),
                "device_ms_median_of_medians": median_of("device_ms"),
                "wall_ms_median_of_medians": median_of("wall_ms"),
                "stage_ms_median_of_medians": median_of("stage_ms"),
                "compute_ms_median_of_medians": median_of("compute_ms"),
                "exposed_copy_ms_median_of_medians": median_of("exposed_copy_ms"),
                "hidden_copy_ratio_median_of_medians": median_of("hidden_copy_ratio"),
                "device_gib_per_s_median": statistics.median(
                    float(value["device_gib_per_s_median"]) for value in values
                ),
            }
        )
    return {
        "schema": "llama-wackmall-expert-transport-summary-v1",
        "device": device,
        "config": config,
        "run_count": len(runs),
        "results": summaries,
    }


def add_cpu_lower_bounds(
    summary: dict[str, object], stats_path: Path, layers: list[dict[str, object]]
) -> None:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("timing_enabled") is not True:
        raise ProfileError("expert stats JSON does not contain enabled CPU timing")
    timing_by_layer = {item["layer"]: item for item in stats.get("layers", [])}
    pinned = {
        item["bytes"]: item
        for item in summary["results"]
        if item["mode"] == "pinned_h2d"
    }
    bounds: list[dict[str, object]] = []
    for layout in layers:
        layer = layout["layer"]
        timing = timing_by_layer.get(layer)
        if timing is None or timing.get("cold_hits", 0) <= 0:
            continue
        cold_hits = timing["cold_hits"]
        full_cpu = timing["cpu_cold_ms"] / cold_hits
        gate_up_cpu = timing["cpu_gate_up_ms"] / cold_hits
        down_cpu = timing["cpu_down_ms"] / cold_hits

        def bound(bytes_value: int, cpu_ms: float) -> dict[str, object] | None:
            transfer = pinned.get(bytes_value)
            if transfer is None or cpu_ms <= 0:
                return None
            copy_ms = transfer["device_ms_median_of_medians"]
            return {
                "bytes": bytes_value,
                "cpu_ms_per_cold_selection": cpu_ms,
                "pinned_h2d_ms": copy_ms,
                "copy_to_cpu_ratio": copy_ms / cpu_ms,
                "optimistic_reuses_before_copy_break_even": math.ceil(copy_ms / cpu_ms),
            }

        parts = layout["parts"]
        down_bytes = parts.get("down")
        bounds.append(
            {
                "layer": layer,
                "cold_hits": cold_hits,
                "full": bound(layout["full_bytes"], full_cpu),
                "gate_up": bound(layout["gate_up_bytes"], gate_up_cpu),
                "down": bound(down_bytes, down_cpu) if down_bytes else None,
            }
        )
    summary["cpu_timing_source"] = str(stats_path)
    summary["optimistic_transport_lower_bounds"] = bounds
    summary["lower_bound_caveat"] = (
        "Break-even excludes GPU expert compute, synchronization, contention, and prediction cost. "
        "It is therefore necessary but not sufficient for transient GPU execution."
    )


def run(args: argparse.Namespace) -> None:
    model_path = args.model.resolve()
    binary_path = args.binary.resolve()
    output_dir = args.output_dir.resolve(strict=False)
    if not model_path.is_file():
        raise ProfileError(f"model does not exist: {model_path}")
    if not binary_path.is_file():
        raise ProfileError(f"benchmark binary does not exist: {binary_path}")
    if output_dir.exists():
        raise ProfileError(f"refusing to reuse existing output directory: {output_dir}")
    if not output_dir.parent.is_dir():
        raise ProfileError(f"output parent does not exist: {output_dir.parent}")

    model, layers, sizes = model_layout(model_path)
    output_dir.mkdir()
    layout_path = output_dir / "model-layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "schema": "llama-wackmall-expert-layout-v1",
                "model": str(model_path),
                "architecture": model.architecture,
                "layers": model.dimensions.n_layer,
                "experts": model.dimensions.n_expert,
                "expert_tensors": layers,
                "benchmarked_sizes": sizes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    run_paths: list[Path] = []
    for run_index in range(1, args.runs + 1):
        run_path = output_dir / f"transport-run{run_index}.json"
        command = [
            str(binary_path),
            "--device",
            str(args.device),
            "--repeats",
            str(args.repeats),
            "--warmups",
            str(args.warmups),
            "--working-set-mib",
            str(args.working_set_mib),
            "--overlap-us",
            str(args.overlap_us),
            "--json",
            str(run_path),
        ]
        for size in sizes:
            command.extend(("--segment", f"bytes-{size}={size}"))
        subprocess.run(command, check=True)
        run_paths.append(run_path)

    summary = aggregate_runs(run_paths)
    summary["model_layout"] = str(layout_path)
    if args.stats_json:
        add_cpu_lower_bounds(summary, args.stats_json.resolve(), layers)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"model layers={model.dimensions.n_layer} experts={model.dimensions.n_expert}")
    print("sizes=" + ",".join(str(size) for size in sizes))
    print(f"summary={summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build-hybrid/bin/llama-expert-transport-bench"),
    )
    parser.add_argument("--stats-json", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--working-set-mib", type=int, default=16)
    parser.add_argument("--overlap-us", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.runs < 1 or args.repeats < 1 or args.warmups < 0:
            raise ProfileError("runs/repeats must be positive and warmups non-negative")
        if args.device < 0 or args.working_set_mib < 1 or args.overlap_us < 0:
            raise ProfileError("device, working-set-mib, or overlap-us is invalid")
        run(args)
    except (GGUFError, ProfileError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
