#!/usr/bin/env python3
"""Build a q* bandwidth profile from transport + CPU expert timing."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from convert_luce_spark_profile import ProfileError


BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0


def gbs(nbytes: int, ms: float) -> float:
    if ms <= 0:
        return 0.0
    return nbytes / (ms / 1000.0) / BYTES_PER_GIB


def recommend(cpu_gbs: float, pcie_gbs: float, threshold: float) -> str:
    if cpu_gbs <= 0 or pcie_gbs <= 0:
        return "unknown"
    if cpu_gbs > threshold * pcie_gbs:
        return "cpu-heavy"  # q* small: keep misses on CPU (W=0)
    if pcie_gbs > threshold * cpu_gbs:
        return "pcie-heavy"  # fill misses over PCIe
    return "hybrid"  # split misses


def profile_from_summary(
    summary: dict[str, object],
    *,
    threshold: float,
    expert_bytes: int | None,
) -> dict[str, object]:
    results = summary.get("results")
    if not isinstance(results, list):
        raise ProfileError("transport summary has no results")

    bounds = summary.get("optimistic_transport_lower_bounds")
    want_bytes = expert_bytes
    if want_bytes is None and isinstance(bounds, list):
        full_sizes = [
            int(item["full"]["bytes"])
            for item in bounds
            if isinstance(item, dict) and isinstance(item.get("full"), dict)
        ]
        if full_sizes:
            want_bytes = full_sizes[0]

    def pick(mode: str) -> dict[str, object] | None:
        matches = [item for item in results if item.get("mode") == mode]
        if want_bytes is not None:
            sized = [item for item in matches if int(item.get("bytes", -1)) == want_bytes]
            if sized:
                matches = sized
        if not matches:
            return None
        return min(matches, key=lambda item: abs(int(item["bytes"]) - (want_bytes or 0)))

    pinned = pick("pinned_h2d")
    overlap = pick("pinned_h2d_cpu_overlap")
    if pinned is None:
        raise ProfileError("transport summary lacks pinned_h2d")

    nbytes = int(pinned["bytes"])
    pcie_ms = float(pinned["device_ms_median_of_medians"])
    pcie_gbs = gbs(nbytes, pcie_ms)
    cpu_gbs = None
    cpu_ms = None
    source = "standalone-pinned"

    if isinstance(bounds, list) and bounds:
        ratios = []
        cpu_samples = []
        for item in bounds:
            full = item.get("full")
            if not isinstance(full, dict):
                continue
            if int(full["bytes"]) != nbytes:
                continue
            cpu_samples.append(float(full["cpu_ms_per_cold_selection"]))
            ratios.append(float(full["copy_to_cpu_ratio"]))
        if cpu_samples:
            cpu_ms = statistics.median(cpu_samples)
            cpu_gbs = gbs(nbytes, cpu_ms)
            source = "cpu-timing+pinned_h2d"

    if overlap is not None:
        pcie_ms = float(overlap["device_ms_median_of_medians"])
        pcie_gbs = gbs(int(overlap["bytes"]), pcie_ms)
        cpu_ms = float(overlap["compute_ms_median_of_medians"])
        cpu_gbs = gbs(int(overlap["bytes"]), cpu_ms)
        nbytes = int(overlap["bytes"])
        source = "pinned_h2d_cpu_overlap"

    if cpu_gbs is None or cpu_ms is None:
        raise ProfileError("no CPU bandwidth: pass --stats-json or a summary with overlap-cpu")

    q_star = pcie_gbs / cpu_gbs if cpu_gbs > 0 else 0.0
    q_star = max(0.0, min(1.0, q_star))
    backend = recommend(cpu_gbs, pcie_gbs, threshold)
    return {
        "schema": "llama-wackmall-expert-bw-v1",
        "source": source,
        "expert_bytes": nbytes,
        "pcie_gbs": round(pcie_gbs, 3),
        "cpu_gbs": round(cpu_gbs, 3),
        "q_star": round(q_star, 4),
        "fetch_fraction": round(q_star, 4),
        "threshold": threshold,
        "recommended": backend,
        "note": (
            "q_star = pcie_gbs / cpu_gbs; fetch that fraction of decode misses "
            "over PCIe and compute the rest on CPU. cpu-heavy means keep W=0."
        ),
    }


def run(args: argparse.Namespace) -> None:
    summary = json.loads(args.transport_json.read_text(encoding="utf-8"))
    if summary.get("schema") not in (
        "llama-wackmall-expert-transport-summary-v1",
        "llama-wackmall-expert-transport-v1",
    ):
        raise ProfileError(f"unexpected transport schema {summary.get('schema')!r}")
    if summary.get("schema") == "llama-wackmall-expert-transport-v1":
        # Single-run bench JSON: flatten segment results into the summary shape.
        flat = []
        for segment in summary.get("segments", []):
            for item in segment.get("results", []):
                row = dict(item)
                row["bytes"] = int(segment.get("bytes", item.get("bytes", 0)))
                row["device_ms_median_of_medians"] = float(item["device_ms"]["median"])
                row["compute_ms_median_of_medians"] = float(item["compute_ms"]["median"])
                flat.append(row)
        summary = {"schema": "llama-wackmall-expert-transport-summary-v1", "results": flat}

    if args.stats_json:
        from bench_expert_transport import add_cpu_lower_bounds, model_layout

        if args.model is None:
            raise ProfileError("--stats-json requires --model for expert byte sizes")
        _, layers, _ = model_layout(args.model.resolve())
        add_cpu_lower_bounds(summary, args.stats_json.resolve(), layers)

    profile = profile_from_summary(
        summary, threshold=args.threshold, expert_bytes=args.expert_bytes
    )
    output_path = args.output.resolve(strict=False)
    if output_path.exists():
        raise ProfileError(f"refusing to overwrite existing output: {output_path}")
    if not output_path.parent.is_dir():
        raise ProfileError(f"output parent does not exist: {output_path.parent}")
    output_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(
        f"pcie={profile['pcie_gbs']:.2f} GB/s cpu={profile['cpu_gbs']:.2f} GB/s "
        f"q*={profile['q_star']:.3f} recommend={profile['recommended']}"
    )
    print(f"output={output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--expert-bytes", type=int)
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="cpu-heavy when CPU GB/s > threshold * PCIe GB/s (default 2.0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.threshold <= 0:
            raise ProfileError("threshold must be positive")
        if args.expert_bytes is not None and args.expert_bytes <= 0:
            raise ProfileError("expert-bytes must be positive")
        run(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, ProfileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
