#!/usr/bin/env python3
"""Build a prompt-balanced wackMall expert profile from session-local CSVs."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

from convert_luce_spark_profile import (
    GGUFError,
    INT64_MAX,
    ProfileDimensions,
    ProfileError,
    SparkProfile,
    read_gguf_metadata,
    validate_wack_profile,
    write_wack_profile,
)


def parse_weight(value: str) -> Fraction:
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"invalid positive weight {value!r}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError(f"weight must be positive, got {value!r}")
    return result


def dense_counts(path: Path, dims: ProfileDimensions) -> list[list[int]]:
    if not path.is_file():
        raise ProfileError(f"input profile does not exist: {path}")
    sparse = validate_wack_profile(path, dims)
    return [
        [sparse.get((layer, expert), 0) for expert in range(dims.n_expert)]
        for layer in range(dims.n_layer)
    ]


def apportion(values: list[Fraction], total: int) -> list[int]:
    """Round non-negative fractions to integers with an exact requested sum."""
    if total < 0:
        raise ProfileError(f"apportion total must be non-negative, got {total}")
    floors = [value.numerator // value.denominator for value in values]
    missing = total - sum(floors)
    if missing < 0 or missing > len(values):
        raise ProfileError("internal normalization error while apportioning counts")
    order = sorted(
        range(len(values)),
        key=lambda index: (-(values[index] - floors[index]), index),
    )
    for index in order[:missing]:
        floors[index] += 1
    return floors


def aggregate_per_layer(
    profiles: list[list[list[int]]],
    weights: list[Fraction],
    dims: ProfileDimensions,
    scale: int,
) -> list[list[int]]:
    weight_total = sum(weights, Fraction())
    output: list[list[int]] = []
    for layer in range(dims.n_layer):
        layer_totals = [sum(profile[layer]) for profile in profiles]
        for index, total in enumerate(layer_totals):
            if total == 0:
                raise ProfileError(
                    f"input {index + 1} has no selections for layer {layer}; "
                    "per-layer normalization would make a silent ordering assumption"
                )
        probabilities = [
            sum(
                (
                    weights[index]
                    * Fraction(profiles[index][layer][expert], layer_totals[index])
                    for index in range(len(profiles))
                ),
                Fraction(),
            )
            / weight_total
            for expert in range(dims.n_expert)
        ]
        output.append(apportion([value * scale for value in probabilities], scale))
    return output


def aggregate_global(
    profiles: list[list[list[int]]],
    weights: list[Fraction],
    dims: ProfileDimensions,
    scale: int,
) -> list[list[int]]:
    totals = [sum(sum(row) for row in profile) for profile in profiles]
    for index, total in enumerate(totals):
        if total == 0:
            raise ProfileError(f"input {index + 1} contains no expert selections")
    weight_total = sum(weights, Fraction())
    probabilities: list[Fraction] = []
    for layer in range(dims.n_layer):
        for expert in range(dims.n_expert):
            probabilities.append(
                sum(
                    (
                        weights[index]
                        * Fraction(profiles[index][layer][expert], totals[index])
                        for index in range(len(profiles))
                    ),
                    Fraction(),
                )
                / weight_total
            )
    flat = apportion([value * scale for value in probabilities], scale)
    return [
        flat[layer * dims.n_expert : (layer + 1) * dims.n_expert]
        for layer in range(dims.n_layer)
    ]


def top_k_coverage(counts: list[list[int]], top_k: int) -> Fraction:
    total = sum(sum(row) for row in counts)
    if total == 0:
        return Fraction()
    hits = sum(sum(sorted(row, reverse=True)[:top_k]) for row in counts)
    return Fraction(hits, total)


def run(args: argparse.Namespace) -> None:
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise ProfileError(f"model does not exist: {model_path}")
    model = read_gguf_metadata(model_path)
    dims = model.dimensions

    inputs = [path.resolve() for path in args.input]
    if len(set(inputs)) != len(inputs):
        raise ProfileError("the same input profile was supplied more than once")
    weights = args.weight or [Fraction(1) for _ in inputs]
    if len(weights) != len(inputs):
        raise ProfileError(
            f"received {len(weights)} weights for {len(inputs)} input profiles"
        )
    profiles = [dense_counts(path, dims) for path in inputs]

    if args.normalization == "per-layer":
        counts = aggregate_per_layer(profiles, weights, dims, args.scale)
    else:
        counts = aggregate_global(profiles, weights, dims, args.scale)

    output = args.output.resolve(strict=False)
    result = SparkProfile(dims, counts)
    write_wack_profile(result, output, drop_zero=args.drop_zero)

    print(f"model: {model_path}")
    print(
        f"dimensions: layers={dims.n_layer} experts={dims.n_expert} "
        f"used={dims.n_expert_used}"
    )
    for index, (path, weight, profile) in enumerate(zip(inputs, weights, profiles), 1):
        totals = [sum(row) for row in profile]
        coverage = float(top_k_coverage(profile, args.top_k)) * 100.0
        print(
            f"input[{index}]: path={path} weight={weight} total={sum(totals)} "
            f"layer_min={min(totals)} layer_max={max(totals)} "
            f"top{args.top_k}={coverage:.3f}%"
        )
    result_totals = [sum(row) for row in counts]
    result_coverage = float(top_k_coverage(counts, args.top_k)) * 100.0
    print(
        f"aggregate: normalization={args.normalization} scale={args.scale} "
        f"total={sum(result_totals)} layer_min={min(result_totals)} "
        f"layer_max={max(result_totals)} top{args.top_k}={result_coverage:.3f}%"
    )
    print(f"output: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="session-local wackMall usage CSV; repeat for each prompt",
    )
    parser.add_argument(
        "--weight",
        type=parse_weight,
        action="append",
        help="optional positive weight aligned with each --input",
    )
    parser.add_argument("--model", type=Path, required=True, help="target GGUF")
    parser.add_argument("--output", type=Path, required=True, help="new aggregate CSV")
    parser.add_argument(
        "--normalization",
        choices=("per-layer", "global"),
        default="per-layer",
        help="prompt balancing mode (default: per-layer)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1_000_000,
        help="integer count mass per layer, or for the whole profile in global mode",
    )
    parser.add_argument("--top-k", type=int, default=33, help="coverage statistic")
    parser.add_argument("--drop-zero", action="store_true", help="omit zero rows")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.scale <= 0 or args.scale > INT64_MAX:
            raise ProfileError(f"scale must be in [1, {INT64_MAX}]")
        if args.top_k <= 0:
            raise ProfileError("top-k must be positive")
        run(args)
    except (GGUFError, ProfileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
