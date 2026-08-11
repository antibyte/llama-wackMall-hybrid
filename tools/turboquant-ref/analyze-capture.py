#!/usr/bin/env python3
# Copyright (c) 2026 llama-wackMall contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate and aggregate llama-turboquant-capture output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys
from typing import Any


FORMATS = ("turbo3", "turbo4")


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def load_manifest(capture_dir: pathlib.Path) -> dict[int, dict[str, int]]:
    path = capture_dir / "manifest.csv"
    if not path.is_file():
        fail(f"missing manifest: {path}")

    required = {
        "event",
        "graph",
        "layer",
        "tokens",
        "head_dim",
        "q_heads",
        "kv_heads",
        "key_rows",
        "pair_rows",
    }
    layers: dict[int, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            fail("manifest columns do not match the capture format")
        for line, row in enumerate(reader, start=2):
            try:
                values = {name: int(row[name]) for name in required}
            except (KeyError, TypeError, ValueError) as error:
                fail(f"invalid integer in manifest line {line}: {error}")
            positive_fields = required - {"event", "layer"}
            if values["event"] < 0 or values["layer"] < 0 or any(values[name] <= 0 for name in positive_fields):
                fail(f"non-positive capture dimension in manifest line {line}")

            layer = values["layer"]
            current = layers.setdefault(
                layer,
                {
                    "head_dim": values["head_dim"],
                    "q_heads": values["q_heads"],
                    "kv_heads": values["kv_heads"],
                    "tokens": 0,
                    "key_rows": 0,
                    "pair_rows": 0,
                    "events": 0,
                },
            )
            for name in ("head_dim", "q_heads", "kv_heads"):
                if current[name] != values[name]:
                    fail(f"layer {layer} changes {name} within one capture")
            current["tokens"] += values["tokens"]
            current["key_rows"] += values["key_rows"]
            current["pair_rows"] += values["pair_rows"]
            current["events"] += 1

    if not layers:
        fail("manifest contains no capture rows")
    return layers


def validate_files(capture_dir: pathlib.Path, layers: dict[int, dict[str, int]]) -> None:
    for layer, metadata in layers.items():
        stem = f"layer-{layer:03d}"
        keys = capture_dir / f"{stem}-keys.f32"
        pairs = capture_dir / f"{stem}-qk-pairs.f32"
        if not keys.is_file() or not pairs.is_file():
            fail(f"missing capture file for layer {layer}")
        key_bytes = metadata["key_rows"] * metadata["head_dim"] * 4
        pair_bytes = metadata["pair_rows"] * metadata["head_dim"] * 4
        if keys.stat().st_size != key_bytes:
            fail(f"layer {layer} key file size does not match manifest")
        if pairs.stat().st_size != pair_bytes:
            fail(f"layer {layer} Q/K pair file size does not match manifest")
        if metadata["pair_rows"] % 2 != 0:
            fail(f"layer {layer} Q/K pair row count is odd")


def run_analyzer(
    analyzer: pathlib.Path,
    capture_dir: pathlib.Path,
    layer: int,
    metadata: dict[str, int],
    format_name: str,
) -> dict[str, Any]:
    input_path = capture_dir / f"layer-{layer:03d}-qk-pairs.f32"
    key_input_path = capture_dir / f"layer-{layer:03d}-keys.f32"
    command = [
        str(analyzer),
        "--input",
        str(input_path),
        "--key-input",
        str(key_input_path),
        "--row-size",
        str(metadata["head_dim"]),
        "--tokens",
        str(metadata["tokens"]),
        "--q-heads",
        str(metadata["q_heads"]),
        "--kv-heads",
        str(metadata["kv_heads"]),
        "--format",
        format_name,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        fail(
            f"analyzer failed for layer {layer} {format_name}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        fail(f"analyzer returned invalid JSON for layer {layer} {format_name}: {error}")
    if result.get("format") != format_name or result.get("row_size") != metadata["head_dim"]:
        fail(f"analyzer metadata mismatch for layer {layer} {format_name}")
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    logical_values = sum(int(item["logical_values"]) for item in results)
    rows = sum(int(item["rows"]) for item in results)
    dot_pairs = sum(int(item["dot_pairs"]) for item in results)
    encoded_bytes = sum(int(item["encoded_bytes"]) for item in results)
    if logical_values <= 0 or rows <= 0 or dot_pairs <= 0:
        fail("analyzer produced empty metrics")

    mse = sum(float(item["mse"]) * int(item["logical_values"]) for item in results) / logical_values
    weighted_rows = lambda name: sum(float(item[name]) * int(item["rows"]) for item in results) / rows
    weighted_dots = lambda name: sum(float(item[name]) * int(item["dot_pairs"]) for item in results) / dot_pairs
    return {
        "layers": len(results),
        "rows": rows,
        "logical_values": logical_values,
        "encoded_bytes": encoded_bytes,
        "bits_per_value": encoded_bytes * 8.0 / logical_values,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "weighted_mean_relative_l2": weighted_rows("relative_l2"),
        "max_abs_error": max(float(item["max_abs_error"]) for item in results),
        "weighted_mean_cosine": weighted_rows("mean_cosine"),
        "weighted_mean_norm_ratio": weighted_rows("mean_norm_ratio"),
        "dot_pairs": dot_pairs,
        "mean_normalized_dot_error": weighted_dots("mean_normalized_dot_error"),
        "max_normalized_dot_error": max(float(item["max_normalized_dot_error"]) for item in results),
        "attention_queries": sum(int(item["attention_queries"]) for item in results),
        "mean_abs_scaled_logit_error": weighted_dots("mean_abs_scaled_logit_error"),
        "max_abs_scaled_logit_error": max(float(item["max_abs_scaled_logit_error"]) for item in results),
        "mean_softmax_kl": (
            sum(float(item["mean_softmax_kl"]) * int(item["attention_queries"]) for item in results)
            / sum(int(item["attention_queries"]) for item in results)
        ),
        "max_softmax_kl": max(float(item["max_softmax_kl"]) for item in results),
        "mean_max_attention_probability_error": (
            sum(float(item["mean_max_attention_probability_error"]) * int(item["attention_queries"]) for item in results)
            / sum(int(item["attention_queries"]) for item in results)
        ),
        "max_attention_probability_error": max(float(item["max_attention_probability_error"]) for item in results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=pathlib.Path)
    parser.add_argument("--analyzer", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--max-layers",
        type=positive_int,
        help="analyze only the first N manifest layers (diagnostic use)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_dir = args.capture_dir.resolve()
    analyzer = args.analyzer.resolve()
    if not capture_dir.is_dir():
        fail(f"capture directory does not exist: {capture_dir}")
    if not analyzer.is_file() or not analyzer.stat().st_mode & 0o111:
        fail(f"analyzer is not executable: {analyzer}")
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            fail(f"output already exists: {output}")
        if output == capture_dir or capture_dir in output.parents:
            fail("output must not be written inside the immutable capture directory")
    else:
        output = None

    layers = load_manifest(capture_dir)
    validate_files(capture_dir, layers)
    layer_ids = sorted(layers)
    if args.max_layers is not None:
        layer_ids = layer_ids[: args.max_layers]

    by_format: dict[str, Any] = {}
    for format_name in FORMATS:
        per_layer = []
        for layer in layer_ids:
            metrics = run_analyzer(
                analyzer,
                capture_dir,
                layer,
                layers[layer],
                format_name,
            )
            per_layer.append({"layer": layer, **metrics})
        by_format[format_name] = {
            "aggregate": aggregate(per_layer),
            "layers": per_layer,
        }

    result = {
        "schema": "llama-wackmall-turboquant-capture-analysis-v1",
        "capture_dir": str(capture_dir),
        "layer_count": len(layer_ids),
        "layer_ids": layer_ids,
        "formats": by_format,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
    else:
        output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
