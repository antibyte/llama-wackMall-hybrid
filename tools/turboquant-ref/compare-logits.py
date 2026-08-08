#!/usr/bin/env python3
# Copyright (c) 2026 llama-wackMall contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare immutable raw little-endian F32 logit vectors or row matrices."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import NoReturn

try:
    import numpy as np
except ImportError as error:  # pragma: no cover
    print(f"error: NumPy is required: {error}", file=sys.stderr)
    raise SystemExit(2)


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def load_logits(path: pathlib.Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0 or path.stat().st_size % 4 != 0:
        fail(f"invalid raw F32 logit file: {path}")
    values = np.memmap(path, dtype="<f4", mode="r")
    if not np.all(np.isfinite(values)):
        fail(f"logit file contains non-finite values: {path}")
    return values


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)
    return probabilities


def load_window_metadata(path: pathlib.Path) -> dict[str, object] | None:
    metadata_path = pathlib.Path(str(path) + ".json")
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "llama-wackmall-logits-window-v1":
        fail(f"unsupported logits window metadata: {metadata_path}")
    for field in ("rows", "vocab_size", "start_token", "prompt_token_count"):
        if not isinstance(metadata.get(field), int):
            fail(f"invalid {field} in {metadata_path}")
    if metadata["rows"] <= 0 or metadata["vocab_size"] <= 0:
        fail(f"invalid dimensions in {metadata_path}")
    return metadata


def compare_row(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    ref64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    difference = candidate64 - ref64
    reference_norm2 = float(np.dot(ref64, ref64))
    candidate_norm2 = float(np.dot(candidate64, candidate64))
    error_norm2 = float(np.dot(difference, difference))
    probabilities_reference = softmax(reference)
    probabilities_candidate = softmax(candidate)
    epsilon = np.finfo(np.float64).tiny
    kld = float(
        np.sum(
            probabilities_reference
            * (
                np.log(np.maximum(probabilities_reference, epsilon))
                - np.log(np.maximum(probabilities_candidate, epsilon))
            )
        )
    )
    reference_top = int(np.argmax(reference))
    candidate_top = int(np.argmax(candidate))
    top_n = min(10, reference.size)
    reference_top_n = np.argpartition(reference, -top_n)[-top_n:]
    candidate_top_n = np.argpartition(candidate, -top_n)[-top_n:]
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "absolute_error_sum": float(np.sum(np.abs(difference))),
        "squared_error_sum": error_norm2,
        "squared_reference_sum": reference_norm2,
        "squared_candidate_sum": candidate_norm2,
        "relative_l2_error": math.sqrt(error_norm2 / max(reference_norm2, epsilon)),
        "softmax_kld_reference_to_candidate": kld,
        "maximum_probability_error": float(
            np.max(np.abs(probabilities_candidate - probabilities_reference))
        ),
        "reference_top_token": reference_top,
        "candidate_top_token": candidate_top,
        "same_top_token": reference_top == candidate_top,
        "top10_overlap": int(
            len(set(reference_top_n.tolist()) & set(candidate_top_n.tolist()))
        ),
        "reference_top_probability": float(probabilities_reference[reference_top]),
        "candidate_probability_at_reference_top": float(
            probabilities_candidate[reference_top]
        ),
        "dot_product": float(np.dot(ref64, candidate64)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--row-width",
        type=int,
        help="logits per row; inferred from capture metadata when available",
    )
    args = parser.parse_args()

    reference_path = args.reference.resolve()
    candidate_path = args.candidate.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        fail(f"output already exists: {output_path}")
    if output_path in (reference_path, candidate_path):
        fail("output must not overwrite an input")

    reference = load_logits(reference_path)
    candidate = load_logits(candidate_path)
    if reference.shape != candidate.shape:
        fail("logit files have different element counts")

    reference_metadata = load_window_metadata(reference_path)
    candidate_metadata = load_window_metadata(candidate_path)
    if (reference_metadata is None) != (candidate_metadata is None):
        fail("only one input has logits window metadata")
    if reference_metadata is not None and candidate_metadata is not None:
        compared_fields = (
            "rows",
            "vocab_size",
            "start_token",
            "prompt_token_count",
            "prompt_token_hash_fnv1a64",
        )
        for field in compared_fields:
            if reference_metadata.get(field) != candidate_metadata.get(field):
                fail(f"logits window metadata differs in {field}")
        metadata_row_width = int(reference_metadata["vocab_size"])
        if args.row_width is not None and args.row_width != metadata_row_width:
            fail("--row-width conflicts with logits window metadata")
        row_width = metadata_row_width
    else:
        row_width = args.row_width if args.row_width is not None else int(reference.size)
    if row_width <= 0 or reference.size % row_width != 0:
        fail("row width does not divide the logit element count")
    row_count = int(reference.size // row_width)

    row_results = []
    for row in range(row_count):
        start = row*row_width
        row_result = compare_row(
            reference[start:start + row_width], candidate[start:start + row_width]
        )
        row_result["row"] = row
        if reference_metadata is not None:
            row_result["prompt_token"] = int(reference_metadata["start_token"]) + row
        row_results.append(row_result)

    error_norm2 = sum(float(row["squared_error_sum"]) for row in row_results)
    reference_norm2 = sum(float(row["squared_reference_sum"]) for row in row_results)
    candidate_norm2 = sum(float(row["squared_candidate_sum"]) for row in row_results)
    dot_product = sum(float(row["dot_product"]) for row in row_results)
    epsilon = np.finfo(np.float64).tiny
    relative_l2_rows = np.asarray(
        [row["relative_l2_error"] for row in row_results], dtype=np.float64
    )
    kld_rows = np.asarray(
        [row["softmax_kld_reference_to_candidate"] for row in row_results],
        dtype=np.float64,
    )
    top10_rows = np.asarray([row["top10_overlap"] for row in row_results])
    same_top_count = sum(bool(row["same_top_token"]) for row in row_results)

    result = {
        "schema": "llama-wackmall-logit-comparison-v2",
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "logit_count": int(reference.size),
        "row_count": row_count,
        "row_width": row_width,
        "byte_identical": bool(np.array_equal(reference, candidate)),
        "maximum_absolute_error": max(
            float(row["maximum_absolute_error"]) for row in row_results
        ),
        "mean_absolute_error": sum(
            float(row["absolute_error_sum"]) for row in row_results
        ) / reference.size,
        "root_mean_square_error": math.sqrt(error_norm2 / reference.size),
        "relative_l2_error": math.sqrt(error_norm2 / max(reference_norm2, epsilon)),
        "cosine_similarity": float(
            dot_product / math.sqrt(max(reference_norm2 * candidate_norm2, epsilon))
        ),
        "mean_row_softmax_kld_reference_to_candidate": float(np.mean(kld_rows)),
        "p95_row_softmax_kld_reference_to_candidate": float(np.percentile(kld_rows, 95)),
        "maximum_row_softmax_kld_reference_to_candidate": float(np.max(kld_rows)),
        "maximum_probability_error": max(
            float(row["maximum_probability_error"]) for row in row_results
        ),
        "same_top_token_count": same_top_count,
        "same_top_token_ratio": same_top_count / row_count,
        "mean_top10_overlap": float(np.mean(top10_rows)),
        "minimum_top10_overlap": int(np.min(top10_rows)),
        "mean_row_relative_l2_error": float(np.mean(relative_l2_rows)),
        "p95_row_relative_l2_error": float(np.percentile(relative_l2_rows, 95)),
        "maximum_row_relative_l2_error": float(np.max(relative_l2_rows)),
        "first_top_token_divergence_row": next(
            (int(row["row"]) for row in row_results if not row["same_top_token"]), None
        ),
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key
                not in (
                    "absolute_error_sum",
                    "squared_error_sum",
                    "squared_reference_sum",
                    "squared_candidate_sum",
                    "dot_product",
                )
            }
            for row in row_results
        ],
    }
    if row_count == 1:
        result.update(
            {
                "softmax_kld_reference_to_candidate": row_results[0][
                    "softmax_kld_reference_to_candidate"
                ],
                "reference_top_token": row_results[0]["reference_top_token"],
                "candidate_top_token": row_results[0]["candidate_top_token"],
                "same_top_token": row_results[0]["same_top_token"],
                "top10_overlap": row_results[0]["top10_overlap"],
                "reference_top_probability": row_results[0][
                    "reference_top_probability"
                ],
                "candidate_probability_at_reference_top": row_results[0][
                    "candidate_probability_at_reference_top"
                ],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
