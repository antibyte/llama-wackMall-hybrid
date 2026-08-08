#!/usr/bin/env python3
# Copyright (c) 2026 llama-wackMall contributors
# SPDX-License-Identifier: Apache-2.0

"""Offline, non-mutating TriAttention retention simulation for Q/K captures.

The simulator deliberately does not edit a KV cache.  It learns query
statistics from a prefix of a capture, chooses a fixed keep-set at a simulated
prune boundary, and measures retained *exact* attention mass on later queries.
An oracle keep-set supplies an upper bound for any global, fixed-position
eviction policy with the same token budget.

This is an independent implementation of the scoring equations documented by
atomicmilkshake/llama-cpp-turboquant.  No source code from that project is
included here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, NoReturn

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - exercised on dependency failure
    print(
        "error: NumPy is required (on Ubuntu: sudo apt install python3-numpy): "
        f"{error}",
        file=sys.stderr,
    )
    raise SystemExit(2)


SCHEMA = "llama-wackmall-triattention-simulation-v1"


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def budgets_arg(value: str) -> list[int]:
    try:
        budgets = sorted(set(int(part.strip()) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from error
    if not budgets or budgets[0] <= 0:
        raise argparse.ArgumentTypeError("budgets must be positive")
    return budgets


@dataclass
class LayerCapture:
    layer: int
    tokens: int
    head_dim: int
    q_heads: int
    kv_heads: int
    key_rows: int
    pair_rows: int


@dataclass
class MetricAccumulator:
    retained_mass: list[np.ndarray] = field(default_factory=list)
    old_mass: list[np.ndarray] = field(default_factory=list)
    retained_old_mass: list[np.ndarray] = field(default_factory=list)
    top1_retained: int = 0
    top8_retained: int = 0
    query_count: int = 0

    def add(self, probabilities: np.ndarray, keep_old: np.ndarray, prune_at: int) -> None:
        # probabilities has already been causally masked and normalized.
        retained_old = probabilities[:, :prune_at][:, keep_old].sum(axis=1)
        old = probabilities[:, :prune_at].sum(axis=1)
        fresh = probabilities[:, prune_at:].sum(axis=1)
        retained = retained_old + fresh

        self.retained_mass.append(retained.astype(np.float64, copy=False))
        self.old_mass.append(old.astype(np.float64, copy=False))
        self.retained_old_mass.append(retained_old.astype(np.float64, copy=False))

        top1 = np.argmax(probabilities, axis=1)
        top1_old_index = np.minimum(top1, prune_at - 1)
        self.top1_retained += int(
            np.count_nonzero((top1 >= prune_at) | keep_old[top1_old_index])
        )

        top_n = min(8, probabilities.shape[1])
        top = np.argpartition(probabilities, -top_n, axis=1)[:, -top_n:]
        kept = (top >= prune_at) | keep_old[np.minimum(top, prune_at - 1)]
        self.top8_retained += int(np.count_nonzero(kept))
        self.query_count += probabilities.shape[0]

    def finish(self) -> dict[str, Any]:
        if self.query_count == 0:
            fail("metric accumulator is empty")
        retained = np.concatenate(self.retained_mass)
        old = np.concatenate(self.old_mass)
        retained_old = np.concatenate(self.retained_old_mass)
        valid_old = old > 1.0e-12
        old_fraction = retained_old[valid_old] / old[valid_old]
        return {
            "attention_queries": self.query_count,
            "mean_attention_mass_retained": float(np.mean(retained)),
            "p05_attention_mass_retained": float(np.quantile(retained, 0.05)),
            "minimum_attention_mass_retained": float(np.min(retained)),
            "mean_attention_mass_dropped": float(np.mean(1.0 - retained)),
            "weighted_old_attention_fraction_retained": float(
                np.sum(retained_old) / max(float(np.sum(old)), 1.0e-30)
            ),
            "mean_per_query_old_attention_fraction_retained": float(np.mean(old_fraction)),
            "top1_attention_position_retained_ratio": self.top1_retained / self.query_count,
            "top8_attention_position_recall": self.top8_retained / (self.query_count * 8),
        }


@dataclass
class AttentionOutputAccumulator:
    relative_l2: list[np.ndarray] = field(default_factory=list)
    cosine_similarity: list[np.ndarray] = field(default_factory=list)
    squared_error = 0.0
    squared_reference = 0.0
    vector_count = 0

    def add(
        self,
        probabilities: np.ndarray,
        values: np.ndarray,
        keep_old: np.ndarray,
        prune_at: int,
    ) -> None:
        if probabilities.shape[1] != values.shape[0]:
            fail("attention probabilities and values disagree on token count")
        available = np.ones(probabilities.shape[1], dtype=bool)
        available[:prune_at] = keep_old
        retained = probabilities[:, available]
        retained_mass = retained.sum(axis=1, keepdims=True)
        if np.any(retained_mass <= 1.0e-30):
            fail("pruned attention has no retained probability mass")

        reference = np.matmul(probabilities, values)
        if np.all(available):
            # Preserve the no-eviction invariant exactly; a redundant
            # normalization can otherwise introduce roundoff-only error.
            pruned = reference
        else:
            pruned = np.matmul(retained / retained_mass, values[available])
        difference = pruned - reference
        error2 = np.sum(difference.astype(np.float64) ** 2, axis=1)
        reference2 = np.sum(reference.astype(np.float64) ** 2, axis=1)
        pruned2 = np.sum(pruned.astype(np.float64) ** 2, axis=1)
        denominator = np.maximum(reference2, 1.0e-30)
        relative = np.sqrt(error2 / denominator)
        cosine = np.sum(
            reference.astype(np.float64) * pruned.astype(np.float64), axis=1
        ) / np.sqrt(np.maximum(reference2 * pruned2, 1.0e-30))

        self.relative_l2.append(relative)
        self.cosine_similarity.append(np.clip(cosine, -1.0, 1.0))
        self.squared_error += float(np.sum(error2))
        self.squared_reference += float(np.sum(reference2))
        self.vector_count += probabilities.shape[0]

    def finish(self) -> dict[str, Any]:
        if self.vector_count == 0:
            fail("attention-output accumulator is empty")
        relative = np.concatenate(self.relative_l2)
        cosine = np.concatenate(self.cosine_similarity)
        return {
            "vectors": self.vector_count,
            "squared_error_sum": self.squared_error,
            "squared_reference_sum": self.squared_reference,
            "normalized_rmse": math.sqrt(
                self.squared_error / max(self.squared_reference, 1.0e-30)
            ),
            "mean_relative_l2": float(np.mean(relative)),
            "p95_relative_l2": float(np.quantile(relative, 0.95)),
            "maximum_relative_l2": float(np.max(relative)),
            "mean_cosine_similarity": float(np.mean(cosine)),
            "p05_cosine_similarity": float(np.quantile(cosine, 0.05)),
            "minimum_cosine_similarity": float(np.min(cosine)),
        }


def load_manifest(capture_dir: pathlib.Path) -> list[LayerCapture]:
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
    aggregate: dict[int, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            fail("manifest columns do not match the capture format")
        for line, row in enumerate(reader, start=2):
            try:
                values = {name: int(row[name]) for name in required}
            except (KeyError, TypeError, ValueError) as error:
                fail(f"invalid manifest line {line}: {error}")
            if values["layer"] < 0 or any(
                values[name] <= 0
                for name in ("tokens", "head_dim", "q_heads", "kv_heads", "key_rows", "pair_rows")
            ):
                fail(f"invalid dimensions in manifest line {line}")
            layer = values["layer"]
            item = aggregate.setdefault(
                layer,
                {
                    "tokens": 0,
                    "head_dim": values["head_dim"],
                    "q_heads": values["q_heads"],
                    "kv_heads": values["kv_heads"],
                    "key_rows": 0,
                    "pair_rows": 0,
                },
            )
            for name in ("head_dim", "q_heads", "kv_heads"):
                if item[name] != values[name]:
                    fail(f"layer {layer} changes {name} within the capture")
            item["tokens"] += values["tokens"]
            item["key_rows"] += values["key_rows"]
            item["pair_rows"] += values["pair_rows"]

    result: list[LayerCapture] = []
    for layer, item in sorted(aggregate.items()):
        expected_keys = item["tokens"] * item["kv_heads"]
        expected_pairs = item["tokens"] * item["q_heads"] * 2
        if item["key_rows"] != expected_keys or item["pair_rows"] != expected_pairs:
            fail(f"layer {layer} row totals disagree with token/head dimensions")
        result.append(LayerCapture(layer=layer, **item))
    if not result:
        fail("manifest contains no layers")
    return result


def validate_capture_files(
    capture_dir: pathlib.Path,
    layer: LayerCapture,
    require_values: bool = False,
) -> None:
    key_path = capture_dir / f"layer-{layer.layer:03d}-keys.f32"
    pair_path = capture_dir / f"layer-{layer.layer:03d}-qk-pairs.f32"
    expected_key_bytes = layer.key_rows * layer.head_dim * 4
    expected_pair_bytes = layer.pair_rows * layer.head_dim * 4
    if not key_path.is_file() or key_path.stat().st_size != expected_key_bytes:
        fail(f"layer {layer.layer} key file size does not match manifest")
    if not pair_path.is_file() or pair_path.stat().st_size != expected_pair_bytes:
        fail(f"layer {layer.layer} Q/K pair file size does not match manifest")
    value_path = capture_dir / f"layer-{layer.layer:03d}-values.f32"
    if require_values:
        expected_value_bytes = layer.tokens * layer.kv_heads * layer.head_dim * 4
        if not value_path.is_file() or value_path.stat().st_size != expected_value_bytes:
            fail(f"layer {layer.layer} V file size does not match manifest")


def load_layer(capture_dir: pathlib.Path, layer: LayerCapture) -> tuple[np.ndarray, np.ndarray]:
    key_path = capture_dir / f"layer-{layer.layer:03d}-keys.f32"
    pair_path = capture_dir / f"layer-{layer.layer:03d}-qk-pairs.f32"
    keys = np.memmap(key_path, dtype="<f4", mode="r").reshape(
        layer.tokens, layer.kv_heads, layer.head_dim
    )
    pairs = np.memmap(pair_path, dtype="<f4", mode="r").reshape(
        layer.tokens, layer.q_heads, 2, layer.head_dim
    )
    queries = pairs[:, :, 0, :]
    # The duplicated K rows in qk-pairs are independently validated on a few
    # boundary points.  Full validation is already available in
    # llama-turboquant-ref and would add a large memory scan here.
    q_per_kv = layer.q_heads // layer.kv_heads
    if q_per_kv * layer.kv_heads != layer.q_heads:
        fail(f"layer {layer.layer} Q heads are not divisible by KV heads")
    for token in sorted(set((0, layer.tokens // 2, layer.tokens - 1))):
        for q_head in (0, layer.q_heads - 1):
            kv_head = q_head // q_per_kv
            if not np.array_equal(pairs[token, q_head, 1], keys[token, kv_head]):
                fail(f"layer {layer.layer} duplicated K row validation failed")
    return queries, keys


def load_values(capture_dir: pathlib.Path, layer: LayerCapture) -> np.ndarray:
    value_path = capture_dir / f"layer-{layer.layer:03d}-values.f32"
    return np.memmap(value_path, dtype="<f4", mode="r").reshape(
        layer.tokens, layer.kv_heads, layer.head_dim
    )


def layer_manifest_spans(capture_dir: pathlib.Path, layer_id: int) -> list[dict[str, int]]:
    spans: list[dict[str, int]] = []
    token_start = 0
    with (capture_dir / "manifest.csv").open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            try:
                current_layer = int(row["layer"])
                tokens = int(row["tokens"])
                graph = int(row["graph"])
            except (KeyError, TypeError, ValueError) as error:
                fail(f"invalid replay manifest row: {error}")
            if current_layer != layer_id:
                continue
            if tokens <= 0 or graph <= 0:
                fail("replay manifest contains invalid graph or token count")
            spans.append({"graph": graph, "token_start": token_start, "tokens": tokens})
            token_start += tokens
    if not spans:
        fail(f"capture manifest has no layer {layer_id}")
    return spans


def export_attention_delta(
    export_dir: pathlib.Path,
    capture_dir: pathlib.Path,
    capture_summary: dict[str, Any] | None,
    layer: LayerCapture,
    keep_old: np.ndarray,
    train_tokens: int,
    eval_tokens: int,
    policy: str,
    budget: int,
) -> dict[str, Any]:
    if export_dir.exists():
        fail(f"replay export directory already exists: {export_dir}")
    if export_dir == capture_dir or capture_dir in export_dir.parents:
        fail("replay export must not be placed inside the immutable capture directory")
    if capture_summary is None:
        fail("replay export requires a capture summary")
    required_summary = (
        "model",
        "model_bytes",
        "prompt_tokens",
        "prompt_token_hash_fnv1a64",
        "context",
        "batch",
        "ubatch",
        "cache_type_k",
        "cache_type_v",
    )
    missing = [name for name in required_summary if name not in capture_summary]
    if missing:
        fail("replay export capture summary is missing: " + ", ".join(missing))
    if int(capture_summary["prompt_tokens"]) != layer.tokens:
        fail("replay export requires capture length to equal the truncated prompt length")

    queries, keys = load_layer(capture_dir, layer)
    values = load_values(capture_dir, layer)
    end = train_tokens + eval_tokens
    if end > layer.tokens or keep_old.shape != (train_tokens,):
        fail("replay export range or keep set is inconsistent")
    if end != layer.tokens:
        fail("replay export currently requires the evaluation window to end at the prompt boundary")

    available = np.ones(end, dtype=bool)
    available[:train_tokens] = keep_old
    delta = np.zeros((eval_tokens, layer.q_heads, layer.head_dim), dtype="<f4")
    if not np.all(available):
        scale = 1.0 / math.sqrt(layer.head_dim)
        q_per_kv = layer.q_heads // layer.kv_heads
        for q_head in range(layer.q_heads):
            kv_head = q_head // q_per_kv
            logits = np.matmul(
                np.asarray(queries[train_tokens:end, q_head], dtype=np.float32),
                np.asarray(keys[:end, kv_head], dtype=np.float32).T,
            ) * scale
            probabilities = softmax_causal(logits, train_tokens)
            retained = probabilities[:, available]
            retained_mass = retained.sum(axis=1, keepdims=True)
            if np.any(retained_mass <= 1.0e-30):
                fail("replay export retained no attention probability")
            head_values = np.asarray(values[:end, kv_head], dtype=np.float32)
            reference = np.matmul(probabilities, head_values)
            pruned = np.matmul(retained / retained_mass, head_values[available])
            delta[:, q_head, :] = np.asarray(pruned - reference, dtype="<f4")

    spans = layer_manifest_spans(capture_dir, layer.layer)
    records: list[dict[str, int]] = []
    byte_offset = 0
    export_dir.mkdir(parents=True)
    keep_path = export_dir / "attention-keep.u8"
    keep_data = np.ascontiguousarray(keep_old, dtype=np.uint8)
    keep_path.write_bytes(keep_data.tobytes(order="C"))
    if keep_path.stat().st_size != train_tokens:
        fail("replay export produced an invalid keep-set file")
    delta_path = export_dir / "attention-delta.f32"
    with delta_path.open("wb") as output:
        for span in spans:
            span_start = span["token_start"]
            span_end = span_start + span["tokens"]
            overlap_start = max(train_tokens, span_start)
            overlap_end = min(end, span_end)
            if overlap_start >= overlap_end:
                continue
            source_start = overlap_start - train_tokens
            source_end = overlap_end - train_tokens
            chunk = np.ascontiguousarray(delta[source_start:source_end], dtype="<f4")
            output.write(chunk.tobytes(order="C"))
            records.append(
                {
                    "graph": span["graph"],
                    "layer": layer.layer,
                    "tensor_token_offset": overlap_start - span_start,
                    "tokens": overlap_end - overlap_start,
                    "head_dim": layer.head_dim,
                    "heads": layer.q_heads,
                    "byte_offset": byte_offset,
                }
            )
            byte_offset += chunk.nbytes
    if not records or delta_path.stat().st_size != byte_offset:
        fail("replay export produced an invalid delta file")

    manifest_path = export_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as output:
        for name in required_summary:
            output.write(f"# {name}={capture_summary[name]}\n")
        output.write(f"# schema=llama-wackmall-attention-delta-replay-v1\n")
        output.write(f"# policy={policy}\n")
        output.write(f"# budget={budget}\n")
        output.write(f"# train_tokens={train_tokens}\n")
        output.write(f"# eval_tokens={eval_tokens}\n")
        writer = csv.DictWriter(output, fieldnames=tuple(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    metrics = {
        "schema": "llama-wackmall-attention-delta-replay-v1",
        "observational_only": True,
        "kv_cache_modified": False,
        "capture_dir": str(capture_dir),
        "layer": layer.layer,
        "policy": policy,
        "budget": budget,
        "train_tokens": train_tokens,
        "eval_tokens": eval_tokens,
        "records": len(records),
        "delta_bytes": byte_offset,
        "keep_set_bytes": train_tokens,
        "retained_prefix_tokens": int(np.count_nonzero(keep_old)),
        "evicted_prefix_tokens": int(train_tokens - np.count_nonzero(keep_old)),
        "delta_max_abs": float(np.max(np.abs(delta))),
        "delta_rms": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
    }
    (export_dir / "summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def invert_half_rope(
    values: np.ndarray,
    positions: np.ndarray,
    omega: np.ndarray,
    rope_dims: int,
) -> np.ndarray:
    """Invert half-layout RoPE on the rotated prefix of each head."""
    head_dim = values.shape[-1]
    if rope_dims <= 0 or rope_dims > head_dim or rope_dims % 2 != 0:
        fail("rope-dims must be positive, even, and no larger than head-dim")
    if omega.shape != (rope_dims // 2,):
        fail("RoPE dimensions are inconsistent")
    freq_count = rope_dims // 2
    angle = positions.astype(np.float64)[:, None] * omega.astype(np.float64)[None, :]
    cosine = np.cos(angle).astype(np.float32)
    sine = np.sin(angle).astype(np.float32)
    while cosine.ndim < values.ndim:
        cosine = cosine[:, None, :]
        sine = sine[:, None, :]
    real = values[..., :freq_count]
    imag = values[..., freq_count:rope_dims]
    result = np.array(values, dtype=np.float32, copy=True)
    result[..., :freq_count] = real * cosine + imag * sine
    result[..., freq_count:rope_dims] = imag * cosine - real * sine
    return result


def paired_query_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return complex query mean and norm excess for a paired dimension range."""
    if values.shape[-1] % 2 != 0:
        fail("paired query range has odd width")
    half = values.shape[-1] // 2
    real = values[..., :half]
    imag = values[..., half:]
    mean_real = np.mean(real, axis=0, dtype=np.float64).astype(np.float32)
    mean_imag = np.mean(imag, axis=0, dtype=np.float64).astype(np.float32)
    abs_mean = np.mean(np.hypot(real, imag), axis=0, dtype=np.float64).astype(np.float32)
    mean_abs = np.hypot(mean_real, mean_imag)
    return mean_real, mean_imag, np.maximum(abs_mean - mean_abs, 0.0)


def triattention_scores(
    queries: np.ndarray,
    keys: np.ndarray,
    train_tokens: int,
    recent_window: int,
    omega: np.ndarray,
    offsets: np.ndarray,
    rope_dims: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw TriAttention, per-head-normalized, and norm token scores."""
    positions = np.arange(train_tokens, dtype=np.int64)
    pre_queries = invert_half_rope(queries[:train_tokens], positions, omega, rope_dims)
    pre_keys = invert_half_rope(keys[:train_tokens], positions, omega, rope_dims)
    freq_count = rope_dims // 2
    q_per_kv = queries.shape[1] // keys.shape[1]

    q_mean_real, q_mean_imag, extra = paired_query_statistics(pre_queries[..., :rope_dims])
    static_dims = queries.shape[-1] - rope_dims
    if static_dims:
        static_mean_real, static_mean_imag, static_extra = paired_query_statistics(
            pre_queries[..., rope_dims:]
        )

    candidate_count = train_tokens - recent_window
    key_positions = positions[:candidate_count]
    base_delta = train_tokens - key_positions
    # Mean trigonometric phase over the same geometric future offsets used by
    # Atomic's implementation.  [candidate, frequency]
    angles = (
        base_delta[:, None, None].astype(np.float64)
        + offsets[None, :, None].astype(np.float64)
    ) * omega[None, None, :].astype(np.float64)
    cos_mean = np.mean(np.cos(angles), axis=1).astype(np.float32)
    sin_mean = np.mean(np.sin(angles), axis=1).astype(np.float32)

    raw_heads: list[np.ndarray] = []
    normalized_heads: list[np.ndarray] = []
    norm_heads: list[np.ndarray] = []
    for q_head in range(queries.shape[1]):
        kv_head = q_head // q_per_kv
        key = pre_keys[:candidate_count, kv_head]
        k_real = key[:, :freq_count]
        k_imag = key[:, freq_count:rope_dims]
        k_magnitude = np.hypot(k_real, k_imag)
        conj_real = q_mean_real[q_head] * k_real + q_mean_imag[q_head] * k_imag
        conj_imag = q_mean_imag[q_head] * k_real - q_mean_real[q_head] * k_imag
        trig = np.sum(conj_real * cos_mean - conj_imag * sin_mean, axis=1, dtype=np.float64)
        norm = np.sum(extra[q_head] * k_magnitude, axis=1, dtype=np.float64)
        if static_dims:
            static_half = static_dims // 2
            static_key = key[:, rope_dims:]
            static_real = static_key[:, :static_half]
            static_imag = static_key[:, static_half:]
            static_magnitude = np.hypot(static_real, static_imag)
            trig += np.sum(
                static_mean_real[q_head] * static_real + static_mean_imag[q_head] * static_imag,
                axis=1,
                dtype=np.float64,
            )
            norm += np.sum(
                static_extra[q_head] * static_magnitude,
                axis=1,
                dtype=np.float64,
            )
        score = (trig + norm).astype(np.float32)
        raw_heads.append(score)
        standard_deviation = float(np.std(score, dtype=np.float64))
        if standard_deviation > 1.0e-12:
            normalized_heads.append(
                ((score - float(np.mean(score, dtype=np.float64))) / standard_deviation).astype(np.float32)
            )
        else:
            normalized_heads.append(np.zeros_like(score))
        norm_heads.append(norm.astype(np.float32))

    return (
        np.stack(raw_heads),
        np.stack(normalized_heads),
        np.stack(norm_heads),
    )


def select_keep_old(scores: np.ndarray, budget: int, train_tokens: int, recent_window: int) -> np.ndarray:
    if budget < recent_window:
        fail(f"budget {budget} is smaller than recent window {recent_window}")
    if budget > train_tokens:
        fail(f"budget {budget} exceeds prune boundary {train_tokens}")
    candidate_count = train_tokens - recent_window
    choose = budget - recent_window
    keep = np.zeros(train_tokens, dtype=bool)
    keep[candidate_count:] = True
    if choose == candidate_count:
        keep[:candidate_count] = True
    elif choose > 0:
        indices = np.argpartition(scores, -choose)[-choose:]
        keep[indices] = True
    return keep


def softmax_causal(logits: np.ndarray, query_start: int) -> np.ndarray:
    result = logits.astype(np.float64, copy=True)
    key_positions = np.arange(result.shape[1])[None, :]
    query_positions = np.arange(query_start, query_start + result.shape[0])[:, None]
    result[key_positions > query_positions] = -np.inf
    row_max = np.max(result, axis=1, keepdims=True)
    np.exp(result - row_max, out=result)
    result[key_positions > query_positions] = 0.0
    result /= np.sum(result, axis=1, keepdims=True)
    return result


def softmax_causal_available(
    logits: np.ndarray,
    query_start: int,
    available: np.ndarray,
) -> np.ndarray:
    if logits.shape[1] != available.size:
        fail("availability mask does not match attention logits")
    result = logits.astype(np.float64, copy=True)
    key_positions = np.arange(result.shape[1])[None, :]
    query_positions = np.arange(query_start, query_start + result.shape[0])[:, None]
    masked = (key_positions > query_positions) | ~available[None, :]
    result[masked] = -np.inf
    row_max = np.max(result, axis=1, keepdims=True)
    if np.any(~np.isfinite(row_max)):
        fail("recursive policy removed every key for a query")
    np.exp(result - row_max, out=result)
    result[masked] = 0.0
    result /= np.sum(result, axis=1, keepdims=True)
    return result


def select_available(
    scores: np.ndarray,
    available: np.ndarray,
    budget: int,
    recent_window: int,
) -> np.ndarray:
    prune_at = available.size
    recent_start = prune_at - recent_window
    if scores.shape != (prune_at,):
        fail("recursive score vector has the wrong size")
    if budget < recent_window:
        fail("recursive budget is smaller than the recent window")
    if not np.all(available[recent_start:]):
        fail("recursive policy lost a protected recent token")
    keep = np.zeros(prune_at, dtype=bool)
    keep[recent_start:] = True
    candidate_indices = np.flatnonzero(available[:recent_start])
    choose = min(budget - recent_window, candidate_indices.size)
    if choose:
        candidate_scores = scores[candidate_indices]
        selected = np.argpartition(candidate_scores, -choose)[-choose:]
        keep[candidate_indices[selected]] = True
    return keep


def recursive_simulation(
    capture_dir: pathlib.Path,
    layers: list[LayerCapture],
    budget: int,
    recent_window: int,
    history_tokens: int,
    max_events: int | None,
    attention_output: bool,
) -> dict[str, Any]:
    total_tokens = layers[0].tokens
    first_prune = budget + recent_window
    if first_prune + recent_window > total_tokens:
        fail(f"capture is too short for recursive budget {budget}")

    policies = ("recency", "history_attention_global", "oracle_global_fixed")
    accumulated = {policy: MetricAccumulator() for policy in policies}
    accumulated_per_layer = {
        policy: {layer.layer: MetricAccumulator() for layer in layers}
        for policy in policies
    }
    output_accumulated = {
        policy: AttentionOutputAccumulator() for policy in policies
    } if attention_output else None
    output_accumulated_per_layer = {
        policy: {layer.layer: AttentionOutputAccumulator() for layer in layers}
        for policy in policies
    } if attention_output else None
    available: dict[str, np.ndarray] = {
        policy: np.ones(first_prune, dtype=bool) for policy in policies
    }
    events: list[dict[str, Any]] = []
    scale = 1.0 / math.sqrt(layers[0].head_dim)
    prune_at = first_prune

    while prune_at + recent_window <= total_tokens:
        if max_events is not None and len(events) >= max_events:
            break
        eval_end = prune_at + recent_window
        history_start = max(0, prune_at - history_tokens)
        history_importance = np.zeros(prune_at, dtype=np.float64)
        oracle_importance = np.zeros(prune_at, dtype=np.float64)

        for layer in layers:
            queries, keys = load_layer(capture_dir, layer)
            q_per_kv = layer.q_heads // layer.kv_heads
            for q_head in range(layer.q_heads):
                kv_head = q_head // q_per_kv
                history_logits = np.matmul(
                    np.asarray(queries[history_start:prune_at, q_head], dtype=np.float32),
                    np.asarray(keys[:prune_at, kv_head], dtype=np.float32).T,
                ) * scale
                history_probabilities = softmax_causal_available(
                    history_logits,
                    history_start,
                    available["history_attention_global"],
                )
                history_importance += np.sum(history_probabilities, axis=0)

                future_logits = np.matmul(
                    np.asarray(queries[prune_at:eval_end, q_head], dtype=np.float32),
                    np.asarray(keys[:eval_end, kv_head], dtype=np.float32).T,
                ) * scale
                future_probabilities = softmax_causal(future_logits, prune_at)
                oracle_importance += np.sum(future_probabilities[:, :prune_at], axis=0)

        recency_scores = np.arange(prune_at, dtype=np.float64)
        keep = {
            "recency": select_available(
                recency_scores,
                available["recency"],
                budget,
                recent_window,
            ),
            "history_attention_global": select_available(
                history_importance,
                available["history_attention_global"],
                budget,
                recent_window,
            ),
            "oracle_global_fixed": select_available(
                oracle_importance,
                available["oracle_global_fixed"],
                budget,
                recent_window,
            ),
        }
        event_metrics = {policy: MetricAccumulator() for policy in policies}
        event_output = {
            policy: AttentionOutputAccumulator() for policy in policies
        } if attention_output else None

        for layer in layers:
            queries, keys = load_layer(capture_dir, layer)
            values = load_values(capture_dir, layer) if attention_output else None
            q_per_kv = layer.q_heads // layer.kv_heads
            for q_head in range(layer.q_heads):
                kv_head = q_head // q_per_kv
                future_logits = np.matmul(
                    np.asarray(queries[prune_at:eval_end, q_head], dtype=np.float32),
                    np.asarray(keys[:eval_end, kv_head], dtype=np.float32).T,
                ) * scale
                future_probabilities = softmax_causal(future_logits, prune_at)
                for policy in policies:
                    accumulated[policy].add(future_probabilities, keep[policy], prune_at)
                    accumulated_per_layer[policy][layer.layer].add(
                        future_probabilities, keep[policy], prune_at
                    )
                    event_metrics[policy].add(future_probabilities, keep[policy], prune_at)
                    if attention_output:
                        assert values is not None
                        assert output_accumulated is not None
                        assert output_accumulated_per_layer is not None
                        assert event_output is not None
                        output_accumulated[policy].add(
                            future_probabilities,
                            np.asarray(values[:eval_end, kv_head], dtype=np.float32),
                            keep[policy],
                            prune_at,
                        )
                        output_accumulated_per_layer[policy][layer.layer].add(
                            future_probabilities,
                            np.asarray(values[:eval_end, kv_head], dtype=np.float32),
                            keep[policy],
                            prune_at,
                        )
                        event_output[policy].add(
                            future_probabilities,
                            np.asarray(values[:eval_end, kv_head], dtype=np.float32),
                            keep[policy],
                            prune_at,
                        )

        events.append(
            {
                "event": len(events),
                "prune_at": prune_at,
                "eval_end": eval_end,
                "policies": {
                    policy: {
                        **event_metrics[policy].finish(),
                        **(
                            {"attention_output": event_output[policy].finish()}
                            if event_output is not None else {}
                        ),
                    }
                    for policy in policies
                },
            }
        )

        next_prune = prune_at + recent_window
        for policy in policies:
            next_available = np.ones(next_prune, dtype=bool)
            next_available[:prune_at] = keep[policy]
            available[policy] = next_available
        prune_at = next_prune

    return {
        "budget": budget,
        "recent_window": recent_window,
        "history_tokens": history_tokens,
        "first_prune": first_prune,
        "event_count": len(events),
        "last_eval_end": events[-1]["eval_end"] if events else None,
        "policies": {
            policy: {
                **accumulated[policy].finish(),
                **(
                    {"attention_output": output_accumulated[policy].finish()}
                    if output_accumulated is not None else {}
                ),
                "layers": {
                    str(layer.layer): {
                        **accumulated_per_layer[policy][layer.layer].finish(),
                        **(
                            {
                                "attention_output":
                                    output_accumulated_per_layer[policy][layer.layer].finish()
                            }
                            if output_accumulated_per_layer is not None else {}
                        ),
                    }
                    for layer in layers
                },
            }
            for policy in policies
        },
        "events": events,
    }


def evaluate_pass(
    capture_dir: pathlib.Path,
    layers: list[LayerCapture],
    train_tokens: int,
    eval_tokens: int,
    keep_sets: dict[str, dict[int, np.ndarray]],
    collect_oracle: bool,
    attention_output: bool,
) -> tuple[
    dict[str, dict[int, MetricAccumulator]],
    dict[str, dict[int, dict[int, MetricAccumulator]]],
    dict[str, dict[int, AttentionOutputAccumulator]] | None,
    dict[str, dict[int, dict[int, AttentionOutputAccumulator]]] | None,
    np.ndarray,
]:
    accumulators = {
        policy: {budget: MetricAccumulator() for budget in budgets}
        for policy, budgets in keep_sets.items()
    }
    per_layer = {
        policy: {
            budget: {layer.layer: MetricAccumulator() for layer in layers}
            for budget in budgets
        }
        for policy, budgets in keep_sets.items()
    }
    output_accumulators = {
        policy: {budget: AttentionOutputAccumulator() for budget in budgets}
        for policy, budgets in keep_sets.items()
    } if attention_output else None
    output_per_layer = {
        policy: {
            budget: {layer.layer: AttentionOutputAccumulator() for layer in layers}
            for budget in budgets
        }
        for policy, budgets in keep_sets.items()
    } if attention_output else None
    oracle_importance = np.zeros(train_tokens, dtype=np.float64)
    scale = 1.0 / math.sqrt(layers[0].head_dim)
    end = train_tokens + eval_tokens
    for layer in layers:
        queries, keys = load_layer(capture_dir, layer)
        values = load_values(capture_dir, layer) if attention_output else None
        q_per_kv = layer.q_heads // layer.kv_heads
        for q_head in range(layer.q_heads):
            kv_head = q_head // q_per_kv
            logits = np.matmul(
                np.asarray(queries[train_tokens:end, q_head], dtype=np.float32),
                np.asarray(keys[:end, kv_head], dtype=np.float32).T,
            ) * scale
            probabilities = softmax_causal(logits, train_tokens)
            if collect_oracle:
                oracle_importance += np.sum(probabilities[:, :train_tokens], axis=0)
            for policy, budget_sets in keep_sets.items():
                for budget, keep in budget_sets.items():
                    accumulators[policy][budget].add(probabilities, keep, train_tokens)
                    per_layer[policy][budget][layer.layer].add(probabilities, keep, train_tokens)
                    if attention_output:
                        assert values is not None
                        assert output_accumulators is not None
                        assert output_per_layer is not None
                        head_values = np.asarray(values[:end, kv_head], dtype=np.float32)
                        output_accumulators[policy][budget].add(
                            probabilities, head_values, keep, train_tokens
                        )
                        output_per_layer[policy][budget][layer.layer].add(
                            probabilities, head_values, keep, train_tokens
                        )
    return (
        accumulators,
        per_layer,
        output_accumulators,
        output_per_layer,
        oracle_importance,
    )


def historical_attention_importance(
    capture_dir: pathlib.Path,
    layers: list[LayerCapture],
    train_tokens: int,
    history_tokens: int,
) -> np.ndarray:
    """Aggregate exact past attention without looking beyond the prune point."""
    query_start = train_tokens - history_tokens
    importance = np.zeros(train_tokens, dtype=np.float64)
    scale = 1.0 / math.sqrt(layers[0].head_dim)
    for layer in layers:
        queries, keys = load_layer(capture_dir, layer)
        q_per_kv = layer.q_heads // layer.kv_heads
        for q_head in range(layer.q_heads):
            kv_head = q_head // q_per_kv
            logits = np.matmul(
                np.asarray(queries[query_start:train_tokens, q_head], dtype=np.float32),
                np.asarray(keys[:train_tokens, kv_head], dtype=np.float32).T,
            ) * scale
            probabilities = softmax_causal(logits, query_start)
            importance += np.sum(probabilities, axis=0)
    return importance


def summarize(
    accumulators: dict[str, dict[int, MetricAccumulator]],
    per_layer: dict[str, dict[int, dict[int, MetricAccumulator]]],
    keep_sets: dict[str, dict[int, np.ndarray]],
    train_tokens: int,
    output_accumulators: dict[str, dict[int, AttentionOutputAccumulator]] | None,
    output_per_layer: dict[str, dict[int, dict[int, AttentionOutputAccumulator]]] | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for policy, by_budget in accumulators.items():
        output[policy] = {}
        for budget, metrics in by_budget.items():
            keep = keep_sets[policy][budget]
            output[policy][str(budget)] = {
                "budget": budget,
                "retained_prefix_tokens": int(np.count_nonzero(keep)),
                "evicted_prefix_tokens": train_tokens - int(np.count_nonzero(keep)),
                **metrics.finish(),
                "layers": {
                    str(layer): {
                        **layer_metrics.finish(),
                        **(
                            {
                                "attention_output": output_per_layer[policy][budget][layer].finish()
                            }
                            if output_per_layer is not None else {}
                        ),
                    }
                    for layer, layer_metrics in per_layer[policy][budget].items()
                },
            }
            if output_accumulators is not None:
                output[policy][str(budget)]["attention_output"] = (
                    output_accumulators[policy][budget].finish()
                )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--train-tokens", type=positive_int, default=1024)
    parser.add_argument("--eval-tokens", type=positive_int, default=128)
    parser.add_argument(
        "--history-tokens",
        type=positive_int,
        default=128,
        help="past queries used by the causal heavy-hitter comparison",
    )
    parser.add_argument("--budgets", type=budgets_arg, default=budgets_arg("256,512,768"))
    parser.add_argument("--recent-window", type=nonnegative_int, default=128)
    parser.add_argument("--rope-theta", type=float, default=10_000_000.0)
    parser.add_argument(
        "--rope-dims",
        type=positive_int,
        help="number of rotated head dimensions (default: complete head)",
    )
    parser.add_argument("--offset-max", type=positive_int, default=65536)
    parser.add_argument("--max-layers", type=positive_int)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="simulate repeated pruning where evicted positions cannot return",
    )
    parser.add_argument(
        "--recursive-max-events",
        type=positive_int,
        help="limit recursive events for diagnostic runs",
    )
    parser.add_argument(
        "--attention-output",
        action="store_true",
        help="measure reconstructed attention-output error (requires captured V tensors)",
    )
    parser.add_argument(
        "--export-replay-dir",
        type=pathlib.Path,
        help="export one non-recursive attention-output delta replay directory",
    )
    parser.add_argument("--export-replay-layer", type=nonnegative_int)
    parser.add_argument(
        "--export-replay-policy",
        default="history_attention_global",
        choices=(
            "recency",
            "triattention_global",
            "triattention_global_normalized",
            "norm_global",
            "history_attention_global",
            "oracle_global_fixed",
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_dir = args.capture_dir.resolve()
    output_path = args.output.resolve()
    if not capture_dir.is_dir():
        fail(f"capture directory does not exist: {capture_dir}")
    if output_path.exists():
        fail(f"output already exists: {output_path}")
    if output_path == capture_dir or capture_dir in output_path.parents:
        fail("output must not be placed inside the immutable capture directory")
    if not math.isfinite(args.rope_theta) or args.rope_theta <= 1.0:
        fail("--rope-theta must be finite and greater than one")
    if args.export_replay_dir is not None:
        if args.recursive:
            fail("replay export currently supports one non-recursive pruning event")
        if not args.attention_output:
            fail("replay export requires --attention-output")
        if args.export_replay_layer is None:
            fail("replay export requires --export-replay-layer")
        if len(args.budgets) != 1:
            fail("replay export requires exactly one budget")
    layers = load_manifest(capture_dir)
    if args.max_layers is not None:
        layers = layers[: args.max_layers]
    reference = layers[0]
    for layer in layers:
        validate_capture_files(capture_dir, layer, require_values=args.attention_output)
        for name in ("tokens", "head_dim", "q_heads", "kv_heads"):
            if getattr(layer, name) != getattr(reference, name):
                fail(f"selected layers disagree on {name}")

    summary_path = capture_dir / "summary.json"
    capture_summary = None
    if summary_path.is_file():
        capture_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if args.recursive:
        if args.recent_window <= 0:
            fail("recursive simulation requires a positive recent window")
        if args.history_tokens > reference.tokens:
            fail("--history-tokens exceeds captured tokens")
        recursive_results = {}
        for budget in args.budgets:
            if budget < args.recent_window:
                fail("every recursive budget must be at least the recent window")
            recursive_results[str(budget)] = recursive_simulation(
                capture_dir,
                layers,
                budget,
                args.recent_window,
                args.history_tokens,
                args.recursive_max_events,
                args.attention_output,
            )
        result = {
            "schema": "llama-wackmall-heavy-hitter-recursive-v1",
            "capture_dir": str(capture_dir),
            "capture_summary": capture_summary,
            "simulation": {
                "observational_only": True,
                "kv_cache_modified": False,
                "recursive": True,
                "layer_ids": [layer.layer for layer in layers],
                "layer_count": len(layers),
                "q_heads": reference.q_heads,
                "kv_heads": reference.kv_heads,
                "head_dim": reference.head_dim,
                "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                "attention_output_measured": args.attention_output,
                "limitations": [
                    "queries and keys come from the full-cache baseline and do not model state divergence",
                    "oracle uses future attention and is an unattainable upper bound",
                    "only the captured representative layers are evaluated",
                    *(
                        ["downstream residual and logits are not captured"]
                        if args.attention_output else
                        ["V, downstream residual, and logits are not captured"]
                    ),
                ],
            },
            "budgets": recursive_results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {output_path}")
        return 0

    if args.recent_window >= args.train_tokens:
        fail("--recent-window must be smaller than --train-tokens")
    if args.history_tokens > args.train_tokens:
        fail("--history-tokens must not exceed --train-tokens")
    if args.train_tokens + args.eval_tokens > reference.tokens:
        fail("train-tokens + eval-tokens exceed captured tokens")
    for budget in args.budgets:
        if budget < args.recent_window or budget > args.train_tokens:
            fail("every budget must be between recent-window and train-tokens")

    rope_dims = args.rope_dims if args.rope_dims is not None else reference.head_dim
    if rope_dims > reference.head_dim or rope_dims % 2 != 0:
        fail("--rope-dims must be even and no larger than head-dim")
    if (reference.head_dim - rope_dims) % 2 != 0:
        fail("the non-RoPE dimension count must be even")
    freq_count = rope_dims // 2
    omega = np.power(
        args.rope_theta,
        -2.0 * np.arange(freq_count, dtype=np.float64) / reference.head_dim,
    ).astype(np.float32)
    offsets = []
    value = 1
    while value <= args.offset_max:
        offsets.append(value)
        value *= 2
    offsets_array = np.asarray(offsets, dtype=np.float32)

    candidate_count = args.train_tokens - args.recent_window
    raw_global = np.full(candidate_count, -np.inf, dtype=np.float32)
    normalized_global = np.full(candidate_count, -np.inf, dtype=np.float32)
    norm_global = np.full(candidate_count, -np.inf, dtype=np.float32)
    for layer in layers:
        queries, keys = load_layer(capture_dir, layer)
        raw, normalized, norm = triattention_scores(
            queries,
            keys,
            args.train_tokens,
            args.recent_window,
            omega,
            offsets_array,
            rope_dims,
        )
        raw_global = np.maximum(raw_global, np.max(raw, axis=0))
        normalized_global = np.maximum(normalized_global, np.max(normalized, axis=0))
        norm_global = np.maximum(norm_global, np.max(norm, axis=0))

    recency_scores = np.arange(candidate_count, dtype=np.float32)
    history_scores = historical_attention_importance(
        capture_dir,
        layers,
        args.train_tokens,
        args.history_tokens,
    )[:candidate_count]
    score_map = {
        "recency": recency_scores,
        "triattention_global": raw_global,
        "triattention_global_normalized": normalized_global,
        "norm_global": norm_global,
        "history_attention_global": history_scores,
    }
    keep_sets = {
        policy: {
            budget: select_keep_old(
                scores,
                budget,
                args.train_tokens,
                args.recent_window,
            )
            for budget in args.budgets
        }
        for policy, scores in score_map.items()
    }

    (
        accumulators,
        per_layer,
        output_accumulators,
        output_per_layer,
        oracle_importance,
    ) = evaluate_pass(
        capture_dir,
        layers,
        args.train_tokens,
        args.eval_tokens,
        keep_sets,
        collect_oracle=True,
        attention_output=args.attention_output,
    )
    oracle_keep = {
        budget: select_keep_old(
            oracle_importance[:candidate_count],
            budget,
            args.train_tokens,
            args.recent_window,
        )
        for budget in args.budgets
    }
    oracle_sets = {"oracle_global_fixed": oracle_keep}
    (
        oracle_accumulators,
        oracle_per_layer,
        oracle_output_accumulators,
        oracle_output_per_layer,
        _,
    ) = evaluate_pass(
        capture_dir,
        layers,
        args.train_tokens,
        args.eval_tokens,
        oracle_sets,
        collect_oracle=False,
        attention_output=args.attention_output,
    )
    keep_sets.update(oracle_sets)
    accumulators.update(oracle_accumulators)
    per_layer.update(oracle_per_layer)
    if output_accumulators is not None:
        assert oracle_output_accumulators is not None
        assert output_per_layer is not None
        assert oracle_output_per_layer is not None
        output_accumulators.update(oracle_output_accumulators)
        output_per_layer.update(oracle_output_per_layer)

    replay_export = None
    if args.export_replay_dir is not None:
        selected_layer = next(
            (layer for layer in layers if layer.layer == args.export_replay_layer),
            None,
        )
        if selected_layer is None:
            fail(f"replay export layer {args.export_replay_layer} is not in the capture")
        budget = args.budgets[0]
        replay_export = export_attention_delta(
            args.export_replay_dir.resolve(),
            capture_dir,
            capture_summary,
            selected_layer,
            keep_sets[args.export_replay_policy][budget],
            args.train_tokens,
            args.eval_tokens,
            args.export_replay_policy,
            budget,
        )

    result = {
        "schema": SCHEMA,
        "capture_dir": str(capture_dir),
        "capture_summary": capture_summary,
        "replay_export": replay_export,
        "simulation": {
            "observational_only": True,
            "kv_cache_modified": False,
            "capture_stage": "post-rope",
            "rope_style": "half",
            "rope_theta": args.rope_theta,
            "rope_dims": rope_dims,
            "non_rope_dims": reference.head_dim - rope_dims,
            "train_tokens": args.train_tokens,
            "prune_at": args.train_tokens,
            "eval_tokens": args.eval_tokens,
            "history_tokens": args.history_tokens,
            "recent_window": args.recent_window,
            "budgets": args.budgets,
            "offset_max": args.offset_max,
            "offsets": offsets,
            "layer_ids": [layer.layer for layer in layers],
            "layer_count": len(layers),
            "q_heads": reference.q_heads,
            "kv_heads": reference.kv_heads,
            "head_dim": reference.head_dim,
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "attention_output_measured": args.attention_output,
            "limitations": [
                "single pruning event rather than recursive production eviction",
                "oracle uses future attention and is an unattainable upper bound",
                "calibration and evaluation come from one prompt but use disjoint token ranges",
                "only the captured representative layers are evaluated",
                "history-attention is causal but would require runtime attention-score accounting",
                "partial-RoPE scoring of static dimensions is an independent extension, not Atomic runtime behavior",
                *(
                    ["downstream residual and logits are not captured"]
                    if args.attention_output else
                    ["V, downstream residual, and logits are not captured"]
                ),
            ],
        },
        "policy_descriptions": {
            "recency": "keep the newest prefix positions",
            "norm_global": "keep positions with the largest max-over-head paired norm score",
            "triattention_global": "Atomic-style trigonometric score with max-over-head aggregation",
            "triattention_global_normalized": "same score after per-head z-score normalization",
            "history_attention_global": "causal attention mass received over preceding queries",
            "oracle_global_fixed": "fixed keep-set optimized using future evaluation attention",
        },
        "policies": summarize(
            accumulators,
            per_layer,
            keep_sets,
            args.train_tokens,
            output_accumulators,
            output_per_layer,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
