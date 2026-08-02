#!/usr/bin/env python3
"""Optimize static per-layer expert slots under a model-derived byte budget."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from convert_luce_spark_profile import (
    GGUFError,
    ProfileError,
    _GGUFReader,
    read_gguf_metadata,
    validate_wack_profile,
)


EXPERT_TENSOR_RE = re.compile(
    r"blk\.([0-9]+)\.ffn_(?:gate|up|down|gate_up)_exps\.weight"
)

# GGUF ggml_type -> (elements per block, bytes per block). Keep this table in
# sync with ggml/include/ggml.h and ggml/src/ggml-common.h. Unsupported future
# types fail explicitly instead of guessing from neighboring tensor offsets.
GGML_TYPE_LAYOUT: dict[int, tuple[int, int]] = {
    0: (1, 4),       # f32
    1: (1, 2),       # f16
    2: (32, 18),     # q4_0
    3: (32, 20),     # q4_1
    6: (32, 22),     # q5_0
    7: (32, 24),     # q5_1
    8: (32, 34),     # q8_0
    9: (32, 36),     # q8_1
    10: (256, 84),   # q2_K
    11: (256, 110),  # q3_K
    12: (256, 144),  # q4_K
    13: (256, 176),  # q5_K
    14: (256, 210),  # q6_K
    15: (256, 292),  # q8_K
    16: (256, 66),   # iq2_xxs
    17: (256, 74),   # iq2_xs
    18: (256, 98),   # iq3_xxs
    19: (256, 50),   # iq1_s
    20: (32, 18),    # iq4_nl
    21: (256, 110),  # iq3_s
    22: (256, 82),   # iq2_s
    23: (256, 136),  # iq4_xs
    24: (1, 1),      # i8
    25: (1, 2),      # i16
    26: (1, 4),      # i32
    27: (1, 8),      # i64
    28: (1, 8),      # f64
    29: (256, 56),   # iq1_m
    30: (1, 2),      # bf16
    34: (256, 54),   # tq1_0
    35: (256, 66),   # tq2_0
    39: (32, 17),    # mxfp4
    40: (64, 36),    # nvfp4
    41: (128, 18),   # q1_0
    42: (64, 18),    # q2_0
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dimensions: tuple[int, ...]
    type_id: int
    offset: int
    nbytes: int


@dataclass(frozen=True)
class PlacementResult:
    slots: tuple[int, ...]
    experts: tuple[tuple[int, ...], ...]
    fixed_budget_bytes: int
    fixed_bytes_used: int
    predicted_hits: int
    predicted_total: int


def tensor_nbytes(dimensions: tuple[int, ...], type_id: int) -> int:
    if not dimensions or any(value <= 0 for value in dimensions):
        raise GGUFError(f"invalid tensor dimensions {dimensions}")
    layout = GGML_TYPE_LAYOUT.get(type_id)
    if layout is None:
        raise GGUFError(
            f"unsupported ggml tensor type id {type_id}; update GGML_TYPE_LAYOUT explicitly"
        )
    block, block_bytes = layout
    if dimensions[0] % block:
        raise GGUFError(
            f"tensor row size {dimensions[0]} is not divisible by type-{type_id} block {block}"
        )
    return dimensions[0] // block * block_bytes * math.prod(dimensions[1:])


def read_gguf_tensors(path: Path) -> list[TensorInfo]:
    try:
        file_size = path.stat().st_size
        stream = path.open("rb")
    except OSError as exc:
        raise GGUFError(f"cannot open GGUF model {path}: {exc}") from exc

    with stream:
        reader = _GGUFReader(stream, file_size)
        if reader.read_exact(4) != b"GGUF":
            raise GGUFError(f"{path} does not start with GGUF magic")
        version = reader.unpack("<I")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = reader.unpack("<Q")
        metadata_count = reader.unpack("<Q")
        if tensor_count > 10_000_000 or metadata_count > 1_000_000:
            raise GGUFError("implausible GGUF tensor or metadata count")
        for _ in range(metadata_count):
            reader.read_string(decode=False)
            reader.skip_value(reader.unpack("<I"))

        result: list[TensorInfo] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            name = reader.read_string()
            assert name is not None
            if name in names:
                raise GGUFError(f"duplicate GGUF tensor name {name!r}")
            names.add(name)
            n_dimensions = reader.unpack("<I")
            if n_dimensions < 1 or n_dimensions > 4:
                raise GGUFError(f"tensor {name!r} has invalid dimension count {n_dimensions}")
            dimensions = tuple(reader.unpack("<Q") for _ in range(n_dimensions))
            type_id = reader.unpack("<I")
            offset = reader.unpack("<Q")
            result.append(
                TensorInfo(
                    name=name,
                    dimensions=dimensions,
                    type_id=type_id,
                    offset=offset,
                    nbytes=tensor_nbytes(dimensions, type_id),
                )
            )
        return result


def expert_slot_bytes(
    tensors: list[TensorInfo], n_layer: int, n_expert: int
) -> tuple[int, ...]:
    result = [0] * n_layer
    tensor_counts = [0] * n_layer
    for tensor in tensors:
        match = EXPERT_TENSOR_RE.fullmatch(tensor.name)
        if not match:
            continue
        layer = int(match.group(1))
        if layer >= n_layer:
            continue  # appended MTP/next-token layers are not in the target tier
        if tensor.dimensions[-1] != n_expert:
            raise ProfileError(
                f"expert tensor {tensor.name!r} has trailing dimension "
                f"{tensor.dimensions[-1]}, expected {n_expert}"
            )
        if tensor.nbytes % n_expert:
            raise ProfileError(
                f"expert tensor {tensor.name!r} has {tensor.nbytes} bytes, not divisible "
                f"by {n_expert} experts"
            )
        result[layer] += tensor.nbytes // n_expert
        tensor_counts[layer] += 1
    missing = [layer for layer, count in enumerate(tensor_counts) if count == 0]
    if missing:
        raise ProfileError(f"GGUF has no recognized routed-expert tensors for layers {missing}")
    return tuple(result)


def read_layer_costs(path: Path | None, n_layer: int) -> tuple[Fraction, ...]:
    if path is None:
        return tuple(Fraction(1) for _ in range(n_layer))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ProfileError(f"cannot read layer costs {path}: {exc}") from exc
    if not lines or lines[0] != "layer,cost_multiplier":
        raise ProfileError("layer cost CSV must start with 'layer,cost_multiplier'")
    values: dict[int, Fraction] = {}
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split(",")
        if len(fields) != 2:
            raise ProfileError(f"layer cost line {line_number} must have two fields")
        try:
            layer = int(fields[0])
            multiplier = Fraction(fields[1])
        except (ValueError, ZeroDivisionError) as exc:
            raise ProfileError(f"invalid layer cost line {line_number}: {line!r}") from exc
        if layer < 0 or layer >= n_layer or multiplier <= 0:
            raise ProfileError(f"invalid layer or multiplier on cost line {line_number}")
        if layer in values:
            raise ProfileError(f"duplicate layer {layer} in cost CSV")
        values[layer] = multiplier
    if set(values) != set(range(n_layer)):
        missing = sorted(set(range(n_layer)) - set(values))
        raise ProfileError(f"layer cost CSV is incomplete; missing layers {missing}")
    return tuple(values[layer] for layer in range(n_layer))


def read_layer_costs_json(path: Path, n_layer: int) -> tuple[Fraction, ...]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read timing stats JSON {path}: {exc}") from exc
    if data.get("timing_enabled") is not True or not isinstance(data.get("layers"), list):
        raise ProfileError("timing stats JSON does not contain enabled per-layer timing")
    values: dict[int, Fraction] = {}
    for item in data["layers"]:
        if not isinstance(item, dict):
            raise ProfileError("timing stats layer entry is not an object")
        layer = item.get("layer")
        cold_hits = item.get("cold_hits")
        cpu_cold_ms = item.get("cpu_cold_ms")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or layer >= n_layer
            or isinstance(cold_hits, bool)
            or not isinstance(cold_hits, int)
            or cold_hits <= 0
            or not isinstance(cpu_cold_ms, (int, Decimal))
            or cpu_cold_ms <= 0
        ):
            raise ProfileError(f"invalid or unmeasured timing entry for layer {layer!r}")
        if layer in values:
            raise ProfileError(f"duplicate layer {layer} in timing stats JSON")
        values[layer] = Fraction(cpu_cold_ms) / cold_hits
    if set(values) != set(range(n_layer)):
        missing = sorted(set(range(n_layer)) - set(values))
        raise ProfileError(f"timing stats JSON is incomplete; missing layers {missing}")
    return tuple(values[layer] for layer in range(n_layer))


def optimize(
    counts: list[list[int]],
    slot_bytes: tuple[int, ...],
    layer_costs: tuple[Fraction, ...],
    *,
    fixed_budget_bytes: int,
    min_slots: int,
    max_slots: int,
    objective: str,
) -> PlacementResult:
    n_layer = len(counts)
    if n_layer == 0 or len(slot_bytes) != n_layer or len(layer_costs) != n_layer:
        raise ProfileError("placement dimensions are inconsistent")
    n_expert = len(counts[0])
    if n_expert == 0 or any(len(row) != n_expert for row in counts):
        raise ProfileError("profile rows have inconsistent expert counts")
    if min_slots < 1 or max_slots < min_slots or max_slots > n_expert:
        raise ProfileError(
            f"slot bounds must satisfy 1 <= min <= max <= {n_expert}, got "
            f"{min_slots}..{max_slots}"
        )
    minimum_bytes = min_slots * sum(slot_bytes)
    if fixed_budget_bytes < minimum_bytes:
        raise ProfileError(
            f"fixed budget {fixed_budget_bytes} bytes cannot hold min_slots={min_slots}; "
            f"at least {minimum_bytes} bytes are required"
        )

    rankings = [
        sorted(range(n_expert), key=lambda expert: (-counts[layer][expert], expert))
        for layer in range(n_layer)
    ]
    selected = [ranking[:min_slots] for ranking in rankings]
    slots = [min_slots] * n_layer
    used = minimum_bytes
    queue: list[tuple[Fraction, int, int]] = []

    def priority(layer: int, rank: int) -> Fraction:
        benefit = Fraction(counts[layer][rankings[layer][rank]]) * layer_costs[layer]
        if objective == "counts-per-byte":
            benefit /= slot_bytes[layer]
        return benefit

    for layer in range(n_layer):
        if min_slots < max_slots:
            heapq.heappush(queue, (-priority(layer, min_slots), layer, min_slots))

    while queue:
        _, layer, rank = heapq.heappop(queue)
        if used + slot_bytes[layer] > fixed_budget_bytes:
            continue
        selected[layer].append(rankings[layer][rank])
        slots[layer] += 1
        used += slot_bytes[layer]
        next_rank = rank + 1
        if next_rank < max_slots:
            heapq.heappush(queue, (-priority(layer, next_rank), layer, next_rank))

    predicted_total = sum(sum(row) for row in counts)
    predicted_hits = sum(
        counts[layer][expert]
        for layer in range(n_layer)
        for expert in selected[layer]
    )
    return PlacementResult(
        slots=tuple(slots),
        experts=tuple(tuple(experts) for experts in selected),
        fixed_budget_bytes=fixed_budget_bytes,
        fixed_bytes_used=used,
        predicted_hits=predicted_hits,
        predicted_total=predicted_total,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: Path,
    *,
    model_architecture: str,
    model_file_size: int,
    n_expert: int,
    profile_sha256: str,
    objective: str,
    min_slots: int,
    max_slots: int,
    slot_bytes: tuple[int, ...],
    result: PlacementResult,
) -> None:
    if not path.parent.is_dir():
        raise ProfileError(f"output directory does not exist: {path.parent}")
    if os.path.lexists(path):
        raise ProfileError(f"refusing to overwrite existing output: {path}")
    sentinel_bytes = sum(slot_bytes)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
            delete=False,
        ) as stream:
            temp_name = stream.name
            stream.write("# llama-wackmall-expert-placement-v1\n")
            stream.write(f"# model_architecture={model_architecture}\n")
            stream.write(f"# model_file_size={model_file_size}\n")
            stream.write(f"# model_layers={len(slot_bytes)}\n")
            stream.write(f"# model_experts={n_expert}\n")
            stream.write(f"# profile_sha256={profile_sha256}\n")
            stream.write(f"# objective={objective}\n")
            stream.write(f"# fixed_budget_bytes={result.fixed_budget_bytes}\n")
            stream.write(f"# fixed_bytes_used={result.fixed_bytes_used}\n")
            stream.write(f"# sentinel_bytes={sentinel_bytes}\n")
            stream.write(f"# min_slots={min_slots}\n")
            stream.write(f"# max_slots={max_slots}\n")
            stream.write("layer,fixed_slots,slot_bytes\n")
            for layer, (slots, size) in enumerate(zip(result.slots, slot_bytes)):
                stream.write(f"{layer},{slots},{size}\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp = Path(temp_name)
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise ProfileError(f"refusing to overwrite output created concurrently: {path}") from exc
        temp.unlink()
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def run(args: argparse.Namespace) -> None:
    model_path = args.model.resolve()
    profile_path = args.profile.resolve()
    if not model_path.is_file() or not profile_path.is_file():
        raise ProfileError("model and profile must both exist")
    model = read_gguf_metadata(model_path)
    sparse = validate_wack_profile(profile_path, model.dimensions)
    counts = [
        [sparse.get((layer, expert), 0) for expert in range(model.dimensions.n_expert)]
        for layer in range(model.dimensions.n_layer)
    ]
    for layer, row in enumerate(counts):
        if sum(row) == 0:
            raise ProfileError(f"profile has no observations for layer {layer}")

    tensors = read_gguf_tensors(model_path)
    sizes = expert_slot_bytes(tensors, model.dimensions.n_layer, model.dimensions.n_expert)
    if args.layer_stats_json:
        costs = read_layer_costs_json(args.layer_stats_json.resolve(), len(sizes))
    else:
        costs = read_layer_costs(args.layer_costs.resolve() if args.layer_costs else None, len(sizes))

    if args.reference_slots is not None:
        if args.reference_slots < 1 or args.reference_slots > model.dimensions.n_expert:
            raise ProfileError(
                f"reference-slots must be in [1, {model.dimensions.n_expert}]"
            )
        budget = args.reference_slots * sum(sizes)
    else:
        budget_fraction = Fraction(str(args.fixed_budget_mib)) * 1024 * 1024
        budget = budget_fraction.numerator // budget_fraction.denominator
        if budget <= 0:
            raise ProfileError("fixed-budget-mib must be positive")

    max_slots = args.max_slots or model.dimensions.n_expert
    result = optimize(
        counts,
        sizes,
        costs,
        fixed_budget_bytes=budget,
        min_slots=args.min_slots,
        max_slots=max_slots,
        objective=args.objective,
    )
    output = args.output.resolve(strict=False)
    profile_digest = sha256_file(profile_path)
    write_manifest(
        output,
        model_architecture=model.architecture,
        model_file_size=model.file_size,
        n_expert=model.dimensions.n_expert,
        profile_sha256=profile_digest,
        objective=args.objective,
        min_slots=args.min_slots,
        max_slots=max_slots,
        slot_bytes=sizes,
        result=result,
    )

    coverage = 100.0 * result.predicted_hits / result.predicted_total
    print(f"model: {model_path}")
    print(f"profile: {profile_path} sha256={profile_digest}")
    print(
        f"dimensions: layers={model.dimensions.n_layer} experts={model.dimensions.n_expert} "
        f"used={model.dimensions.n_expert_used}"
    )
    print(
        f"slot-bytes: min={min(sizes)} max={max(sizes)} sum={sum(sizes)} "
        f"sentinel_mib={sum(sizes) / 1048576:.3f}"
    )
    print(
        f"placement: objective={args.objective} slots={sum(result.slots)} "
        f"range={min(result.slots)}..{max(result.slots)} "
        f"fixed_used_mib={result.fixed_bytes_used / 1048576:.3f} "
        f"fixed_budget_mib={result.fixed_budget_bytes / 1048576:.3f} "
        f"coverage={coverage:.3f}%"
    )
    print("slots-by-layer: " + ",".join(str(value) for value in result.slots))
    print(f"output: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="wackMall usage CSV")
    parser.add_argument("--model", type=Path, required=True, help="target GGUF")
    parser.add_argument("--output", type=Path, required=True, help="new placement manifest")
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument(
        "--reference-slots",
        type=int,
        metavar="S",
        help="same fixed-expert bytes as uniform S slots per layer",
    )
    budget.add_argument(
        "--fixed-budget-mib",
        type=float,
        help="explicit fixed-expert budget; one sentinel per layer is additional",
    )
    parser.add_argument("--min-slots", type=int, default=1, help="per-layer floor")
    parser.add_argument("--max-slots", type=int, help="per-layer cap")
    parser.add_argument(
        "--objective",
        choices=("counts-per-byte", "counts"),
        default="counts-per-byte",
    )
    costs = parser.add_mutually_exclusive_group()
    costs.add_argument(
        "--layer-costs",
        type=Path,
        help="optional complete CSV with header layer,cost_multiplier",
    )
    costs.add_argument(
        "--layer-stats-json",
        type=Path,
        help="LLAMA_EXPERT_TIMING stats; uses cpu_cold_ms/cold_hits per layer",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (GGUFError, ProfileError, OSError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
