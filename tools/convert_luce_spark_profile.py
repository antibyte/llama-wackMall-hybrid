#!/usr/bin/env python3
"""Convert a LuceBox Spark hotness table to a wackMall expert heat CSV."""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


UINT64_MAX = (1 << 64) - 1
INT64_MAX = (1 << 63) - 1

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: "<B",
    GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_BOOL: "<B",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}

SPARK_HEADER_RE = re.compile(
    r"# hotness table: n_layer=([0-9]+) n_expert=([0-9]+) n_expert_used=([0-9]+)"
)
SPARK_FORMAT_LINE = (
    "# format: one row per layer, columns are expert activation counts "
    "(expert 0..N-1)"
)
UNSIGNED_RE = re.compile(r"0|[1-9][0-9]*")


class ProfileError(ValueError):
    pass


class GGUFError(ValueError):
    pass


@dataclass(frozen=True)
class ProfileDimensions:
    n_layer: int
    n_expert: int
    n_expert_used: int


@dataclass(frozen=True)
class SparkProfile:
    dimensions: ProfileDimensions
    counts: list[list[int]]


@dataclass(frozen=True)
class ModelMetadata:
    path: Path
    file_size: int
    gguf_version: int
    tensor_count: int
    metadata_count: int
    architecture: str
    name: str | None
    block_count_total: int
    nextn_predict_layers: int
    dimensions: ProfileDimensions
    file_type: int | None
    quantization_version: int | None


class _GGUFReader:
    def __init__(self, stream: BinaryIO, file_size: int):
        self.stream = stream
        self.file_size = file_size

    def read_exact(self, size: int) -> bytes:
        if size < 0 or self.stream.tell() + size > self.file_size:
            raise GGUFError(
                f"truncated GGUF at offset {self.stream.tell()} while reading {size} bytes"
            )
        data = self.stream.read(size)
        if len(data) != size:
            raise GGUFError(
                f"truncated GGUF at offset {self.stream.tell() - len(data)}"
            )
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read_exact(struct.calcsize(fmt)))[0]

    def read_string(self, *, decode: bool = True) -> str | None:
        size = self.unpack("<Q")
        if size > self.file_size - self.stream.tell():
            raise GGUFError(
                f"invalid GGUF string length {size} at offset {self.stream.tell()}"
            )
        if not decode:
            self.stream.seek(size, os.SEEK_CUR)
            return None
        raw = self.read_exact(size)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError(f"invalid UTF-8 GGUF string: {exc}") from exc

    def skip(self, size: int) -> None:
        if size < 0 or size > self.file_size - self.stream.tell():
            raise GGUFError(
                f"invalid GGUF skip length {size} at offset {self.stream.tell()}"
            )
        self.stream.seek(size, os.SEEK_CUR)

    def read_scalar(self, value_type: int):
        if value_type == GGUF_TYPE_STRING:
            return self.read_string()
        fmt = SCALAR_FORMATS.get(value_type)
        if fmt is None:
            raise GGUFError(f"metadata type {value_type} is not scalar")
        value = self.unpack(fmt)
        if value_type == GGUF_TYPE_BOOL:
            if value not in (0, 1):
                raise GGUFError(f"invalid GGUF bool value {value}")
            return bool(value)
        return value

    def skip_value(self, value_type: int) -> None:
        if value_type == GGUF_TYPE_STRING:
            self.read_string(decode=False)
            return
        if value_type == GGUF_TYPE_ARRAY:
            element_type = self.unpack("<I")
            count = self.unpack("<Q")
            if element_type == GGUF_TYPE_ARRAY:
                raise GGUFError("nested GGUF metadata arrays are invalid")
            if element_type == GGUF_TYPE_STRING:
                for _ in range(count):
                    self.read_string(decode=False)
                return
            fmt = SCALAR_FORMATS.get(element_type)
            if fmt is None:
                raise GGUFError(f"unknown GGUF array element type {element_type}")
            element_size = struct.calcsize(fmt)
            if count > (self.file_size - self.stream.tell()) // element_size:
                raise GGUFError(
                    f"invalid GGUF array length {count} at offset {self.stream.tell()}"
                )
            self.skip(count * element_size)
            return
        fmt = SCALAR_FORMATS.get(value_type)
        if fmt is None:
            raise GGUFError(f"unknown GGUF metadata type {value_type}")
        self.skip(struct.calcsize(fmt))


def _parse_unsigned(value: str, where: str, *, maximum: int = UINT64_MAX) -> int:
    if not UNSIGNED_RE.fullmatch(value):
        raise ProfileError(f"{where}: expected an unsigned decimal integer, got {value!r}")
    parsed = int(value)
    if parsed > maximum:
        raise ProfileError(f"{where}: value {parsed} exceeds supported maximum {maximum}")
    return parsed


def parse_spark_profile(path: Path) -> SparkProfile:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ProfileError(f"cannot read Spark profile {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ProfileError(f"Spark profile is not valid UTF-8: {exc}") from exc

    nonempty = [(number, line) for number, line in enumerate(lines, 1) if line]
    if len(nonempty) < 2:
        raise ProfileError("Spark profile is missing its two required header lines")

    header_number, header = nonempty[0]
    match = SPARK_HEADER_RE.fullmatch(header)
    if not match:
        raise ProfileError(
            f"line {header_number}: invalid Spark dimension header; expected "
            "'# hotness table: n_layer=<N> n_expert=<N> n_expert_used=<N>'"
        )

    dims = ProfileDimensions(*(int(value) for value in match.groups()))
    if dims.n_layer <= 0 or dims.n_expert <= 0 or dims.n_expert_used <= 0:
        raise ProfileError("Spark dimensions must all be greater than zero")
    if dims.n_expert_used > dims.n_expert:
        raise ProfileError(
            f"Spark n_expert_used={dims.n_expert_used} exceeds n_expert={dims.n_expert}"
        )

    format_number, format_line = nonempty[1]
    if format_line != SPARK_FORMAT_LINE:
        raise ProfileError(
            f"line {format_number}: invalid Spark format declaration; no column-order "
            "assumption will be made"
        )

    data: list[tuple[int, str]] = []
    for line_number, line in nonempty[2:]:
        if line.startswith("#"):
            continue
        data.append((line_number, line))

    if len(data) != dims.n_layer:
        raise ProfileError(
            f"Spark row count {len(data)} does not match n_layer={dims.n_layer}"
        )

    counts: list[list[int]] = []
    for layer, (line_number, line) in enumerate(data):
        fields = line.split(",")
        if len(fields) != dims.n_expert:
            raise ProfileError(
                f"line {line_number} (layer {layer}): found {len(fields)} expert columns, "
                f"expected {dims.n_expert}"
            )
        row = [
            _parse_unsigned(
                field,
                f"line {line_number}, expert {expert}",
                maximum=INT64_MAX,
            )
            for expert, field in enumerate(fields)
        ]
        counts.append(row)

    return SparkProfile(dims, counts)


def read_gguf_metadata(path: Path) -> ModelMetadata:
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
            raise GGUFError(
                f"unsupported GGUF version {version}; only metadata layouts 2 and 3 are supported"
            )
        tensor_count = reader.unpack("<Q")
        metadata_count = reader.unpack("<Q")
        if metadata_count > 1_000_000:
            raise GGUFError(f"implausible GGUF metadata count {metadata_count}")

        values: dict[str, object] = {}
        wanted_general = {
            "general.architecture",
            "general.name",
            "general.file_type",
            "general.quantization_version",
        }
        wanted_suffixes = (
            ".block_count",
            ".expert_count",
            ".expert_used_count",
            ".nextn_predict_layers",
        )

        for _ in range(metadata_count):
            key = reader.read_string()
            assert key is not None
            value_type = reader.unpack("<I")
            wanted = key in wanted_general or key.endswith(wanted_suffixes)
            if wanted:
                if value_type == GGUF_TYPE_ARRAY:
                    raise GGUFError(f"required metadata {key!r} unexpectedly has array type")
                if key in values:
                    raise GGUFError(f"duplicate GGUF metadata key {key!r}")
                values[key] = reader.read_scalar(value_type)
            else:
                reader.skip_value(value_type)

    architecture = values.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise GGUFError("GGUF is missing string metadata 'general.architecture'")

    def require_positive_int(key: str) -> int:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GGUFError(f"GGUF is missing positive integer metadata {key!r}")
        return value

    block_count_total = require_positive_int(f"{architecture}.block_count")
    nextn_key = f"{architecture}.nextn_predict_layers"
    nextn_value = values.get(nextn_key, 0)
    if isinstance(nextn_value, bool) or not isinstance(nextn_value, int) or nextn_value < 0:
        raise GGUFError(f"GGUF metadata {nextn_key!r} is not a non-negative integer")
    if nextn_value >= block_count_total:
        raise GGUFError(
            f"GGUF nextn_predict_layers={nextn_value} must be smaller than "
            f"block_count={block_count_total}"
        )

    dims = ProfileDimensions(
        block_count_total - nextn_value,
        require_positive_int(f"{architecture}.expert_count"),
        require_positive_int(f"{architecture}.expert_used_count"),
    )
    if dims.n_expert_used > dims.n_expert:
        raise GGUFError(
            f"GGUF expert_used_count={dims.n_expert_used} exceeds expert_count={dims.n_expert}"
        )

    name = values.get("general.name")
    if name is not None and not isinstance(name, str):
        raise GGUFError("GGUF metadata 'general.name' is not a string")

    def optional_int(key: str) -> int | None:
        value = values.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise GGUFError(f"GGUF metadata {key!r} is not an integer")
        return value

    return ModelMetadata(
        path=path,
        file_size=file_size,
        gguf_version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        architecture=architecture,
        name=name,
        block_count_total=block_count_total,
        nextn_predict_layers=nextn_value,
        dimensions=dims,
        file_type=optional_int("general.file_type"),
        quantization_version=optional_int("general.quantization_version"),
    )


def validate_dimensions(
    profile: SparkProfile,
    model: ModelMetadata,
    *,
    label: str = "target model",
) -> None:
    if profile.dimensions != model.dimensions:
        raise ProfileError(
            f"Spark dimensions {profile.dimensions} do not match {label} GGUF dimensions "
            f"{model.dimensions} (architecture={model.architecture!r})"
        )


def validate_models_compatible(source: ModelMetadata, target: ModelMetadata) -> None:
    if source.architecture != target.architecture:
        raise ProfileError(
            f"source architecture {source.architecture!r} does not match target "
            f"architecture {target.architecture!r}"
        )
    if source.dimensions != target.dimensions:
        raise ProfileError(
            f"source GGUF dimensions {source.dimensions} do not match target dimensions "
            f"{target.dimensions}"
        )
    comparable = (
        ("file type", source.file_type, target.file_type),
        (
            "quantization version",
            source.quantization_version,
            target.quantization_version,
        ),
    )
    for label, source_value, target_value in comparable:
        if (
            source_value is not None
            and target_value is not None
            and source_value != target_value
        ):
            raise ProfileError(
                f"source GGUF {label} {source_value} does not match target "
                f"{label} {target_value}"
            )


def make_placement_seed(profile: SparkProfile, placement_slots: int) -> SparkProfile:
    """Keep only top-S placement order and bound online-adaptation seed scores."""
    if placement_slots <= 0 or placement_slots > profile.dimensions.n_expert:
        raise ProfileError(
            f"placement slots must be in [1, {profile.dimensions.n_expert}], "
            f"got {placement_slots}"
        )
    output: list[list[int]] = []
    for row in profile.counts:
        ranked = sorted(range(len(row)), key=lambda expert: (-row[expert], expert))
        scores = [0] * len(row)
        for rank, expert in enumerate(ranked[:placement_slots]):
            scores[expert] = placement_slots - rank
        output.append(scores)
    return SparkProfile(profile.dimensions, output)


def validate_wack_profile(
    path: Path,
    dimensions: ProfileDimensions,
    *,
    expected: SparkProfile | None = None,
    require_complete: bool = False,
) -> dict[tuple[int, int], int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ProfileError(f"cannot validate wackMall profile {path}: {exc}") from exc

    if not lines or lines[0] != "layer,expert,count":
        raise ProfileError("wackMall output is missing exact header 'layer,expert,count'")

    values: dict[tuple[int, int], int] = {}
    for line_number, line in enumerate(lines[1:], 2):
        if not line:
            raise ProfileError(f"wackMall output line {line_number} is empty")
        fields = line.split(",")
        if len(fields) != 3:
            raise ProfileError(
                f"wackMall output line {line_number} has {len(fields)} fields, expected 3"
            )
        layer = _parse_unsigned(fields[0], f"wackMall line {line_number} layer")
        expert = _parse_unsigned(fields[1], f"wackMall line {line_number} expert")
        count = _parse_unsigned(
            fields[2], f"wackMall line {line_number} count", maximum=INT64_MAX
        )
        if layer >= dimensions.n_layer:
            raise ProfileError(
                f"wackMall line {line_number}: layer {layer} is outside [0, {dimensions.n_layer})"
            )
        if expert >= dimensions.n_expert:
            raise ProfileError(
                f"wackMall line {line_number}: expert {expert} is outside [0, {dimensions.n_expert})"
            )
        key = (layer, expert)
        if key in values:
            raise ProfileError(
                f"wackMall line {line_number}: duplicate layer/expert pair {key}"
            )
        values[key] = count

    full_count = dimensions.n_layer * dimensions.n_expert
    if require_complete and len(values) != full_count:
        raise ProfileError(
            f"wackMall output contains {len(values)} rows, expected complete table of {full_count}"
        )

    if expected is not None:
        for layer, row in enumerate(expected.counts):
            for expert, count in enumerate(row):
                output_count = values.get((layer, expert), 0)
                if output_count != count:
                    raise ProfileError(
                        f"wackMall output changed count at layer {layer}, expert {expert}: "
                        f"{output_count} != {count}"
                    )
    return values


def write_wack_profile(
    profile: SparkProfile,
    output: Path,
    *,
    drop_zero: bool = False,
) -> None:
    output_parent = output.parent
    if not output_parent.is_dir():
        raise ProfileError(f"output directory does not exist: {output_parent}")
    if os.path.lexists(output):
        raise ProfileError(f"refusing to overwrite existing output: {output}")

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output.name}.tmp-",
            dir=output_parent,
            delete=False,
        ) as temp:
            temp_name = temp.name
            temp.write("layer,expert,count\n")
            for layer, row in enumerate(profile.counts):
                for expert, count in enumerate(row):
                    if drop_zero and count == 0:
                        continue
                    temp.write(f"{layer},{expert},{count}\n")
            temp.flush()
            os.fsync(temp.fileno())

        temp_path = Path(temp_name)
        validate_wack_profile(
            temp_path,
            profile.dimensions,
            expected=profile,
            require_complete=not drop_zero,
        )

        try:
            os.link(temp_path, output)
        except FileExistsError as exc:
            raise ProfileError(f"refusing to overwrite output created concurrently: {output}") from exc
        except OSError as exc:
            raise ProfileError(f"cannot publish output {output} without overwriting: {exc}") from exc
        temp_path.unlink()
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


def _format_metadata(model: ModelMetadata) -> str:
    dims = model.dimensions
    return (
        f"path={model.path} size={model.file_size} version={model.gguf_version} "
        f"tensors={model.tensor_count} metadata={model.metadata_count} "
        f"architecture={model.architecture} name={model.name!r} "
        f"blocks_total={model.block_count_total} nextn={model.nextn_predict_layers} "
        f"base_layers={dims.n_layer} experts={dims.n_expert} used={dims.n_expert_used} "
        f"file_type={model.file_type} quantization_version={model.quantization_version}"
    )


def convert(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    output_path = args.output.resolve(strict=False)
    model_path = args.model.resolve()

    if input_path == output_path:
        raise ProfileError("input and output paths resolve to the same file")
    if not input_path.is_file():
        raise ProfileError(f"input profile does not exist: {input_path}")
    if not model_path.is_file():
        raise ProfileError(f"target model does not exist: {model_path}")

    profile = parse_spark_profile(input_path)
    target = read_gguf_metadata(model_path)
    validate_dimensions(profile, target)

    source: ModelMetadata | None = None
    if args.source_model is not None:
        source_path = args.source_model.resolve()
        if not source_path.is_file():
            raise ProfileError(f"source model does not exist: {source_path}")
        source = read_gguf_metadata(source_path)
        validate_dimensions(profile, source, label="source model")
        validate_models_compatible(source, target)

    output_profile = profile
    if args.placement_slots is not None:
        output_profile = make_placement_seed(profile, args.placement_slots)

    write_wack_profile(output_profile, output_path, drop_zero=args.drop_zero)

    flat = [count for row in profile.counts for count in row]
    output_flat = [count for row in output_profile.counts for count in row]
    layer_totals = [sum(row) for row in profile.counts]
    nonzero = sum(count != 0 for count in flat)
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"target-model: {_format_metadata(target)}")
    if source is not None:
        print(f"source-model: {_format_metadata(source)}")
    else:
        print(
            "source-model: not supplied; Spark CSV contains dimensions but no model "
            "name, fingerprint, quantization, or tensor hashes"
        )
    print(
        f"profile: layers={profile.dimensions.n_layer} "
        f"experts={profile.dimensions.n_expert} used={profile.dimensions.n_expert_used}"
    )
    print(
        f"counts: total={sum(flat)} nonzero={nonzero} zero={len(flat) - nonzero} "
        f"min={min(flat)} max={max(flat)}"
    )
    if args.placement_slots is None:
        print("score-mode: raw-counts (placement and online-adaptation seed)")
    else:
        print(
            f"score-mode: placement-only top-{args.placement_slots}; output scores are "
            f"bounded ranks {args.placement_slots}..1 and all other scores are zero"
        )
    print(
        f"layer-totals: min={min(layer_totals)} max={max(layer_totals)} "
        f"sum={sum(layer_totals)}"
    )
    output_nonzero = sum(count != 0 for count in output_flat)
    print(f"output-rows: {output_nonzero if args.drop_zero else len(output_flat)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="LuceBox Spark dense CSV")
    parser.add_argument("--output", type=Path, required=True, help="new wackMall sparse CSV")
    parser.add_argument("--model", type=Path, required=True, help="target wackMall GGUF")
    parser.add_argument(
        "--source-model",
        type=Path,
        help="optional GGUF that produced the Spark profile for additional metadata checks",
    )
    parser.add_argument(
        "--drop-zero",
        action="store_true",
        help="omit zero-count rows; ordering and dimensions are still validated",
    )
    parser.add_argument(
        "--placement-slots",
        type=int,
        metavar="S",
        help=(
            "emit a placement-only top-S seed with bounded rank scores instead of "
            "raw lifetime counts; use the same S at runtime"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        convert(args)
    except (ProfileError, GGUFError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
