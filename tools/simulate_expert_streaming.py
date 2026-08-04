#!/usr/bin/env python3
"""Replay lookahead traces under measured expert transport constraints."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from convert_luce_spark_profile import ProfileError


@dataclass(frozen=True)
class Timing:
    full_cpu_ms: float
    gate_up_cpu_ms: float
    down_cpu_ms: float


@dataclass
class Totals:
    actual_cold: int = 0
    ready_hits: int = 0
    late_hits: int = 0
    copies: int = 0
    useful_copies: int = 0
    wasted_copies: int = 0
    copied_bytes: int = 0
    useful_bytes: int = 0
    copy_ms: float = 0.0
    exposed_copy_ms: float = 0.0
    cpu_saving_ms: float = 0.0
    gpu_compute_ms: float = 0.0


class Arena:
    def __init__(self, kind: str, slots: int):
        self.kind = kind
        self.slots = slots
        self.global_entries: collections.OrderedDict[tuple[int, int], None] = (
            collections.OrderedDict()
        )
        self.layer_entries: dict[
            int, collections.OrderedDict[tuple[int, int], None]
        ] = {}

    def _entries(self, layer: int) -> collections.OrderedDict[tuple[int, int], None]:
        if self.kind == "global":
            return self.global_entries
        return self.layer_entries.setdefault(layer, collections.OrderedDict())

    def contains(self, layer: int, expert: int) -> bool:
        entries = self._entries(layer)
        key = (layer, expert)
        if key not in entries:
            return False
        entries.move_to_end(key)
        return True

    def insert(self, layer: int, expert: int) -> None:
        entries = self._entries(layer)
        key = (layer, expert)
        if key in entries:
            entries.move_to_end(key)
            return
        while len(entries) >= self.slots:
            entries.popitem(last=False)
        entries[key] = None


def comma_ints(text: str, name: str, minimum: int) -> list[int]:
    try:
        values = [int(item) for item in text.split(",")]
    except ValueError as error:
        raise ProfileError(f"invalid {name}: {text!r}") from error
    if not values or any(value < minimum for value in values):
        raise ProfileError(f"invalid {name}: {text!r}")
    return values


def read_traces(paths: list[Path]) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    reference: dict[str, object] | None = None
    request_offset = 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "llama-wackmall-router-lookahead-v1":
            raise ProfileError(f"unexpected trace schema in {path}")
        config = data.get("config")
        if not isinstance(config, dict):
            raise ProfileError(f"missing trace config in {path}")
        key = {
            "model": data.get("model"),
            "n_layer": config.get("n_layer"),
            "n_expert": config.get("n_expert"),
            "actual_top_k": config.get("actual_top_k"),
            "distance": config.get("distance"),
            "point": config.get("point"),
            "norm": config.get("norm"),
        }
        if reference is None:
            reference = key
        elif key != reference:
            raise ProfileError(f"trace {path} is incompatible with earlier inputs")
        raw_records = data.get("records")
        if not isinstance(raw_records, list):
            raise ProfileError(f"trace {path} has no records")
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise ProfileError(f"trace {path} has an invalid record")
            record = dict(raw)
            record["request_index"] = request_offset
            records.append(record)
        request_offset += 1
    if reference is None or not records:
        raise ProfileError("no trace records were loaded")
    records.sort(
        key=lambda item: (
            item["request_index"],
            item["token_index"],
            item["target_layer"],
        )
    )
    return records, reference


def read_cpu_timing(path: Path, n_layer: int) -> dict[int, Timing]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("timing_enabled") is not True:
        raise ProfileError("CPU stats JSON does not contain enabled timing")
    result: dict[int, Timing] = {}
    for item in data.get("layers", []):
        layer = item.get("layer")
        cold = item.get("cold_hits")
        if not isinstance(layer, int) or not isinstance(cold, int) or cold <= 0:
            continue
        try:
            result[layer] = Timing(
                full_cpu_ms=float(item["cpu_cold_ms"]) / cold,
                gate_up_cpu_ms=float(item["cpu_gate_up_ms"]) / cold,
                down_cpu_ms=float(item["cpu_down_ms"]) / cold,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProfileError(f"invalid CPU timing for layer {layer}") from error
    missing = sorted(set(range(n_layer)) - set(result))
    if missing:
        raise ProfileError(f"CPU stats are missing measured layers {missing}")
    return result


def read_layout(path: Path, n_layer: int) -> dict[int, dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "llama-wackmall-expert-layout-v1":
        raise ProfileError("unexpected model layout schema")
    result: dict[int, dict[str, int]] = {}
    for item in data.get("expert_tensors", []):
        layer = item.get("layer")
        parts = item.get("parts")
        if not isinstance(layer, int) or not isinstance(parts, dict):
            raise ProfileError("invalid expert layout entry")
        converted = {name: int(value) for name, value in parts.items()}
        converted["gate_up"] = int(item["gate_up_bytes"])
        converted["full"] = int(item["full_bytes"])
        result[layer] = converted
    if set(result) != set(range(n_layer)):
        raise ProfileError("model layout does not cover every base layer")
    return result


def read_transport(path: Path) -> dict[tuple[int, str], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "llama-wackmall-expert-transport-summary-v1":
        raise ProfileError("unexpected transport summary schema")
    result: dict[tuple[int, str], float] = {}
    for item in data.get("results", []):
        result[(int(item["bytes"]), str(item["mode"]))] = float(
            item["device_ms_median_of_medians"]
        )
    return result


def read_gpu_compute(paths: list[Path], timing_kind: str) -> dict[tuple[int, str], float]:
    collected: dict[tuple[int, str], list[float]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        schema = data.get("schema")
        if schema == "llama-wackmall-expert-compute-summary-v1":
            for item in data.get("by_size_mode", []):
                key = (int(item["weight_bytes"]), str(item["mode"]))
                collected.setdefault(key, []).append(float(item[f"{timing_kind}_ms_median"]))
            continue
        if schema != "llama-wackmall-expert-compute-v1":
            raise ProfileError(f"unexpected expert compute schema in {path}")
        for item in data.get("results", []):
            key = (int(item["weight_bytes"]), str(item["mode"]))
            collected.setdefault(key, []).append(float(item[f"{timing_kind}_ms"]["median"]))
    return {key: statistics.median(values) for key, values in collected.items()}


def rank_precision(
    records: list[dict[str, object]], n_layer: int, top_m: int
) -> dict[int, list[float]]:
    hits = [[0] * top_m for _ in range(n_layer)]
    total = [[0] * top_m for _ in range(n_layer)]
    for record in records:
        layer = int(record["target_layer"])
        actual = set(int(value) for value in record["actual"])
        predicted = record["predicted"][:top_m]
        fixed = record["predicted_fixed"][:top_m]
        for rank, (expert, is_fixed) in enumerate(zip(predicted, fixed)):
            if is_fixed:
                continue
            total[layer][rank] += 1
            if int(expert) in actual:
                hits[layer][rank] += 1
    return {
        layer: [
            hits[layer][rank] / total[layer][rank] if total[layer][rank] else 0.0
            for rank in range(top_m)
        ]
        for layer in range(n_layer)
    }


def phase_values(mode: str, timing: Timing, parts: dict[str, int]) -> tuple[float, int]:
    if mode == "full":
        return timing.full_cpu_ms, parts["full"]
    if mode == "gate-up":
        return timing.gate_up_cpu_ms, parts["gate_up"]
    if mode == "down":
        if "down" not in parts:
            raise ProfileError("down-only mode requires a separate down tensor")
        return timing.down_cpu_ms, parts["down"]
    raise ProfileError(f"unsupported streaming mode {mode}")


def simulate(
    records: list[dict[str, object]],
    n_layer: int,
    timing: dict[int, Timing],
    layout: dict[int, dict[str, int]],
    transport: dict[tuple[int, str], float],
    gpu_compute: dict[tuple[int, str], float],
    precision: dict[int, list[float]],
    *,
    policy: str,
    mode: str,
    arena_kind: str,
    arena_slots: int,
    top_m: int,
    max_copies_layer: int,
    max_bytes_token: int,
    lead_ms: float | None,
) -> dict[str, object]:
    arena = Arena(arena_kind, arena_slots)
    totals = Totals()
    current_token: tuple[int, int] | None = None
    bytes_this_token = 0
    for record in records:
        token = (int(record["request_index"]), int(record["token_index"]))
        if token != current_token:
            current_token = token
            bytes_this_token = 0
        layer = int(record["target_layer"])
        actual = {int(value) for value in record["actual"]}
        actual_fixed = {
            int(expert)
            for expert, is_fixed in zip(record["actual"], record["actual_fixed"])
            if is_fixed
        }
        cold = actual - actual_fixed
        totals.actual_cold += len(cold)
        cpu_ms, bytes_per_expert = phase_values(mode, timing[layer], layout[layer])
        copy_ms = transport.get((bytes_per_expert, "pinned_h2d"))
        if copy_ms is None:
            raise ProfileError(
                f"transport summary lacks pinned_h2d size {bytes_per_expert} for layer {layer}"
            )
        gpu_ms = gpu_compute.get((bytes_per_expert, mode))
        if gpu_compute and gpu_ms is None:
            raise ProfileError(
                f"GPU compute measurements lack mode {mode} size {bytes_per_expert} for layer {layer}"
            )
        gpu_ms = gpu_ms or 0.0
        net_compute_benefit = max(0.0, cpu_ms - gpu_ms)

        ready_before = {expert for expert in cold if arena.contains(layer, expert)}
        candidates: list[tuple[float, int]] = []
        if policy == "oracle":
            candidates = [
                (net_compute_benefit / bytes_per_expert, expert) for expert in sorted(cold)
            ]
        else:
            seen: set[int] = set()
            for rank, (raw_expert, is_fixed) in enumerate(
                zip(record["predicted"][:top_m], record["predicted_fixed"][:top_m])
            ):
                expert = int(raw_expert)
                if is_fixed or expert in seen:
                    continue
                seen.add(expert)
                probability = precision[layer][rank]
                candidates.append(
                    (probability * net_compute_benefit / bytes_per_expert, expert)
                )
            candidates.sort(key=lambda item: (-item[0], item[1]))

        selected: list[int] = []
        for utility, expert in candidates:
            if len(selected) >= max_copies_layer:
                break
            if utility <= 0 or arena.contains(layer, expert):
                continue
            if bytes_this_token + bytes_per_expert > max_bytes_token:
                continue
            selected.append(expert)
            bytes_this_token += bytes_per_expert

        ready_now: set[int] = set()
        late_now: set[int] = set()
        for expert in selected:
            useful = expert in actual
            totals.copies += 1
            totals.copied_bytes += bytes_per_expert
            totals.copy_ms += copy_ms
            if useful:
                totals.useful_copies += 1
                totals.useful_bytes += bytes_per_expert
            else:
                totals.wasted_copies += 1
            if lead_ms is None or copy_ms <= lead_ms:
                arena.insert(layer, expert)
                if useful:
                    ready_now.add(expert)
            else:
                late_now.add(expert)
            if lead_ms is not None:
                totals.exposed_copy_ms += max(0.0, copy_ms - lead_ms)

        ready = ready_before | ready_now
        totals.ready_hits += len(ready)
        totals.late_hits += len(cold & late_now)
        totals.cpu_saving_ms += len(ready) * cpu_ms
        totals.gpu_compute_ms += len(ready) * gpu_ms
        for expert in selected:
            if expert in late_now:
                arena.insert(layer, expert)

    max_part = max(phase_values(mode, timing[layer], layout[layer])[1] for layer in range(n_layer))
    if arena_kind == "global":
        scratch_bytes = arena_slots * max_part
    else:
        scratch_bytes = arena_slots * sum(
            phase_values(mode, timing[layer], layout[layer])[1] for layer in range(n_layer)
        )
    useful_ratio = totals.useful_copies / totals.copies if totals.copies else 0.0
    useful_bytes_ratio = totals.useful_bytes / totals.copied_bytes if totals.copied_bytes else 0.0
    ready_recall = totals.ready_hits / totals.actual_cold if totals.actual_cold else 0.0
    return {
        "policy": policy,
        "mode": mode,
        "arena": arena_kind,
        "arena_slots": arena_slots,
        "top_m": top_m,
        "max_copies_per_layer": max_copies_layer,
        "max_bytes_per_token": max_bytes_token,
        "lead_ms": lead_ms,
        "deadline_model": "optimistic-ready" if lead_ms is None else "constant-explicit",
        "actual_cold": totals.actual_cold,
        "ready_hits": totals.ready_hits,
        "late_hits": totals.late_hits,
        "ready_recall": ready_recall,
        "copies": totals.copies,
        "useful_copies": totals.useful_copies,
        "wasted_copies": totals.wasted_copies,
        "useful_prefetch_ratio": useful_ratio,
        "copied_bytes": totals.copied_bytes,
        "useful_bytes": totals.useful_bytes,
        "useful_bytes_ratio": useful_bytes_ratio,
        "copy_ms": totals.copy_ms,
        "exposed_copy_ms": totals.exposed_copy_ms,
        "theoretical_cpu_saving_ms": totals.cpu_saving_ms,
        "gpu_compute_ms": totals.gpu_compute_ms,
        "theoretical_net_saving_ms": (
            totals.cpu_saving_ms - totals.gpu_compute_ms - totals.exposed_copy_ms
        ),
        "scratch_bytes": scratch_bytes,
    }


def zero_copy_bounds(
    records: list[dict[str, object]],
    timing: dict[int, Timing],
    layout: dict[int, dict[str, int]],
    transport: dict[tuple[int, str], float],
    mode: str,
) -> dict[str, object]:
    actual_cold = 0
    cpu_ms = 0.0
    mapped_ms = 0.0
    for record in records:
        layer = int(record["target_layer"])
        cold_count = sum(not fixed for fixed in record["actual_fixed"])
        phase_cpu, bytes_per_expert = phase_values(mode, timing[layer], layout[layer])
        phase_mapped = transport.get((bytes_per_expert, "mapped_read"))
        if phase_mapped is None:
            raise ProfileError(f"transport summary lacks mapped_read size {bytes_per_expert}")
        actual_cold += cold_count
        cpu_ms += cold_count * phase_cpu
        mapped_ms += cold_count * phase_mapped
    return {
        "mode": mode,
        "actual_cold": actual_cold,
        "cpu_phase_ms": cpu_ms,
        "mapped_sequential_read_ms": mapped_ms,
        "optimistic_net_ms_before_gpu_compute": cpu_ms - mapped_ms,
        "caveat": "Mapped read is a sequential checksum lower bound, not quantized expert compute.",
    }


def run(args: argparse.Namespace) -> None:
    trace_paths = [path.resolve() for path in args.trace]
    records, trace_config = read_traces(trace_paths)
    calibration_paths = [path.resolve() for path in args.calibration_trace]
    if calibration_paths:
        calibration_records, calibration_config = read_traces(calibration_paths)
        if calibration_config != trace_config:
            raise ProfileError("calibration traces are incompatible with replay traces")
        calibration_scope = "held-out"
    else:
        calibration_records = records
        calibration_scope = "in-sample-optimistic"
    n_layer = int(trace_config["n_layer"])
    timing = read_cpu_timing(args.stats_json.resolve(), n_layer)
    layout = read_layout(args.layout_json.resolve(), n_layer)
    transport = read_transport(args.transport_json.resolve())
    gpu_compute = read_gpu_compute(
        [path.resolve() for path in args.compute_json], args.compute_timing
    )
    top_values = comma_ints(args.top_m, "top-m", 1)
    slot_values = comma_ints(args.arena_slots, "arena-slots", 1)
    copy_values = comma_ints(args.max_copies_per_layer, "max-copies-per-layer", 1)
    byte_values = comma_ints(args.max_mib_per_token, "max-mib-per-token", 1)
    max_top = max(top_values)
    if any(len(record["predicted"]) < max_top for record in records):
        raise ProfileError("a trace does not contain enough predictor ranks")
    precision = rank_precision(calibration_records, n_layer, max_top)

    simulations: list[dict[str, object]] = []
    for policy in ("predictor", "oracle"):
        for mode in args.mode:
            for arena_kind in args.arena:
                for slots in slot_values:
                    for top_m in top_values:
                        for max_copies in copy_values:
                            for max_mib in byte_values:
                                simulations.append(
                                    simulate(
                                        records,
                                        n_layer,
                                        timing,
                                        layout,
                                        transport,
                                        gpu_compute,
                                        precision,
                                        policy=policy,
                                        mode=mode,
                                        arena_kind=arena_kind,
                                        arena_slots=slots,
                                        top_m=top_m,
                                        max_copies_layer=max_copies,
                                        max_bytes_token=max_mib * 1024 * 1024,
                                        lead_ms=args.lead_ms,
                                    )
                                )
    output = {
        "schema": "llama-wackmall-expert-streaming-simulation-v1",
        "scope": (
            "upper bound; resident GPU expert compute is included, predictor runtime is excluded"
            if gpu_compute
            else "upper bound; GPU expert compute and predictor runtime are excluded"
        ),
        "trace_config": trace_config,
        "trace_files": [str(path) for path in trace_paths],
        "calibration_trace_files": [str(path) for path in calibration_paths],
        "predictor_calibration": calibration_scope,
        "record_count": len(records),
        "request_count": len({record["request_index"] for record in records}),
        "residency_source": "fixed booleans embedded in each trace record",
        "transport_source": str(args.transport_json.resolve()),
        "gpu_compute_sources": [str(path.resolve()) for path in args.compute_json],
        "gpu_compute_timing": args.compute_timing,
        "cpu_timing_source": str(args.stats_json.resolve()),
        "zero_copy_lower_bounds": [
            zero_copy_bounds(records, timing, layout, transport, mode)
            for mode in args.mode
        ],
        "simulations": simulations,
    }
    output_path = args.output.resolve(strict=False)
    if output_path.exists():
        raise ProfileError(f"refusing to overwrite existing output: {output_path}")
    if not output_path.parent.is_dir():
        raise ProfileError(f"output parent does not exist: {output_path.parent}")
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    best = sorted(
        simulations,
        key=lambda item: item["theoretical_net_saving_ms"],
        reverse=True,
    )[:5]
    print(f"records={len(records)} simulations={len(simulations)}")
    for item in best:
        print(
            f"{item['policy']} {item['mode']} {item['arena']}:{item['arena_slots']} "
            f"top={item['top_m']} copies/layer={item['max_copies_per_layer']} "
            f"budget={item['max_bytes_per_token']} ready={item['ready_recall']:.4f} "
            f"useful_bytes={item['useful_bytes_ratio']:.4f} "
            f"net_upper_ms={item['theoretical_net_saving_ms']:.3f}"
        )
    print(f"output={output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument(
        "--calibration-trace",
        type=Path,
        action="append",
        default=[],
        help="held-out compatible trace used to calibrate rank precision",
    )
    parser.add_argument("--transport-json", type=Path, required=True)
    parser.add_argument("--layout-json", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--compute-json", type=Path, action="append", default=[])
    parser.add_argument("--compute-timing", choices=("latency", "queued"), default="latency")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-m", default="8,12,16")
    parser.add_argument("--arena", action="append", choices=("global", "per-layer"), default=[])
    parser.add_argument("--arena-slots", default="1,2,4")
    parser.add_argument("--max-copies-per-layer", default="1,2")
    parser.add_argument("--max-mib-per-token", default="8,16,32,64")
    parser.add_argument("--mode", action="append", choices=("full", "gate-up", "down"), default=[])
    parser.add_argument(
        "--lead-ms",
        type=float,
        help="explicit measured constant lead; omitted means an optimistic always-ready upper bound",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.lead_ms is not None and args.lead_ms < 0:
            raise ProfileError("lead-ms must be non-negative")
        if not args.arena:
            args.arena = ["global", "per-layer"]
        if not args.mode:
            args.mode = ["full", "gate-up", "down"]
        run(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, ProfileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
