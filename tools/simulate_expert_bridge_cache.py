#!/usr/bin/env python3
"""Simulate a small persistent LRU cache for bridge Gate+Up candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path


class CacheSimulationError(RuntimeError):
    pass


def cold_values(record: dict[str, object], values_key: str, fixed_key: str) -> list[int]:
    values = record.get(values_key, [])
    fixed = record.get(fixed_key, [])
    if not isinstance(values, list) or not isinstance(fixed, list) or len(values) != len(fixed):
        raise CacheSimulationError(f"invalid {values_key}/{fixed_key} arrays")
    return [int(value) for value, is_fixed in zip(values, fixed) if not bool(is_fixed)]


def layer_steps(
    records: list[dict[str, object]], layer: int, mode: str, k: int
) -> list[tuple[list[int], set[int]]]:
    selected = [record for record in records if int(record.get("target_layer", -1)) == layer]
    selected.sort(key=lambda record: int(record.get("token_index", -1)))
    steps = []
    previous_actual: list[int] = []
    for record in selected:
        actual = cold_values(record, "actual", "actual_fixed")
        if mode == "lookahead":
            candidates = cold_values(record, "predicted", "predicted_fixed")[:k]
        elif mode == "recurrence":
            candidates = previous_actual[:k]
        else:
            raise CacheSimulationError(f"unknown prediction mode: {mode}")
        steps.append((list(dict.fromkeys(candidates)), set(actual)))
        previous_actual = actual
    return steps


def simulate_lru(
    steps: list[tuple[list[int], set[int]]], slots: int, expert_bytes: int
) -> dict[str, object]:
    if slots < 1 or expert_bytes < 1:
        raise CacheSimulationError("cache slots and expert bytes must be positive")
    cache: OrderedDict[int, None] = OrderedDict()
    candidates = 0
    copies = 0
    cache_hits = 0
    useful_candidates = 0
    useful_cache_hits = 0
    for predicted, actual in steps:
        for expert in predicted:
            candidates += 1
            useful = expert in actual
            useful_candidates += int(useful)
            if expert in cache:
                cache_hits += 1
                useful_cache_hits += int(useful)
                cache.move_to_end(expert)
                continue
            copies += 1
            cache[expert] = None
            if len(cache) > slots:
                cache.popitem(last=False)
    avoided = candidates - copies
    return {
        "steps": len(steps),
        "slots": slots,
        "candidates": candidates,
        "copies": copies,
        "avoided_copies": avoided,
        "avoided_copy_ratio": avoided / candidates if candidates else 0.0,
        "cache_hits": cache_hits,
        "cache_hit_ratio": cache_hits / candidates if candidates else 0.0,
        "useful_candidates": useful_candidates,
        "useful_cache_hits": useful_cache_hits,
        "copied_bytes": copies * expert_bytes,
        "avoided_bytes": avoided * expert_bytes,
        "scratch_bytes": slots * expert_bytes,
    }


def simulate_trace(
    data: dict[str, object], layers: list[int], modes: list[str], k_values: list[int],
    slot_values: list[int], expert_bytes: int
) -> list[dict[str, object]]:
    if data.get("schema") != "llama-wackmall-router-lookahead-v1":
        raise CacheSimulationError("unexpected routing trace schema")
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise CacheSimulationError("routing trace has no records")
    results = []
    for layer in layers:
        for mode in modes:
            for k in k_values:
                steps = layer_steps(records, layer, mode, k)
                if not steps:
                    raise CacheSimulationError(f"trace has no records for layer {layer}")
                for slots in slot_values:
                    result = simulate_lru(steps, slots, expert_bytes)
                    result.update({"layer": layer, "mode": mode, "k": k})
                    results.append(result)
    return results


def parse_int_list(text: str, name: str, minimum: int = 1) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise CacheSimulationError(f"invalid {name}: {text!r}") from error
    if not values or any(value < minimum for value in values):
        raise CacheSimulationError(f"invalid {name}: {text!r}")
    return list(dict.fromkeys(values))


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve(strict=False)
    if output.exists():
        raise CacheSimulationError(f"refusing to overwrite output: {output}")
    if not output.parent.is_dir():
        raise CacheSimulationError(f"output parent does not exist: {output.parent}")
    traces = []
    for path in args.trace:
        data = json.loads(path.read_text(encoding="utf-8"))
        traces.append(
            {
                "path": str(path),
                "results": simulate_trace(
                    data, args.layers, args.modes, args.k_values, args.slot_values,
                    args.expert_bytes
                ),
            }
        )
    result = {
        "schema": "llama-wackmall-expert-bridge-cache-simulation-v1",
        "config": {
            "layers": args.layers,
            "modes": args.modes,
            "k_values": args.k_values,
            "slot_values": args.slot_values,
            "expert_bytes": args.expert_bytes,
            "caveat": "No-MTP token traces approximate bridge candidate order; MTP graph batching is not modeled.",
        },
        "traces": traces,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for trace in traces:
        print(trace["path"])
        for item in trace["results"]:
            print(
                f"layer={item['layer']} mode={item['mode']} k={item['k']} slots={item['slots']} "
                f"avoided={100.0*item['avoided_copy_ratio']:.2f}%"
            )
    print(f"output={output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="1,2")
    parser.add_argument("--modes", default="lookahead,recurrence")
    parser.add_argument("--k", dest="k_values", default="1,2,3")
    parser.add_argument("--slots", dest="slot_values", default="2,4,8")
    parser.add_argument("--expert-bytes", type=int, default=1179648)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.trace = [path.resolve() for path in args.trace]
        if not all(path.is_file() for path in args.trace):
            raise CacheSimulationError("every trace path must exist")
        args.layers = parse_int_list(args.layers, "layers", minimum=0)
        args.modes = list(dict.fromkeys(args.modes.split(",")))
        if not args.modes or any(mode not in {"lookahead", "recurrence"} for mode in args.modes):
            raise CacheSimulationError(f"invalid modes: {args.modes!r}")
        args.k_values = parse_int_list(args.k_values, "K values")
        args.slot_values = parse_int_list(args.slot_values, "slot values")
        run(args)
    except (CacheSimulationError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
