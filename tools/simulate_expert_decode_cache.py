#!/usr/bin/env python3
"""Replay routing traces: static S vs LRU-from-empty vs fixed S plus overflow W."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from convert_luce_spark_profile import ProfileError
from simulate_expert_streaming import read_traces


class LruCache:
    def __init__(self, slots: int):
        self.slots = slots
        self.entries: collections.OrderedDict[tuple[int, int], None] = collections.OrderedDict()

    def lookup(self, key: tuple[int, int]) -> bool:
        if key not in self.entries:
            return False
        self.entries.move_to_end(key)
        return True

    def admit(self, key: tuple[int, int]) -> None:
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        while self.slots > 0 and len(self.entries) >= self.slots:
            self.entries.popitem(last=False)
        if self.slots > 0:
            self.entries[key] = None


def unique_ints(values: list[object]) -> list[int]:
    seen: dict[int, None] = {}
    for value in values:
        seen.setdefault(int(value), None)
    return list(seen)


def admit_budget(n_miss: int, warm_slots: int, admit_fraction: float) -> int:
    if n_miss <= 0 or warm_slots <= 0 or admit_fraction <= 0:
        return 0
    frac = min(1.0, float(admit_fraction))
    q = int(frac * n_miss + 0.5)
    return min(max(0, q), n_miss, warm_slots)


def admit_from_counts(peak_count: int, tmax: int) -> bool:
    return peak_count > 0 and tmax > 0 and peak_count <= tmax


def simulate_static(records: list[dict[str, object]]) -> dict[str, object]:
    selected = 0
    hits = 0
    miss_steps: list[int] = []
    for record in records:
        actual = record["actual"]
        fixed = record["actual_fixed"]
        if len(actual) != len(fixed):
            raise ProfileError("actual and actual_fixed length mismatch")
        selected += len(actual)
        hits += sum(1 for is_fixed in fixed if is_fixed)
        missed = [int(expert) for expert, is_fixed in zip(actual, fixed) if not is_fixed]
        miss_steps.append(len(unique_ints(missed)))
    misses = selected - hits
    return {
        "policy": "static",
        "selected": selected,
        "hits": hits,
        "misses": misses,
        "miss_rate": misses / selected if selected else 0.0,
        "miss_steps": miss_steps,
    }


def simulate_lru(
    records: list[dict[str, object]],
    *,
    global_slots: int | None,
    slots_per_layer: int | None,
    n_layer: int,
    admit_fraction: float = 1.0,
    min_fill: bool = False,
) -> dict[str, object]:
    if global_slots is not None:
        caches = {None: LruCache(global_slots)}
        policy = "global-lru"
        slots = global_slots
    else:
        if slots_per_layer is None:
            raise ProfileError("per-layer LRU needs slots_per_layer")
        caches = {layer: LruCache(slots_per_layer) for layer in range(n_layer)}
        policy = "per-layer-lru"
        slots = slots_per_layer
    if admit_fraction != 1.0:
        policy = policy + "-q*"
    selected = 0
    hits = 0
    miss_steps: list[int] = []
    copy_steps: list[int] = []
    for record in records:
        layer = int(record["target_layer"])
        cache = caches[None] if None in caches else caches[layer]
        uniques = unique_ints(record["actual"])
        misses: list[int] = []
        for expert in uniques:
            key = (layer, expert)
            if cache.lookup(key):
                hits += 1
            else:
                misses.append(expert)
        q = int(round(admit_fraction * len(misses)))
        if min_fill and misses:
            q = max(1, q)
        q = min(max(0, q), len(misses))
        for expert in misses[:q]:
            cache.admit((layer, expert))
        selected += len(record["actual"])
        hits += len(record["actual"]) - len(uniques)  # repeats follow the first
        miss_steps.append(len(misses))
        copy_steps.append(q)
    misses = selected - hits
    return {
        "policy": policy,
        "slots": slots,
        "selected": selected,
        "hits": hits,
        "misses": misses,
        "miss_rate": misses / selected if selected else 0.0,
        "miss_steps": miss_steps,
        "copy_steps": copy_steps,
        "admit_fraction": admit_fraction,
        "min_fill": min_fill,
    }


def simulate_fixed_overflow(
    records: list[dict[str, object]],
    *,
    warm_slots: int,
    n_layer: int,
    global_warm: bool = False,
    admit_fraction: float = 1.0,
    min_fill: bool = False,
) -> dict[str, object]:
    """Fixed experts always hit; LRU overflow holds only cold experts."""
    if warm_slots < 0:
        raise ProfileError(f"invalid warm_slots: {warm_slots}")
    if global_warm:
        caches = {None: LruCache(warm_slots * n_layer)}
        policy = "fixed-overflow-global"
        slots = warm_slots * n_layer
    else:
        caches = {layer: LruCache(warm_slots) for layer in range(n_layer)}
        policy = "fixed-overflow"
        slots = warm_slots
    if admit_fraction != 1.0:
        policy = policy + "-q*"
    selected = 0
    hits = 0
    miss_steps: list[int] = []
    copy_steps: list[int] = []
    for record in records:
        actual = record["actual"]
        fixed = record["actual_fixed"]
        if len(actual) != len(fixed):
            raise ProfileError("actual and actual_fixed length mismatch")
        layer = int(record["target_layer"])
        cache = caches[None] if None in caches else caches[layer]
        counts: dict[int, int] = {}
        seen_miss: dict[int, None] = {}
        misses: list[int] = []
        for expert, is_fixed in zip(actual, fixed):
            selected += 1
            expert_id = int(expert)
            if is_fixed:
                hits += 1
                continue
            key = (layer, expert_id)
            if cache.lookup(key):
                hits += 1
                continue
            counts[expert_id] = counts.get(expert_id, 0) + 1
            if expert_id not in seen_miss:
                seen_miss[expert_id] = None
                misses.append(expert_id)
        misses.sort(key=lambda expert_id: (-counts[expert_id], expert_id))
        q = admit_budget(len(misses), cache.slots, admit_fraction)
        if min_fill and misses:
            q = max(q, 1 if cache.slots > 0 else 0)
            q = min(q, len(misses), cache.slots)
        for expert in misses[:q]:
            cache.admit((layer, expert))
        miss_steps.append(len(misses))
        copy_steps.append(q)
    misses_n = selected - hits
    return {
        "policy": policy,
        "slots": slots,
        "warm_slots": warm_slots,
        "selected": selected,
        "hits": hits,
        "misses": misses_n,
        "miss_rate": misses_n / selected if selected else 0.0,
        "miss_steps": miss_steps,
        "copy_steps": copy_steps,
        "admit_fraction": admit_fraction,
        "min_fill": min_fill,
    }


BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0


def ms_for_bytes(nbytes: int, gbs: float) -> float:
    if gbs <= 0:
        return 0.0
    return nbytes / (gbs * BYTES_PER_GIB) * 1000.0


def exposed_ms(n_miss: int, cpu_ms: float, copy_ms: float, q_star: float, min_fill: bool) -> float:
    if n_miss <= 0:
        return 0.0
    if copy_ms <= 0 or q_star <= 0:
        return n_miss * cpu_ms
    q = int(round(q_star * n_miss))
    if min_fill:
        q = max(1, q)
    q = min(max(0, q), n_miss)
    return max(q * copy_ms, (n_miss - q) * cpu_ms)


def attach_costs(
    sim: dict[str, object], cpu_ms: float, copy_ms: float, q_star: float
) -> dict[str, object]:
    steps = sim.get("miss_steps") or []
    copies = sim.get("copy_steps")
    out = dict(sim)
    out.pop("miss_steps", None)
    out.pop("copy_steps", None)
    out["cpu_only_ms"] = sum(n * cpu_ms for n in steps)
    if copies is not None and len(copies) == len(steps):
        out["qstar_ms"] = sum(
            max(c * copy_ms, (m - c) * cpu_ms) for m, c in zip(steps, copies)
        )
    else:
        out["qstar_ms"] = sum(exposed_ms(n, cpu_ms, copy_ms, q_star, False) for n in steps)
    out["fill_all_ms"] = sum(exposed_ms(n, cpu_ms, copy_ms, 1.0, True) for n in steps)
    out["qstar_min_fill_ms"] = sum(exposed_ms(n, cpu_ms, copy_ms, q_star, True) for n in steps)
    return out


def load_bw_profile(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "llama-wackmall-expert-bw-v1":
        raise ProfileError(f"unexpected bw schema in {path}")
    nbytes = int(data["expert_bytes"])
    return {
        "cpu_ms": ms_for_bytes(nbytes, float(data["cpu_gbs"])),
        "copy_ms": ms_for_bytes(nbytes, float(data["pcie_gbs"])),
        "q_star": float(data["q_star"]),
    }


def run(args: argparse.Namespace) -> None:
    records, trace_config = read_traces([path.resolve() for path in args.trace])
    n_layer = int(trace_config["n_layer"])
    slot_values = []
    for item in args.slots_per_layer.split(","):
        value = int(item)
        if value < 1:
            raise ProfileError(f"invalid slots-per-layer: {item!r}")
        slot_values.append(value)

    warm_values: list[int] = []
    if args.warm_slots.strip():
        for item in args.warm_slots.split(","):
            value = int(item)
            if value < 0:
                raise ProfileError(f"invalid warm-slots: {item!r}")
            warm_values.append(value)

    costs = None
    if args.bw_profile is not None:
        costs = load_bw_profile(args.bw_profile.resolve())
    elif args.cpu_ms is not None or args.copy_ms is not None or args.q_star is not None:
        if args.cpu_ms is None or args.copy_ms is None:
            raise ProfileError("--cpu-ms and --copy-ms are required together")
        costs = {
            "cpu_ms": float(args.cpu_ms),
            "copy_ms": float(args.copy_ms),
            "q_star": float(args.q_star if args.q_star is not None else 0.0),
        }

    simulations = [simulate_static(records)]
    for slots in slot_values:
        simulations.append(
            simulate_lru(
                records,
                global_slots=None,
                slots_per_layer=slots,
                n_layer=n_layer,
            )
        )
        simulations.append(
            simulate_lru(
                records,
                global_slots=slots * n_layer,
                slots_per_layer=None,
                n_layer=n_layer,
            )
        )
        if costs is not None and costs["q_star"] < 1.0:
            simulations.append(
                simulate_lru(
                    records,
                    global_slots=None,
                    slots_per_layer=slots,
                    n_layer=n_layer,
                    admit_fraction=costs["q_star"],
                    min_fill=False,
                )
            )

    for warm in warm_values:
        if warm == 0:
            continue
        for global_warm in (False, True):
            simulations.append(
                simulate_fixed_overflow(
                    records,
                    warm_slots=warm,
                    n_layer=n_layer,
                    global_warm=global_warm,
                )
            )
            if costs is not None and costs["q_star"] < 1.0:
                simulations.append(
                    simulate_fixed_overflow(
                        records,
                        warm_slots=warm,
                        n_layer=n_layer,
                        global_warm=global_warm,
                        admit_fraction=costs["q_star"],
                        min_fill=False,
                    )
                )

    if costs is not None:
        simulations = [
            attach_costs(item, costs["cpu_ms"], costs["copy_ms"], costs["q_star"])
            for item in simulations
        ]
    else:
        simulations = [
            {k: v for k, v in item.items() if k not in ("miss_steps", "copy_steps")}
            for item in simulations
        ]

    output = {
        "schema": "llama-wackmall-expert-decode-cache-v1",
        "trace_config": trace_config,
        "trace_files": [str(path.resolve()) for path in args.trace],
        "record_count": len(records),
        "slots_per_layer": slot_values,
        "warm_slots": warm_values,
        "simulations": simulations,
    }
    if costs is not None:
        output["cost_model"] = costs
    static = simulations[0]
    output["vs_static"] = []
    for item in simulations[1:]:
        row = {
            "policy": item["policy"],
            "slots": item["slots"],
            "miss_rate": item["miss_rate"],
            "static_miss_rate": static["miss_rate"],
            "miss_rate_delta": static["miss_rate"] - item["miss_rate"],
        }
        if "cpu_only_ms" in item:
            row["cpu_only_ms"] = item["cpu_only_ms"]
            row["qstar_ms"] = item["qstar_ms"]
            row["qstar_min_fill_ms"] = item["qstar_min_fill_ms"]
            row["cpu_only_vs_static_ms"] = item["cpu_only_ms"] - static["cpu_only_ms"]
            row["qstar_vs_static_ms"] = item["qstar_ms"] - static["cpu_only_ms"]
        output["vs_static"].append(row)

    output_path = args.output.resolve(strict=False)
    if output_path.exists():
        raise ProfileError(f"refusing to overwrite existing output: {output_path}")
    if not output_path.parent.is_dir():
        raise ProfileError(f"output parent does not exist: {output_path.parent}")
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"records={len(records)} static_miss={static['miss_rate']:.4f} "
        f"simulations={len(simulations)}"
    )
    for item in output["vs_static"]:
        line = (
            f"{item['policy']} slots={item['slots']} "
            f"miss={item['miss_rate']:.4f} delta={item['miss_rate_delta']:+.4f}"
        )
        if "qstar_vs_static_ms" in item:
            line += (
                f" cpu_only={item['cpu_only_vs_static_ms']:+.1f}ms "
                f"q*={item['qstar_vs_static_ms']:+.1f}ms"
            )
        print(line)
    print(f"output={output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--slots-per-layer",
        default="28",
        help="comma-separated S values; global LRU uses S * n_layer slots",
    )
    parser.add_argument(
        "--warm-slots",
        default="1,2,4",
        help="comma-separated overflow W values on top of static S; 0 is skipped",
    )
    parser.add_argument("--bw-profile", type=Path, help="llama-wackmall-expert-bw-v1 JSON")
    parser.add_argument("--cpu-ms", type=float, help="CPU cold ms per missed expert")
    parser.add_argument("--copy-ms", type=float, help="pinned H2D ms per expert")
    parser.add_argument("--q-star", type=float, help="fetch fraction of misses (default 0)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, ProfileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
