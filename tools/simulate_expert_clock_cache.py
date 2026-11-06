#!/usr/bin/env python3
"""Expert-Seiten-Cache simulieren: LRU gegen CLOCK ueber Use-Counts.

Der Fork liest kalte Experten aus dem mmap; was im Page-Cache liegt, kostet keinen
Platten-Read. Welche Experten dort liegen, entscheidet der Kernel heute nicht - er
laesst es dem Betriebssystem. TensorFold haelt dafuer eine eigene Regel: ein CLOCK
mit 2-Bit-Use-Count (zig/src/core/fsr/cache.zig). Ein einmal benutzter Experte geht
dann zuerst wieder raus, einer der oft genutzt wurde bleibt.

Dieses Werkzeug replayt einen Lookahead-Trace (Schema llama-wackmall-router-lookahead-v1)
und vergleicht die Regeln, ohne Hardware:

  lru    - least recently used (der Linux-Page-Cache als naechstes)
  clock  - TensorFold: CLOCK-Hand, uses = min(uses+1, 3) je Treffer, der Hand
           senkt uses > 0 um 1, ein Slot mit uses == 0 ist das Opfer
  static - der Hot-Satz des Forks: die im Trace als "fixed" markierten Experten
           treffen immer, der Rest geht wie bei lru in den Cache

Aufruf:
  python3 tools/simulate_expert_clock_cache.py TRACE.json [TRACE.json ...] \\
      --slots-per-layer N --expert-bytes B [--per-layer|--global] [--gbps S]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from simulate_expert_streaming import read_traces


class ClockCache:
    """CLOCK ueber 2-Bit-Use-Counts, nach TensorFold core/fsr/cache.zig.

    uses: Treffer minus Hand-Durchlaeufe. Ein neuer Experte startet bei 0, also
    geht ein einmal benutzter zuerst wieder raus. Treffer erhoehen uses auf
    maximal 3 (gesaettigte Addition), der Hand senkt uses > 0 um 1.
    """

    MAX_USES = 3
    SWEEPS = 4  # maxInt(u2) + 1

    def __init__(self, slots: int):
        if slots <= 0:
            raise ValueError("slots muessen groesser als 0 sein")
        self.slots = slots
        self.map: dict[tuple[int, int], int] = {}
        self.entry: list[tuple[int, int] | None] = [None] * slots
        self.uses = [0] * slots
        self.hand = 0
        self.evictions = 0

    def lookup(self, key: tuple[int, int]) -> bool:
        at = self.map.get(key)
        if at is None:
            return False
        self.uses[at] = min(self.uses[at] + 1, self.MAX_USES)
        return True

    def admit(self, key: tuple[int, int]) -> None:
        if key in self.map:
            return
        visits = 0
        while visits < self.slots * self.SWEEPS:
            visits += 1
            i = self.hand
            self.hand = (i + 1) % self.slots
            if self.uses[i] > 0:
                self.uses[i] -= 1
                continue
            old = self.entry[i]
            if old is not None:
                del self.map[old]
                self.evictions += 1
            self.entry[i] = key
            self.uses[i] = 0
            self.map[key] = i
            return
        raise RuntimeError("kein evictbarer Slot")

    def resident_count(self) -> int:
        return len(self.map)


class LruCache:
    def __init__(self, slots: int):
        if slots <= 0:
            raise ValueError("slots muessen groesser als 0 sein")
        self.slots = slots
        self.entries: collections.OrderedDict[tuple[int, int], None] = collections.OrderedDict()
        self.evictions = 0

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
            self.evictions += 1
        if self.slots > 0:
            self.entries[key] = None


def simulate(records, policy: str, slots_per_layer: int | None, global_slots: int | None,
             n_layer: int) -> dict[str, object]:
    if global_slots is not None:
        caches = {None: (ClockCache(global_slots) if policy == "clock" else LruCache(global_slots))}
        slots = global_slots
    else:
        if slots_per_layer is None:
            raise ValueError("slots_per_layer fehlt")
        caches = {
            layer: (ClockCache(slots_per_layer) if policy == "clock" else LruCache(slots_per_layer))
            for layer in range(n_layer)
        }
        slots = slots_per_layer

    selected = 0
    hits = 0
    misses = 0
    evictions = 0
    for record in records:
        layer = int(record["target_layer"])
        actual = [int(e) for e in record["actual"]]
        fixed = [bool(f) for f in record["actual_fixed"]]
        if len(actual) != len(fixed):
            raise ValueError("actual und actual_fixed haben verschiedene Laenge")
        cache = caches[None] if None in caches else caches[layer]
        selected += len(actual)
        hits += sum(1 for f in fixed if f)
        cold: list[int] = []
        for expert, is_fixed in zip(actual, fixed):
            if is_fixed:
                continue
            if expert not in cold:
                cold.append(expert)
        for expert in cold:
            key = (layer, expert)
            if cache.lookup(key):
                hits += 1
            else:
                misses += 1
                cache.admit(key)
        evictions += getattr(cache, "evictions", 0)
    total = selected
    return {
        "policy": policy,
        "slots": slots,
        "selected": total,
        "hits": hits,
        "misses": misses,
        "miss_rate": misses / total if total else 0.0,
        "evictions": evictions,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--slots-per-layer", type=int, default=None)
    ap.add_argument("--global-slots", type=int, default=None)
    ap.add_argument("--expert-bytes", type=int, default=0,
                    help="Bytes je Experten-Slice (Layer) fuer die Byte- und Zeitrechnung")
    ap.add_argument("--gbps", type=float, default=0.0,
                    help="gemessene SSD-Bandbreite in GB/s fuer die Zeitrechnung")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    records, config = read_traces(args.traces)
    n_layer = int(config["n_layer"])
    n_expert = int(config["n_expert"])
    print(f"Trace: {len(records)} Records, {n_layer} Layer, {n_expert} Experten, "
          f"top-{config['actual_top_k']}")

    budgets: list[tuple[str, int | None, int | None]] = []
    if args.slots_per_layer is not None:
        for s in (2, 4, 8, 16, 32, 64, 128):
            budgets.append(("per-layer", s, None))
    if args.global_slots is not None:
        for s in (128, 256, 512, 1024, 2048):
            budgets.append(("global", None, s))

    rows = []
    for kind, spl, gsl in budgets:
        for policy in ("lru", "clock"):
            res = simulate(records, policy, spl, gsl, n_layer)
            res["scope"] = kind
            res["slots_per_layer"] = spl
            rows.append(res)
            miss_bytes = res["misses"] * args.expert_bytes
            line = (f"{kind:10s} slots={str(res['slots']):>5s} {policy:6s} "
                    f"hits={res['hits']:6d} misses={res['misses']:6d} "
                    f"miss_rate={res['miss_rate']*100:6.2f}%")
            if args.expert_bytes:
                secs = miss_bytes / (args.gbps * 1e9) if args.gbps else 0.0
                line += (f"  neu lesen={miss_bytes/1073741824:8.2f} GiB"
                         + (f"  = {secs:7.2f} s" if args.gbps else ""))
            print(line)

    if rows:
        print("\nCLOCK gegen LRU, je Budget:")
        for kind, spl, gsl in budgets:
            pair = [r for r in rows if r["scope"] == kind and r["slots"] == (spl if kind == "per-layer" else gsl)]
            if len(pair) != 2:
                continue
            lru, clock = pair
            if lru["misses"] == 0:
                continue
            delta = (clock["misses"] - lru["misses"]) / lru["misses"] * 100.0
            print(f"  {kind:10s} slots={str(lru['slots']):>5s}: {delta:+6.2f}% kalte Bytes "
                  f"({lru['misses']} -> {clock['misses']})")

    if args.json:
        args.json.write_text(json.dumps({"config": config, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nJSON nach {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
