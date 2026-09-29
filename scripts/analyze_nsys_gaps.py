#!/usr/bin/env python3
"""Summarize CUDA kernel occupancy vs inter-kernel GPU idle from an nsys sqlite export."""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


def first_table(conn: sqlite3.Connection, names: list[str]) -> str | None:
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for n in names:
        if n in have:
            return n
    return None


def cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def string_lookup(conn: sqlite3.Connection) -> dict[int, str]:
    table = first_table(conn, ["StringIds", "StringTable"])
    if not table:
        return {}
    return {int(i): str(v) for i, v in conn.execute(f"SELECT id, value FROM {table}")}


def resolve_name(raw, strings: dict[int, str]) -> str:
    if raw is None:
        return "?"
    if isinstance(raw, int) or (isinstance(raw, str) and raw.isdigit()):
        return strings.get(int(raw), str(raw))
    return str(raw)


def fetch_kernels(conn: sqlite3.Connection) -> list[tuple[int, int, str]]:
    table = first_table(conn, ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"])
    if not table:
        return []
    strings = string_lookup(conn)
    cset = cols(conn, table)
    name_col = "shortName" if "shortName" in cset else ("name" if "name" in cset else None)
    start = "start" if "start" in cset else "timestamp"
    end = "end" if "end" in cset else None
    if not name_col or not end:
        return []
    rows = []
    for s, e, n in conn.execute(f"SELECT {start}, {end}, {name_col} FROM {table}"):
        if s is None or e is None:
            continue
        rows.append((int(s), int(e), resolve_name(n, strings)))
    rows.sort()
    return rows


def fetch_runtime(conn: sqlite3.Connection) -> dict[str, float]:
    table = first_table(conn, ["CUPTI_ACTIVITY_KIND_RUNTIME"])
    out: dict[str, float] = defaultdict(float)
    if not table:
        return out
    strings = string_lookup(conn)
    cset = cols(conn, table)
    if "nameId" in cset:
        name_col = "nameId"
    elif "name" in cset:
        name_col = "name"
    else:
        return out
    sql = f"SELECT {name_col}, SUM(end-start) FROM {table} GROUP BY {name_col}"
    for n, tot in conn.execute(sql):
        out[resolve_name(n, strings)] = float(tot or 0)
    return out


def union_busy(intervals: list[tuple[int, int]]) -> tuple[int, int]:
    if not intervals:
        return 0, 0
    ev: list[tuple[int, int]] = []
    for s, e in intervals:
        ev.append((s, 1))
        ev.append((e, -1))
    ev.sort()
    act = 0
    last = None
    busy = 0
    idle = 0
    for t, d in ev:
        if last is not None:
            dt = t - last
            if act > 0:
                busy += dt
            else:
                idle += dt
        act += d
        last = t
    return busy, idle


def ns_to_ms(x: float) -> float:
    return x / 1e6


def window_stats(kernels: list[tuple[int, int, str]]) -> dict:
    if not kernels:
        return {
            "n_kernels": 0,
            "kernel_ms": 0.0,
            "span_ms": 0.0,
            "busy_frac": 0.0,
            "gap_frac": 0.0,
            "mmvq_ms": 0.0,
            "mmvq_frac": 0.0,
        }
    busy, idle = union_busy([(s, e) for s, e, _ in kernels])
    wall = busy + idle
    ktime = sum(e - s for s, e, _ in kernels)
    mmvq = sum(e - s for s, e, n in kernels if "mul_mat_vec_q" in n)
    return {
        "n_kernels": len(kernels),
        "kernel_ms": ns_to_ms(ktime),
        "span_ms": ns_to_ms(wall),
        "busy_frac": (busy / wall) if wall else 0.0,
        "gap_frac": (idle / wall) if wall else 0.0,
        "mmvq_ms": ns_to_ms(mmvq),
        "mmvq_frac": (mmvq / busy) if busy else 0.0,
    }


def analyze(path: Path) -> dict:
    conn = sqlite3.connect(str(path))
    kernels = fetch_kernels(conn)
    runtime = fetch_runtime(conn)
    conn.close()

    if not kernels:
        return {"file": str(path), "n_kernels": 0}

    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for s, e, n in kernels:
        short = n.split("(")[0]
        by_name[short][0] += e - s
        by_name[short][1] += 1
    top = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:8]

    decode_marks = [s for s, e, n in kernels if "mul_mat_vec_q" in n]
    decode_kernels = kernels
    if decode_marks:
        d0, d1 = min(decode_marks), max(e for s, e, n in kernels if "mul_mat_vec_q" in n)
        decode_kernels = [(s, e, n) for s, e, n in kernels if e > d0 and s < d1]

    all_s = window_stats(kernels)
    dec_s = window_stats(decode_kernels)

    launch_ns = sum(v for k, v in runtime.items() if "Launch" in k or "launch" in k)
    sync_ns = sum(v for k, v in runtime.items() if "Synchronize" in k or "Memcpy" in k)
    graph_ns = sum(v for k, v in runtime.items() if "Graph" in k or "graph" in k)

    return {
        "file": path.name,
        "n_kernels": all_s["n_kernels"],
        "unique_kernels": len(by_name),
        "kernel_ms": all_s["kernel_ms"],
        "span_ms": all_s["span_ms"],
        "gap_ms": all_s["span_ms"] * all_s["gap_frac"],
        "gap_frac": all_s["gap_frac"],
        "busy_frac": all_s["busy_frac"],
        "decode_n": dec_s["n_kernels"],
        "decode_span_ms": dec_s["span_ms"],
        "decode_busy_frac": dec_s["busy_frac"],
        "decode_gap_frac": dec_s["gap_frac"],
        "decode_mmvq_frac": dec_s["mmvq_frac"],
        "n_gaps": 0,
        "median_gap_us": 0.0,
        "p95_gap_us": 0.0,
        "launch_api_ms": ns_to_ms(launch_ns),
        "sync_memcpy_api_ms": ns_to_ms(sync_ns),
        "graph_api_ms": ns_to_ms(graph_ns),
        "top": [(n, ns_to_ms(t), c) for n, (t, c) in top],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    rows = [analyze(Path(p)) for p in args.sqlite]
    hdr = [
        "file", "n_kernels", "unique_kernels", "kernel_ms", "span_ms", "gap_ms",
        "gap_frac", "busy_frac", "decode_n", "decode_span_ms", "decode_busy_frac",
        "decode_gap_frac", "decode_mmvq_frac", "launch_api_ms", "sync_memcpy_api_ms",
        "graph_api_ms",
    ]
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write(",".join(hdr) + "\n")
            for r in rows:
                if r.get("n_kernels", 0) == 0:
                    f.write(f"{r.get('file','')},0,,,,,,,,,,,,\n")
                    continue
                f.write(
                    ",".join(
                        str(r[k]) if not isinstance(r[k], float) else f"{r[k]:.4f}"
                        for k in hdr
                    )
                    + "\n"
                )
    for r in rows:
        print(f"== {r.get('file')} ==")
        if not r.get("n_kernels"):
            print("  no kernel table")
            continue
        print(
            f"  all: kernels={r['n_kernels']} unique={r['unique_kernels']} "
            f"busy={r['busy_frac']*100:.1f}% gap={r['gap_frac']*100:.1f}% "
            f"span={r['span_ms']:.1f}ms"
        )
        print(
            f"  decode window: n={r.get('decode_n')} busy={r.get('decode_busy_frac',0)*100:.1f}% "
            f"gap={r.get('decode_gap_frac',0)*100:.1f}% mmvq={r.get('decode_mmvq_frac',0)*100:.1f}% "
            f"span={r.get('decode_span_ms',0):.1f}ms"
        )
        print(
            f"  API launch={r['launch_api_ms']:.1f}ms sync/memcpy={r['sync_memcpy_api_ms']:.1f}ms "
            f"graph={r['graph_api_ms']:.1f}ms"
        )
        print("  top kernels (ms, count):")
        for n, ms, c in r["top"]:
            print(f"    {ms:8.1f}  n={c:5d}  {n}")


if __name__ == "__main__":
    main()
