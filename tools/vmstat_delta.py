#!/usr/bin/env python3
"""Report the vmstat delta of a command: reclaim, stalls and faults.

Usage: vmstat_delta.py <command> [args...]
"""
import subprocess
import sys

KEYS = ("pgfault", "pgmajfault", "pgscan_direct", "pgsteal_direct", "pgrefill",
        "allocstall_normal", "allocstall_movable", "compact_stall", "pgactivate",
        "pgdeactivate", "pgfree", "pswpin", "pswpout")


def read():
    out = {}
    with open("/proc/vmstat") as f:
        for line in f:
            k, v = line.split()
            if k in KEYS:
                out[k] = int(v)
    return out


def main():
    before = read()
    rc = subprocess.call(sys.argv[1:])
    after = read()
    print(f"exit {rc}")
    for k in KEYS:
        if k in before and k in after and after[k] != before[k]:
            print(f"  {k:20s} {after[k] - before[k]:+d}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
