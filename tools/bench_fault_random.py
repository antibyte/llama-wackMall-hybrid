#!/usr/bin/env python3
"""Fault-Pfad-Nachbau mit Lings Zugriffsmuster: verstreute Expertenbloecke.

Der Decode-Pfad holt die Experten ueber madvise(MADV_WILLNEED) in 128-kB-Fenstern
und beruehrt danach jede Seite (ggml_fault_expert_slabs in ggml-cpu.c). Der
Decode erreicht dabei nur ~0,85 GB/s, die SSD kann mit verstreuten 1-MB-Bloecken
aber ~2,9 GB/s. Dieses Skript stellt die Varianten gegeneinander:

  madvise128 : 8 x 128 kB MADV_WILLNEED je Block, dann Touch (der Ist-Zustand)
  madvise1m  : 1 x 1 MiB MADV_WILLNEED je Block, dann Touch
  pread      : Block mit pread in einen privaten Puffer (kein Page-Cache)

Aufruf:
    python3 tools/bench_fault_random.py <file> [threads] [blocks] [span_gb]
"""

import mmap
import os
import random
import sys
import threading
import time

FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
PAGE = os.sysconf("SC_PAGE_SIZE")


def build_offsets(blocks, span, block):
    rnd = random.Random(20261106)
    offs = []
    for _ in range(blocks):
        off = rnd.randrange(0, max(1, span - block))
        offs.append(off - (off % 4096))
    return offs


def touch(mm, off, length):
    acc = 0
    end = off + length
    pos = off
    while pos < end:
        acc |= mm[pos]
        pos += PAGE
    return acc


def run_variant(name, path, offsets, threads, chunk, span):
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    per = [offsets[i::threads] for i in range(threads)]
    got = [0] * threads

    def worker(tid):
        total = 0
        if name == "pread":
            buf = bytearray(chunk)
            mv = memoryview(buf)
            for off in per[tid]:
                n = os.preadv(fd, [mv], off)
                total += n if n > 0 else 0
        else:
            for off in per[tid]:
                if name == "madvise1m":
                    mm.madvise(mmap.MADV_WILLNEED, off, chunk)
                else:
                    c = off
                    end = off + chunk
                    while c < end:
                        mm.madvise(mmap.MADV_WILLNEED, c, min(128 * 1024, end - c))
                        c += 128 * 1024
            for off in per[tid]:
                total += touch(mm, off, chunk)
        got[tid] = total

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    mm.close()
    os.close(fd)
    return len(offsets) * chunk / dt / 1e6


def main():
    path = os.path.expanduser(sys.argv[1])
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 384
    span_gb = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0

    size = os.path.getsize(path)
    span = int(min(span_gb * (1 << 30), size))
    chunk = 1024 * 1024
    offsets = build_offsets(blocks, span, chunk)

    print(f"file {path}")
    print(f"{len(offsets)} Bloecke x 1 MiB, span {span/2**30:.1f} GiB, threads {threads}")
    for name in ("madvise128", "madvise1m", "pread"):
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, span, FADV_DONTNEED)
        os.close(fd)
        time.sleep(0.5)
        mbs = run_variant(name, path, offsets, threads, chunk, span)
        print(f"  {name:11s} {mbs:7.0f} MB/s")


if __name__ == "__main__":
    main()
