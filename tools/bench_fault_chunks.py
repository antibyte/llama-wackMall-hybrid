#!/usr/bin/env python3
"""Chunk-Groesse des Readaheads mit Lings Zugriffsmuster durchsuchen.

Der Ist-Zustand (ggml-cpu.c) ruft madvise(MADV_WILLNEED) in 128-kB-Fenstern
und beruehrt danach jede Seite. Der Kernel liest je Aufruf nur ein
Readahead-Fenster; die Fenstergroesse bestimmt also, wie viele Aufrufe und
damer wie viel Overhead ein Expertenblock kostet. Das Skript misst die
Bandbreite fuer Fenster von 32 kB bis 1 MiB.

Aufruf:
    python3 tools/bench_fault_chunks.py <file> [threads] [blocks] [span_gb]
"""

import mmap
import os
import random
import sys
import threading
import time

FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
PAGE = os.sysconf("SC_PAGE_SIZE")


def run_chunk(path, offsets, threads, chunk_kb, block):
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    per = [offsets[i::threads] for i in range(threads)]
    window = chunk_kb * 1024
    bytes_done = [0] * threads

    def worker(tid):
        total = 0
        for off in per[tid]:
            c = off
            end = off + block
            while c < end:
                mm.madvise(mmap.MADV_WILLNEED, c, min(window, end - c))
                c += window
            pos = off
            acc = 0
            while pos < end:
                acc |= mm[pos]
                pos += PAGE
            total += end - off
        bytes_done[tid] = total

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    mm.close()
    os.close(fd)
    return sum(bytes_done) / dt / 1e6


def main():
    path = os.path.expanduser(sys.argv[1])
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 384
    span_gb = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0

    size = os.path.getsize(path)
    span = int(min(span_gb * (1 << 30), size))
    block = 1024 * 1024
    rnd = random.Random(20261106)
    offsets = []
    for _ in range(blocks):
        off = rnd.randrange(0, max(1, span - block))
        offsets.append(off - (off % 4096))

    print(f"file {path}")
    print(f"{len(offsets)} Bloecke x 1 MiB, span {span/2**30:.1f} GiB, threads {threads}")
    for chunk_kb in (32, 64, 128, 256, 512, 1024):
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, span, FADV_DONTNEED)
        os.close(fd)
        time.sleep(0.4)
        mbs = run_chunk(path, offsets, threads, chunk_kb, block)
        print(f"  Fenster {chunk_kb:5d} kB: {mbs:7.0f} MB/s")


if __name__ == "__main__":
    main()
