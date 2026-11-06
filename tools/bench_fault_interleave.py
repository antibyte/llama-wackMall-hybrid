#!/usr/bin/env python3
"""Fault-Pfad mit Lings Zugriffsmuster: Hinweis und Zugriff nebeneinander oder
getrennt.

Der Decode-Pfad holt die Experten in einem Durchgang: madvise(WILLNEED) fuer den
Experten und danach der Zugriff auf dieselben Seiten (ggml_fault_expert_slabs in
ggml-cpu.c). Die Warteschlange der SSD bleibt dabei flach, weil der Zugriff auf
den Hinweis folgt. Steht der Hinweis am Anfang, sammelt sich eine tiefe
Warteschlange und die SSD liefert mehr.

Varianten (je 8 Threads, verstreute 1-MiB-Bloecke):
  interleave : Hinweis, dann Zugriff, je Block (der Ist-Zustand)
  batched    : alle Hinweise, dann alle Zugriffe (tiefe Warteschlange)
  pread      : pread in einen privaten Puffer (kein Page-Cache)
  fadvise    : posix_fadvise(WILLNEED) statt madvise, Hinweis und Zugriff je Block

Aufruf:
    python3 tools/bench_fault_interleave.py <file> [threads] [blocks] [span_gb]
"""

import mmap
import os
import random
import sys
import threading
import time

FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
FADV_WILLNEED = getattr(os, "POSIX_FADV_WILLNEED", 3)
PAGE = os.sysconf("SC_PAGE_SIZE")
CHUNK = 128 * 1024


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


def hint(mm, fd, off, length, use_fadvise):
    if use_fadvise:
        for c in range(off, off + length, CHUNK):
            os.posix_fadvise(fd, c, min(CHUNK, off + length - c), FADV_WILLNEED)
    else:
        for c in range(off, off + length, CHUNK):
            mm.madvise(mmap.MADV_WILLNEED, c, min(CHUNK, off + length - c))


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
        elif name == "batched":
            for off in per[tid]:
                hint(mm, fd, off, chunk, False)
            for off in per[tid]:
                total += touch(mm, off, chunk)
        elif name == "fadvise":
            for off in per[tid]:
                hint(mm, fd, off, chunk, True)
                total += touch(mm, off, chunk)
        else:
            for off in per[tid]:
                hint(mm, fd, off, chunk, False)
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
    for name in ("interleave", "batched", "fadvise", "pread"):
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, span, FADV_DONTNEED)
        os.close(fd)
        time.sleep(0.5)
        mbs = run_variant(name, path, offsets, threads, chunk, span)
        print(f"  {name:11s} {mbs:7.0f} MB/s")


if __name__ == "__main__":
    main()
