#!/usr/bin/env python3
"""Fault-Pfad-Varianten mit Lings Zugriffsmuster vergleichen.

Der Ist-Pfad holt die Experten mit madvise(MADV_WILLNEED) in 128-kB-Fenstern
und beruehrt danach jede Seite (ggml-cpu.c). Varianten:

  madvise128 : 8 x 128 kB MADV_WILLNEED je Block, dann Touch (Ist-Zustand)
  readahead  : ein readahead(2) je Block mit so vielen Seiten wie der Block
               gross ist - fuellt den Page-Cache wie madvise, aber mit einem
               anderen Kernel-Pfad
  pread      : Block mit pread in einen privaten Puffer (kein Page-Cache)

readahead und pread fuellen unterschiedliche Ziele: readahead befuellt den
Page-Cache (der Compute liest ohne Kopie daraus), pread braucht einen privaten
Puffer (der Compute liest daraus, kostet aber eine Kopie).

Aufruf:
    python3 tools/bench_fault_variants.py <file> [threads] [blocks] [span_gb]
"""

import ctypes
import mmap
import os
import random
import sys
import threading
import time

FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
FADV_WILLNEED = getattr(os, "POSIX_FADV_WILLNEED", 3)
PAGE = os.sysconf("SC_PAGE_SIZE")

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.readahead.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_size_t]
_libc.readahead.restype = ctypes.c_int


def readahead(fd, offset, count_pages):
    return _libc.readahead(fd, ctypes.c_longlong(offset), ctypes.c_size_t(count_pages))


def touch(mm, off, length):
    acc = 0
    end = off + length
    pos = off
    while pos < end:
        acc |= mm[pos]
        pos += PAGE
    return acc


def run_variant(name, path, offsets, threads, block):
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    per = [offsets[i::threads] for i in range(threads)]
    done = [0] * threads

    def worker(tid):
        total = 0
        if name == "pread":
            buf = bytearray(block)
            mv = memoryview(buf)
            for off in per[tid]:
                n = os.preadv(fd, [mv], off)
                total += n if n > 0 else 0
        else:
            for off in per[tid]:
                if name == "readahead":
                    readahead(fd, off, block // PAGE)
                elif name == "fadvise":
                    os.posix_fadvise(fd, off, block, FADV_WILLNEED)
                elif name == "fadvise128":
                    c = off
                    end = off + block
                    while c < end:
                        os.posix_fadvise(fd, c, min(128 * 1024, end - c), FADV_WILLNEED)
                        c += 128 * 1024
                else:
                    c = off
                    end = off + block
                    while c < end:
                        mm.madvise(mmap.MADV_WILLNEED, c, min(128 * 1024, end - c))
                        c += 128 * 1024
            for off in per[tid]:
                total += touch(mm, off, block)
        done[tid] = total

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    mm.close()
    os.close(fd)
    return len(offsets) * block / dt / 1e6


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
        offsets.append(off - (off % PAGE))

    print(f"file {path}")
    print(f"{len(offsets)} Bloecke x 1 MiB, span {span/2**30:.1f} GiB, threads {threads}")
    for name in ("madvise128", "fadvise", "fadvise128", "pread"):
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, span, FADV_DONTNEED)
        os.close(fd)
        time.sleep(0.5)
        mbs = run_variant(name, path, offsets, threads, block)
        print(f"  {name:11s} {mbs:7.0f} MB/s")


if __name__ == "__main__":
    main()
