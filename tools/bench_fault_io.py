#!/usr/bin/env python3
"""Micro-benchmark of the expert fault path: madvise(WILLNEED) vs. plain pread.

The decode path of Ling-3.0-flash is SSD bound (0.9-1.2 GB/s measured during
decode), while the same device reaches ~5 GB/s with enough parallel streams.
This script isolates the two ways of pulling a range of the model file in:

  madvise:  map the file, queue MADV_WILLNEED in chunks, then touch the pages
            (this is what ggml_fault_expert_slabs does)
  pread:    read the same bytes into anonymous buffers with N parallel readers

For each variant the page cache is dropped for the tested range first, so the
numbers are device numbers and not cache numbers.

Usage:
    python3 tools/bench_fault_io.py <file> [span_gb] [threads] [chunk_kb]
"""

import mmap
import os
import sys
import threading
import time

FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
PAGE = os.sysconf("SC_PAGE_SIZE")


def drop_cache(path, offset, length):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, offset, length, FADV_DONTNEED)
    finally:
        os.close(fd)


def touch_range(mm, offset, length):
    """Touch one byte per page to force the fault, the way the kernel path does."""
    acc = 0
    end = offset + length
    pos = offset
    while pos < end:
        acc |= mm[pos]
        pos += PAGE
    return acc


def variant_madvise(path, offset, length, threads, chunk):
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        # queue the readahead in chunks, T threads splitting the range
        def queue(tid):
            per = length // threads
            start = offset + tid * per
            stop = start + per if tid < threads - 1 else offset + length
            c = start
            while c < stop:
                n = min(chunk, stop - c)
                try:
                    mm.madvise(mmap.MADV_WILLNEED, c, n)
                except OSError:
                    pass
                c += n

        ts = [threading.Thread(target=queue, args=(i,)) for i in range(threads)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        t_queue = time.perf_counter() - t0

        def do_touch(tid):
            per = length // threads
            start = offset + tid * per
            stop = start + per if tid < threads - 1 else offset + length
            touch_range(mm, start, stop - start)

        ts = [threading.Thread(target=do_touch, args=(i,)) for i in range(threads)]
        t1 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        t_touch = time.perf_counter() - t1
        mm.close()
        return t_queue, t_touch
    finally:
        os.close(fd)


def variant_pread(path, offset, length, threads, chunk):
    per = length // threads

    def worker(tid):
        start = offset + tid * per
        stop = start + per if tid < threads - 1 else offset + length
        buf = bytearray(chunk)
        mv = memoryview(buf)
        fd = os.open(path, os.O_RDONLY)
        try:
            c = start
            while c < stop:
                n = min(chunk, stop - c)
                got = os.preadv(fd, [mv[:n]], c)
                if got <= 0:
                    break
                c += got
        finally:
            os.close(fd)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return 0.0, time.perf_counter() - t0


def main():
    path = os.path.expanduser(sys.argv[1])
    span_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    chunk_kb = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    size = os.path.getsize(path)
    span = int(min(span_gb * (1 << 30), size))
    chunk = chunk_kb * 1024

    print(f"file {path}")
    print(f"span {span/2**30:.2f} GiB, threads {threads}, chunk {chunk_kb} KiB")
    print(f"{'variant':10s} {'queue s':>9s} {'wait s':>9s} {'total s':>9s} {'MB/s':>8s}")
    for name, fn in (("madvise", variant_madvise), ("pread", variant_pread)):
        drop_cache(path, 0, span)
        time.sleep(0.5)
        tq, tw = fn(path, 0, span, threads, chunk)
        tot = tq + tw
        print(f"{name:10s} {tq:9.3f} {tw:9.3f} {tot:9.3f} {span/tot/1e6:8.0f}")


if __name__ == "__main__":
    main()
