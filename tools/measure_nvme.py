#!/usr/bin/env python3
"""Read-only throughput probe for the model file on the SSD.

Measures sequential read bandwidth with the page cache dropped for the tested
range (POSIX_FADV_DONTNEED), so the number reflects the device and not the
kernel cache. Also measures a "many small streams" pattern, which is what the
expert faulting does when several worker threads read different experts.

Usage:
    python3 tools/measure_nvme.py <file> [gb] [streams]
"""

import os
import sys
import time

FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)


def drop_cache(fd, offset, length):
    try:
        os.posix_fadvise(fd, offset, length, FADV_DONTNEED)
    except OSError:
        pass


def read_range(fd, offset, length, bufsize=4 << 20):
    os.lseek(fd, offset, os.SEEK_SET)
    remaining = length
    got = 0
    while remaining > 0:
        chunk = os.read(fd, min(bufsize, remaining))
        if not chunk:
            break
        got += len(chunk)
        remaining -= len(chunk)
    return got


def main():
    path = sys.argv[1]
    total_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    streams = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    size = os.path.getsize(path)
    span = int(min(total_gb * (1 << 30), size))

    fd = os.open(path, os.O_RDONLY)
    try:
        print(f"file      : {path}")
        print(f"size      : {size/2**30:.1f} GiB")
        print(f"span      : {span/2**30:.2f} GiB, streams={streams}")
        print()

        drop_cache(fd, 0, span)
        time.sleep(1.0)

        if streams == 1:
            t0 = time.perf_counter()
            got = read_range(fd, 0, span)
            dt = time.perf_counter() - t0
            print(f"sequential: {got/2**30:.2f} GiB in {dt:.2f} s -> {got/dt/1e6:.0f} MB/s")
        else:
            # One stream per worker: each thread reads its own contiguous slab.
            per = span // streams
            chunks = []
            for s in range(streams):
                off = s * per
                chunks.append((off, per))
            t0 = time.perf_counter()
            import threading

            results = [0] * streams

            def worker(idx):
                off, ln = chunks[idx]
                results[idx] = read_range(fd, off, ln)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(streams)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            dt = time.perf_counter() - t0
            got = sum(results)
            print(f"parallel  : {got/2**30:.2f} GiB in {dt:.2f} s -> {got/dt/1e6:.0f} MB/s "
                  f"({streams} streams)")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
