#!/usr/bin/env python3
"""Lese-Bandbreite der SSD fuer das Ling-Zugriffsmuster messen.

Lings Experten liegen verstreut in der GGUF-Datei: 512 Experten je Layer, je
rund 1,05 MB (gate/up) bzw. 1,54 MB (down), ueber 36,9 GiB pro Split verteilt.
Der Decode liest also viele kleine Bloecke an verstreuten Offsets, nicht einen
langen sequenziellen Strom. Ein sequenzieller Test (tools/measure_nvme.py) sagt
darueber nichts aus.

Der Test liest N Threads mal Bloecke der Groesse S an zufaelligen Offsets aus
einem Bereich und misst die erreichte Bandbreite. Der Page-Cache wird vorher
per posix_fadvise(DONTNEED) verworfen.

Aufruf:
    python3 tools/measure_nvme_random.py <file> [threads] [block_kb] [blocks] [span_gb]
"""

import os
import random
import sys
import threading
import time

FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)


def drop_cache(fd, offset, length):
    try:
        os.posix_fadvise(fd, offset, length, FADV_DONTNEED)
    finally:
        pass


def main():
    path = os.path.expanduser(sys.argv[1])
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    block_kb = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    span_gb = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0

    size = os.path.getsize(path)
    span = int(min(span_gb * (1 << 30), size))
    block = block_kb * 1024
    nbytes = blocks * block

    fd = os.open(path, os.O_RDONLY)
    drop_cache(fd, 0, span)
    time.sleep(0.5)

    offsets = []
    rnd = random.Random(20261106)
    for _ in range(blocks):
        off = rnd.randrange(0, max(1, span - block))
        offsets.append(off - (off % 4096))

    per_thread = [offsets[i::threads] for i in range(threads)]
    results = [0] * threads

    def worker(tid):
        buf = bytearray(block)
        mv = memoryview(buf)
        got = 0
        for off in per_thread[tid]:
            n = os.preadv(fd, [mv], off)
            if n <= 0:
                continue
            got += n
        results[tid] = got

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    total = sum(results)
    os.close(fd)

    print(f"file      : {path}")
    print(f"span      : {span/2**30:.2f} GiB, threads {threads}, block {block_kb} KiB, "
          f"{blocks} Bloecke")
    print(f"gelesen   : {total/2**20:.1f} MiB in {dt:.2f} s -> {total/dt/1e6:.0f} MB/s "
          f"({total/2**20/dt:.0f} MiB/s)")


if __name__ == "__main__":
    main()
