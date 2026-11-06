#!/usr/bin/env python3
"""RAM-Bandbreite messen: Kopieren groesser Bloecke mit N Threads.

Der private Experten-Puffer (pread) kostet eine Kopie der Gewichte durch den
RAM. Diese Kopie ist der Grund, warum der Puffer den schnelleren SSD-Zugriff
nicht bezahlt macht. Das Skript misst die Kopierbandbreite, die der Puffer
kosten wuerde.

Aufruf:
    python3 tools/measure_ram_bandwidth.py [threads] [mb_per_thread]
"""
import sys
import threading
import time


def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    mb = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    size = mb * 1024 * 1024
    src = bytearray(size)
    dst = bytearray(size)
    half = size // 2
    results = [0] * threads

    def worker(tid):
        moved = 0
        chunk = 1 << 20
        for off in range(0, half, chunk):
            dst[off:off + chunk] = src[off:off + chunk]
            moved += chunk
        results[tid] = moved

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    moved = sum(results)
    print(f"threads {threads}: {moved/2**20:.0f} MiB kopiert in {dt:.2f} s -> "
          f"{moved/dt/1e9:.1f} GB/s")


if __name__ == "__main__":
    main()
