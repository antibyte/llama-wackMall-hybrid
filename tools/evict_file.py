#!/usr/bin/env python3
"""Wirft eine Datei aus dem Page-Cache (fadvise DONTNEED ueber die ganze Datei).

Aufruf: evict_file.py <datei>

Hinweis: ein Evict nur ueber einen Teilbereich (offset, length) hat auf dieser
Maschine keine Wirkung - nach dem Evict lagen die Seiten weiter im Page-Cache
(65.782 MiB/s statt 2.759 MiB/s). Der Aufruf ueber die ganze Datei (offset 0,
length 0) wirkt. Das Harness (tools/mbench.py --evict) macht es so.
"""
import os
import sys

path = sys.argv[1]
fd = os.open(path, os.O_RDONLY)
try:
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    print(f"evictiert: {path}")
finally:
    os.close(fd)
