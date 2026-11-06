#!/usr/bin/env python3
"""Tensor-Infos (Name, Dims, Offset, Typ) aus einem GGUF-Kopf lesen.

Wird gebraucht, um den Per-Experten-Stride (nb02) der Gewichte zu bestimmen:
der Readahead-Pfad in ggml-cpu.c verarbeitet nur Slabs bis 8 MiB, groessere
Experten werden also ueberhaupt nicht vorgezogen.

Aufruf: python3 tools/gguf_tensor_info.py MODELL.gguf [FILTER]
"""

from __future__ import annotations

import struct
import sys

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

# GGML-Typen (ggml.h) mit Blockgroesse in Byte
GGML_TYPE = {
    0: ("F32", 4), 1: ("F16", 2), 2: ("Q4_0", 18), 3: ("Q4_1", 20),
    7: ("Q8_0", 34), 8: ("Q5_0", 22), 9: ("Q5_1", 24), 10: ("Q2_K", 84),
    11: ("Q3_K", 110), 12: ("Q4_K", 144), 13: ("Q5_K", 176), 14: ("Q6_K", 210),
    15: ("Q8_K", 292), 16: ("IQ2_XXS", 66), 17: ("IQ2_XS", 74),
    18: ("IQ3_XXS", 98), 19: ("IQ1_S", 50), 20: ("IQ4_NL", 18),
    21: ("IQ3_S", 110), 22: ("IQ2_S", 82), 23: ("IQ4_XS", 136),
    24: ("I8", 1), 25: ("I16", 2), 26: ("I32", 4), 27: ("I64", 8),
    28: ("F64", 8), 29: ("IQ1_M", 56), 30: ("BF16", 2),
}
# Quantisierte Typen: Bytes pro Element (Blockgroesse / Elemente im Block)
BLK = {
    "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32, "Q8_0": 32, "Q2_K": 256,
    "Q3_K": 256, "Q4_K": 256, "Q5_K": 256, "Q6_K": 256, "Q8_K": 256,
    "IQ2_XXS": 256, "IQ2_XS": 256, "IQ3_XXS": 256, "IQ1_S": 256,
    "IQ4_NL": 32, "IQ3_S": 256, "IQ2_S": 256, "IQ4_XS": 256, "IQ1_M": 256,
}


def read_prefix(path: str, limit: int) -> tuple[dict, list]:
    with open(path, "rb") as f:
        data = f.read(limit)
    if data[:4] != b"GGUF":
        raise ValueError("keine GGUF-Datei")
    n_tensors, n_kv = struct.unpack_from("<QQ", data, 8)
    pos = 24
    kv: dict = {}

    def rstr(buf, p):
        n, = struct.unpack_from("<Q", buf, p)
        p += 8
        return buf[p:p + n].decode("utf-8", "replace"), p + n

    for _ in range(n_kv):
        key, pos = rstr(data, pos)
        t, = struct.unpack_from("<I", data, pos)
        pos += 4
        if t == 8:
            val, pos = rstr(data, pos)
        elif t == 9:
            et, = struct.unpack_from("<I", data, pos)
            n, = struct.unpack_from("<Q", data, pos + 4)
            pos += 12
            if et == 8:
                items = []
                for _ in range(n):
                    s, pos = rstr(data, pos)
                    items.append(s)
                val = items
            else:
                size = SCALAR.get(et, 4)
                raw = data[pos:pos + size * n]
                pos += size * n
                fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                       10: "Q", 11: "q", 12: "d"}.get(et, "I")
                val = list(struct.unpack(f"<{n}{fmt}", raw))
        else:
            size = SCALAR.get(t, 4)
            raw = data[pos:pos + size]
            pos += size
            fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                   7: "?", 10: "Q", 11: "q", 12: "d"}.get(t, "I")
            val = struct.unpack(f"<{fmt}", raw)[0]
        kv[key] = val

    tensors = []
    for _ in range(n_tensors):
        name, pos = rstr(data, pos)
        n_dims, = struct.unpack_from("<I", data, pos)
        pos += 4
        dims = struct.unpack_from(f"<{n_dims}Q", data, pos)
        pos += 8 * n_dims
        ttype, = struct.unpack_from("<I", data, pos)
        pos += 4
        off, = struct.unpack_from("<Q", data, pos)
        pos += 8
        tensors.append((name, list(dims), ttype, off))
    return kv, tensors


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    for limit in (16 << 20, 64 << 20, 256 << 20):
        try:
            kv, tensors = read_prefix(path, limit)
            break
        except (struct.error, IndexError):
            continue
    else:
        raise ValueError("Tensor-Infos nicht lesbar")

    arch = kv.get("general.architecture", "")
    pref = f"{arch}." if arch else ""
    print(f"{path}")
    for key in ("embedding_length", "expert_count", "expert_feed_forward_length",
                "expert_shared_feed_forward_length", "block_count"):
        if pref + key in kv:
            print(f"  {pref+key} = {kv[pref+key]}")

    print(f"  {'Tensor':52s} {'Typ':7s} {'Dims':24s} {'Groesse':>10s}")
    for name, dims, ttype, off in tensors:
        if filt and filt not in name:
            continue
        tname, tsize = GGML_TYPE.get(ttype, (f"T{ttype}", 4))
        n = 1
        for d in dims:
            n *= d
        if tname in BLK:
            per = tsize / BLK[tname]
            size = n * per
        else:
            size = n * tsize
        print(f"  {name:52s} {tname:7s} {str(dims):24s} {size/2**20:9.2f}M")
    return 0


if __name__ == "__main__":
    sys.exit(main())
