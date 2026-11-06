#!/usr/bin/env python3
"""Expert-Groessen aus einem GGUF bestimmen (Summe der Bytes je Experte).

Liest nur die Tensor-Info (Namen, Typ, Offset) und rechnet die Groessen aus den
Nachbartensoren. Faellt auf 0 Bytes, wenn die Datei kein Tensorverzeichnis im
erwarteten Format hat.

Aufruf: python3 tools/gguf_expert_bytes.py MODELL.gguf [experten_pro_layer]
"""

from __future__ import annotations

import re
import struct
import sys
from collections import defaultdict

# Bytes je Element. Die quantisierten Typen haben Blockgroessen, darum der Quotient.
GGML_TYPE_SIZE = {
    0: 1, 1: 1, 2: 18 / 32, 3: 20 / 32, 4: 4, 5: 4, 6: 22 / 32, 7: 24 / 32,
    8: 34 / 32, 9: 36 / 32, 10: 84 / 256, 11: 110 / 256, 12: 144 / 256, 13: 176 / 256,
    14: 210 / 256, 15: 292 / 256, 16: 66 / 256, 17: 74 / 256, 18: 98 / 256,
    19: 50 / 256, 20: 18 / 32, 21: 110 / 256, 22: 82 / 256, 23: 136 / 256,
    24: 56 / 256, 25: 2, 26: 54 / 256, 27: 66 / 256, 28: 0.5, 29: 0.5, 30: 1,
}


def read_tensor_info(path: str) -> tuple[list[tuple[str, int, int, int, tuple[int, ...]]], int]:
    """Liefert (name, typ, n_bytes, offset, dims) je Tensor."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"GGUF":
        raise ValueError("keine GGUF-Datei")
    version, = struct.unpack_from("<I", data, 4)
    n_tensors, n_kv = struct.unpack_from("<QQ", data, 8)
    pos = 24

    def rstr(buf: bytes, p: int) -> tuple[str, int]:
        n, = struct.unpack_from("<Q", buf, p)
        p += 8
        return buf[p:p + n].decode("utf-8", "replace"), p + n

    for _ in range(n_kv):
        _, pos = rstr(data, pos)
        t, = struct.unpack_from("<I", data, pos)
        pos += 4
        if t == 8:
            _, pos = rstr(data, pos)
        elif t == 9:
            et, = struct.unpack_from("<I", data, pos)
            n, = struct.unpack_from("<Q", data, pos + 4)
            pos += 12
            if et == 8:
                for _ in range(n):
                    _, pos = rstr(data, pos)
            else:
                pos += GGML_TYPE_SIZE.get(et, 4) * n
        else:
            pos += {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(t, 4)

    out: list[tuple[str, int, int, int, tuple[int, ...]]] = []
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
        nelem = 1
        for d in dims:
            nelem *= d
        out.append((name, ttype, int(nelem * GGML_TYPE_SIZE.get(ttype, 0)), off, tuple(dims)))
    return out, len(data)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    tensors, total = read_tensor_info(path)
    TENSOR_DIMS = {name: dims for name, _t, _b, _o, dims in tensors}
    per_expert: dict[tuple[int, int], int] = defaultdict(int)
    layers: dict[int, set[int]] = defaultdict(set)
    other = 0
    n_expert_total = 0
    # Die Experten liegen als 3D-Tensor (n_embd, n_ff, n_expert) im GGUF, nicht je
    # Experte einzeln. Der Byte-Anteil je Experte ist tensor_bytes / n_expert.
    pat = re.compile(r"blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.")
    for name, ttype, nbytes, off, _dims in tensors:
        m = pat.search(name)
        if not m:
            other += nbytes
            continue
        layer = int(m.group(1))
        layers[layer].add(-1)
        n_expert_total += 1
        per_expert[(layer, -1)] += nbytes
    if not per_expert:
        print("keine Experten-Tensoren gefunden")
        return 1
    # n_expert aus der Tensorform holen: bei 3D-Tensoren ist ne[2] die Expertenzahl.
    n_expert = 0
    for name, ttype, nbytes, off, _dims in tensors:
        if re.search(r"ffn_(?:gate|up|down)_exps\.", name):
            dims = TENSOR_DIMS.get(name, ())
            if len(dims) == 3:
                n_expert = max(n_expert, dims[2])
    if n_expert == 0:
        print("n_expert nicht aus den Tensorformen ableitbar")
        return 1
    sizes = sorted(v // n_expert for v in per_expert.values())
    n_layers = len(layers)
    print(f"Modell: {path}")
    print(f"Tensoren: {len(tensors)}  Layer: {n_layers}  Experten je Layer: {n_expert}")
    print(f"Expertenbytes je Slice: min {sizes[0]}  median {sizes[len(sizes)//2]}  max {sizes[-1]} "
          f"= {sizes[len(sizes)//2]/1048576:.2f} MiB")
    total_exp = sum(sizes) * n_expert
    print(f"Summe Experten: {total_exp/1073741824:.2f} GiB   Rest: {other/1073741824:.2f} GiB")
    print(f"Datei: {total/1073741824:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
