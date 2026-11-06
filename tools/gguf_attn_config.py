#!/usr/bin/env python3
"""Aufmerksamkeits-Konfiguration aus einem GGUF lesen (Kopfzahl, GQA-Rate).

Fuer die FA-Arbeitsaufteilung ist die GQA-Rate (n_head / n_head_kv) entscheidend:
sie begrenzt, wie viele Query-Koepfe sich ein KV-Kopf teilen, also wie breit die
GQA-Kachel sein darf, ohne dass Arbeit vergewendet wird.

Aufruf: python3 tools/gguf_attn_config.py MODELL.gguf [...]
"""

from __future__ import annotations

import struct
import sys

# GGUF-Typen: 0=uint8 1=int8 2=uint16 3=int16 4=uint32 5=int32 6=float32
# 7=bool 8=string 9=array 10=uint64 11=int64 12=float64
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_kv(path: str, limit: int = 64 * 1024 * 1024) -> dict[str, object]:
    """Liefert die Metadaten-Schluessel als dict (Arrays als Liste).

    Gelesen wird nur der Dateianfang: die Metadaten stehen vor den Tensoren.
    Reicht der Ausschnitt nicht, wird er vergroessert.
    """
    for size in (limit, 8 * limit, 32 * limit):
        try:
            return _read_kv_prefix(path, size)
        except (struct.error, IndexError):
            continue
    raise ValueError("Metadaten nicht lesbar")


def _read_kv_prefix(path: str, limit: int) -> dict[str, object]:
    with open(path, "rb") as f:
        data = f.read(limit)
    if data[:4] != b"GGUF":
        raise ValueError("keine GGUF-Datei")
    n_tensors, n_kv = struct.unpack_from("<QQ", data, 8)
    pos = 24
    kv: dict[str, object] = {}

    def rstr(buf: bytes, p: int) -> tuple[str, int]:
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
    return kv


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for path in sys.argv[1:]:
        kv = read_kv(path)
        # Die Metadaten sind architektur-praefixed, z. B. "qwen3.attention.head_count".
        arch = kv.get("general.architecture", "")
        pref = f"{arch}." if arch else ""

        def get(key: str, default=None):
            if pref + key in kv:
                return kv[pref + key]
            return kv.get(key, default)

        n_head = get("attention.head_count")
        n_head_kv = get("attention.head_count_kv", n_head)
        n_embd = get("embedding_length")
        key_len = get("attention.key_length", n_embd // n_head if n_head and n_embd else None)
        n_layer = get("block_count")
        n_expert = get("expert_count")
        gqa = (n_head // n_head_kv) if (n_head and n_head_kv) else None
        print(f"Modell: {path}")
        print(f"  Layer {n_layer}   n_embd {n_embd}   n_head {n_head}   n_head_kv {n_head_kv}"
              f"   head_dim {key_len}")
        print(f"  GQA-Rate (n_head/n_head_kv): {gqa}")
        if n_expert:
            print(f"  Experten je Layer: {n_expert}")
        if n_head and n_head_kv and key_len:
            # KV-Bytes je Token, f16
            kv_bytes = n_layer * n_head_kv * key_len * 2 * 2
            print(f"  KV-Bytes je Token (f16): {kv_bytes/1024:.1f} KiB")
            for n in (4096, 16384, 32768):
                print(f"    bei {n:>6d} Tokens: {kv_bytes*n/1073741824:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
