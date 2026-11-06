#!/usr/bin/env python3
"""Alle GGUF-Modelle nach Attention-Konfiguration scannen.

Relevant ist die GQA-Rate (n_head / n_head_kv): nur wenn sie groesser als 8 ist,
gibt es Spielraum fuer eine breitere GQA-Kachel im FA-Kernel. Der Upstream
deckelt die Kachelbreite bei min(gqa_ratio, 8) ab.

Aufruf: python3 tools/gguf_attn_scan.py [MODELLVERZEICHNIS] [--min-gqa N]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_attn_config import read_kv  # noqa: E402


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    min_gqa = 1
    for a in sys.argv[1:]:
        if a.startswith("--min-gqa"):
            min_gqa = int(a.split("=", 1)[1]) if "=" in a else int(a.split()[1])
    root = Path(args[0]) if args else Path.home() / "models"

    rows = []
    for gguf in sorted(root.rglob("*.gguf")):
        try:
            kv = read_kv(str(gguf))
        except Exception as e:  # defekte oder exotische Dateien ueberspringen
            print(f"  (!) {gguf.name}: {e}", file=sys.stderr)
            continue
        arch = kv.get("general.architecture", "?")
        pref = f"{arch}."

        def get(key, default=None):
            if pref + key in kv:
                return kv[pref + key]
            return kv.get(key, default)

        n_head = get("attention.head_count")
        n_head_kv = get("attention.head_count_kv", n_head)
        n_embd = get("embedding_length")
        head_dim = get("attention.key_length") or get("kda.head_dim")
        if not head_dim and n_head and n_embd:
            head_dim = n_embd // n_head
        layers_full = None
        if isinstance(n_head_kv, list):
            # Liste je Layer: die Vollattention-Layer haben den groessten Wert
            layers_full = sum(1 for v in n_head_kv if isinstance(v, int))
            kv_heads = sorted({v for v in n_head_kv if isinstance(v, int)})
            n_head_kv = kv_heads[-1] if kv_heads else None
        if not n_head or not n_head_kv or not head_dim:
            continue
        gqa = n_head // n_head_kv
        rows.append((gqa, arch, n_head, n_head_kv, head_dim, get("block_count"), layers_full, gguf))

    rows.sort(reverse=True)
    print(f"{'gqa':>4} {'arch':<14} {'n_head':>6} {'n_kv':>5} {'head_dim':>8} {'layer':>5} {'voll':>4}  Datei")
    for gqa, arch, nh, nhk, hd, nl, lf, path in rows:
        flag = "  <<<" if gqa >= min_gqa else ""
        print(f"{gqa:>4} {arch:<14} {nh:>6} {nhk:>5} {hd:>8} {str(nl or '-'):>5} {str(lf or '-'):>4}  "
              f"{path.relative_to(root)}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
