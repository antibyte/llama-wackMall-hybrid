#!/usr/bin/env python3
"""Rewrite an upstream llama.cpp Inkling GGUF (arch inkling2_loop, one block per
physical layer) into the inkling2 layout this fork loads (one block per visit,
as in the Unsloth files). Only metadata and the loop tensors change; the weights
are copied as they are.

- block_count and the per-layer arrays cover the visits: layers before the loop,
  the loop layers loop.count times, the layers after it.
- loop.start/end/count -> loop.core_start/core_end and num_loops.
- loop_{q,k,v,ffn}_ctrl_* -> loop_{attn,k,v,mlp}_*.
- the per-layer loop_msg_query (4 rows each) -> the global loop_res_query and
  loop_kv_res_query: rows [res 2i, kv 2i, kv 2i+1, res 2i+1] of loop layer i.

Checked on ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-IQ3_S against the Unsloth
UD-Q4_K_M file: every key and type matches, the loop tensors are bit-equal.

Usage: gguf_inkling2_from_upstream.py IN.gguf OUT.gguf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402

OLD, NEW = "inkling2_loop.", "inkling2."
RENAME = {"loop.start": "loop.core_start", "loop.end": "loop.core_end", "loop.count": "num_loops"}
PER_LAYER = {"attention.head_count", "attention.head_count_kv", "attention.sliding_window_pattern"}
CTRL = {".loop_q_ctrl_": ".loop_attn_", ".loop_k_ctrl_": ".loop_k_", ".loop_v_ctrl_": ".loop_v_", ".loop_ffn_ctrl_": ".loop_mlp_"}


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src = GGUFReader(sys.argv[1])
    arch = src.fields[gguf.Keys.General.ARCHITECTURE].contents()
    if arch != OLD[:-1]:
        sys.exit(f"architecture is {arch!r}, expected {OLD[:-1]!r}")

    def key(name: str):
        return src.fields[OLD + name].contents()

    start, end, count, n_layer = key("loop.start"), key("loop.end"), key("loop.count"), key("block_count")
    if key("loop.varying"):
        sys.exit("loop.varying is set: the inkling2 layout shares the weights of all loop visits")
    visits = list(range(start)) + [start + v % (end - start) for v in range((end - start) * count)] + list(range(end, n_layer))

    w = GGUFWriter(sys.argv[2], arch=NEW[:-1], endianess=src.endianess)
    for f in src.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        name, val = f.name, f.contents()
        if name.startswith(OLD):
            k = name[len(OLD):]
            if k == "loop.varying":
                continue
            if k == "block_count":
                val = len(visits)
            elif k in PER_LAYER:
                if len(val) != n_layer:
                    sys.exit(f"{name} has {len(val)} entries, expected {n_layer}")
                val = [val[p] for p in visits]
            name = NEW + RENAME.get(k, k)
        sub = f.types[-1] if f.types[0] == GGUFValueType.ARRAY else None
        w.add_key_value(name, val, f.types[0], sub_type=sub)

    n_loop = end - start
    res: list = [None] * (2 * n_loop)
    kv: list = [None] * (2 * n_loop)
    tensors = []
    for t in src.tensors:
        name = t.name
        if name.endswith(".loop_msg_query.weight"):
            i = int(name.split(".")[1]) - start
            m = np.asarray(t.data, dtype=np.float32)
            if not 0 <= i < n_loop or m.shape[0] != 4:
                sys.exit(f"unexpected {name} {m.shape}")
            res[2 * i], kv[2 * i], kv[2 * i + 1], res[2 * i + 1] = m[0], m[1], m[2], m[3]
            continue
        for a, b in CTRL.items():
            name = name.replace(a, b)
        tensors.append((name, t.data, t.tensor_type))
    for name, rows in (("loop_res_query.weight", res), ("loop_kv_res_query.weight", kv)):
        if any(r is None for r in rows):
            sys.exit(f"loop_msg_query rows missing for {name}")
        tensors.append((name, np.ascontiguousarray(np.stack(rows)), gguf.GGMLQuantizationType.F32))

    for name, data, tt in tensors:
        w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, tt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, data, _ in tensors:
        w.write_tensor_data(data, tensor_endianess=src.endianess)
    w.close()
    print(f"{sys.argv[2]}: {len(visits)} visits of {n_layer} layers, {len(tensors)} tensors")


if __name__ == "__main__":
    main()
