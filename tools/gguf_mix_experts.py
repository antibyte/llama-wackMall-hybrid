#!/usr/bin/env python3
"""Write one GGUF with the metadata and tensors of a base model, taking the tensors
whose name matches a pattern from a donor quantization of the same model.

Used to keep attention, routers and shared experts at the base quantization while the
routed experts (which stream from disk) come from a smaller one.

Usage: gguf_mix_experts.py --base A-00001-of-00002.gguf [A-00002-of-00002.gguf ...]
           --donor B-00001-of-00002.gguf [...] --out mixed.gguf [--pattern REGEX]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))

import gguf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--donor", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default=r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$")
    args = ap.parse_args()

    pattern = re.compile(args.pattern)
    base = [gguf.GGUFReader(p) for p in args.base]
    donor = {t.name: t for p in args.donor for t in gguf.GGUFReader(p).tensors}

    arch = base[0].get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.out, arch=arch, endianess=base[0].endianess)

    for field in base[0].fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith(("GGUF.", "split.")):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    tensors = []
    n_swapped = 0
    bytes_base = 0
    bytes_out = 0
    for reader in base:
        for t in reader.tensors:
            bytes_base += t.n_bytes
            if pattern.search(t.name):
                d = donor.get(t.name)
                if d is None or list(d.shape) != list(t.shape):
                    raise SystemExit(f"donor has no matching tensor for {t.name}")
                t = d
                n_swapped += 1
            bytes_out += t.n_bytes
            tensors.append(t)
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    print(f"{n_swapped} of {len(tensors)} tensors from the donor, {bytes_base / 2**30:.1f} -> {bytes_out / 2**30:.1f} GiB")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in tensors:
        writer.write_tensor_data(t.data, tensor_endianess=base[0].endianess)
    writer.close()


if __name__ == "__main__":
    main()
