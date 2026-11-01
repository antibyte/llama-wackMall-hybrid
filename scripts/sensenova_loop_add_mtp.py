#!/usr/bin/env python3
# Append the SenseNova-Loop MTP head (mtp.safetensors of
# sensenova/SenseNova-Loop-9B-Preview) to a sensenova_loop GGUF that lacks it,
# as block n_layer with nextn_predict_layers = 1. Trunk tensors and metadata
# are copied byte for byte. Conversion rules match conversion/qwen.py
# (_Qwen35MtpMixin + Qwen3NextModel): every *norm.weight gets +1, projections
# keep their HF [out, in] layout.
#
#   python3 scripts/sensenova_loop_add_mtp.py IN.gguf mtp.safetensors OUT.gguf [--type q4_0]
import argparse, json, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gguf-py'))
import gguf
from gguf.quants import quantize

ap = argparse.ArgumentParser()
ap.add_argument('src_gguf')
ap.add_argument('mtp_safetensors')
ap.add_argument('out_gguf')
ap.add_argument('--type', default='q4_0', help='type of the 2D MTP weights (q8_0, q5_0, q4_0, f16)')
ap.add_argument('--ffn-type', default=None, help='override for ffn_gate/up/down')
ap.add_argument('--attn-type', default=None, help='override for attn_q/k/v/output')
ap.add_argument('--eh-type', default=None, help='override for nextn.eh_proj')
args = ap.parse_args()

QT = {'q8_0': gguf.GGMLQuantizationType.Q8_0, 'q5_0': gguf.GGMLQuantizationType.Q5_0,
      'q4_0': gguf.GGMLQuantizationType.Q4_0, 'f16': gguf.GGMLQuantizationType.F16}

reader = gguf.GGUFReader(args.src_gguf)
arch = reader.fields['general.architecture'].contents()
assert arch == 'sensenova_loop', arch
n_layer = int(reader.fields[f'{arch}.block_count'].contents())
assert f'{arch}.nextn_predict_layers' not in reader.fields, 'source already has MTP'
il = n_layer

# HF name -> (GGUF name, is_norm, group)
P = 'mtp.layers.0.'
B = f'blk.{il}.'
MAP = {
    'mtp.fc.weight':                         (B + 'nextn.eh_proj.weight',          False, 'eh'),
    'mtp.pre_fc_norm_embedding.weight':      (B + 'nextn.enorm.weight',            True,  None),
    'mtp.pre_fc_norm_hidden.weight':         (B + 'nextn.hnorm.weight',            True,  None),
    'mtp.norm.weight':                       (B + 'nextn.shared_head_norm.weight', True,  None),
    P + 'input_layernorm.weight':            (B + 'attn_norm.weight',              True,  None),
    P + 'post_attention_layernorm.weight':   (B + 'post_attention_norm.weight',    True,  None),
    P + 'self_attn.q_proj.weight':           (B + 'attn_q.weight',                 False, 'attn'),
    P + 'self_attn.k_proj.weight':           (B + 'attn_k.weight',                 False, 'attn'),
    P + 'self_attn.v_proj.weight':           (B + 'attn_v.weight',                 False, 'attn'),
    P + 'self_attn.o_proj.weight':           (B + 'attn_output.weight',            False, 'attn'),
    P + 'self_attn.q_norm.weight':           (B + 'attn_q_norm.weight',            True,  None),
    P + 'self_attn.k_norm.weight':           (B + 'attn_k_norm.weight',            True,  None),
    P + 'mlp.gate_proj.weight':              (B + 'ffn_gate.weight',               False, 'ffn'),
    P + 'mlp.up_proj.weight':                (B + 'ffn_up.weight',                 False, 'ffn'),
    P + 'mlp.down_proj.weight':              (B + 'ffn_down.weight',               False, 'ffn'),
}

def group_type(g):
    t = {'ffn': args.ffn_type, 'attn': args.attn_type, 'eh': args.eh_type}.get(g) or args.type
    return QT[t]

with open(args.mtp_safetensors, 'rb') as f:
    hlen = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hlen))
base = 8 + hlen
mm = np.memmap(args.mtp_safetensors, dtype=np.uint8, mode='r')

def load(name):
    t = hdr[name]
    s, e = t['data_offsets']
    raw = mm[base + s: base + e]
    if t['dtype'] == 'BF16':
        v = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif t['dtype'] == 'F32':
        v = raw.view(np.float32).copy()
    else:
        raise ValueError(t['dtype'])
    return v.reshape(t['shape'])

names = [k for k in hdr if k != '__metadata__']
missing = set(MAP) - set(names)
extra = set(names) - set(MAP)
assert not missing and not extra, (missing, extra)

new_tensors = []
for hn in names:
    gn, is_norm, grp = MAP[hn]
    x = load(hn)
    if is_norm:
        x = x.astype(np.float32) + 1.0
        q, qt = x.astype(np.float32), gguf.GGMLQuantizationType.F32
    else:
        qt = group_type(grp)
        q = quantize(np.ascontiguousarray(x, dtype=np.float32), qt)
    print(f'{hn:45s} -> {gn:36s} {qt.name:5s} {list(x.shape)} mean {float(x.mean()):+.4f}', flush=True)
    new_tensors.append((gn, q, qt))

writer = gguf.GGUFWriter(args.out_gguf, arch=arch, endianess=reader.endianess)

for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if field.name == f'{arch}.block_count':
        val = n_layer + 1
    elif field.name == f'{arch}.attention.recurrent_layers':
        assert len(val) == n_layer
        val = list(val) + [False]
    writer.add_key_value(field.name, val, vt, sub_type=st)
writer.add_key_value(f'{arch}.nextn_predict_layers', 1, gguf.GGUFValueType.UINT32)

for t in reader.tensors:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
for gn, q, qt in new_tensors:
    writer.add_tensor_info(gn, q.shape, q.dtype, q.nbytes, qt)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in reader.tensors:
    writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
for gn, q, qt in new_tensors:
    writer.write_tensor_data(q)
writer.close()
print('wrote', args.out_gguf)
