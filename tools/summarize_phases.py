#!/usr/bin/env python3
"""Wertet die Phasen des CPU-MoE-Kerns aus (LLAMA_EXPERT_TIMING=1).

Aufruf: summarize_phases.py <label> [<label> ...]
"""
import re
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__)) + "/../runs"

RE_LAYER = re.compile(
    r"layer\s+(\d+): cpu_cold_ms ([0-9.]+) cpu_gate_up_ms ([0-9.]+) "
    r"cpu_activation_ms ([0-9.]+) cpu_down_ms ([0-9.]+) cpu_cold_runs (\d+)")
RE_EVAL = re.compile(r"eval time\s+=\s+([0-9.]+) ms /\s+(\d+) tokens")
RE_PPEVAL = re.compile(r"prompt eval time\s+=\s+([0-9.]+) ms /\s+(\d+) tokens")
RE_FAULT = re.compile(r"slab faults (\d+) missing of (\d+) checked \(([0-9.]+)%\).*?fault wait ([0-9.]+) ms")


def analyse(label):
    d = f"{BASE}/{label}"
    log = open(f"{d}/server.log", errors="ignore").read()
    layers = [(int(m.group(1)), float(m.group(2)), float(m.group(3)),
               float(m.group(4)), float(m.group(5)), int(m.group(6)))
              for m in RE_LAYER.finditer(log)]
    if not layers:
        return None
    cold = sum(l[1] for l in layers)
    gate_up = sum(l[2] for l in layers)
    act = sum(l[3] for l in layers)
    down = sum(l[4] for l in layers)
    runs = sum(l[5] for l in layers)
    calc = gate_up + act + down
    rest = cold - calc
    ev = RE_EVAL.findall(log)
    pev = RE_PPEVAL.findall(log)
    dec_tok = sum(int(t) for _, t in ev)
    dec_ms = sum(float(x) for x, _ in ev)
    pre_tok = sum(int(t) for _, t in pev)
    pre_ms = sum(float(x) for x, _ in pev)
    fault = RE_FAULT.search(log)
    return {
        "label": label,
        "layers": len(layers),
        "cold_s": cold / 1000.0,
        "calc_s": calc / 1000.0,
        "rest_s": rest / 1000.0,
        "runs": runs,
        "ms_per_run": cold / runs if runs else 0,
        "calc_ms_per_run": calc / runs if runs else 0,
        "rest_ms_per_run": rest / runs if runs else 0,
        "dec_tokens": dec_tok,
        "dec_s": dec_ms / 1000.0,
        "ms_per_token": dec_ms / dec_tok if dec_tok else 0,
        "pre_tokens": pre_tok,
        "pre_s": pre_ms / 1000.0,
        "missing_pct": float(fault.group(3)) if fault else None,
        "faultwait_s": float(fault.group(4)) / 1000.0 if fault else None,
    }


labels = sys.argv[1:]
if not labels:
    labels = sorted(d for d in os.listdir(BASE)
                    if d.startswith(("r7d-", "r7p-")) and os.path.isdir(f"{BASE}/{d}"))

rows = []
for label in labels:
    a = analyse(label)
    if a:
        rows.append((label, a))

print(f"{'Arm':<22} {'Laeufe':>7} {'Knoten s':>9} {'Rechnen s':>10} {'Rest s':>8} "
      f"{'ms/Lauf':>8} {'rechnen':>8} {'Rest/Lauf':>10} {'Dec-Token':>10} {'ms/Token':>9} {'fehlend %':>10}")
print("-" * 128)
for label, a in rows:
    print(f"{label:<22} {a['runs']:>7} {a['cold_s']:>9.2f} {a['calc_s']:>10.2f} {a['rest_s']:>8.2f} "
          f"{a['ms_per_run']:>8.2f} {a['calc_ms_per_run']:>8.2f} {a['rest_ms_per_run']:>10.2f} "
          f"{a['dec_tokens']:>10} {a['ms_per_token']:>9.2f} "
          f"{a['missing_pct'] if a['missing_pct'] is not None else '-':>10}")

# Differenz Decode-Arm minus Prefill-Arm: das ist der Decode-Anteil.
pre = [a for _, a in rows if "prefill" in _]
dec = [a for _, a in rows if "decode" in _]
if pre and dec:
    p = min(pre, key=lambda x: x["runs"])
    d = min(dec, key=lambda x: x["runs"])
    d_runs = d["runs"] - p["runs"]
    if d_runs > 0:
        cold = d["cold_s"] - p["cold_s"]
        calc = d["calc_s"] - p["calc_s"]
        rest = d["rest_s"] - p["rest_s"]
        tok = d["dec_tokens"] - p["dec_tokens"]
        print()
        print("Decode-Anteil (Decode-Arm minus Prefill-Arm):")
        print(f"  Laeufe            {d_runs}")
        print(f"  Knoten            {cold:.2f} s  -> {cold/d_runs*1000:.3f} ms je Lauf")
        print(f"  davon Rechnen     {calc:.2f} s  -> {calc/d_runs*1000:.3f} ms je Lauf"
              f"  ({calc/cold*100:.1f} % des Knotens)")
        print(f"  davon Rest        {rest:.2f} s  -> {rest/d_runs*1000:.3f} ms je Lauf"
              f"  ({rest/cold*100:.1f} % des Knotens)")
        if tok:
            print(f"  dekodierte Token  {tok}  -> {cold/tok*1000:.3f} ms Knoten je Token,"
                  f" {calc/tok*1000:.3f} ms Rechnen je Token, {rest/tok*1000:.3f} ms Rest je Token")

print()
print("Knoten s: Summe cpu_cold_ms ueber alle Layer (Wanduhr des kalten MoE-Kerns aus Sicht von Thread 0, mit der Fault-Phase)")
print("Rechnen s: gate_up + activation + down. Rest: Fault-Phase, Kopie, Barrieren, Teile ausserhalb der drei Phasen.")
print("ms/Token: gemessene Tokenzeit des Decodes (eval time / tokens).")

