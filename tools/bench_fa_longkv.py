#!/usr/bin/env python3
"""Decode bei langem KV ueber den Server messen (FA-Kachelbreite).

llama-bench faehrt Prefill und Decode als getrennte Tests: die tg-Zeile startet bei
n_prompt=0, also bei leerem Kontext. Fuer den Decode mit langem KV braucht es einen
Server, der den langen Prompt erst verarbeitet und dann dekodiert. Die Antwort
liefert timings.predicted_per_second.

Der Prompt wird auf eine echte Tokenzahl kalibriert: /tokenize zaehlt die Tokens,
Wortzahl taugt nicht (BPE splittet jedes Wort in mehrere Tokens).

Aufruf: python3 tools/bench_fa_longkv.py --variant 8 --variant 16 ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import string
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 18097
URL = f"http://127.0.0.1:{PORT}"


def post(path: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def make_words(n: int, seed: int = 7) -> list[str]:
    """Zufallskoerterpool: gleiche Seed -> gleiche Woerter -> gleicher Prompt."""
    rng = random.Random(seed)
    vocab = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))) for _ in range(4000)]
    return [rng.choice(vocab) for _ in range(n)]


def n_tokens(words: list[str]) -> int:
    res = post("/tokenize", {"content": " ".join(words), "add_special": False})
    return len(res.get("tokens", []))


def fit_prompt(target_tokens: int, budget: int) -> tuple[str, int]:
    """Kalibriert die Wortzahl so, dass der Prompt zwischen target und budget liegt.

    Der Prompt muss zusammen mit n_predict in den Kontext passen, sonst lehnt der
    Server die Anfrage mit HTTP 400 ab.
    """
    lo, hi = 1, max(2, target_tokens)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        words = make_words(mid)
        n = n_tokens(words)
        if n <= budget:
            best = (words, n)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise RuntimeError("auch ein Wort passt nicht in den Kontext")
    words, n = best
    return " ".join(words), n


def wait_health(proc: subprocess.Popen, timeout: float = 600.0) -> float:
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"Server beendete mit {proc.returncode}")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except (urllib.error.URLError, OSError):
            pass
        if time.time() - t0 > timeout:
            raise TimeoutError("Server wurde nicht bereit")
        time.sleep(0.5)


def run_variant(args, variant: str, prompt: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_"))}
    env["GGML_CUDA_FA_NCOLS2"] = variant
    env["LD_LIBRARY_PATH"] = str(Path(args.bin).resolve())
    cmd = [str(Path(args.bin) / "llama-server"), "--host", "127.0.0.1", "--port", str(PORT),
           "--no-ui", "--offline", "-m", args.model, "-c", str(args.ctx), "-b", str(args.batch),
           "-ub", str(args.ubatch), "-t", str(args.threads), "-ngl", "99", "-fa", "on",
           "--temp", "0", "--top-k", "1"]
    log_path = Path(args.out) / f"server-ncols2-{variant}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=Path(args.bin), env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        load_s = wait_health(proc)
        body = {"prompt": prompt, "n_predict": args.n_predict, "cache_prompt": False,
                "seed": 1, "temperature": 0, "top_k": 1}
        t0 = time.time()
        res = post("/completion", body)
        wall = time.time() - t0
        timings = res.get("timings", {})
        return {
            "ncols2": variant,
            "load_s": round(load_s, 1),
            "wall_s": round(wall, 2),
            "prompt_tps": timings.get("prompt_per_second"),
            "predicted_tps": timings.get("predicted_per_second"),
            "prompt_n": timings.get("prompt_n"),
            "predicted_n": timings.get("predicted_n"),
            "content": res.get("content", ""),
        }
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)
        log.close()
        time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./build-main-sm75/bin")
    ap.add_argument("--model", default=os.path.expanduser("~/models/qwen3-1.7b/Qwen_Qwen3-1.7B-Q4_K_M.gguf"))
    ap.add_argument("--ctx", type=int, default=24576)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=20000)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variant", action="append", default=None)
    ap.add_argument("--out", default="benchmark-results/tensorfold-review-20261106/fa-longkv-server")
    args = ap.parse_args()
    args.bin = str(Path(args.bin).resolve())
    variants = args.variant or ["8", "16", "32"]

    Path(args.out).mkdir(parents=True, exist_ok=True)
    rows = []
    prompt = None
    for rep in range(args.reps):
        for variant in variants:
            if prompt is None:
                # Kalibrierung einmal, danach derselbe Prompt fuer alle Varianten.
                prompt, n_prompt = fit_prompt(args.prompt_tokens, args.ctx - args.n_predict - 128)
                print(f"Prompt kalibriert: {n_prompt} Tokens (Ziel {args.prompt_tokens}, "
                      f"ctx {args.ctx})", flush=True)
            row = run_variant(args, variant, prompt)
            row["rep"] = rep
            rows.append(row)
            print(f"  Runde {rep} ncols2={variant:>2s}: decode {row['predicted_tps']} t/s   "
                  f"prefill {row['prompt_tps']} t/s   (prompt_n={row['prompt_n']})", flush=True)
            (Path(args.out) / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Tokengleichheit: alle Varianten muessen dieselbe Ausgabe erzeugen.
    contents = {r["ncols2"]: r.get("content", "") for r in rows}
    same = len(set(contents.values())) == 1
    print(f"\nTokengleichheit ueber alle Varianten: {'JA' if same else 'NEIN'}")
    if not same:
        for v, c in contents.items():
            print(f"  ncols2={v}: {c[:120]!r}")

    print("\nMittel je Variante:")
    base_variant = variants[0]
    base = [r["predicted_tps"] for r in rows if r["ncols2"] == base_variant and r["predicted_tps"]]
    for variant in variants:
        vals = [r["predicted_tps"] for r in rows if r["ncols2"] == variant and r["predicted_tps"]]
        if not vals:
            continue
        rel = ""
        if base and variant != base_variant:
            rel = f"   {(sum(vals)/len(vals)) / (sum(base)/len(base)) * 100 - 100:+.2f} %"
        print(f"  ncols2={variant:>2s}: {sum(vals)/len(vals):8.2f} t/s  "
              f"(Werte: {', '.join(f'{v:.2f}' for v in vals)}){rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
