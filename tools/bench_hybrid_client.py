#!/usr/bin/env python3
"""Streaming client used by scripts/bench_hybrid.sh."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--n-predict", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "n_predict": args.n_predict,
        "ignore_eos": True,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": True,
    }
    request = urllib.request.Request(
        args.url.rstrip("/") + "/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_content_at: float | None = None
    content_parts: list[str] = []
    tokens: list[int] = []
    final: dict[str, object] = {}

    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="strict").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            piece = event.get("content", "")
            if piece and first_content_at is None:
                first_content_at = time.perf_counter()
            if isinstance(piece, str):
                content_parts.append(piece)
            event_tokens = event.get("tokens", [])
            if isinstance(event_tokens, list):
                tokens.extend(int(token) for token in event_tokens)
            if event.get("stop"):
                final = event

    ended = time.perf_counter()
    if not final:
        raise RuntimeError("completion stream ended without a final stop event")
    content = "".join(content_parts)
    token_bytes = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    result = {
        "wall_ms": (ended - started) * 1000.0,
        "ttft_ms": None if first_content_at is None else (first_content_at - started) * 1000.0,
        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "token_ids": tokens,
        "stream_token_count": len(tokens),
        "content_bytes": len(content.encode("utf-8")),
        "timings": final.get("timings", {}),
        "stop_type": final.get("stop_type", ""),
        "tokens_predicted": final.get("tokens_predicted", len(tokens)),
        "truncated": final.get("truncated", False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
