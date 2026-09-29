#!/usr/bin/env python3
"""Completion and persistent chat client used for hybrid benchmarks."""

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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-file", type=Path)
    source.add_argument("--chat-turns-file", type=Path, help="JSON array of user prompts; runs all turns on the same server slot.")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--prompt-suffix-file", type=Path)
    parser.add_argument("--prompt-suffix-repeat", type=int, default=1)
    parser.add_argument("--cache-prompt", action="store_true")
    parser.add_argument("--n-predict", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force n_predict tokens (default: true, matching existing benches).",
    )
    return parser.parse_args()


def run_chat(args: argparse.Namespace) -> int:
    if args.prompt_repeat != 1 or args.prompt_suffix_file is not None or args.prompt_suffix_repeat != 1:
        raise ValueError("prompt repetition and suffix options require --prompt-file")
    if args.n_predict < 1 or args.timeout <= 0:
        raise ValueError("--n-predict and --timeout must be positive")
    source = args.chat_turns_file.read_bytes()
    turns = json.loads(source)
    if not isinstance(turns, list) or not turns or any(not isinstance(turn, str) or not turn.strip() for turn in turns):
        raise ValueError("--chat-turns-file must contain a non-empty JSON array of non-empty strings")

    base_url = args.url.rstrip("/")
    parameters = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": args.seed,
        "max_tokens": args.n_predict,
        "ignore_eos": args.ignore_eos,
        "cache_prompt": args.cache_prompt,
        "id_slot": 0,
        "return_tokens": True,
        "verbose": True,
        "stream": False,
    }
    result = {
        "mode": "chat",
        "metadata": {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "url": base_url,
            "turns_file": str(args.chat_turns_file),
            "turns_sha256": hashlib.sha256(source).hexdigest(),
            "request_parameters": parameters,
        },
        "turns": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    messages = []
    turn_index = 0
    try:
        with urllib.request.urlopen(base_url + "/props", timeout=args.timeout) as response:
            result["metadata"]["server_properties"] = json.load(response)
        for turn_index, prompt in enumerate(turns, 1):
            messages.append({"role": "user", "content": prompt})
            payload = {**parameters, "messages": list(messages)}
            request_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                base_url + "/v1/chat/completions",
                data=request_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                data = json.load(response)
            wall_ms = (time.perf_counter() - started) * 1000.0
            raw = data.get("__verbose", {})
            tokens = raw.get("tokens")
            content = raw.get("content")
            if not isinstance(tokens, list) or any(type(token) is not int for token in tokens) or not isinstance(content, str):
                raise RuntimeError("chat response must include __verbose.content and __verbose.tokens for correctness hashes")
            timings = data["timings"]
            if timings["predicted_n"] > 0 and not tokens:
                raise RuntimeError("chat response omitted generated tokens")
            message = data["choices"][0]["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise RuntimeError("chat response must include an assistant message")
            token_bytes = json.dumps(tokens, separators=(",", ":")).encode("ascii")
            token_count_complete = len(tokens) == timings["predicted_n"]
            row = {
                "turn": turn_index,
                "wall_ms": wall_ms,
                "ttft_ms": None,
                "prompt_tokens_processed": timings["prompt_n"],
                "prompt_tokens_cached": timings["cache_n"],
                "prompt_tps": timings["prompt_n"] * 1000.0 / timings["prompt_ms"] if timings["prompt_ms"] > 0 else None,
                "decode_tps": timings["predicted_n"] * 1000.0 / timings["predicted_ms"] if timings["predicted_ms"] > 0 else None,
                "tokens_predicted": timings["predicted_n"],
                "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "token_sha256": hashlib.sha256(token_bytes).hexdigest() if token_count_complete else None,
                "token_count_complete": token_count_complete,
                "token_ids": tokens,
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "request": payload,
                "timings": timings,
                "response": data,
            }
            result["turns"].append(row)
            messages.append(message)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({key: row[key] for key in ("turn", "prompt_tokens_processed", "prompt_tokens_cached", "prompt_tps", "decode_tps", "tokens_predicted")}))
    except Exception as exc:
        result["error"] = {"turn": turn_index, "type": type(exc).__name__, "message": str(exc)}
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    return 0


def main() -> int:
    args = parse_args()
    if args.chat_turns_file is not None:
        return run_chat(args)
    if args.prompt_repeat < 1:
        raise ValueError("--prompt-repeat must be at least 1")
    if args.prompt_suffix_repeat < 1:
        raise ValueError("--prompt-suffix-repeat must be at least 1")
    prompt_unit = args.prompt_file.read_text(encoding="utf-8").rstrip("\n")
    prompt = "\n".join([prompt_unit] * args.prompt_repeat)
    if args.prompt_suffix_file is not None:
        suffix_unit = args.prompt_suffix_file.read_text(encoding="utf-8").rstrip("\n")
        prompt += "\n" + "\n".join([suffix_unit] * args.prompt_suffix_repeat)
    payload = {
        "prompt": prompt,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": args.seed,
        "n_predict": args.n_predict,
        "ignore_eos": args.ignore_eos,
        "cache_prompt": args.cache_prompt,
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
            if "error" in event:
                raise RuntimeError("completion stream error: " + json.dumps(event["error"], ensure_ascii=False))
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
    timings = final.get("timings", {})
    token_count_complete = len(tokens) == timings.get("predicted_n")
    result = {
        "wall_ms": (ended - started) * 1000.0,
        "ttft_ms": None if first_content_at is None else (first_content_at - started) * 1000.0,
        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "token_sha256": hashlib.sha256(token_bytes).hexdigest() if token_count_complete else None,
        "token_count_complete": token_count_complete,
        "token_ids": tokens,
        "stream_token_count": len(tokens),
        "content_bytes": len(content.encode("utf-8")),
        "content_head": content[:800],
        "content_tail": content[-800:] if len(content) > 800 else content,
        "prompt_bytes": len(prompt.encode("utf-8")),
        "cache_prompt": args.cache_prompt,
        "timings": timings,
        "stop_type": final.get("stop_type", ""),
        "tokens_predicted": final.get("tokens_predicted", len(tokens)),
        "truncated": final.get("truncated", False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
