#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bench_hybrid_client as client


def chat_response(turn: int) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": f"answer {turn}", "reasoning_content": f"reasoning {turn}"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 100 + turn, "completion_tokens": 2},
        "timings": {"prompt_n": 5, "cache_n": 95 + turn, "prompt_ms": 10, "predicted_n": 2, "predicted_ms": 40, "draft_n": 2, "draft_n_accepted": 1},
        "__verbose": {"tokens": [turn, turn + 1], "content": f"<think>reasoning {turn}</think>answer {turn}"},
    }


class HybridClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "result.json"
        self.turns = Path(self.directory.name) / "turns.json"
        self.turns.write_text(json.dumps(["first", "follow-up", "third"]), encoding="utf-8")
        self.requests = []
        self.responses = [(200, chat_response(turn)) for turn in range(1, 4)]
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def send(self, status, data):
                body = data.encode("utf-8") if isinstance(data, str) else json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send(200, {"model_path": "test.gguf", "total_slots": 1})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, payload))
                status, data = owner.responses.pop(0)
                self.send(status, data)

            def log_message(self, *_args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def stop_server(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def run_client(self, *extra):
        arguments = ["bench_hybrid_client.py", "--url", self.url, "--output", str(self.output), "--n-predict", "2", *extra]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            return client.main()

    def test_three_turns_preserve_reasoning_and_measure_uncached_tokens(self):
        self.assertEqual(self.run_client("--chat-turns-file", str(self.turns), "--cache-prompt"), 0)
        result = json.loads(self.output.read_text())
        self.assertEqual(result["metadata"]["server_properties"]["model_path"], "test.gguf")
        self.assertEqual(len(result["turns"]), 3)
        for index, (path, payload) in enumerate(self.requests):
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(len(payload["messages"]), 2 * index + 1)
            self.assertTrue(payload["cache_prompt"])
            self.assertTrue(payload["return_tokens"])
            self.assertTrue(payload["verbose"])
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["id_slot"], 0)
            self.assertEqual(payload["seed"], 42)
            if index:
                self.assertEqual(payload["messages"][-2], chat_response(index)["choices"][0]["message"])
            row = result["turns"][index]
            self.assertEqual(row["request"], payload)
            self.assertEqual(row["prompt_tps"], 500)
            self.assertEqual(row["decode_tps"], 50)
            self.assertEqual(row["prompt_tokens_cached"], 96 + index)
            self.assertIsNone(row["ttft_ms"])
            self.assertEqual(row["response"], chat_response(index + 1))
            expected_tokens = json.dumps([index + 1, index + 2], separators=(",", ":")).encode("ascii")
            self.assertTrue(row["token_count_complete"])
            self.assertEqual(row["token_sha256"], hashlib.sha256(expected_tokens).hexdigest())
            expected_content = chat_response(index + 1)["__verbose"]["content"].encode("utf-8")
            self.assertEqual(row["output_sha256"], hashlib.sha256(expected_content).hexdigest())

    def test_http_failure_keeps_completed_turns(self):
        self.responses[1] = (500, {"error": "test failure"})
        with self.assertRaises(urllib.error.HTTPError):
            self.run_client("--chat-turns-file", str(self.turns))
        result = json.loads(self.output.read_text())
        self.assertEqual(len(result["turns"]), 1)
        self.assertEqual(result["error"]["turn"], 2)
        self.assertEqual(result["error"]["type"], "HTTPError")
        self.assertEqual(len(self.requests), 2)

    def test_missing_tokens_cannot_pass_as_a_correctness_hash(self):
        del self.responses[0][1]["__verbose"]["tokens"]
        with self.assertRaisesRegex(RuntimeError, "correctness hashes"):
            self.run_client("--chat-turns-file", str(self.turns))
        result = json.loads(self.output.read_text())
        self.assertEqual(result["turns"], [])
        self.assertEqual(result["error"]["turn"], 1)

    def test_completion_stream_remains_compatible(self):
        prompt = Path(self.directory.name) / "prompt.txt"
        prompt.write_text("original prompt\n", encoding="utf-8")
        events = [{"content": "answer", "tokens": [11, 12], "stop": False}, {"stop": True, "timings": {"predicted_n": 2}, "tokens_predicted": 2, "stop_type": "limit"}]
        self.responses = [(200, "".join("data: " + json.dumps(event) + "\n\n" for event in events))]
        self.assertEqual(self.run_client("--prompt-file", str(prompt)), 0)
        path, payload = self.requests[0]
        self.assertEqual(path, "/completion")
        self.assertEqual(payload["prompt"], "original prompt")
        self.assertTrue(payload["stream"])
        self.assertFalse(payload["cache_prompt"])
        result = json.loads(self.output.read_text())
        self.assertEqual(result["stream_token_count"], 2)
        self.assertEqual(result["token_ids"], [11, 12])
        self.assertTrue(result["token_count_complete"])
        self.assertEqual(result["token_sha256"], hashlib.sha256(b"[11,12]").hexdigest())
        self.assertEqual(result["output_sha256"], hashlib.sha256(b"answer").hexdigest())
        self.assertEqual(result["stop_type"], "limit")

    def test_utf8_stream_with_missing_token_ids_has_no_correctness_hash(self):
        prompt = Path(self.directory.name) / "prompt.txt"
        prompt.write_text("test", encoding="utf-8")
        timings = {"predicted_n": 2, "predicted_ms": 40, "predicted_per_second": 50}
        # The server buffers the first UTF-8 fragment, then sends only the completing token ID.
        events = [{"content": "\u251c", "tokens": [12], "stop": False}, {"content": "", "tokens": [], "stop": True, "timings": timings, "tokens_predicted": 2, "stop_type": "limit"}]
        self.responses = [(200, "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events))]
        self.assertEqual(self.run_client("--prompt-file", str(prompt)), 0)
        result = json.loads(self.output.read_text())
        self.assertFalse(result["token_count_complete"])
        self.assertIsNone(result["token_sha256"])
        self.assertEqual(result["token_ids"], [12])
        self.assertEqual(result["stream_token_count"], 1)
        self.assertEqual(result["tokens_predicted"], 2)
        self.assertEqual(result["timings"], timings)
        self.assertEqual(result["content_head"], "\u251c")
        self.assertEqual(result["output_sha256"], hashlib.sha256("\u251c".encode("utf-8")).hexdigest())

    def test_invalid_turns_fail_before_requesting_inference(self):
        for turns in ([], [""], [1], {"turns": ["wrong schema"]}):
            with self.subTest(turns=turns):
                self.turns.write_text(json.dumps(turns), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "JSON array"):
                    self.run_client("--chat-turns-file", str(self.turns))
        self.assertEqual(self.requests, [])

    def test_completion_stream_surfaces_server_error(self):
        prompt = Path(self.directory.name) / "prompt.txt"
        prompt.write_text("test", encoding="utf-8")
        self.responses = [(200, 'data: {"error":{"code":500,"message":"out of memory"}}\n\n')]
        with self.assertRaisesRegex(RuntimeError, "completion stream error:.*out of memory"):
            self.run_client("--prompt-file", str(prompt))


if __name__ == "__main__":
    unittest.main()
