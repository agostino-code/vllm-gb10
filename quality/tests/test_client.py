#!/usr/bin/env python3
"""Offline tests for the stdlib OpenAI client against a local fake server.

No GPU, no network: a threaded ``http.server`` plays the role of vLLM.

    python3 -m unittest discover -s quality/tests -v
"""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

QUALITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALITY_DIR))

from qlib.client import ChatClient, ClientError, ServerUnavailable  # noqa: E402


class FakeVLLM:
    """Tiny OpenAI-compatible fake server with recorded request payloads."""

    def __init__(self):
        self.payloads: list = []
        self.fail_next = 0
        self.stream_status = 200
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _json(self, status, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self._json(200, {"status": "ok"})
                elif self.path == "/v1/models":
                    self._json(200, {"object": "list", "data": [
                        {"id": "azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ",
                         "object": "model", "max_model_len": 409600},
                    ]})
                else:
                    self._json(404, {"error": "not found: %s" % self.path})

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self._json(404, {"error": "not found: %s" % self.path})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                outer.payloads.append(payload)
                if outer.fail_next > 0:
                    outer.fail_next -= 1
                    self._json(500, {"error": "transient"})
                    return
                if payload.get("stream") and outer.stream_status != 200:
                    self._json(outer.stream_status, {"error": "prompt too long"})
                    return
                if payload.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    chunks = [
                        {"choices": [{"delta": {"content": "quasar"}}]},
                        {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
                        {"choices": [{"delta": {"content": "-4419-zebra"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 123, "completion_tokens": 4}},
                    ]
                    for chunk in chunks:
                        self.wfile.write(("data: %s\n\n" % json.dumps(chunk)).encode("utf-8"))
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                tools = payload.get("tools") or []
                if tools:
                    message = {"role": "assistant", "content": "", "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "get_weather",
                                     "arguments": json.dumps({"city": "Rome", "unit": "celsius"})},
                    }]}
                else:
                    message = {"role": "assistant", "content": "Hello there.",
                               "reasoning_content": "let me think"}
                self._json(200, {
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                })

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class ClientTests(unittest.TestCase):
    def test_base_url_without_v1_is_normalised(self):
        client = ChatClient(base_url="http://127.0.0.1:8000", enable_thinking=False)
        self.assertTrue(client.base_url.endswith("/v1"))
        client = ChatClient(base_url="http://127.0.0.1:8000/v1/", enable_thinking=False)
        self.assertTrue(client.base_url.endswith("/v1"))

    def test_health_and_models(self):
        with FakeVLLM() as fake:
            client = ChatClient(base_url="http://127.0.0.1:%d/v1" % fake.port)
            self.assertTrue(client.is_healthy())
            model = client.resolve_model(None)
            self.assertEqual(model, "azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ")
            self.assertEqual(client.model_max_len(), 409600)

    def test_model_preference_and_auto_detect(self):
        with FakeVLLM() as fake:
            client = ChatClient(base_url="http://127.0.0.1:%d/v1" % fake.port)
            self.assertEqual(client.resolve_model("nonexistent/model"),
                             "azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ")
            client.model = None
            self.assertEqual(client.resolve_model(None),
                             "azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ")

    def test_chat_sends_thinking_flag_and_parses_message(self):
        with FakeVLLM() as fake:
            client = ChatClient(base_url="http://127.0.0.1:%d/v1" % fake.port, enable_thinking=False)
            client.model = "azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ"
            result = client.chat([{"role": "user", "content": "hi"}], max_tokens=16, seed=1)
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.content, "Hello there.")
            self.assertEqual(result.reasoning, "let me think")
            self.assertEqual(result.finish_reason, "stop")
            self.assertEqual(result.usage["prompt_tokens"], 10)
            payload = fake.payloads[-1]
            self.assertEqual(payload["model"], "azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ")
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(payload["max_tokens"], 16)
            self.assertEqual(payload["seed"], 1)

    def test_chat_thinking_on(self):
        with FakeVLLM() as fake:
            client = ChatClient(base_url="http://127.0.0.1:%d/v1" % fake.port, enable_thinking=True)
            client.model = "m"
            client.chat([{"role": "user", "content": "hi"}], max_tokens=16)
            self.assertEqual(fake.payloads[-1]["chat_template_kwargs"], {"enable_thinking": True})

    def test_tool_calls_are_parsed(self):
        with FakeVLLM() as fake:
            client = ChatClient(base_url="http://127.0.0.1:%d/v1" % fake.port)
            client.model = "m"
            tools = [{"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }}]
            result = client.chat([{"role": "user", "content": "weather in Rome"}],
                                 max_tokens=64, tools=tools)
            self.assertEqual(len(result.tool_calls), 1)
            call = result.tool_calls[0]
            self.assertEqual(call["name"], "get_weather")
            self.assertEqual(call["arguments"], {"city": "Rome", "unit": "celsius"})
            self.assertIsNone(call["parse_error"])
            self.assertEqual(fake.payloads[-1]["tool_choice"], "auto")

    def test_streaming_ttft_and_usage(self):
        with FakeVLLM() as fake:
            client = ChatClient(base_url="http://127.0.0.1:%d/v1" % fake.port)
            client.model = "m"
            result = client.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8)
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.content, "quasar-4419-zebra")
            self.assertIsNotNone(result.ttft_s)
            self.assertLessEqual(result.ttft_s, result.latency_s)
            self.assertEqual(result.usage["prompt_tokens"], 123)
            self.assertTrue(fake.payloads[-1]["stream_options"]["include_usage"])

    def test_streaming_4xx_is_client_error_not_retried(self):
        with FakeVLLM() as fake:
            # Fake rejects stream requests with 400 the same way vLLM rejects
            # an over-long prompt; we must fail the item, not abort the run.
            fake.stream_status = 400
            client = ChatClient(base_url="http://127.0.0.1:%d/v1" % fake.port,
                                retries=2, retry_backoff=0.01)
            client.model = "m"
            with self.assertRaises(ClientError):
                client.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8)
            self.assertEqual(len(fake.payloads), 1)  # no retries

    def test_retry_on_5xx(self):
        with FakeVLLM() as fake:
            fake.fail_next = 2
            client = ChatClient(base_url="http://127.0.0.1:%d/v1" % fake.port,
                                retries=2, retry_backoff=0.01)
            client.model = "m"
            result = client.chat([{"role": "user", "content": "hi"}], max_tokens=8)
            self.assertEqual(result.status, "ok")
            self.assertEqual(len(fake.payloads), 3)

    def test_http_error_is_reported(self):
        with FakeVLLM() as fake:
            client = ChatClient(base_url="http://127.0.0.1:%d/v1" % fake.port, retries=0)
            client.model = "m"
            # A 404 path is unreachable through chat(), so exercise the raw helper.
            with self.assertRaises(ClientError):
                client._request_json("GET", "/definitely-not-a-route")

    def test_unreachable_server(self):
        client = ChatClient(base_url="http://127.0.0.1:9/v1", retries=0, timeout=2)
        self.assertFalse(client.is_healthy(timeout=1))
        with self.assertRaises(ServerUnavailable):
            client.chat([{"role": "user", "content": "hi"}], max_tokens=4)

    def test_wait_ready_times_out(self):
        client = ChatClient(base_url="http://127.0.0.1:9/v1", retries=0, timeout=1)
        started = time.perf_counter()
        self.assertFalse(client.wait_ready(max_wait=0.6, poll_interval=0.2))
        self.assertLess(time.perf_counter() - started, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
