#!/usr/bin/env python3
"""mock_vllm.py — tiny OpenAI-compatible mock server for harness self-tests.

It emulates just enough of vLLM to exercise bench/suite.py end-to-end without a
GPU: health/models endpoints, SSE chat completions with **multi-token chunks**
(speculative-decoding style), token usage, and a Prometheus /metrics endpoint
with the vLLM spec-decode counter names.

  python3 bench/tests/mock_vllm.py --port 18000            # with /metrics
  MOCK_NO_METRICS=1 python3 bench/tests/mock_vllm.py ...   # /metrics -> 404
  MOCK_NO_SPEC_COUNTERS=1 ...                              # metrics without spec_decode_*
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE_LOCK = threading.Lock()
STATE = {
    "spec_drafts": 0.0,
    "spec_draft_tokens": 0.0,
    "spec_accepted_tokens": 0.0,
    "prompt_tokens": 0.0,
    "generation_tokens": 0.0,
    "requests_success": 0.0,
    "running": 0.0,
    "total_spec": 12,
}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def metrics_text() -> str:
    with STATE_LOCK:
        s = dict(STATE)
    lines = [
        "# HELP vllm:num_requests_running Number of requests currently running",
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{engine="0",model_name="mock"}} {s["running"]}',
        f'vllm:num_requests_waiting{{engine="0",model_name="mock"}} 0.0',
        f'vllm:kv_cache_usage_perc{{engine="0",model_name="mock"}} 0.12',
        f'vllm:cache_config_info{{engine="0",kv_cache_size_tokens="750933",num_gpu_blocks="242",'
        f'block_size="3456",kv_cache_max_concurrency="1.83",enable_prefix_caching="True"}} 1.0',
        f'vllm:prompt_tokens_total{{engine="0",model_name="mock"}} {s["prompt_tokens"]}',
        f'vllm:generation_tokens_total{{engine="0",model_name="mock"}} {s["generation_tokens"]}',
        f'vllm:request_success_total{{engine="0",model_name="mock"}} {s["requests_success"]}',
    ]
    if not os.environ.get("MOCK_NO_SPEC_COUNTERS"):
        lines += [
            f'vllm:spec_decode_num_drafts_total{{engine="0",model_name="mock"}} {s["spec_drafts"]}',
            f'vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="mock"}} {s["spec_draft_tokens"]}',
            f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="mock"}} {s["spec_accepted_tokens"]}',
        ]
        for pos in range(4):
            lines.append(
                'vllm:spec_decode_num_accepted_tokens_per_pos_total'
                f'{{engine="0",model_name="mock",position="{pos}"}} '
                f'{max(0.0, s["spec_accepted_tokens"] * (0.8 - 0.2 * pos))}'
            )
    lines += [
        "# HELP vllm:time_to_first_token_seconds Time to first token",
        "# TYPE vllm:time_to_first_token_seconds histogram",
        'vllm:time_to_first_token_seconds_bucket{engine="0",model_name="mock",le="0.1"} 1.0',
        'vllm:time_to_first_token_seconds_bucket{engine="0",model_name="mock",le="0.5"} 3.0',
        'vllm:time_to_first_token_seconds_bucket{engine="0",model_name="mock",le="+Inf"} 5.0',
        'vllm:time_to_first_token_seconds_sum{engine="0",model_name="mock"} 1.2',
        'vllm:time_to_first_token_seconds_count{engine="0",model_name="mock"} 5.0',
    ]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    tokens_per_step = 3  # simulate speculative decoding: ~3 tokens per SSE chunk

    def log_message(self, *args):  # silence
        pass

    def _send(self, status: int, body: bytes, ctype: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, b"OK", "text/plain")
        elif self.path == "/v1/models":
            self._send(200, json.dumps({"data": [{"id": "mock/model-AZ", "object": "model"}]}).encode())
        elif self.path == "/metrics":
            if os.environ.get("MOCK_NO_METRICS"):
                self._send(404, b"no metrics", "text/plain")
            else:
                self._send(200, metrics_text().encode(), "text/plain; version=0.0.4")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:  # noqa: BLE001
            self._send(400, b'{"error":"bad json"}')
            return
        content = " ".join(m.get("content", "") for m in body.get("messages", []))
        prompt_tokens = estimate_tokens(content)
        max_tokens = int(body.get("max_tokens") or 32)
        ignore_eos = bool(body.get("ignore_eos"))
        seed = int(body.get("seed") or 0)
        # deterministic "natural stop": some prompts end early unless ignore_eos
        completion = max_tokens
        if not ignore_eos:
            completion = min(max_tokens, 48 + (len(content) + seed) % 80)
        completion = max(1, completion)
        with STATE_LOCK:
            STATE["running"] += 1.0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write_chunk(payload: dict):
            data = ("data: " + json.dumps(payload) + "\n\n").encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        try:
            write_chunk({"id": "mock", "object": "chat.completion.chunk", "model": "mock/model-AZ",
                         "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            time.sleep(0.03)  # prefill
            produced = 0
            steps = 0
            while produced < completion:
                batch = min(self.tokens_per_step, completion - produced)
                text = ("token " * batch)
                produced += batch
                steps += 1
                write_chunk({"id": "mock", "object": "chat.completion.chunk", "model": "mock/model-AZ",
                             "choices": [{"index": 0, "delta": {"content": text},
                                          "finish_reason": None if produced < completion else "length"}]})
                time.sleep(0.02)
            with STATE_LOCK:
                STATE["spec_drafts"] += steps
                STATE["spec_draft_tokens"] += steps * STATE["total_spec"]
                STATE["spec_accepted_tokens"] += max(0, produced - steps)
                STATE["prompt_tokens"] += prompt_tokens
                STATE["generation_tokens"] += produced
                STATE["requests_success"] += 1.0
                STATE["running"] = max(0.0, STATE["running"] - 1.0)
            write_chunk({"id": "mock", "object": "chat.completion.chunk", "model": "mock/model-AZ",
                         "choices": [], "usage": {"prompt_tokens": prompt_tokens,
                                                  "completion_tokens": produced,
                                                  "total_tokens": prompt_tokens + produced,
                                                  "prompt_tokens_details": {"cached_tokens": 0},
                                                  "completion_tokens_details": {"reasoning_tokens": 0}}})
            self.wfile.write(b"0\r\n\r\n")  # end chunked stream
        except (BrokenPipeError, ConnectionResetError):
            with STATE_LOCK:
                STATE["running"] = max(0.0, STATE["running"] - 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tokens-per-step", type=int, default=3)
    args = parser.parse_args()
    Handler.tokens_per_step = args.tokens_per_step
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock vLLM on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
