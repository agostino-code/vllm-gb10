#!/usr/bin/env python3
"""Replicate myllmbox's published measurement style on our box.

Their numbers: "code (TypeScript) c=1: 86.9 tok/s", "mixed c=1: 60.5 tok/s",
"thinking-on request: 56.5 thinking / 102.8 writing code", measured as averages of 3 runs
of one FULL answer, default recipe setting (thinking off unless stated).

We measure:
- per prompt: 3 full-answer runs at the server's sampled temperature (0.6) and at 0,
  streaming, decode tok/s = completion_tokens / (time between first and last delta),
  TTFT, and spec acceptance from /metrics deltas.
- long TypeScript code answer (~1024 tokens) and a mixed code+prose answer.
- a thinking-on run of the code prompt.

Usage: python3 matrix/pubcheck.py
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.request

BASE = "http://127.0.0.1:8000"

CODE_TS = (
    "Write a complete TypeScript implementation of a debounce utility for async functions. "
    "Include full type definitions, overloads, cancellation support, and a short usage example. "
    "Write the code and nothing else."
)
MIXED = (
    "First write a TypeScript function that debounces an async call, then explain in prose when "
    "debouncing is the wrong choice for a UI event handler. Include code and explanation."
)
CODE_THINK = (
    "Design and write a complete TypeScript implementation of a generic debounce utility with "
    "cancellation, then explain the tradeoffs versus throttle."
)


def http_json(url, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=20) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            pass
    return out


def run(model, prompt, max_tokens, temperature, thinking, label):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = None
    last = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
                    chunks += 1
    end = time.time()
    m_before = run._before
    m_after = metrics()
    keys = {
        "drafts": "vllm:spec_decode_num_drafts_total",
        "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
        "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    }
    delta = {k: m_after.get(v, 0.0) - m_before.get(v, 0.0) for k, v in keys.items()}
    run._before = m_after
    ct = (usage or {}).get("completion_tokens")
    decode_s = (last - first) if first else None
    toks = ct if ct else chunks
    return {
        "label": label,
        "completion_tokens": ct,
        "chunks": chunks,
        "ttft_s": (first - t0) if first else None,
        "total_s": end - t0,
        "decode_s": decode_s,
        "decode_tok_s": (toks / decode_s) if decode_s else None,
        "acceptance": (delta["accepted"] / delta["draft_tokens"]) if delta["draft_tokens"] else None,
        "tokens_per_step": ((delta["accepted"] + delta["drafts"]) / delta["drafts"]) if delta["drafts"] else None,
    }


def main():
    model = http_json(BASE + "/v1/models")["data"][0]["id"]
    print("model:", model, flush=True)
    # warmup
    run._before = metrics()
    run(model, "Say READY.", 8, 0.0, False, "warmup")
    cases = [
        ("code-ts T=0.6", CODE_TS, 1024, 0.6, False),
        ("code-ts T=0.0", CODE_TS, 1024, 0.0, False),
        ("mixed T=0.6", MIXED, 1024, 0.6, False),
        ("mixed T=0.0", MIXED, 1024, 0.0, False),
        ("code-think T=0.6", CODE_THINK, 2048, 0.6, True),
    ]
    all_rows = {}
    run._before = metrics()
    for label, prompt, mt, temp, thinking in cases:
        rows = []
        for i in range(3):
            r = run(model, prompt, mt, temp, thinking, label)
            rows.append(r)
            print(
                f"  {label} rep{i+1}: {r['completion_tokens']} tok in {r['decode_s']:.2f}s decode "
                f"= {r['decode_tok_s']:.1f} tok/s, ttft={r['ttft_s']*1000:.0f}ms, "
                f"accept={r['acceptance']:.3f} tok/step={r['tokens_per_step']:.2f}",
                flush=True,
            )
        dec = statistics.median([x["decode_tok_s"] for x in rows if x["decode_tok_s"]])
        ttft = statistics.median([x["ttft_s"] for x in rows if x["ttft_s"]])
        acc = statistics.median([x["acceptance"] for x in rows if x["acceptance"]])
        all_rows[label] = {"decode_tok_s": dec, "ttft_ms": ttft * 1000, "acceptance": acc, "runs": rows}
        print(f"  => {label}: median {dec:.1f} tok/s, ttft {ttft*1000:.0f} ms, accept {acc:.3f}\n", flush=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in all_rows.items()}, indent=2))
    open("bench/results/pubcheck.json", "w").write(
        json.dumps({"model": model, "results": all_rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
