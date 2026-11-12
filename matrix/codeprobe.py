#!/usr/bin/env python3
"""Code-content probe: compare single-stream throughput on code-only full answers.

The study's screening metric was a cross-category median at 256 tokens (dominated by prose).
myllmbox's published numbers are per-prompt full answers: code/JSON get much higher draft
acceptance. This probe measures the same code prompts for both configs.
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
PROMPTS = {
    "py-code": (
        "Write a complete, production-quality Python module that implements a red-black tree with "
        "insert, delete, search, range query and in-order traversal, plus a full unit-test suite. "
        "Output only code, no explanation, no markdown fences."
    ),
    "ts-code": (
        "Write a complete TypeScript implementation of a debounce utility for async functions. "
        "Include full type definitions, overloads, cancellation support, and a short usage example. "
        "Write the code and nothing else."
    ),
}


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=20).read().decode()
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        n = line.split("{")[0].split(" ")[0]
        try:
            out[n] = out.get(n, 0.0) + float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            pass
    return out


def run(model, prompt, max_tokens=2048):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    m0 = metrics()
    t0 = time.time()
    first = last = None
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                obj = json.loads(p)
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
    m1 = metrics()
    acc = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    dt = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    dr = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    ct = (usage or {}).get("completion_tokens")
    return {
        "tokens": ct,
        "ttft_ms": (first - t0) * 1000 if first else None,
        "tok_s": ct / (last - first) if first and ct and last > first else None,
        "accept": acc / dt if dt else None,
        "tok_step": (acc + dr) / dr if dr else None,
    }


def main():
    model = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=20).read())["data"][0]["id"]
    out = {"model": model, "prompts": {}}
    # warmup
    run(model, "Say READY.", 8)
    for name, prompt in PROMPTS.items():
        rows = [run(model, prompt) for _ in range(3)]
        for i, r in enumerate(rows):
            print(f"  {name} rep{i+1}: {r['tokens']} tok, {r['tok_s']:.1f} tok/s, "
                  f"ttft {r['ttft_ms']:.0f} ms, accept {r['accept']:.3f}, tok/step {r['tok_step']:.2f}", flush=True)
        med = statistics.median([r["tok_s"] for r in rows if r["tok_s"]])
        out["prompts"][name] = {"median_tok_s": med, "runs": rows}
        print(f"  => {name}: median {med:.1f} tok/s", flush=True)
    print(json.dumps({k: v["median_tok_s"] for k, v in out["prompts"].items()}))
    with open("/tmp/code-probe.json", "w") as f:
        f.write(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
