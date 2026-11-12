#!/usr/bin/env python3
"""Quick lead-side probe: single-stream decode + small concurrency + spec acceptance.

This is a sanity/spot check, not the official suite (bench/suite.py owns that).
Usage: python3 matrix/probe.py [--base http://127.0.0.1:8000] [--c4-seconds 20]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

CODE_PROMPT = (
    "Write a complete Python implementation of a red-black tree with insert, delete, "
    "search, and in-order traversal. Include all rotation methods."
)
MIXED_PROMPT = (
    "First write a TypeScript function that debounces an async call, then explain in prose "
    "when debouncing is the wrong choice."
)


def http_json(url, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def stream_chat(base, model, prompt, max_tokens=512, thinking=False):
    """Returns dict with ttft, tokens, total, tpot list, tok/s."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    t_tokens = []
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as r:
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
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    t_tokens.append(now)
    t_end = time.time()
    n = len(t_tokens)
    # inter-token deltas from the 2nd token onward
    tpot = [b - a for a, b in zip(t_tokens, t_tokens[1:])]
    return {
        "prompt": prompt[:40],
        "out_tokens": n,
        "usage_completion": (usage or {}).get("completion_tokens"),
        "usage_prompt": (usage or {}).get("prompt_tokens"),
        "ttft": ttft,
        "total_s": t_end - t0,
        "decode_s": (t_end - t_tokens[0]) if t_tokens else None,
        "tok_s": (n / (t_end - t_tokens[0])) if n and t_tokens else None,
        "tpot_p50": statistics.median(tpot) if tpot else None,
    }


def metrics(base):
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=20) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        out[name] = out.get(name, 0.0) + val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--c4-seconds", type=float, default=20.0)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    model = http_json(base + "/v1/models")["data"][0]["id"]
    print(f"model: {model}")

    before = metrics(base)
    # warmup
    stream_chat(base, model, "Say READY.", max_tokens=8)
    print("warmup done")

    # 3 single-stream reps
    reps = []
    for i in range(3):
        r = stream_chat(base, model, CODE_PROMPT, max_tokens=512)
        reps.append(r)
        print(
            f"  code c=1 rep{i+1}: {r['out_tokens']} tok in {r['total_s']:.2f}s "
            f"ttft={r['ttft']*1000:.0f}ms tok/s={r['tok_s']:.1f} tpot_p50={r['tpot_p50']*1000:.1f}ms"
        )
    med = statistics.median([r["tok_s"] for r in reps if r["tok_s"]])
    ttft_med = statistics.median([r["ttft"] for r in reps if r["ttft"]])
    print(f"  => code c=1 median tok/s={med:.1f} ttft={ttft_med*1000:.0f}ms")

    r = stream_chat(base, model, MIXED_PROMPT, max_tokens=512)
    print(f"  mixed c=1: {r['out_tokens']} tok, tok/s={r['tok_s']:.1f}, ttft={r['ttft']*1000:.0f}ms")

    # concurrency 4, ~20s
    prompts = [
        CODE_PROMPT,
        MIXED_PROMPT,
        "Explain the CAP theorem and give one example system for each pair of tradeoffs.",
        "Write a Python LRU cache from scratch with O(1) get/put, including type hints.",
        "Generate a JSON array of 10 employees with name, age, department, salary, email.",
        "Write a complete Python trie with insert, search, startsWith, delete, autocomplete.",
        "Compare REST, GraphQL and gRPC for an internal microservice mesh. Be specific.",
        "Implement quicksort with a random pivot in Python plus complexity analysis.",
    ]
    n_streams = 4
    results = []
    lock = threading.Lock()
    stop_at = time.time() + args.c4_seconds

    def worker(i):
        count_tokens = 0
        latencies = []
        while time.time() < stop_at:
            p = prompts[i % len(prompts)]
            t0 = time.time()
            try:
                r = stream_chat(base, model, p, max_tokens=256)
            except Exception as exc:  # noqa: BLE001
                print(f"  stream{i} error: {exc}")
                break
            if r["out_tokens"]:
                count_tokens += r["out_tokens"]
                latencies.append(time.time() - t0)
            with lock:
                results.append((i, count_tokens, sum(latencies)))

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0
    total = sum(tokens for _, tokens, _ in results)
    print(f"  c={n_streams}: {total} tokens in {elapsed:.1f}s = {total/elapsed:.1f} tok/s total")

    after = metrics(base)
    keys = {
        "drafts": "vllm:spec_decode_num_drafts_total",
        "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
        "accepted": "vllm:spec_decode_num_accepted_tokens_total",
        "emitted": "vllm:spec_decode_num_emitted_tokens_total",
        "generated": "vllm:generation_tokens_total",
    }
    delta = {k: after.get(v, 0.0) - before.get(v, 0.0) for k, v in keys.items()}
    if delta["draft_tokens"]:
        acc = delta["accepted"] / delta["draft_tokens"]
        steps = delta["drafts"]
        tps = delta["emitted"] / steps if steps else None
        print(
            f"  spec: drafts={delta['drafts']:.0f} draft_tokens={delta['draft_tokens']:.0f} "
            f"accepted={delta['accepted']:.0f} acceptance={acc*100:.1f}% "
            f"tokens/step={tps:.2f}"
        )
    print(f"  generated tokens (whole probe): {delta['generated']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
