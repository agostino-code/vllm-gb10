#!/usr/bin/env python3
"""Soak test: keep C concurrent streaming requests running for N seconds and report
error rate, throughput drift (first half vs second half), and spec acceptance delta.

Usage: python3 matrix/soak.py --base http://127.0.0.1:8000 --seconds 480 --concurrency 8 \
           --out bench/results/<run>/soak.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

PROMPTS = [
    "Write a complete Python implementation of a red-black tree with insert, delete, search and traversal.",
    "Explain the CAP theorem with one example system for each pair of tradeoffs. Be specific.",
    "Write a TypeScript function that debounces an async call, then explain when debouncing is wrong.",
    "Implement a Python LRU cache from scratch with O(1) get/put, type hints and tests.",
    "Generate a JSON array of 12 employees with name, age, department, salary, email, skills.",
    "Write a Python trie with insert, search, startsWith, delete and autocomplete plus unit tests.",
    "Compare REST, GraphQL and gRPC for an internal service mesh. Give concrete tradeoffs.",
    "Write a Python HTTP server from scratch using only the socket module (GET/POST/routing).",
]


def http_json(url, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def metrics(base):
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=20) as r:
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


def stream_chat(base, model, prompt, max_tokens, temperature, thinking):
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
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    tokens = 0
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
                    if ttft is None:
                        ttft = time.time() - t0
                    tokens += 1
    t_end = time.time()
    comp = (usage or {}).get("completion_tokens") or tokens
    return {"tokens": comp, "ttft": ttft, "duration": t_end - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--seconds", type=float, default=480)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    model = http_json(base + "/v1/models")["data"][0]["id"]
    m_before = metrics(base)
    deadline = time.time() + args.seconds
    lock = threading.Lock()
    results = []  # (t_start, tokens, duration, err)
    errors = []

    def worker(i):
        j = 0
        while time.time() < deadline:
            p = PROMPTS[(i + j) % len(PROMPTS)]
            t0 = time.time()
            try:
                r = stream_chat(base, model, p, args.max_tokens, args.temperature, args.thinking)
                with lock:
                    results.append((t0, r["tokens"], r["duration"], None, r["ttft"]))
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(str(exc))
                    results.append((t0, 0, time.time() - t0, str(exc), None))
            j += 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t_start
    m_after = metrics(base)

    ok = [r for r in results if r[3] is None]
    total_tokens = sum(r[1] for r in ok)
    half = t_start + args.seconds / 2
    first = sum(r[1] for r in ok if r[0] < half) / max(1e-9, (half - t_start))
    second = sum(r[1] for r in ok if r[0] >= half) / max(1e-9, (t_start + args.seconds - half))
    keys = {"drafts": "vllm:spec_decode_num_drafts_total", "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
            "accepted": "vllm:spec_decode_num_accepted_tokens_total", "generated": "vllm:generation_tokens_total"}
    delta = {k: m_after.get(v, 0.0) - m_before.get(v, 0.0) for k, v in keys.items()}
    out = {
        "model": model,
        "concurrency": args.concurrency,
        "seconds_requested": args.seconds,
        "wall_s": wall,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "requests_ok": len(ok),
        "requests_failed": len(errors),
        "errors_sample": errors[:5],
        "total_completion_tokens": total_tokens,
        "throughput_tok_s": total_tokens / wall,
        "first_half_tok_s": first,
        "second_half_tok_s": second,
        "drift_pct": (second - first) / first * 100 if first else None,
        "ttft_ms_median": statistics.median([r[4] * 1000 for r in ok if r[4]]) if ok else None,
        "ttft_ms_p90": sorted([r[4] * 1000 for r in ok if r[4]])[int(0.9 * len(ok)) - 1] if ok else None,
        "duration_s_median": statistics.median([r[2] for r in ok]) if ok else None,
        "spec": {
            "drafts": delta["drafts"],
            "draft_tokens": delta["draft_tokens"],
            "accepted": delta["accepted"],
            "acceptance_rate": (delta["accepted"] / delta["draft_tokens"]) if delta["draft_tokens"] else None,
            "tokens_per_step": ((delta["accepted"] + delta["drafts"]) / delta["drafts"]) if delta["drafts"] else None,
        },
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
