#!/usr/bin/env python3
"""suite.py — standardized, reproducible measurement suite for an
OpenAI-compatible vLLM server.

One command runs a complete config measurement (warmup → decode → concurrency →
prefill → optional 128k needle smoke → Prometheus spec-decode accounting) and
writes a machine-readable run directory::

    results/<run_id>/
        meta.json          launch manifest (written by serve.py up/adopt)
        summary.json       aggregated, aggregation-friendly results
        requests.jsonl     one line per HTTP request (timings, tokens, errors)
        metrics/*.txt      raw Prometheus snapshots per phase
        server.log         server log copied by serve.py (if available)
        quality.json       optional quality-hook output

Profiles: ``quick`` (≤ ~4-5 min) and ``full`` (≤ ~10 min incl. 128k smoke).

Examples
--------
  python3 bench/suite.py --base-url http://127.0.0.1:8000 \\
      --run-dir bench/results/m0-reference-full --config-id m0-reference --profile full

  python3 bench/suite.py --base-url http://127.0.0.1:8000 --profile quick \\
      --concurrency 1,2,4 --decode-max-tokens 128
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import prompts as P  # noqa: E402

SUITE_VERSION = "1.0"

PROFILES: dict[str, dict] = {
    "quick": {
        "repeats": 2,
        "decode_max_tokens": 192,
        "decode_prompts_per_category": 1,
        "concurrency": [1, 4],
        "conc_waves": 1,
        "conc_max_tokens": 160,
        "prefill_targets": [1024, 8192],
        "long_smoke": False,
        "warmup_requests": 2,
    },
    "full": {
        "repeats": 3,
        "decode_max_tokens": 256,
        "decode_prompts_per_category": 2,
        "concurrency": [1, 2, 4, 8],
        "conc_waves": 2,
        "conc_max_tokens": 192,
        "prefill_targets": [1024, 8192, 32768],
        "long_smoke": True,
        "warmup_requests": 2,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def parse_int_list(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def light_metrics_snapshot(base_url: str, timeout: float = 5.0) -> dict:
    """Cheap /metrics poll for the background sampler (no raw text retained)."""
    status, text = C.http_text(C.url_join(base_url, "/metrics"), timeout=timeout)
    if status != 200 or not text:
        return {}
    parsed = C.parse_metrics_text(text)
    return {
        "kv_usage": C.kv_cache_usage(parsed),
        "running": C.metric_gauge(parsed, "vllm:num_requests_running"),
        "waiting": C.metric_gauge(parsed, "vllm:num_requests_waiting"),
    }


class MetricsSampler:
    """Background sampler for peak KV-cache usage / running requests."""

    def __init__(self, base_url: str, interval: float = 0.5):
        self.base_url = base_url
        self.interval = interval
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.samples: list[dict] = []
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="metrics-sampler", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                snap = light_metrics_snapshot(self.base_url)
            except Exception:  # noqa: BLE001
                continue
            if snap:
                with self.lock:
                    self.samples.append(snap)

    def stop(self) -> dict:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        with self.lock:
            samples = list(self.samples)
        peaks = {
            "n_samples": len(samples),
            "peak_kv_cache_usage_perc": None,
            "peak_num_requests_running": None,
            "peak_num_requests_waiting": None,
        }
        if samples:
            def peak(key):
                values = [s[key] for s in samples if s.get(key) is not None]
                return max(values) if values else None

            peaks["peak_kv_cache_usage_perc"] = peak("kv_usage")
            peaks["peak_num_requests_running"] = peak("running")
            peaks["peak_num_requests_waiting"] = peak("waiting")
        return peaks


def request_with_retry(base_url: str, body: dict, timeout: float, retries: int) -> tuple[C.StreamingResult, list[dict]]:
    """Run a streaming chat request, retrying network/HTTP failures once."""
    attempts: list[dict] = []
    result = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            result = C.stream_chat(base_url, body, timeout=timeout)
            error = result.get("error")
        except Exception as exc:  # noqa: BLE001 - stream_chat should not raise, but be safe
            result = C.StreamingResult(
                ok=False, status="error", error=f"{type(exc).__name__}: {exc}", n_tokens=0,
                n_content_chunks=0, ttft_s=None, duration_s=time.perf_counter() - started,
                decode_tok_s=None, tpot_ms=None, e2e_tok_s=None, usage=None,
                chunk_at=[], itl_ms=[], text="", finish_reason=None, token_count_source="none",
            )
            error = result["error"]
        attempts.append({"attempt": attempt + 1, "ok": result["ok"], "error": error,
                         "http_status": result.get("http_status")})
        if result["ok"] and (result.get("n_tokens") or 0) > 0:
            break
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    assert result is not None
    return result, attempts


def result_record(phase: str, result: C.StreamingResult, attempts: list[dict], **extra) -> dict:
    record = {
        "ts": C.now_iso(),
        "phase": phase,
        "ok": bool(result["ok"] and (result.get("n_tokens") or 0) > 0),
        "attempts": len(attempts),
        "http_status": result.get("http_status"),
        "error": result.get("error"),
        "ttft_ms": round(result["ttft_s"] * 1000.0, 3) if result.get("ttft_s") is not None else None,
        "duration_s": round(result["duration_s"], 4) if result.get("duration_s") is not None else None,
        "completion_tokens": result.get("n_tokens"),
        "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
        "cached_tokens": ((result.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "reasoning_tokens": ((result.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "decode_tok_s": round(result["decode_tok_s"], 3) if result.get("decode_tok_s") else None,
        "e2e_tok_s": round(result["e2e_tok_s"], 3) if result.get("e2e_tok_s") else None,
        "tpot_ms": round(result["tpot_ms"], 3) if result.get("tpot_ms") else None,
        "n_chunks": result.get("n_content_chunks"),
        "finish_reason": result.get("finish_reason"),
        "token_count_source": result.get("token_count_source"),
    }
    n_chunks = result.get("n_content_chunks")
    n_tokens = result.get("n_tokens")
    record["tokens_per_chunk"] = round(n_tokens / n_chunks, 3) if n_chunks and n_tokens else None
    itl = result.get("itl_ms") or []
    record["chunk_itl_ms_median"] = round(C.percentile(itl, 50), 3) if itl else None
    record.update(extra)
    return record


def phase_stats(records: list[dict], key: str) -> dict:
    values = [r[key] for r in records if r.get("ok") and isinstance(r.get(key), (int, float))]
    return C.stat_block(values)


def phase_status(records: list[dict]) -> str:
    if not records:
        return "failed"
    ok = sum(1 for r in records if r.get("ok"))
    if ok == 0:
        return "failed"
    if ok < len(records):
        return "partial"
    return "ok"


def client_server_consistency(ctx: SuiteContext, phase: dict, phase_end: dict,
                              client_tokens: int | None, what: str = "completion") -> float | None:
    """Compare client-counted tokens with the server's Prometheus counter delta."""
    server_tokens = phase_end.get("server_generation_tokens") if what == "completion" else phase_end.get("server_prompt_tokens")
    if not server_tokens or client_tokens is None:
        return None
    ratio = client_tokens / server_tokens if server_tokens else None
    if ratio is not None and not (0.75 <= ratio <= 1.35):
        ctx.warnings.append(
            f"{phase['name']}: client {what} tokens ({client_tokens}) vs server counter "
            f"({server_tokens:.0f}) ratio {ratio:.2f} — possible external traffic or "
            f"accounting mismatch (multi-token chunks are counted from usage)"
        )
    return round(ratio, 4) if ratio is not None else None


def log_request(record: dict) -> None:
    phase = record.get("phase", "?")
    detail = ""
    if phase == "decode":
        detail = f"{record.get('prompt_id')} rep={record.get('rep')}"
    elif phase == "concurrency":
        detail = f"c={record.get('concurrency')} wave={record.get('wave')}"
    elif phase == "prefill":
        detail = f"target={record.get('target_tokens')} tok"
    elif phase == "long_context":
        detail = f"target={record.get('target_tokens')} tok"
    if record.get("ok"):
        ttft = record.get("ttft_ms")
        rate = record.get("decode_tok_s") or record.get("prefill_tok_s")
        C.log(f"  ✓ {phase:<13} {detail:<26} ttft={ttft:8.1f}ms tok/s={rate if rate else '-'}")
    else:
        C.log(f"  ✗ {phase:<13} {detail:<26} ERROR: {str(record.get('error'))[:140]}")


# ──────────────────────────────────────────────────────────────────────────────
# suite context
# ──────────────────────────────────────────────────────────────────────────────


class SuiteContext:
    def __init__(self, args, run_dir: C.RunDir, model: str, meta: dict):
        self.args = args
        self.run_dir = run_dir
        self.model = model
        self.meta = meta
        self.errors: list[dict] = []
        self.warnings: list[str] = []
        self.metrics_snapshots: dict[str, dict] = {}
        self.phase_metrics: dict[str, dict] = {}
        self.suite_started = time.perf_counter()

    def body(self, prompt_text: str, max_tokens: int, **overrides) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
            "temperature": self.args.temperature,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "seed": self.args.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(self.args.thinking)},
        }
        body.update(overrides)
        return body

    def take_metrics(self, label: str) -> dict:
        snap = C.snapshot_metrics(self.args.base_url, timeout=self.args.metrics_timeout)
        if not snap.get("ok"):
            self.warnings.append(f"metrics snapshot '{label}' unavailable: {snap.get('error')}")
            self.metrics_snapshots[label] = {"ok": False}
            return {"ok": False, "spec_decode": {}, "raw": ""}
        self.run_dir.write_text(f"metrics/{label}.txt", snap["raw"])
        self.metrics_snapshots[label] = {
            "ok": True,
            "cache_config": cache_info_from_metrics(snap),
            "spec_decode": snap["spec_decode"],
            "kv_cache_usage_perc": snap["kv_cache_usage_perc"],
            "num_requests_running": snap["num_requests_running"],
            "num_requests_waiting": snap["num_requests_waiting"],
            "prompt_tokens_total": snap["prompt_tokens_total"],
            "generation_tokens_total": snap["generation_tokens_total"],
            "request_success_total": snap["request_success_total"],
        }
        return snap

    def metrics_delta(self, before: dict, after: dict) -> dict:
        return C.spec_decode_delta(
            before.get("spec_decode", {}), after.get("spec_decode", {}),
            C.spec_num_tokens(self.meta.get("vllm", {}).get("argv") or []) if self.meta else None,
        )

    def begin_phase(self, name: str) -> dict:
        snapshot = self.take_metrics(f"{name}_before")
        return {"name": name, "started_at": C.now_iso(), "t0": time.perf_counter(), "before": snapshot}

    def end_phase(self, phase: dict, snapshot_after: bool = True) -> dict:
        after = self.take_metrics(f"{phase['name']}_after") if snapshot_after else {"ok": False, "spec_decode": {}}
        delta = self.metrics_delta(phase["before"], after)
        self.phase_metrics[phase["name"]] = delta
        duration = round(time.perf_counter() - phase["t0"], 3)
        before = phase["before"]

        def _delta(key):
            first, last = before.get(key), after.get(key)
            return None if (first is None or last is None) else last - first

        # Server-side token counters validate the client-side accounting (and
        # reveal traffic from other clients hitting the same server).
        server_generation = _delta("generation_tokens_total")
        server_prompt = _delta("prompt_tokens_total")
        return {
            "started_at": phase["started_at"],
            "duration_s": duration,
            "spec_decode": delta,
            "kv_cache_usage_perc_after": after.get("kv_cache_usage_perc"),
            "server_prompt_tokens": server_prompt,
            "server_generation_tokens": server_generation,
            "server_generation_tok_s": (round(server_generation / duration, 3)
                                        if server_generation is not None and duration > 0 else None),
        }


def _request(ctx: SuiteContext, prompt_text: str, max_tokens: int, timeout: float,
             phase: str, extra: dict, quiet: bool = False, enrich=None):
    """Run one measured request, append its record to requests.jsonl, return it.

    ``enrich(record, result)`` may add phase-specific fields (they land in the
    same single jsonl line, never a duplicate).
    """
    body = ctx.body(prompt_text, max_tokens)
    result, attempts = request_with_retry(ctx.args.base_url, body, timeout=timeout,
                                          retries=ctx.args.retries)
    record = result_record(phase, result, attempts, **extra)
    if enrich is not None:
        enrich(record, result)
    ctx.run_dir.append_jsonl(record)
    if not quiet:
        log_request(record)
    if not record["ok"]:
        ctx.errors.append({"phase": phase, "detail": extra, "error": record.get("error"),
                           "attempts": len(attempts)})
    return result, attempts, record


# ──────────────────────────────────────────────────────────────────────────────
# phases
# ──────────────────────────────────────────────────────────────────────────────


def phase_warmup(ctx: SuiteContext) -> dict:
    n = ctx.args.warmup_requests
    records = []
    for i in range(n):
        prompt = P.DECODE_PROMPTS[i % len(P.DECODE_PROMPTS)]
        text = f"[nonce {P.nonce(ctx.args.run_id, 'warmup', i)}]\n{prompt['prompt']}"
        result, attempts, _rec = _request(ctx, text, min(64, ctx.args.decode_max_tokens),
                                          ctx.args.request_timeout, "warmup",
                                          {"prompt_id": prompt["id"]}, quiet=True)
        records.append(result_record("warmup", result, attempts))
    return {"status": phase_status(records), "requests": len(records),
            "ok": sum(1 for r in records if r["ok"])}


def phase_decode(ctx: SuiteContext) -> dict:
    per_category = ctx.args.decode_prompts_per_category
    prompts: list[dict] = []
    seen: dict[str, int] = {}
    for prompt in P.DECODE_PROMPTS:
        seen[prompt["category"]] = seen.get(prompt["category"], 0) + 1
        if seen[prompt["category"]] <= per_category:
            prompts.append(prompt)
    repeats = ctx.args.repeats
    C.log(f"decode: {len(prompts)} prompts × {repeats} repeats (max_tokens={ctx.args.decode_max_tokens})")
    phase = ctx.begin_phase("decode")
    records: list[dict] = []
    index = 0
    for rep in range(1, repeats + 1):
        for prompt in prompts:
            nonce = P.nonce(ctx.args.run_id, f"decode-r{rep}", index)
            index += 1
            text = f"[nonce {nonce}]\n{prompt['prompt']}"
            result, attempts, _rec = _request(
                ctx, text, ctx.args.decode_max_tokens, ctx.args.request_timeout, "decode",
                {"prompt_id": prompt["id"], "category": prompt["category"], "rep": rep},
            )
            records.append(result_record("decode", result, attempts,
                                         prompt_id=prompt["id"], category=prompt["category"], rep=rep))
            time.sleep(ctx.args.inter_request_delay)

    ok_records = [r for r in records if r["ok"]]
    by_category: dict[str, dict] = {}
    for category in sorted({p["category"] for p in prompts}):
        subset = [r for r in ok_records if r.get("category") == category]
        by_category[category] = {
            "n": len(subset),
            "out_tok_s": phase_stats(subset, "decode_tok_s"),
            "ttft_ms": phase_stats(subset, "ttft_ms"),
            "tpot_ms": phase_stats(subset, "tpot_ms"),
        }
    ttft = phase_stats(ok_records, "ttft_ms")
    tpot = phase_stats(ok_records, "tpot_ms")
    out = phase_stats(ok_records, "decode_tok_s")
    e2e = phase_stats(ok_records, "e2e_tok_s")
    cached = sum(r.get("cached_tokens") or 0 for r in ok_records)
    prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in ok_records)
    if cached and len(ok_records) and cached > 0.05 * max(1, prompt_tokens):
        ctx.warnings.append(
            f"decode phase had {cached} cached prompt tokens (prefix-cache hits); TTFT may be optimistic"
        )
    phase_end = ctx.end_phase(phase)
    completion_total = sum(r.get("completion_tokens") or 0 for r in ok_records)
    ratio = client_server_consistency(ctx, phase, phase_end, completion_total)
    return {
        "status": phase_status(records),
        "n_requests": len(records),
        "n_ok": len(ok_records),
        "n_errors": len(records) - len(ok_records),
        "out_tok_s": out,
        "out_tok_s_median": out["median"],
        "ttft_ms": ttft,
        "ttft_ms_p50": ttft["p50"],
        "tpot_ms": tpot,
        "tpot_ms_p50": tpot["p50"],
        "e2e_tok_s": e2e,
        "tokens_per_chunk": phase_stats(ok_records, "tokens_per_chunk"),
        "chunk_itl_ms": phase_stats(ok_records, "chunk_itl_ms_median"),
        "completion_tokens_sum": sum(r.get("completion_tokens") or 0 for r in ok_records),
        "prompt_tokens_sum": prompt_tokens,
        "cached_tokens_sum": cached,
        "by_category": by_category,
        "client_to_server_token_ratio": ratio,
        "prompts": [p["id"] for p in prompts],
        "repeats": repeats,
        **phase_end,
    }


def _conc_worker(ctx: SuiteContext, c: int, wave: int, worker_index: int, max_tokens: int,
                 barrier: threading.Barrier, timeout: float) -> tuple[C.StreamingResult, list[dict], dict]:
    prompt = P.CONCURRENCY_PROMPTS[(wave * c + worker_index) % len(P.CONCURRENCY_PROMPTS)]
    nonce = P.nonce(ctx.args.run_id, f"conc-c{c}-w{wave}", worker_index)
    text = f"[nonce {nonce}]\n{prompt['prompt']}"
    meta = {"prompt_id": prompt["id"], "category": prompt["category"]}
    try:
        barrier.wait(timeout=30)
    except threading.BrokenBarrierError:
        pass
    started = time.perf_counter()
    body = ctx.body(text, max_tokens)
    result, attempts = request_with_retry(ctx.args.base_url, body, timeout=timeout, retries=ctx.args.retries)
    return result, attempts, {**meta, "started": started, "ended": time.perf_counter()}


def phase_concurrency(ctx: SuiteContext) -> dict:
    levels = list(ctx.args.concurrency)
    if ctx.args.include_16 and 16 not in levels:
        levels.append(16)
    levels = sorted(set(levels))
    waves = ctx.args.conc_waves
    C.log(f"concurrency: levels={levels} waves={waves} max_tokens={ctx.args.conc_max_tokens}")
    phase = ctx.begin_phase("concurrency")
    sampler = MetricsSampler(ctx.args.base_url, interval=0.5)
    sampler.start()
    level_rows: list[dict] = []
    all_records: list[dict] = []
    peak_cache: dict[str, dict] = {}
    try:
        for c in levels:
            level_records: list[dict] = []
            wave_rows: list[dict] = []
            for wave in range(waves):
                barrier = threading.Barrier(c)
                wave_start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=c, thread_name_prefix=f"c{c}") as pool:
                    futures = [
                        pool.submit(_conc_worker, ctx, c, wave, i,
                                    ctx.args.conc_max_tokens, barrier, ctx.args.request_timeout)
                        for i in range(c)
                    ]
                    outputs = [f.result() for f in futures]
                wave_end = time.perf_counter()
                tokens = 0
                errors = 0
                for result, attempts, meta in outputs:
                    record = result_record("concurrency", result, attempts, concurrency=c, wave=wave,
                                           prompt_id=meta["prompt_id"], category=meta["category"])
                    record["wave_duration_s"] = round(wave_end - wave_start, 4)
                    ctx.run_dir.append_jsonl(record)
                    level_records.append(record)
                    all_records.append(record)
                    tokens += record.get("completion_tokens") or 0
                    if not record["ok"]:
                        errors += 1
                        ctx.errors.append({"phase": "concurrency", "detail": {"c": c, "wave": wave},
                                           "error": record.get("error"), "attempts": record.get("attempts")})
                wave_wall = wave_end - wave_start
                wave_rows.append({
                    "wave": wave,
                    "wall_s": round(wave_wall, 4),
                    "completion_tokens": tokens,
                    "total_out_tok_s": round(tokens / wave_wall, 3) if wave_wall > 0 else None,
                    "errors": errors,
                })
                C.log(f"  c={c:<2} wave={wave}: {tokens} tok / {wave_wall:.2f}s = "
                      f"{tokens / wave_wall if wave_wall > 0 else 0:.1f} tok/s ({errors} errors)")
            ok = [r for r in level_records if r["ok"]]
            total_wall = sum(w["wall_s"] for w in wave_rows)
            total_tokens = sum(w["completion_tokens"] for w in wave_rows)
            level_rows.append({
                "c": c,
                "n_requests": len(level_records),
                "n_ok": len(ok),
                "n_errors": len(level_records) - len(ok),
                "waves": wave_rows,
                "wall_s": round(total_wall, 4),
                "completion_tokens": total_tokens,
                "total_out_tok_s": round(total_tokens / total_wall, 3) if total_wall > 0 else None,
                "per_stream_tok_s": C.stat_block([r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]),
                "ttft_ms": C.stat_block([r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]),
                "tpot_ms": C.stat_block([r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None]),
            })
    finally:
        peak_cache = sampler.stop()
    phase_end = ctx.end_phase(phase)
    ok_all = [r for r in all_records if r["ok"]]
    completion_total = sum(r.get("completion_tokens") or 0 for r in ok_all)
    ratio = client_server_consistency(ctx, phase, phase_end, completion_total)
    return {
        "status": phase_status(all_records),
        "client_to_server_token_ratio": ratio,
        "levels": level_rows,
        "peaks": peak_cache,
        "max_total_out_tok_s": max((row["total_out_tok_s"] or 0) for row in level_rows) if level_rows else None,
        "n_requests": len(all_records),
        "n_ok": len(ok_all),
        "n_errors": len(all_records) - len(ok_all),
        "ttft_ms": C.stat_block([r["ttft_ms"] for r in ok_all if r.get("ttft_ms") is not None]),
        **phase_end,
    }


def phase_prefill(ctx: SuiteContext) -> dict:
    targets = list(ctx.args.prefill_targets)
    C.log(f"prefill: targets={targets} tokens (max_tokens=1, cold via nonce)")
    phase = ctx.begin_phase("prefill")
    records: list[dict] = []
    samples: list[dict] = []
    def _prefill_enrich(record, _result):
        prompt_tokens = record.get("prompt_tokens")
        ttft_ms = record.get("ttft_ms")
        record["prefill_tok_s"] = (
            round(prompt_tokens / (ttft_ms / 1000.0), 2)
            if prompt_tokens and ttft_ms and record["ok"] else None
        )

    for target in targets:
        nonce = P.nonce(ctx.args.run_id, f"prefill-{target}", 0)
        text = P.make_long_prompt(target, nonce)
        timeout = ctx.args.prefill_timeout
        result, attempts, record = _request(ctx, text, 1, timeout, "prefill",
                                            {"target_tokens": target}, enrich=_prefill_enrich)
        prompt_tokens = record.get("prompt_tokens")
        ttft_ms = record.get("ttft_ms")
        records.append(record)
        samples.append({
            "target_tokens": target,
            "ok": record["ok"],
            "prompt_tokens": prompt_tokens,
            "ttft_ms": ttft_ms,
            "prefill_tok_s": record.get("prefill_tok_s"),
            "cached_tokens": record.get("cached_tokens"),
            "error": record.get("error"),
        })
        if record["ok"]:
            C.log(f"  ✓ prefill target={target:<6} prompt_tokens={prompt_tokens} "
                  f"ttft={ttft_ms:.0f}ms prefill={record['prefill_tok_s']} tok/s")
        else:
            C.log(f"  ✗ prefill target={target}: {str(record.get('error'))[:160]}")
    phase_end = ctx.end_phase(phase)
    prompt_total = sum(s.get("prompt_tokens") or 0 for s in samples if s.get("ok"))
    ratio = client_server_consistency(ctx, phase, phase_end, prompt_total, what="prompt")
    by_target = {str(s["target_tokens"]): s for s in samples}
    return {
        "status": phase_status(records),
        "samples": samples,
        "by_target": by_target,
        "client_to_server_prompt_ratio": ratio,
        "prefill_tok_s": C.stat_block([s["prefill_tok_s"] for s in samples
                                       if s["ok"] and s.get("prefill_tok_s")]),
        "ttft_ms": C.stat_block([s["ttft_ms"] for s in samples if s["ok"] and s.get("ttft_ms")]),
        **phase_end,
    }


def phase_long_context(ctx: SuiteContext) -> dict:
    target = ctx.args.long_target_tokens
    code = P.needle_code(ctx.args.run_id)
    nonce = P.nonce(ctx.args.run_id, "long", 0)
    text = P.make_long_prompt(target, nonce, needle_code=code)
    C.log(f"long-context smoke: ~{target} tokens, needle at ~45% depth, max_tokens=32")
    phase = ctx.begin_phase("long_context")

    def _long_enrich(record, result):
        prompt_tokens = record.get("prompt_tokens")
        ttft_ms = record.get("ttft_ms")
        record["prefill_tok_s"] = (
            round(prompt_tokens / (ttft_ms / 1000.0), 2)
            if prompt_tokens and ttft_ms and record["ok"] else None
        )
        answer = (result.get("text") or "").strip()
        record["expected_code"] = code
        record["needle_ok"] = code.lower() in answer.lower()
        record["answer_head"] = answer[:200]

    result, attempts, record = _request(ctx, text, 32, ctx.args.long_timeout, "long_context",
                                        {"target_tokens": target}, enrich=_long_enrich)
    prompt_tokens = record.get("prompt_tokens")
    ttft_ms = record.get("ttft_ms")
    answer = (result.get("text") or "").strip()
    needle_ok = record.get("needle_ok")
    phase_end = ctx.end_phase(phase)
    if record["ok"]:
        C.log(f"  ✓ long {target} tok prompt_tokens={prompt_tokens} ttft={ttft_ms:.0f}ms "
              f"prefill={record['prefill_tok_s']} tok/s needle={'OK' if needle_ok else 'MISS'} "
              f"answer={answer[:60]!r}")
        if not needle_ok:
            ctx.warnings.append(f"128k needle not found in answer (expected {code})")
    else:
        C.log(f"  ✗ long context: {str(record.get('error'))[:160]}")
    ratio = client_server_consistency(ctx, phase, phase_end, prompt_tokens, what="prompt")
    return {
        "status": "ok" if record["ok"] else "failed",
        "target_tokens": target,
        "needle_ok": needle_ok if record["ok"] else None,
        "expected_code": code,
        "prompt_tokens": prompt_tokens,
        "ttft_ms": ttft_ms,
        "prefill_tok_s": record.get("prefill_tok_s"),
        "decode_tok_s": record.get("decode_tok_s"),
        "client_to_server_prompt_ratio": ratio,
        "answer_head": answer[:400],
        "error": record.get("error"),
        "sample": record,
        **phase_end,
    }


# ──────────────────────────────────────────────────────────────────────────────
# server metadata / metrics summary / quality hook
# ──────────────────────────────────────────────────────────────────────────────


def server_section(ctx: SuiteContext, model: str, models: list[str]) -> dict:
    meta = ctx.meta or {}
    startup = {}
    log_path = Path(ctx.args.server_log) if ctx.args.server_log else ctx.run_dir.path / "server.log"
    if log_path.is_file():
        startup = C.analyze_server_log(log_path)
    elif meta.get("logs", {}).get("startup"):
        startup = meta["logs"]["startup"]

    cache_info: dict = (ctx.metrics_snapshots.get("initial") or {}).get("cache_config") or {}
    kv_cache = {
        "tokens": cache_info.get("kv_cache_size_tokens") or startup.get("kv_cache_tokens"),
        "num_gpu_blocks": cache_info.get("num_gpu_blocks") or startup.get("gpu_blocks"),
        "block_size": cache_info.get("block_size") or startup.get("block_size"),
        "max_concurrency": cache_info.get("kv_cache_max_concurrency") or startup.get("kv_cache_max_concurrency"),
        "memory_gib": startup.get("kv_cache_memory_gib"),
        "max_num_seqs": startup.get("max_num_seqs") or meta.get("vllm", {}).get("max_num_seqs"),
        "max_model_len": startup.get("max_model_len") or meta.get("vllm", {}).get("max_model_len"),
        "source": "metrics:cache_config_info+startup_log" if cache_info else "startup_log",
    }
    launch = meta.get("launch") or {}
    stack = meta.get("stack") or {}
    timing = meta.get("timing") or {}
    return {
        "base_url": ctx.args.base_url,
        "model": model,
        "models_available": models,
        "meta_present": bool(meta),
        "run_id_from_meta": meta.get("run_id"),
        "image": stack.get("image"),
        "image_id": stack.get("image_id"),
        "image_repo_digests": stack.get("repo_digests"),
        "stack": stack.get("stack"),
        "recipe": meta.get("recipe"),
        "container": meta.get("container"),
        "boot_seconds": timing.get("boot_seconds"),
        "vllm": meta.get("vllm"),
        "launch": {
            "env": launch.get("env"),
            "mounts": launch.get("mounts"),
            "docker_shell": launch.get("shell"),
        } if launch else None,
        "startup": startup,
        "kv_cache": kv_cache,
        "apply": {
            "prefix_caching": (meta.get("vllm", {}).get("argv") or []),
        },
    }


def cache_info_from_metrics(snapshot: dict) -> dict:
    parsed = (snapshot or {}).get("parsed") or {}
    samples = parsed.get("vllm:cache_config_info")
    if not samples:
        return {}
    labels = samples[-1].get("labels") or {}
    out: dict = {"raw_labels": labels}
    for key, cast in (
        ("kv_cache_size_tokens", int),
        ("num_gpu_blocks", int),
        ("block_size", int),
        ("kv_cache_max_concurrency", float),
        ("kv_cache_memory_bytes", int),
        ("enable_prefix_caching", lambda v: str(v).lower() == "true"),
        ("mamba_cache_dtype", str),
        ("cache_dtype", str),
        ("use_recoverssm", lambda v: str(v).lower() == "true"),
        ("replayssm_buffer_len", int),
        ("kv_cache_layout", str),
    ):
        if key in labels and labels[key] not in ("None", ""):
            try:
                out[key] = cast(labels[key])
            except Exception:  # noqa: BLE001
                out[key] = labels[key]
    return out


def metrics_section(ctx: SuiteContext) -> dict:
    initial = ctx.metrics_snapshots.get("initial") or {}
    final = ctx.metrics_snapshots.get("final") or {}
    num_spec = C.spec_num_tokens((ctx.meta.get("vllm", {}).get("argv") if ctx.meta else None) or [])
    overall = ctx.metrics_delta(
        {"spec_decode": initial.get("spec_decode", {})},
        {"spec_decode": final.get("spec_decode", {})},
    )
    overall["num_speculative_tokens_configured"] = num_spec
    kv_usage = []
    for label, snap in ctx.metrics_snapshots.items():
        if isinstance(snap, dict) and snap.get("kv_cache_usage_perc") is not None:
            kv_usage.append({"phase": label, "kv_cache_usage_perc": snap["kv_cache_usage_perc"]})
    return {
        "overall": overall,
        "per_phase": ctx.phase_metrics,
        "kv_cache_usage": kv_usage,
        "peak_kv_cache_usage_perc": max((p["kv_cache_usage_perc"] for p in kv_usage), default=None),
        "metric_error": None if initial.get("ok") or final.get("ok") else "no /metrics snapshots",
    }


def run_quality_hook(ctx: SuiteContext, mode: str, model: str) -> dict | None:
    if mode in ("off", "none", None, ""):
        return None
    script = C.REPO_ROOT / "quality" / "run_quality.py"
    if not script.is_file():
        return {"status": "unavailable", "reason": f"{C.repo_rel(script)} not found"}
    out_json = ctx.run_dir.path / "quality.json"
    base = [sys.executable, str(script)]
    variants = [
        base + ["--base-url", ctx.args.base_url, "--model", model, "--mode", mode,
                "--output", str(out_json)],
        base + ["--quick", "--base-url", ctx.args.base_url, "--model", model,
                "--output", str(out_json)],
        base + ["--mode", mode, "--url", ctx.args.base_url, "--model", model,
                "--out", str(out_json)],
    ]
    last = {}
    for variant in variants:
        C.log(f"quality hook: {' '.join(variant[1:])}")
        try:
            proc = C.run_cmd(variant, timeout=ctx.args.quality_timeout)
        except Exception as exc:  # noqa: BLE001
            last = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            continue
        record = {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "command": " ".join(variant),
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "mode": mode,
            "output_file": str(out_json) if out_json.is_file() else None,
        }
        if out_json.is_file():
            try:
                record["scores"] = json.loads(out_json.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                record["scores"] = None
        last = record
        if proc.returncode == 0 or out_json.is_file():
            break
        text = ((proc.stderr or "") + (proc.stdout or "")).lower()
        if "unrecognized argument" not in text and "invalid choice" not in text and "usage:" not in text:
            break
    return last


# ──────────────────────────────────────────────────────────────────────────────
# orchestration
# ──────────────────────────────────────────────────────────────────────────────


def build_summary(args, ctx: SuiteContext, model: str, models: list[str], phases: dict,
                  timings: dict, server: dict, metrics: dict, quality: dict | None) -> dict:
    initial_kv = ctx.metrics_snapshots.get("initial") or {}
    summary = {
        "schema_version": 1,
        "suite_version": SUITE_VERSION,
        "run_id": args.run_id,
        "config_id": args.config_id or args.run_id,
        "tag": args.tag,
        "generated_at": C.now_iso(),
        "profile": args.profile,
        "harness": {
            "suite_version": SUITE_VERSION,
            "command": args.command_line,
            "python": sys.version.split()[0],
            "base_url": args.base_url,
        },
        "suite_args": {
            "temperature": args.temperature,
            "seed": args.seed,
            "thinking": args.thinking,
            "repeats": args.repeats,
            "decode_max_tokens": args.decode_max_tokens,
            "decode_prompts_per_category": args.decode_prompts_per_category,
            "concurrency": args.concurrency,
            "conc_waves": args.conc_waves,
            "conc_max_tokens": args.conc_max_tokens,
            "include_16": args.include_16,
            "prefill_targets": args.prefill_targets,
            "long_smoke": args.long_smoke,
            "long_target_tokens": args.long_target_tokens,
            "request_timeout_s": args.request_timeout,
            "retries": args.retries,
            "quality": args.quality,
        },
        "server": server,
        "phases": phases,
        "metrics": metrics,
        "quality": quality,
        "errors": ctx.errors,
        "warnings": ctx.warnings,
        "timing": {**timings, "duration_s": round(time.perf_counter() - ctx.suite_started, 3)},
        "phase_timing": {name: {"duration_s": p.get("duration_s")} for name, p in phases.items()},
    }
    # flat convenience keys for quick /metrics checks from other tools
    summary["kv_cache_usage_initial_perc"] = initial_kv.get("kv_cache_usage_perc")
    return summary


def print_report(summary: dict, run_dir: C.RunDir) -> None:
    phases = summary["phases"]
    decode = phases.get("decode", {})
    conc = phases.get("concurrency", {})
    prefill = phases.get("prefill", {})
    long_ctx = phases.get("long_context")
    metrics = summary.get("metrics", {}).get("overall", {})
    print()
    print("═" * 78)
    print(f"SUITE {summary['run_id']}  (profile {summary['profile']}, config {summary['config_id']})")
    print(f"  model : {summary['server'].get('model')}")
    image = summary["server"].get("image")
    if image:
        print(f"  image : {image} ({str(summary['server'].get('image_id'))[:19]}…)")
    if summary["server"].get("boot_seconds") is not None:
        print(f"  boot  : {summary['server']['boot_seconds']}s")
    if decode:
        print(f"  decode: {decode.get('out_tok_s_median'):.1f} tok/s median  "
              f"TTFT p50 {decode.get('ttft_ms_p50')} ms  TPOT p50 {decode.get('tpot_ms_p50')} ms  "
              f"({decode.get('n_ok')}/{decode.get('n_requests')} ok)")
    if conc:
        for row in conc.get("levels") or []:
            print(f"  conc C={row['c']:<2}: {row.get('total_out_tok_s')} tok/s total, "
                  f"{row['per_stream_tok_s'].get('median')} tok/s/stream "
                  f"({row.get('n_ok')}/{row.get('n_requests')} ok)")
    if prefill:
        for sample in prefill.get("samples") or []:
            if sample["ok"]:
                print(f"  prefill {sample['target_tokens']:<6} tok: prompt_tokens={sample['prompt_tokens']:<7}"
                      f" TTFT={sample['ttft_ms']:.0f} ms  {sample['prefill_tok_s']} tok/s")
    if long_ctx:
        print(f"  long {long_ctx.get('target_tokens')} tok: TTFT={long_ctx.get('ttft_ms')} ms  "
              f"prefill={long_ctx.get('prefill_tok_s')} tok/s  needle="
              f"{'OK' if long_ctx.get('needle_ok') else 'MISS'}")
    if metrics.get("acceptance_rate") is not None:
        print(f"  spec  : acceptance {metrics['acceptance_rate']:.3f}  "
              f"tokens/step {metrics.get('tokens_per_step')}  "
              f"(draft_tokens={metrics.get('draft_tokens')}, accepted={metrics.get('accepted_tokens')})")
    else:
        print("  spec  : acceptance unavailable (no spec-decode counters moved)")
    quality = summary.get("quality")
    if quality:
        print(f"  quality: {quality.get('status')} {quality.get('scores') if quality.get('scores') else ''}")
    if summary.get("warnings"):
        for warning in summary["warnings"]:
            print(f"  ⚠ {warning}")
    print(f"  result: {run_dir.path / 'summary.json'}")
    print(f"  total : {summary['timing']['duration_s']:.1f}s")
    print("═" * 78)


def add_profile_defaults(args) -> None:
    profile = PROFILES[args.profile]
    for key, value in profile.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-dir", help="results/<run-id> directory (created if missing)")
    parser.add_argument("--run-id", help="run id when --run-dir is not given")
    parser.add_argument("--config-id", help="config id recorded in summary.json")
    parser.add_argument("--tag", default=None, help="free-form label for aggregation")
    parser.add_argument("--model", help="served model name (default: first from /v1/models)")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--server-log", help="server log to parse for KV pool/concurrency")
    parser.add_argument("--quality", choices=["off", "quick", "full"], default="off",
                        help="invoke ./quality/run_quality.py if present")
    parser.add_argument("--quality-timeout", type=float, default=900.0)
    parser.add_argument("--strict-quality", action="store_true")

    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--decode-max-tokens", type=int, default=None)
    parser.add_argument("--decode-prompts-per-category", type=int, default=None)
    parser.add_argument("--concurrency", type=parse_int_list, default=None,
                        help="explicit levels, e.g. --concurrency 1,2,4")
    parser.add_argument("--include-16", action="store_true",
                        help="append C=16 to the concurrency sweep (opt-in: can hang low-headroom configs)")
    parser.add_argument("--conc-waves", type=int, default=None)
    parser.add_argument("--conc-max-tokens", type=int, default=None)
    parser.add_argument("--prefill-targets", type=parse_int_list, default=None)
    parser.add_argument("--long-smoke", dest="long_smoke", action="store_true", default=None)
    parser.add_argument("--no-long-smoke", dest="long_smoke", action="store_false")
    parser.add_argument("--long-target-tokens", type=int, default=131072)
    parser.add_argument("--long-timeout", type=float, default=900.0)
    parser.add_argument("--warmup-requests", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", action="store_true", help="leave thinking enabled (default: off)")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--prefill-timeout", type=float, default=600.0)
    parser.add_argument("--metrics-timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=1, help="retries after a failed request")
    parser.add_argument("--inter-request-delay", type=float, default=0.05)
    parser.add_argument("--strict", action="store_true", help="non-zero exit if any request failed")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.command_line = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    add_profile_defaults(args)
    if args.concurrency is None:
        args.concurrency = list(PROFILES[args.profile]["concurrency"])
    if args.run_dir:
        run_dir = C.RunDir(Path(args.run_dir))
        run_id = args.run_id or run_dir.path.name
    else:
        run_id = args.run_id or C.default_run_id("suite")
        run_dir = C.RunDir(C.RESULTS_DIR / run_id)
    args.run_id = run_id
    args.config_id = args.config_id or run_id
    C.ensure_dir(run_dir.path / "metrics")

    C.log(f"suite {SUITE_VERSION} run_id={run_id} profile={args.profile} base={args.base_url}")
    # preflight
    if not C.health_ok(args.base_url):
        C.log(f"ERROR: {args.base_url}/health is not healthy — nothing measured")
        return 3
    models = C.list_models(args.base_url)
    if not models:
        C.log("ERROR: /v1/models returned no models")
        return 3
    model = args.model or models[0]
    if args.model and args.model not in models:
        C.log(f"WARNING: --model {args.model} not in /v1/models {models}; using it anyway")

    meta = run_dir.read_json("meta.json", default={}) or {}
    ctx = SuiteContext(args, run_dir, model, meta)
    timings: dict = {"started_at": C.now_iso()}
    phases: dict[str, dict] = {}

    ctx.take_metrics("initial")
    C.log("warmup (discarded)")
    phases["warmup"] = phase_warmup(ctx)

    def run_phase(name: str, fn) -> None:
        C.log(f"phase: {name}")
        try:
            phases[name] = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - a phase crash must not kill the suite
            tb = traceback.format_exc()
            ctx.errors.append({"phase": name, "detail": {}, "error": f"{type(exc).__name__}: {exc}",
                               "traceback": tb})
            phases[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                            "traceback": tb}
        C.log(f"phase {name}: {phases[name].get('status')} "
              f"({phases[name].get('duration_s')}s)")

    run_phase("decode", phase_decode)
    run_phase("concurrency", phase_concurrency)
    run_phase("prefill", phase_prefill)
    if args.long_smoke:
        run_phase("long_context", phase_long_context)
    ctx.take_metrics("final")
    timings["finished_at"] = C.now_iso()

    quality = run_quality_hook(ctx, args.quality, model)
    if args.strict_quality and quality and quality.get("status") != "ok":
        ctx.errors.append({"phase": "quality", "detail": {}, "error": "quality hook failed"})

    server = server_section(ctx, model, models)
    metrics = metrics_section(ctx)
    summary = build_summary(args, ctx, model, models, phases, timings, server, metrics, quality)
    run_dir.write_json("summary.json", summary)
    run_dir.write_json("suite.json", {
        "suite_version": SUITE_VERSION,
        "run_id": run_id,
        "profile": args.profile,
        "started_at": timings.get("started_at"),
        "finished_at": timings.get("finished_at"),
        "command": args.command_line,
        "base_url": args.base_url,
        "model": model,
    })
    print_report(summary, run_dir)

    failed = [name for name, phase in phases.items()
              if name not in ("warmup", "long_context") and phase.get("status") == "failed"]
    if failed:
        C.log(f"FAILED phases: {failed}")
        return 2
    if args.strict and ctx.errors:
        C.log(f"strict mode: {len(ctx.errors)} request errors")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
