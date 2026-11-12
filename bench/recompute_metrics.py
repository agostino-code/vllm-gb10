#!/usr/bin/env python3
"""recompute_metrics.py — recompute Prometheus spec-decode deltas for a
finished run from the raw snapshots saved in ``metrics/*.txt`` and patch
``summary.json``.

This is useful when the metric parsing improved after a run, or when a run was
made with ``--no-metrics`` and raw files were created later.  It is also what
aggregate.py uses implicitly: ``summary.json`` is always the source of truth,
but the recomputation is exact because the raw text is preserved.

  python3 bench/recompute_metrics.py --run-dir bench/results/m0-reference-full
  python3 bench/recompute_metrics.py --results-dir bench/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def load_metrics_dir(run_dir: Path) -> dict[str, dict]:
    """label -> parsed metrics dict from metrics/<label>.txt."""
    out: dict[str, dict] = {}
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        return out
    for path in sorted(metrics_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out[path.stem] = C.parse_metrics_text(text)
    return out


def client_token_sums(run_dir: Path) -> dict[str, dict]:
    """phase -> {n_ok, n_errors, prompt_tokens, completion_tokens} from requests.jsonl."""
    path = run_dir / "requests.jsonl"
    sums: dict[str, dict] = {}
    if not path.is_file():
        return sums
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        phase = row.get("phase") or "?"
        bucket = sums.setdefault(phase, {"n_ok": 0, "n_errors": 0,
                                         "prompt_tokens": 0, "completion_tokens": 0})
        if row.get("ok"):
            bucket["n_ok"] += 1
            bucket["prompt_tokens"] += row.get("prompt_tokens") or 0
            bucket["completion_tokens"] += row.get("completion_tokens") or 0
        else:
            bucket["n_errors"] += 1
    return sums


def kv_from_cache_info(parsed: dict) -> dict:
    samples = parsed.get("vllm:cache_config_info")
    if not samples:
        return {}
    labels = samples[-1].get("labels") or {}
    info: dict = {}
    for key, cast in (
        ("kv_cache_size_tokens", int),
        ("num_gpu_blocks", int),
        ("block_size", int),
        ("kv_cache_max_concurrency", float),
        ("kv_cache_memory_bytes", int),
    ):
        if key in labels and labels[key] not in ("None", ""):
            try:
                info[key] = cast(labels[key])
            except Exception:  # noqa: BLE001
                info[key] = labels[key]
    return info


def recompute_run(run_dir: Path, write: bool = True) -> dict | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw = load_metrics_dir(run_dir)
    if not raw:
        return None
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
    num_spec = C.spec_num_tokens((meta.get("vllm") or {}).get("argv") or []) \
        or C.spec_num_tokens((((summary.get("server") or {}).get("vllm") or {}).get("argv")) or [])
    snapshot_label_prefixes = {name: name for name in (summary.get("phases") or {}).keys()}

    def snapshot_after(label: str) -> dict:
        parsed = raw.get(label)
        return C.spec_decode_snapshot(parsed) if parsed else {}

    metrics = summary.setdefault("metrics", {})
    per_phase = {}
    for phase in summary.get("phases") or {}:
        if phase == "warmup":
            continue
        before = raw.get(f"{phase}_before")
        after = raw.get(f"{phase}_after")
        if before is None or after is None:
            continue
        per_phase[phase] = C.spec_decode_delta(
            C.spec_decode_snapshot(before), C.spec_decode_snapshot(after), num_spec)
    # server-side Prometheus token deltas vs client accounting (also detects
    # external traffic that would make a run non-exclusive)
    sums = client_token_sums(run_dir)
    warnings = summary.setdefault("warnings", [])
    for phase, delta in per_phase.items():
        before = raw.get(f"{phase}_before")
        after = raw.get(f"{phase}_after")
        duration = (summary.get("phases", {}).get(phase) or {}).get("duration_s")

        def total(parsed, name):
            return C.metric_total(parsed, name) if parsed else None

        def delta_of(name):
            first, last = total(before, name), total(after, name)
            return None if (first is None or last is None) else last - first

        server_generation = delta_of("vllm:generation_tokens_total")
        server_prompt = delta_of("vllm:prompt_tokens_total")
        entry = dict(delta)
        entry["server_generation_tokens"] = server_generation
        entry["server_prompt_tokens"] = server_prompt
        entry["server_generation_tok_s"] = (round(server_generation / duration, 3)
                                            if server_generation is not None and duration else None)
        phase_sums = sums.get(phase) or {}
        if phase in ("decode", "concurrency"):
            client_tokens = phase_sums.get("completion_tokens")
            server_tokens = server_generation
            kind = "completion"
        else:
            client_tokens = phase_sums.get("prompt_tokens")
            server_tokens = server_prompt
            kind = "prompt"
        if client_tokens is not None and server_tokens:
            ratio = round(client_tokens / server_tokens, 4)
            key = "client_to_server_token_ratio" if kind == "completion" else "client_to_server_prompt_ratio"
            entry[key] = ratio
            entry["client_token_kind"] = kind
            if not (0.75 <= ratio <= 1.35):
                msg = (f"{phase}: client {kind} tokens {client_tokens} vs server counter "
                       f"{server_tokens:.0f} (ratio {ratio:.2f}) — external traffic or accounting mismatch")
                if msg not in warnings:
                    warnings.append(msg)
        per_phase[phase] = entry
        phase_summary = summary.get("phases", {}).get(phase)
        if isinstance(phase_summary, dict):
            phase_summary["server_generation_tokens"] = server_generation
            phase_summary["server_generation_tok_s"] = entry["server_generation_tok_s"]
            phase_summary["server_prompt_tokens"] = server_prompt
            for key in ("client_to_server_token_ratio", "client_to_server_prompt_ratio"):
                if key in entry:
                    phase_summary[key] = entry[key]
    if per_phase:
        metrics["per_phase"] = per_phase

    initial = raw.get("initial")
    final = raw.get("final")
    if initial is not None and final is not None:
        overall = C.spec_decode_delta(
            C.spec_decode_snapshot(initial), C.spec_decode_snapshot(final), num_spec)
        overall["num_speculative_tokens_configured"] = num_spec
        metrics["overall"] = overall

    # kv cache config + usage series from raw snapshots
    if initial is not None:
        cache = kv_from_cache_info(initial)
        server = summary.setdefault("server", {})
        kv = server.setdefault("kv_cache", {})
        if cache.get("kv_cache_size_tokens"):
            kv["tokens"] = cache["kv_cache_size_tokens"]
        if cache.get("num_gpu_blocks"):
            kv["num_gpu_blocks"] = cache["num_gpu_blocks"]
        if cache.get("block_size"):
            kv["block_size"] = cache["block_size"]
        if cache.get("kv_cache_max_concurrency"):
            kv["max_concurrency"] = cache["kv_cache_max_concurrency"]
    usages = []
    for label, parsed in raw.items():
        usage = C.kv_cache_usage(parsed)
        if usage is not None:
            usages.append({"phase": label, "kv_cache_usage_perc": usage})
    if usages:
        metrics["kv_cache_usage"] = usages
        metrics["peak_kv_cache_usage_perc"] = max(u["kv_cache_usage_perc"] for u in usages)

    summary.setdefault("harness", {})["recomputed_at"] = C.now_iso()
    if write:
        tmp = summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", default=[], help="specific results/<run-id> dir")
    parser.add_argument("--results-dir", action="append", default=[], help="scan all runs under this dir")
    parser.add_argument("--no-write", action="store_true", help="print only, do not patch summary.json")
    args = parser.parse_args(argv)
    roots = [Path(p) for p in args.results_dir]
    runs = [Path(p) for p in args.run_dir]
    if not roots and not runs:
        parser.error("pass --run-dir and/or --results-dir")
    for root in roots:
        for path in sorted(root.glob("*/summary.json")):
            runs.append(path.parent)
    updated = 0
    for run_dir in dict.fromkeys(runs):
        summary = recompute_run(run_dir, write=not args.no_write)
        if summary is None:
            C.log(f"{run_dir}: nothing to recompute (no metrics/*.txt or no summary.json)")
            continue
        overall = summary.get("metrics", {}).get("overall") or {}
        C.log(f"{C.repo_rel(run_dir)}: acceptance={overall.get('acceptance_rate')} "
              f"tokens/step={overall.get('tokens_per_step')} avg_depth={overall.get('avg_draft_depth')}")
        updated += 1
    C.log(f"recomputed {updated} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
