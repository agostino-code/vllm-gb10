#!/usr/bin/env python3
"""Summarize eugr-b12x bench runs (bench/results/eugr-b12x-*/summary.json).

Usage: python3 eugr-b12x/summarize.py [--results-dir bench/results]
Writes eugr-b12x/runs/RESULTS-eugr-b12x.md and prints a compact table.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def dig(d, *path, default=None):
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, list) and isinstance(key, int) and -len(cur) <= key < len(cur):
            cur = cur[key]
        else:
            return default
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(PROJECT / "bench" / "results"))
    ap.add_argument("--out", default=str(HERE / "runs" / "RESULTS-eugr-b12x.md"))
    args = ap.parse_args()

    results = Path(args.results_dir)
    rows = []
    for run_dir in sorted(results.glob("eugr-b12x-*")):
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            rows.append({"run_id": run_dir.name, "status": "no summary.json"})
            continue
        try:
            s = json.loads(summary_path.read_text())
        except Exception as exc:  # noqa: BLE001
            rows.append({"run_id": run_dir.name, "status": f"bad json: {exc}"})
            continue

        levels = {}
        for lvl in dig(s, "phases", "concurrency", "levels", default=[]) or []:
            if isinstance(lvl, dict) and "c" in lvl:
                levels[int(lvl["c"])] = lvl
        meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").is_file() else {}

        boot_s = dig(s, "server", "boot_seconds")
        short = run_dir.name.replace("eugr-b12x-", "")
        for cand in (HERE / "runs" / f"{short}-summary.txt",
                     HERE / "runs" / f"eugr-{short}-summary.txt"):
            if boot_s in (None, 0) and cand.is_file():
                for line in cand.read_text().splitlines():
                    if line.startswith("boot_and_health_seconds="):
                        try:
                            boot_s = int(line.split("=", 1)[1])
                        except ValueError:
                            pass
        row = {
            "run_id": run_dir.name,
            "status": s.get("status") or "ok",
            "boot_s": boot_s,
            "decode_tok_s": dig(s, "phases", "decode", "out_tok_s_median"),
            "decode_p50": dig(s, "phases", "decode", "out_tok_s", "p50"),
            "ttft_ms_p50": dig(s, "phases", "decode", "ttft_ms_p50"),
            "tpot_ms_p50": dig(s, "phases", "decode", "tpot_ms_p50"),
            "acceptance": dig(s, "metrics", "overall", "acceptance_rate"),
            "tokens_per_step": dig(s, "metrics", "overall", "tokens_per_step"),
            "avg_draft_depth": dig(s, "metrics", "overall", "avg_draft_depth"),
            "c1": dig(levels.get(1, {}), "total_out_tok_s"),
            "c2": dig(levels.get(2, {}), "total_out_tok_s"),
            "c4": dig(levels.get(4, {}), "total_out_tok_s"),
            "c8": dig(levels.get(8, {}), "total_out_tok_s"),
            "c1_per_stream": dig(levels.get(1, {}), "per_stream_tok_s", "median"),
            "prefill_1k": dig(s, "phases", "prefill", "by_target", "1024", "prefill_tok_s"),
            "prefill_8k": dig(s, "phases", "prefill", "by_target", "8192", "prefill_tok_s"),
            "spec_config": dig(s, "server", "vllm", "speculative_config"),
            "kv_tokens": dig(s, "server", "kv_cache", "tokens"),
            "num_spec": dig(s, "server", "vllm", "num_speculative_tokens"),
            "errors": len(s.get("errors") or []),
            "warnings": len(s.get("warnings") or []),
        }
        rows.append(row)

    def fmt(v, nd=1):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    headers = [
        ("run_id", "run"),
        ("status", "status"),
        ("num_spec", "spec_n"),
        ("boot_s", "boot_s"),
        ("decode_tok_s", "dec tok/s"),
        ("decode_p50", "dec p50"),
        ("ttft_ms_p50", "ttft ms"),
        ("tpot_ms_p50", "tpot ms"),
        ("acceptance", "accept"),
        ("tokens_per_step", "tok/step"),
        ("c1", "c1 tok/s"),
        ("c2", "c2"),
        ("c4", "c4"),
        ("c8", "c8"),
        ("prefill_1k", "prefill1k"),
        ("prefill_8k", "prefill8k"),
        ("kv_tokens", "kv_tokens"),
        ("errors", "err"),
    ]
    lines = [
        "| " + " | ".join(h[1] for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for r in rows:
        cells = []
        for key, _ in headers:
            v = r.get(key)
            if key == "acceptance" and isinstance(v, float):
                v = f"{v * 100:.1f}%"
            cells.append(fmt(v))
        lines.append("| " + " | ".join(cells) + " |")

    out = [
        "# eugr b12x — benchmark results (azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ)",
        "",
        "Generated by `eugr-b12x/summarize.py` from `bench/results/eugr-b12x-*/summary.json`.",
        "",
        *lines,
        "",
        "Notes:",
        "- `dec tok/s` = phases.decode.out_tok_s_median (single stream, code-ish decode).",
        "- `accept` / `tok/step` come from Prometheus spec-decode counters (/metrics).",
        "- `c1/c2/c4/c8` = total output tok/s per concurrency level (multi-request waves).",
        "- `boot_s` = measured container start → /health (launch.sh meta.txt).",
        "",
    ]
    text = "\n".join(out)
    Path(args.out).write_text(text)
    print(text)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
