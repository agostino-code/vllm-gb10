#!/usr/bin/env python3
"""aggregate.py — merge every finished run under bench/results/ into one
ranked comparison: summary.csv + RESULTS.md (+ summary_all.json).

Reads only ``results/<run-id>/summary.json`` (and ``status.json`` when present);
never contacts the server and never touches the GPU, so it is safe to run at any
time, including while a matrix is booting.

  python3 bench/aggregate.py                       # bench/results -> csv + md
  python3 bench/aggregate.py --results-dir matrix/results --md matrix/RESULTS.md
  python3 bench/aggregate.py --run-dir bench/results/m0-reference-full --no-md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COLUMNS = [
    "run_id", "config_id", "tag", "status", "profile",
    "stack", "image", "image_id_short", "recipe",
    "mtp", "spec_method", "kv_cache_dtype", "max_num_seqs", "max_model_len",
    "boot_s",
    "decode_tok_s", "ttft_ms_p50", "tpot_ms_p50", "tokens_per_chunk",
    "acceptance_rate", "tokens_per_step", "avg_draft_depth",
    "accept_pos0", "accept_pos1", "accept_pos2", "accept_pos3",
    "step_ms", "steps_per_s",
    "delta_decode_pct", "delta_c4_pct", "delta_c8_pct", "delta_accept_pp",
    "decode_ratio", "conc_ratio", "prefill_ratio",
    "quality_score",
    "conc_1", "conc_2", "conc_4", "conc_8", "conc_16",
    "prefill_1k_tok_s", "prefill_8k_tok_s", "prefill_32k_tok_s",
    "prefill_32k_ttft_ms", "long_ttft_ms", "long_prefill_tok_s", "long_needle_ok",
    "kv_tokens", "kv_max_concurrency",
    "n_errors", "n_warnings", "duration_s", "flags", "generated_at", "summary_path",
]

NUMERIC_COLUMNS = {
    "mtp", "max_num_seqs", "max_model_len", "boot_s", "decode_tok_s", "ttft_ms_p50",
    "tpot_ms_p50", "tokens_per_chunk", "acceptance_rate", "tokens_per_step",
    "avg_draft_depth", "conc_1", "conc_2", "conc_4", "conc_8", "conc_16",
    "accept_pos0", "accept_pos1", "accept_pos2", "accept_pos3",
    "step_ms", "steps_per_s",
    "decode_ratio", "conc_ratio", "prefill_ratio", "quality_score",
    "prefill_1k_tok_s", "prefill_8k_tok_s", "prefill_32k_tok_s", "prefill_32k_ttft_ms",
    "long_ttft_ms", "long_prefill_tok_s", "kv_tokens", "kv_max_concurrency",
    "n_errors", "n_warnings", "duration_s",
}


def read_status(run_dir: Path) -> str:
    path = run_dir / "status.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("status") or "unknown"
        except Exception:  # noqa: BLE001
            pass
    return "complete"


def num(value, digits: int = 2):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def extract(summary: dict, summary_path: Path) -> dict:
    phases = summary.get("phases") or {}
    server = summary.get("server") or {}
    metrics = summary.get("metrics") or {}
    overall = metrics.get("overall") or {}
    decode = phases.get("decode") or {}
    conc_rows = (phases.get("concurrency") or {}).get("levels") or []
    conc = {row.get("c"): row for row in conc_rows}
    prefill = (phases.get("prefill") or {}).get("by_target") or {}
    long_ctx = phases.get("long_context") or {}
    vllm = server.get("vllm") or {}
    spec = vllm.get("speculative_config") or {}
    kv = server.get("kv_cache") or {}
    recipe = server.get("recipe") or {}
    image_id = server.get("image_id") or ""
    timing = summary.get("timing") or {}
    positions = overall.get("accepted_per_position") or {}
    quality = summary.get("quality") or {}
    quality_scores = quality.get("scores") or {}
    quality_overall = (quality_scores.get("overall") or {}) if isinstance(quality_scores, dict) else {}

    def ratio_of(phase_name: str):
        phase = phases.get(phase_name) or {}
        if not isinstance(phase, dict):
            return None
        for key in ("client_to_server_token_ratio", "client_to_server_prompt_ratio"):
            if phase.get(key) is not None:
                return phase[key]
        return None

    # scaling sanity: aggregate throughput should rise with concurrency
    flags = []
    level_values = {c: (conc.get(c) or {}).get("total_out_tok_s") for c in (1, 2, 4, 8, 16)}
    for lower, higher in ((1, 2), (2, 4), (4, 8), (8, 16)):
        low, high = level_values.get(lower), level_values.get(higher)
        if low and high and high < 0.9 * low:
            flags.append(f"scaling: C{higher} ({high:.1f}) < 0.9*C{lower} ({low:.1f})")
    for c, value in level_values.items():
        ttft = (conc.get(c) or {}).get("ttft_ms", {}).get("p50") if conc.get(c) else None
        if ttft and c <= 8 and ttft > 1500:
            flags.append(f"cold-start stall: C{c} TTFT p50 {ttft:.0f} ms (>1500 ms)")
    for phase_name in ("decode", "concurrency", "prefill"):
        ratio = ratio_of(phase_name)
        if ratio is not None and not (0.95 <= ratio <= 1.05):
            flags.append(f"token-ratio {phase_name}={ratio:.2f} outside 0.95-1.05")
    if overall.get("acceptance_rate") is None:
        flags.append("no spec-decode counters")

    row = {
        "run_id": summary.get("run_id"),
        "config_id": summary.get("config_id"),
        "tag": summary.get("tag"),
        "status": read_status(summary_path.parent),
        "profile": summary.get("profile"),
        "stack": server.get("stack"),
        "image": server.get("image"),
        "image_id_short": str(image_id)[:19] if image_id else None,
        "recipe": recipe.get("path"),
        "mtp": vllm.get("num_speculative_tokens"),
        "spec_method": spec.get("method"),
        "kv_cache_dtype": vllm.get("kv_cache_dtype"),
        "max_num_seqs": vllm.get("max_num_seqs"),
        "max_model_len": vllm.get("max_model_len"),
        "boot_s": server.get("boot_seconds"),
        "decode_tok_s": decode.get("out_tok_s_median"),
        "ttft_ms_p50": decode.get("ttft_ms_p50"),
        "tpot_ms_p50": decode.get("tpot_ms_p50"),
        "tokens_per_chunk": (decode.get("tokens_per_chunk") or {}).get("median"),
        "acceptance_rate": overall.get("acceptance_rate"),
        "tokens_per_step": overall.get("tokens_per_step"),
        "avg_draft_depth": overall.get("avg_draft_depth"),
        "accept_pos0": positions.get("0"),
        "accept_pos1": positions.get("1"),
        "accept_pos2": positions.get("2"),
        "accept_pos3": positions.get("3"),
        "decode_ratio": ratio_of("decode"),
        "conc_ratio": ratio_of("concurrency"),
        "prefill_ratio": ratio_of("prefill"),
        "quality_score": quality_overall.get("score"),
        "flags": flags,
        "conc_1": (conc.get(1) or {}).get("total_out_tok_s"),
        "conc_2": (conc.get(2) or {}).get("total_out_tok_s"),
        "conc_4": (conc.get(4) or {}).get("total_out_tok_s"),
        "conc_8": (conc.get(8) or {}).get("total_out_tok_s"),
        "conc_16": (conc.get(16) or {}).get("total_out_tok_s"),
        "prefill_1k_tok_s": (prefill.get("1024") or {}).get("prefill_tok_s"),
        "prefill_8k_tok_s": (prefill.get("8192") or {}).get("prefill_tok_s"),
        "prefill_32k_tok_s": (prefill.get("32768") or {}).get("prefill_tok_s"),
        "prefill_32k_ttft_ms": (prefill.get("32768") or {}).get("ttft_ms"),
        "long_ttft_ms": long_ctx.get("ttft_ms"),
        "long_prefill_tok_s": long_ctx.get("prefill_tok_s"),
        "long_needle_ok": long_ctx.get("needle_ok"),
        "kv_tokens": kv.get("tokens"),
        "kv_max_concurrency": kv.get("max_concurrency"),
        "n_errors": len(summary.get("errors") or []),
        "n_warnings": len(summary.get("warnings") or []),
        "duration_s": timing.get("duration_s"),
        "generated_at": summary.get("generated_at"),
        "summary_path": C.repo_rel(summary_path),
    }
    decode_rate, tok_step = row.get("decode_tok_s"), row.get("tokens_per_step")
    row["step_ms"] = (round(1000.0 * tok_step / decode_rate, 1)
                      if decode_rate and tok_step and decode_rate > 0 else None)
    row["steps_per_s"] = (round(decode_rate / tok_step, 2)
                          if decode_rate and tok_step and tok_step > 0 else None)
    for key in NUMERIC_COLUMNS:
        row[key] = num(row[key])
    return row


def collect(run_dirs: list[Path], results_dirs: list[Path]) -> list[dict]:
    runs: list[Path] = list(run_dirs)
    for root in results_dirs:
        if not root.is_dir():
            C.log(f"warning: results dir not found: {root}")
            continue
        for path in sorted(root.glob("*/summary.json")):
            runs.append(path.parent)
    rows: list[dict] = []
    seen = set()
    for run_dir in dict.fromkeys(runs):
        path = run_dir / "summary.json"
        if not path.is_file():
            C.log(f"warning: no summary.json in {run_dir}")
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            C.log(f"warning: unreadable summary {path}: {exc}")
            continue
        rows.append(extract(summary, path))
    return rows


def fmt(value, digits: int = 2, dash: str = "—") -> str:
    if value is None or value == "":
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(rows: list[dict], columns: list[tuple[str, str, int]], title: str,
             best: dict[str, str] | None = None) -> list[str]:
    lines = [f"### {title}", ""]
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines += [header, sep]
    for index, row in enumerate(rows, 1):
        cells = []
        for key, _label, digits in columns:
            value = row.get(key)
            text = fmt(value, digits)
            if best and best.get(key) == str(row.get("run_id")):
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def best_row(rows: list[dict], key: str, reverse: bool = True) -> str | None:
    candidates = [r for r in rows if isinstance(r.get(key), (int, float))]
    if not candidates:
        return None
    winner = (max if reverse else min)(candidates, key=lambda r: r[key])
    return str(winner.get("run_id"))


def write_markdown(rows: list[dict], path: Path, results_dirs: list[Path]) -> None:
    ok_rows = [r for r in rows if r.get("status") in ("done", "complete", "already_done", None)
               or r.get("decode_tok_s") is not None]
    failed = [r for r in rows if r.get("status") in ("failed", "suite_failed", "timeout", "interrupted")]
    lines = [
        "# Benchmark results",
        "",
        f"Generated: {C.now_iso()}",
        "",
        f"Run roots: {', '.join(str(p) for p in results_dirs) or 'explicit --run-dir'}",
        f"Runs: {len(rows)} ({len(failed)} failed)",
        "",
        "Notes: `decode_tok_s` = median single-stream decode speed over 3 repeats "
        "(usage.completion_tokens / decode window, thinking off, temp 0, seed 42). "
        "`acceptance_rate` = accepted draft tokens / proposed draft tokens from /metrics deltas; "
        "`tokens_per_step` = emitted tokens / draft step (includes the bonus token). "
        "`conc_N` = aggregate output tok/s at N concurrent streams (2 waves). "
        "`boot_s` includes container start → /health.",
        "",
    ]
    if not ok_rows:
        lines += ["_No successful runs found._", ""]
    else:
        ranked_decode = sorted(ok_rows, key=lambda r: (r.get("decode_tok_s") is None,
                                                       -(r.get("decode_tok_s") or 0)))
        best = {
            "decode_tok_s": best_row(ok_rows, "decode_tok_s"),
            "conc_4": best_row(ok_rows, "conc_4"),
            "conc_8": best_row(ok_rows, "conc_8"),
            "conc_16": best_row(ok_rows, "conc_16"),
            "acceptance_rate": best_row(ok_rows, "acceptance_rate"),
            "ttft_ms_p50": best_row(ok_rows, "ttft_ms_p50", reverse=False),
            "boot_s": best_row(ok_rows, "boot_s", reverse=False),
        }
        lines += md_table(
            ranked_decode,
            [("run_id", "run", 0), ("stack", "stack", 0), ("mtp", "MTP", 0),
             ("kv_cache_dtype", "KV", 0), ("max_num_seqs", "seats", 0),
             ("decode_tok_s", "decode tok/s", 1), ("ttft_ms_p50", "TTFT p50 ms", 0),
             ("tpot_ms_p50", "TPOT p50 ms", 1), ("acceptance_rate", "accept", 3),
             ("tokens_per_step", "tok/step", 2),
             ("delta_decode_pct", "Δdecode %", 1), ("conc_4", "C=4 tok/s", 1),
             ("conc_8", "C=8 tok/s", 1), ("delta_c8_pct", "ΔC8 %", 1),
             ("conc_16", "C=16 tok/s", 1),
             ("boot_s", "boot s", 0), ("n_errors", "errs", 0),
             ("decode_ratio", "ratio", 3), ("quality_score", "quality", 3)],
            "Ranked by single-stream decode throughput", best)

        ranked_c4 = sorted(ok_rows, key=lambda r: -(r.get("conc_4") or 0))
        lines += md_table(
            ranked_c4,
            [("run_id", "run", 0), ("conc_1", "C=1", 1), ("conc_2", "C=2", 1),
             ("conc_4", "C=4", 1), ("conc_8", "C=8", 1), ("conc_16", "C=16", 1),
             ("kv_tokens", "KV tokens", 0), ("kv_max_concurrency", "KV max conc", 2)],
            "Concurrency scaling (aggregate output tok/s)")

        ranked_acc = sorted(ok_rows, key=lambda r: -(r.get("acceptance_rate") or 0))
        lines += md_table(
            ranked_acc,
            [("run_id", "run", 0), ("mtp", "MTP", 0), ("avg_draft_depth", "draft depth", 2),
             ("acceptance_rate", "acceptance", 3), ("delta_accept_pp", "Δaccept pp", 1),
             ("tokens_per_step", "tok/step", 2), ("step_ms", "step ms", 1),
             ("steps_per_s", "steps/s", 1), ("tokens_per_chunk", "tok/chunk", 2),
             ("decode_tok_s", "decode tok/s", 1)],
            "Speculative decoding efficiency")

        ranked_prefill = sorted(ok_rows, key=lambda r: -(r.get("prefill_32k_tok_s") or 0))
        lines += md_table(
            ranked_prefill,
            [("run_id", "run", 0), ("prefill_1k_tok_s", "1k tok/s", 0),
             ("prefill_8k_tok_s", "8k tok/s", 0), ("prefill_32k_tok_s", "32k tok/s", 0),
             ("prefill_32k_ttft_ms", "32k TTFT ms", 0), ("long_ttft_ms", "128k TTFT ms", 0),
             ("long_needle_ok", "128k needle", 0)],
            "Prefill / long-context")

        anomalous = [r for r in ok_rows if r.get("flags")]
        if anomalous:
            lines += ["### Anomalies / warnings (do not rank on the flagged level)", ""]
            for row in anomalous:
                lines.append(f"- **{row.get('run_id')}**: " + "; ".join(row["flags"]))
            lines.append("")

        lines += md_table(
            sorted(ok_rows, key=lambda r: (r.get("boot_s") is None, r.get("boot_s") or 0)),
            [("run_id", "run", 0), ("boot_s", "boot s", 0), ("profile", "profile", 0),
             ("duration_s", "suite s", 0), ("n_errors", "errs", 0),
             ("n_warnings", "warns", 0), ("image_id_short", "image", 0)],
            "Boot & run health")

    if failed:
        lines += ["### Failed runs", ""]
        for row in failed:
            status_path = Path(C.REPO_ROOT / row["summary_path"]).parent / "status.json"
            error = ""
            try:
                error = (json.loads(status_path.read_text(encoding="utf-8")).get("error") or "")[:300]
            except Exception:  # noqa: BLE001
                pass
            lines.append(f"- **{row.get('run_id')}**: {row.get('status')} {error}")
        lines.append("")

    lines += [
        "### Raw rows",
        "",
        "See `summary.csv` for the full machine-readable table and each run's "
        "`summary.json` / `requests.jsonl` for per-request detail.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", action="append", default=[],
                        help="directory containing <run-id>/summary.json (repeatable)")
    parser.add_argument("--run-dir", action="append", default=[], help="explicit run directory")
    parser.add_argument("--csv", help="output CSV path (default <results-dir>/summary.csv)")
    parser.add_argument("--md", help="output markdown path (default <results-dir>/RESULTS.md)")
    parser.add_argument("--json", help="merged JSON output path (default <results-dir>/summary_all.json)")
    parser.add_argument("--no-md", action="store_true", help="skip the markdown report")
    args = parser.parse_args(argv)
    results_dirs = [Path(p) for p in args.results_dir] or [C.RESULTS_DIR]
    run_dirs = [Path(p) for p in args.run_dir]
    rows = collect(run_dirs, results_dirs)
    if not rows:
        C.log("no runs found")
        return 1
    # deltas vs the official reference (M0) when present
    reference = None
    for row in rows:
        if (row.get("run_id") == "m0-reference" or row.get("tag") == "reference"
                or row.get("config_id") == "m0-reference"):
            reference = row
            break

    def _delta_pct(value, ref_value):
        if value is None or ref_value in (None, 0):
            return None
        return round((value - ref_value) / ref_value * 100.0, 2)

    for row in rows:
        ref = reference or {}
        row["delta_decode_pct"] = _delta_pct(row.get("decode_tok_s"), ref.get("decode_tok_s"))
        row["delta_c4_pct"] = _delta_pct(row.get("conc_4"), ref.get("conc_4"))
        row["delta_c8_pct"] = _delta_pct(row.get("conc_8"), ref.get("conc_8"))
        acc, ref_acc = row.get("acceptance_rate"), ref.get("acceptance_rate")
        row["delta_accept_pp"] = (round((acc - ref_acc) * 100.0, 2)
                                  if acc is not None and ref_acc is not None else None)
    primary = results_dirs[0] if results_dirs else run_dirs[0].parent
    csv_path = Path(args.csv) if args.csv else primary / "summary.csv"
    md_path = Path(args.md) if args.md else primary / "RESULTS.md"
    json_path = Path(args.json) if args.json else primary / "summary_all.json"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in COLUMNS}
            if isinstance(out.get("flags"), list):
                out["flags"] = "; ".join(out["flags"])
            writer.writerow(out)
    json_path.write_text(json.dumps({"generated_at": C.now_iso(), "runs": rows}, indent=2) + "\n",
                         encoding="utf-8")
    if not args.no_md:
        write_markdown(rows, md_path, results_dirs)
    C.log(f"aggregated {len(rows)} run(s)")
    C.log(f"  csv: {C.repo_rel(csv_path)}")
    C.log(f"  json: {C.repo_rel(json_path)}")
    if not args.no_md:
        C.log(f"  md: {C.repo_rel(md_path)}")
    # compact console ranking
    ranked = sorted([r for r in rows if r.get("decode_tok_s") is not None],
                    key=lambda r: -r["decode_tok_s"])
    for index, row in enumerate(ranked[:10], 1):
        C.log(f"  {index:>2}. {row['run_id']:<28} decode={fmt(row['decode_tok_s']):>6} tok/s  "
              f"accept={fmt(row['acceptance_rate'], 3):>6}  tok/step={fmt(row['tokens_per_step']):>5}  "
              f"C4={fmt(row['conc_4']):>6}  C8={fmt(row['conc_8']):>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
