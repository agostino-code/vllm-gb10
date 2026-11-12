#!/usr/bin/env python3
"""watch_matrix.py — read-only watchdog for a running bench/run_matrix.py job.

Polls result files only (no docker, no HTTP, no GPU side effects) and prints a
compact, timestamped trace:

  * matrix start / per-config status transitions
  * a one-line metric summary as soon as each summary.json appears
    (decode, acceptance, tokens/step, C=4/C=8, boot, errors, token ratios)
  * warnings for suspicious results: client/server token ratio outside
    0.95-1.05, request errors, failed/partial suites, missing metrics
  * speculative-decoding boot sanity: the "drafter reads N of 21 files" line
    from server.log + a flag when a recipe configures an external draft model
    but the log shows the self-draft path
  * exits when matrix_status.json carries finished_at (matrix finished) or when
    --stop-file exists; prints a final roll-up.

  python3 bench/watch_matrix.py --results-dir bench/results --interval 60
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

BAD_RATIO = (0.95, 1.05)
DRAFTER_RE = re.compile(r"drafter reads\s+(\d+)\s+of\s+(\d+)\s+files")
DRAFT_PATH_RE = re.compile(r'"model":\s*"([^"]*draft[^"]*)"')


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def summarize_result(run_dir: Path) -> dict:
    summary = read_json(run_dir / "summary.json")
    if not summary:
        return {}
    phases = summary.get("phases") or {}
    decode = phases.get("decode") or {}
    conc = {row.get("c"): row for row in (phases.get("concurrency") or {}).get("levels") or []}
    overall = (summary.get("metrics") or {}).get("overall") or {}
    server = summary.get("server") or {}
    warnings = list(summary.get("warnings") or [])
    ratios = {}
    for name, phase in phases.items():
        if not isinstance(phase, dict):
            continue
        for key in ("client_to_server_token_ratio", "client_to_server_prompt_ratio",
                    "client_to_server_ratio"):
            value = phase.get(key)
            if value is not None:
                ratios[name] = value
                break
    bad = {k: v for k, v in ratios.items() if not (BAD_RATIO[0] <= v <= BAD_RATIO[1])}
    return {
        "run_id": summary.get("run_id"),
        "profile": summary.get("profile"),
        "decode": decode.get("out_tok_s_median"),
        "ttft_p50": decode.get("ttft_ms_p50"),
        "acceptance": overall.get("acceptance_rate"),
        "tokens_per_step": overall.get("tokens_per_step"),
        "depth": overall.get("avg_draft_depth"),
        "c1": (conc.get(1) or {}).get("total_out_tok_s"),
        "c4": (conc.get(4) or {}).get("total_out_tok_s"),
        "c8": (conc.get(8) or {}).get("total_out_tok_s"),
        "c16": (conc.get(16) or {}).get("total_out_tok_s"),
        "boot_s": server.get("boot_seconds"),
        "image": server.get("image"),
        "errors": len(summary.get("errors") or []),
        "warnings": warnings,
        "ratios": ratios,
        "bad_ratios": bad,
        "duration_s": (summary.get("timing") or {}).get("duration_s"),
    }


def drafter_lines(run_dir: Path) -> list[str]:
    log = run_dir / "server.log"
    if not log.is_file():
        return []
    out = []
    try:
        with open(log, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "drafter reads" in line or "Loading weights took" in line:
                    match = DRAFTER_RE.search(line)
                    text = line.strip()
                    if len(text) > 180:
                        text = text[:180] + "…"
                    out.append(text if not match else
                               f"drafter reads {match.group(1)}/{match.group(2)} files")
    except OSError:
        return []
    # keep only a few interesting lines
    return out[-4:]


def fmt(value, digits=1, dash="—"):
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--stop-file", default="")
    parser.add_argument("--report-existing", action="store_true",
                        help="also print results that already existed at startup")
    parser.add_argument("--expect-matrix", default="matrix/matrix.yaml",
                        help="matrix file the status file must match (basename), so that other "
                             "run_matrix invocations sharing results_dir do not confuse the watchdog")
    args = parser.parse_args(argv)
    results = Path(args.results_dir)
    stop_file = Path(args.stop_file) if args.stop_file else None
    seen_status: dict[str, str] = {}
    seen_result_mtimes: dict[str, float] = {}
    matrix_finished = False
    matrix_announced = False
    last_matrix_start = None
    foreign_warned = False
    started = time.time()
    started_dt = datetime.now(timezone.utc)

    print(f"[{time.strftime('%H:%M:%S')}] watchdog started on {results} "
          f"(interval {args.interval:.0f}s, read-only)", flush=True)
    if results.is_dir() and not args.report_existing:
        for run_dir in results.iterdir():
            summary = run_dir / "summary.json"
            if summary.is_file():
                seen_result_mtimes[run_dir.name] = summary.stat().st_mtime
        if seen_result_mtimes:
            print(f"[{time.strftime('%H:%M:%S')}] ignoring {len(seen_result_mtimes)} pre-existing "
                  f"result(s) (use --report-existing to print them)", flush=True)

    def is_fresh(iso_text) -> bool:
        """Ignore matrix_status.json left over from an earlier test run."""
        try:
            when = datetime.fromisoformat(str(iso_text).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return False
        return when >= started_dt.replace(microsecond=0) and (when - started_dt).total_seconds() > -120

    while True:
        if stop_file is not None and stop_file.exists():
            print(f"[{time.strftime('%H:%M:%S')}] stop file present — watchdog exiting", flush=True)
            return 0
        matrix = read_json(results / "matrix_status.json") or {}
        ours = Path(str(matrix.get("matrix") or "")).name == Path(args.expect_matrix).name
        fresh = is_fresh(matrix.get("started_at")) and ours
        if not ours and is_fresh(matrix.get("started_at")) and not foreign_warned:
            print(f"[{time.strftime('%H:%M:%S')}] ignoring foreign matrix_status "
                  f"matrix={matrix.get('matrix')} (watching {args.expect_matrix})", flush=True)
            foreign_warned = True
        configs = (matrix.get("configs") or []) if fresh else []
        matrix_start = matrix.get("started_at") if fresh else None
        if matrix_start != last_matrix_start:
            last_matrix_start = matrix_start
            matrix_announced = False
            matrix_finished = False
            if fresh:
                print(f"[{time.strftime('%H:%M:%S')}] NEWMATRIX {matrix_start} "
                      f"matrix={matrix.get('matrix')}", flush=True)
        if fresh and not matrix_announced:
            print(f"[{time.strftime('%H:%M:%S')}] MATRIX STARTED matrix={matrix.get('matrix')} "
                  f"results={matrix.get('results_dir')}", flush=True)
            matrix_announced = True
        if (fresh and matrix.get("finished_at") and len(configs) == 0
                and (time.time() - started) < 600 and not matrix_finished):
            # a --max-configs 0 / --only-no-match smoke test of the runner
            print(f"[{time.strftime('%H:%M:%S')}] ignoring empty matrix completion (configs=0, "
                  f"likely a runner smoke test)", flush=True)
            last_matrix_start = matrix_start
        if fresh and matrix.get("finished_at") and configs and not matrix_finished:
            matrix_finished = True
            print(f"[{time.strftime('%H:%M:%S')}] MATRIX FINISHED finished_at={matrix['finished_at']} "
                  f"configs={len(configs)}", flush=True)
            for entry in configs:
                print(f"    {entry.get('name'):<26} {entry.get('status')}"
                      f"  rc={entry.get('exit_codes')}  {str(entry.get('error') or '')[:140]}",
                      flush=True)
            # keep watching a little for stragglers before exiting
            time.sleep(max(5.0, args.interval))
            pending = [name for name in seen_status if seen_status[name] not in
                       ("done", "suite_failed", "failed", "timeout", "interrupted")]
            if not pending:
                return 0

        for run_dir in sorted(results.iterdir()) if results.is_dir() else []:
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            if name.startswith("."):
                continue
            status = read_json(run_dir / "status.json") or {}
            state = status.get("status")
            if state and seen_status.get(name) != state:
                seen_status[name] = state
                print(f"[{time.strftime('%H:%M:%S')}] STATUS {name:<26} {state}"
                      f"  {str(status.get('error') or '')[:160]}", flush=True)

            summary_path = run_dir / "summary.json"
            if summary_path.is_file():
                mtime = summary_path.stat().st_mtime
                if seen_result_mtimes.get(name) == mtime:
                    continue
                seen_result_mtimes[name] = mtime
                info = summarize_result(run_dir)
                if not info:
                    continue
                print(f"[{time.strftime('%H:%M:%S')}] RESULT {name}", flush=True)
                print(f"    decode={fmt(info['decode'])} tok/s  ttft_p50={fmt(info['ttft_p50'],0)}ms  "
                      f"accept={fmt(info['acceptance'],3)}  tok/step={fmt(info['tokens_per_step'],2)}  "
                      f"depth={fmt(info['depth'],1)}", flush=True)
                print(f"    C1={fmt(info['c1'])}  C4={fmt(info['c4'])}  C8={fmt(info['c8'])}  "
                      f"C16={fmt(info['c16'])}  boot={fmt(info['boot_s'],1)}s  "
                      f"suite={fmt(info['duration_s'],0)}s  errors={info['errors']}", flush=True)
                print(f"    ratios={ {k: round(v, 3) for k, v in info['ratios'].items()} }",
                      flush=True)
                if info["bad_ratios"]:
                    print(f"    ⚠ TOKEN-RATIO OUT OF RANGE: {info['bad_ratios']} "
                          f"(external traffic or accounting drift)", flush=True)
                if info["errors"]:
                    print(f"    ⚠ {info['errors']} request error(s) — inspect summary.errors",
                          flush=True)
                for warning in info["warnings"][:4]:
                    print(f"    ⚠ warning: {warning[:170]}", flush=True)
                draft = DRAFT_PATH_RE.search(json.dumps(
                    (read_json(run_dir / "meta.json") or {}).get("vllm", {})))
                lines = drafter_lines(run_dir)
                if lines:
                    print(f"    boot: {' | '.join(lines)}", flush=True)
                    if draft:
                        joined = " ".join(lines)
                        if "drafter reads 3/21" in joined or "self" in joined.lower():
                            print(f"    ⚠ SPEC-MODEL CONFIGURED but log looks like self-draft "
                                  f"({draft.group(1)}) — CHECK BOOT", flush=True)

        if matrix_finished:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
