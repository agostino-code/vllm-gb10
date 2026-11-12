#!/usr/bin/env python3
"""Run deterministic capability/quality batteries against an OpenAI-compatible
vLLM server and write scores.json + raw outputs.

Typical use (Lead / grader):

    # full default suite (~10 min) against a live server
    python3 quality/run_quality.py --out quality/runs/preset-A

    # fast gate without the long-context needle (~2-3 min)
    python3 quality/run_quality.py --categories quick --label gating

    # just the long-context battery at 32k + 128k
    python3 quality/run_quality.py --categories needle --needle-sizes 32k,128k

    # offline: replay canned model outputs through the graders (no server)
    python3 quality/run_quality.py --self-test

Everything is stdlib-only; the suite never installs packages and never needs
network access (only the local vLLM endpoint).
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qlib import __version__, batteries  # noqa: E402
from qlib.client import ChatClient, ClientError, ServerUnavailable  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CANNED_DEFAULT = ROOT / "tests" / "canned" / "model_outputs.json"
SCHEMA_VERSION = 1

LIMITATIONS = [
    "Scores are deterministic, rule-based proxies for capability; they are not a substitute for human review.",
    "temperature=0 and a fixed seed are requested, but server-side batching can still make sampled tokens differ run-to-run.",
    "Code batteries execute model-written Python in a subprocess with CPU/temp limits and a timeout; this is a robustness guard, not a security sandbox.",
    "Long-context prompts are synthetic English filler; real-world retrieval difficulty varies, so recall here is an upper-ish bound for a given size.",
    "TTFT is measured client-side and includes queueing/batching latency of the moment, so long-context TTFT is indicative, not a benchmark.",
    "Tool-calling checks verify tool name + expected argument values; extra optional arguments are tolerated and reported.",
    "Thinking mode (--thinking) needs generous max_tokens; an empty-content item usually means the CoT consumed the budget.",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _unique_dir(out: Path) -> Path:
    if not out.exists():
        return out
    if not (out / "scores.json").exists() and not any(out.iterdir()):
        return out
    base = out
    index = 2
    while True:
        candidate = Path(str(base) + "-%d" % index)
        if not candidate.exists():
            return candidate
        index += 1


def make_config(args, categories: list, mode: str, model: str | None,
                max_model_len: int | None, client: ChatClient | None) -> dict:
    return {
        "suite": "quality",
        "suite_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "mode": mode,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_revision": git_revision(),
        "argv": sys.argv[1:],
        "base_url": args.base_url,
        "model_requested": args.model,
        "model_resolved": model,
        "served_models": sorted((client.models_info or {}).keys()) if client else [],
        "max_model_len": max_model_len,
        "thinking": bool(args.thinking),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "seed": int(args.seed),
        "categories_requested": categories,
        "mode_arg": getattr(args, "mode", None),
        "needle_sizes_tokens": list(args.parsed_needle_sizes),
        "needle_depth": float(args.depth),
        "needle_chars_per_token": float(args.chars_per_token),
        "workers": int(args.workers),
        "request_timeout_s": float(args.timeout),
        "replay_file": str(args.replay) if args.replay else None,
    }


def build_payload(config: dict, results: dict, runtime_s: float) -> dict:
    categories = results.get("categories") or {}
    run_categories = [c for c, summary in categories.items() if not summary.get("skipped")]
    category_scores = {c: float(categories[c].get("score", 0.0)) for c in run_categories}
    overall_score = round(sum(category_scores.values()) / len(category_scores), 4) if category_scores else 0.0
    items_total = sum(int(categories[c].get("total", 0)) for c in run_categories)
    items_passed = sum(int(categories[c].get("passed", 0)) for c in run_categories)
    payload = {
        "suite": "quality",
        "schema_version": SCHEMA_VERSION,
        "generated_at": config.get("generated_at", now_iso()),
        "mode": config.get("mode"),
        "config": config,
        "runtime_s": round(runtime_s, 2),
        "overall": {
            "score": overall_score,
            "categories_run": len(run_categories),
            "items_total": items_total,
            "items_passed": items_passed,
            "category_scores": category_scores,
            "aborted": bool(results.get("aborted")),
        },
        "categories": categories,
        "limitations": list(LIMITATIONS),
    }
    return payload


def write_outputs(out_dir: Path, payload: dict, records: list, save_prompts: bool,
                  log_lines: list, scores_path: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    by_category: dict = {}
    for record in records:
        by_category.setdefault(record.category, []).append(record)
    for category, cat_records in by_category.items():
        path = raw_dir / ("%s.jsonl" % category)
        with path.open("w", encoding="utf-8") as handle:
            for record in cat_records:
                handle.write(json.dumps(record.to_raw(save_prompts), ensure_ascii=False) + "\n")
    canonical_scores = out_dir / "scores.json"
    canonical_scores.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if scores_path is not None and scores_path != canonical_scores:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        scores_path = canonical_scores
    (out_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(payload), encoding="utf-8")
    return {"scores": str(scores_path), "raw_dir": str(raw_dir)}


def render_markdown(payload: dict) -> str:
    config = payload.get("config") or {}
    lines = [
        "# Quality run summary",
        "",
        "- generated: %s" % payload.get("generated_at"),
        "- mode: %s" % payload.get("mode"),
        "- model: %s" % config.get("model_resolved"),
        "- base_url: %s" % config.get("base_url"),
        "- thinking: %s | temperature: %s | seed: %s" % (
            config.get("thinking"), config.get("temperature"), config.get("seed")),
        "- git: %s | host: %s" % (config.get("git_revision"), config.get("host")),
        "",
        "| category | score | passed | total | runtime_s | notes |",
        "|---|---|---|---|---|---|",
    ]
    for category, summary in (payload.get("categories") or {}).items():
        notes = []
        if summary.get("skipped"):
            notes.append("skipped")
        for key in ("valid_json_rate", "assertion_pass_rate", "check_pass_rate",
                    "recall_rate", "strict_recall_rate", "ttft_s"):
            if key in summary:
                notes.append("%s=%s" % (key, summary[key]))
        if summary.get("metrics"):
            notes.append("degenerate=%s" % summary["metrics"].get("degenerate"))
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            category, summary.get("score"), summary.get("passed"), summary.get("total"),
            summary.get("runtime_s"), "; ".join(notes) or "",
        ))
    overall = payload.get("overall") or {}
    lines += [
        "",
        "**overall score:** %s (%s/%s items passed) in %ss" % (
            overall.get("score"), overall.get("items_passed"), overall.get("items_total"),
            payload.get("runtime_s")),
        "",
    ]
    if overall.get("aborted"):
        lines.append("> Run was ABORTED because the server became unavailable; results are partial.")
        lines.append("")
    limitations = payload.get("limitations") or []
    if limitations:
        lines.append("## Limitations")
        lines.append("")
        lines.extend("- %s" % item for item in limitations)
        lines.append("")
    return "\n".join(lines)


def print_summary(payload: dict) -> None:
    config = payload.get("config") or {}
    print("=" * 78)
    print("quality suite v%s | mode=%s | model=%s" % (
        config.get("suite_version"), payload.get("mode"), config.get("model_resolved")))
    print("base_url=%s | thinking=%s | temperature=%s | seed=%s" % (
        config.get("base_url"), config.get("thinking"), config.get("temperature"), config.get("seed")))
    print("-" * 78)
    print("%-14s %8s %8s %8s %12s" % ("category", "score", "passed", "total", "runtime_s"))
    for category, summary in (payload.get("categories") or {}).items():
        state = "skipped" if summary.get("skipped") else ""
        print("%-14s %8s %8s %8s %12s %s" % (
            category, summary.get("score"), summary.get("passed"), summary.get("total"),
            summary.get("runtime_s"), state))
    overall = payload.get("overall") or {}
    print("-" * 78)
    print("%-14s %8s %8s %8s %12s" % (
        "OVERALL", overall.get("score"), overall.get("items_passed"),
        overall.get("items_total"), payload.get("runtime_s")))
    if overall.get("aborted"):
        print("!! run aborted early: server became unavailable (partial results saved)")
    print("=" * 78)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                        help="OpenAI-compatible base URL (default: %(default)s)")
    parser.add_argument("--model", default=None,
                        help="served model id; auto-detected from /v1/models when omitted")
    parser.add_argument("--api-key", default="EMPTY", help="API key (vLLM ignores it by default)")
    parser.add_argument("--out", default=None,
                        help="output directory (default: quality/runs/<timestamp>_<label>)")
    parser.add_argument("--output", default=None,
                        help="alias for --out; if PATH ends in .json, scores are written exactly "
                             "there and artifacts go to PATH-without-suffix + '.artifacts/'")
    parser.add_argument("--label", default="", help="label used in the default output directory name")
    parser.add_argument("--categories", default="all",
                        help="all | quick | comma list | all,-needle (default: all)")
    parser.add_argument("--mode", default=None, choices=["quick", "full"],
                        help="alias for category selection: quick = all except needle, full = all")
    parser.add_argument("--needle-sizes", default=None,
                        help="comma list in tokens, e.g. 32k,128k,262k (default: 32k,128k,262k)")
    parser.add_argument("--depth", type=float, default=0.5, help="needle depth fraction (default 0.5)")
    parser.add_argument("--chars-per-token", type=float, default=3.5,
                        help="char/token heuristic for needle sizing (default 3.5)")
    parser.add_argument("--thinking", action="store_true",
                        help="enable chain-of-thought (chat_template_kwargs enable_thinking=true)")
    parser.add_argument("--no-thinking", dest="thinking", action="store_false",
                        help="disable thinking (default)")
    parser.set_defaults(thinking=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel requests per text battery (0 = per-category default)")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout seconds")
    parser.add_argument("--max-model-len", type=int, default=409600,
                        help="server context limit; auto-detected from /v1/models when reported")
    parser.add_argument("--wait-server", type=float, default=0.0,
                        help="seconds to wait for /health before giving up (0 = probe once)")
    parser.add_argument("--save-prompts", action="store_true",
                        help="store full prompts in raw/ jsonl (large for needle)")
    parser.add_argument("--replay", default=None,
                        help="offline mode: grade canned model outputs from this JSON file")
    parser.add_argument("--self-test", action="store_true",
                        help="offline mode using quality/tests/canned/model_outputs.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate data/CLI and print the run plan without contacting the server")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--min-score", type=float, default=None,
                        help="exit 2 if the overall score is below this value")
    parser.add_argument("--version", action="version", version="quality suite %s" % __version__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Deterministic capability/quality batteries for an OpenAI-compatible vLLM server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    return parser.parse_args(argv)


def resolve_output(args) -> tuple:
    """Return (artifact_dir, exact_scores_path_or_None) from --out/--output."""
    if args.output:
        path = Path(args.output)
        if path.suffix.lower() == ".json":
            return path.parent / (path.stem + ".artifacts"), path
        return path, None
    return default_out_dir(args), None


def default_out_dir(args) -> Path:
    if args.out:
        return Path(args.out)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = ("_" + args.label) if args.label else ""
    return ROOT / "runs" / (stamp + label)


def check_data(data: dict) -> list:
    problems = []
    expected_counts = {"math": 20, "json": 10, "code": 8, "instruction": 10, "tool": 6, "repetition": 1}
    for category, want in expected_counts.items():
        got = len(data.get(category) or [])
        if got != want:
            problems.append("%s: expected %d items, found %d" % (category, want, got))
    for item in data.get("math") or []:
        if not item.get("answer"):
            problems.append("%s: missing answer" % item.get("id"))
    for item in data.get("code") or []:
        if not item.get("tests") or not item.get("function"):
            problems.append("%s: missing function/tests" % item.get("id"))
    for item in data.get("tool") or []:
        if not (item.get("expect") or {}).get("name"):
            problems.append("%s: missing expect.name" % item.get("id"))
    if not (data.get("needle") or {}).get("fact"):
        problems.append("needle: missing fact")
    return problems


def dry_run(args, categories: list, data: dict, config: dict) -> int:
    problems = check_data(data)
    print("quality suite dry run")
    print("categories:", ", ".join(categories))
    print("thinking:", args.thinking, "| temperature:", args.temperature, "| seed:", args.seed)
    print("needle sizes (tokens):", args.parsed_needle_sizes)
    print("planned requests:")
    total = 0
    for category in categories:
        if category == "needle":
            count = len(args.parsed_needle_sizes)
        else:
            count = len(data.get(category) or [])
        total += count
        print("  %-12s %3d requests   max_tokens=%d" % (
            category, count, batteries.THINKING_MAX_TOKENS[category] if args.thinking
            else batteries.MAX_TOKENS[category]))
    print("total requests:", total, "(needle prompts are generated deterministically)"
          if "needle" in categories else "")
    if problems:
        print("DATA PROBLEMS:")
        for problem in problems:
            print("  -", problem)
        return 2
    print("data OK")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        categories = batteries.parse_categories(args.mode or args.categories)
        args.parsed_needle_sizes = batteries.parse_needle_sizes(args.needle_sizes)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    try:
        data = batteries.load_data()
    except Exception as exc:  # noqa: BLE001
        print("error: cannot load data: %s" % exc, file=sys.stderr)
        return 2

    if args.dry_run:
        return dry_run(args, categories, data, {})

    artifact_dir, exact_scores_path = resolve_output(args)
    out_dir = _unique_dir(artifact_dir)
    log_lines: list = []
    started = time.perf_counter()

    def log(message: str) -> None:
        log_lines.append(message)
        if not args.quiet:
            print(message)

    replay_path = None
    if args.self_test:
        replay_path = CANNED_DEFAULT
    elif args.replay:
        replay_path = Path(args.replay)

    if replay_path is not None:
        if not replay_path.exists():
            print("error: canned outputs file not found: %s" % replay_path, file=sys.stderr)
            return 2
        canned = batteries.load_canned(replay_path)
        cfg = batteries.RunConfig(
            thinking=args.thinking, temperature=args.temperature, top_p=args.top_p,
            seed=args.seed, workers=args.workers, needle_sizes=args.parsed_needle_sizes,
            needle_depth=args.depth, needle_chars_per_token=args.chars_per_token,
            save_prompts=args.save_prompts,
        )
        results = batteries.run_replay(canned, data, cfg, categories,
                                       needle_sizes=args.parsed_needle_sizes)
        records = results.pop("_records", [])
        runtime = time.perf_counter() - started
        config = make_config(args, categories, "replay", "canned", None, None)
        config["replay_file"] = str(replay_path)
        payload = build_payload(config, results, runtime)
        paths = write_outputs(out_dir, payload, records, args.save_prompts, log_lines, exact_scores_path)
        if not args.quiet:
            print_summary(payload)
            print("scores.json:", paths["scores"])
        else:
            print(json.dumps({"overall": payload["overall"], "out": paths["scores"]}))
        if args.min_score is not None and payload["overall"]["score"] < args.min_score:
            return 2
        return 0

    # ---------------------------------------------------------------- live
    client = ChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        retries=1,
        enable_thinking=bool(args.thinking),
    )
    if args.wait_server and args.wait_server > 0:
        log("waiting up to %.0fs for the server at %s ..." % (args.wait_server, args.base_url))
        ready = client.wait_ready(max_wait=args.wait_server)
    else:
        ready = client.is_healthy(timeout=5.0)
    if not ready:
        print("error: no OpenAI-compatible server found at %s" % args.base_url, file=sys.stderr)
        print("       boot/restart the vLLM server, then retry (or pass --wait-server 600, "
              "or --self-test for offline mode).", file=sys.stderr)
        return 3

    try:
        model = client.resolve_model(args.model)
    except (ClientError, ServerUnavailable) as exc:
        print("error: cannot resolve served model: %s" % exc, file=sys.stderr)
        return 3
    client.model = model
    max_model_len = client.model_max_len() or args.max_model_len
    log("server ready: model=%s max_model_len=%s" % (model, max_model_len))
    cfg = batteries.RunConfig(
        thinking=args.thinking, temperature=args.temperature, top_p=args.top_p,
        seed=args.seed, workers=args.workers, needle_sizes=args.parsed_needle_sizes,
        needle_depth=args.depth, needle_chars_per_token=args.chars_per_token,
        save_prompts=args.save_prompts, request_timeout=args.timeout,
    )

    try:
        results = batteries.run_batteries(
            client, data, cfg, categories, max_model_len=max_model_len, log=log,
        )
    except KeyboardInterrupt:
        print("\ninterrupted; no scores written", file=sys.stderr)
        return 130
    records = results.pop("_records", [])
    runtime = time.perf_counter() - started
    config = make_config(args, categories, "live", model, max_model_len, client)
    payload = build_payload(config, results, runtime)

    # Surface suspicious conditions (empty content usually = thinking not disabled).
    empty_items = [r.item_id for r in records if r.result.status == "ok" and not r.result.content
                   and not r.result.tool_calls]
    if empty_items:
        payload["warnings"] = [
            "%d item(s) returned empty content (check --thinking and max_tokens): %s"
            % (len(empty_items), ", ".join(empty_items[:8]))
        ]
        log("WARNING: %d item(s) returned empty content: %s" % (
            len(empty_items), ", ".join(empty_items[:8])))

    paths = write_outputs(out_dir, payload, records, args.save_prompts, log_lines, exact_scores_path)
    if not args.quiet:
        print_summary(payload)
        print("scores.json:", paths["scores"])
        print("raw outputs:", paths["raw_dir"])
    else:
        print(json.dumps({"overall": payload["overall"], "out": paths["scores"]}))

    if payload["overall"]["aborted"]:
        return 4
    if args.min_score is not None and payload["overall"]["score"] < args.min_score:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
