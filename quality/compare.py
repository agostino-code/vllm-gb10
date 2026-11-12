#!/usr/bin/env python3
"""Compare two quality/run_quality.py ``scores.json`` files.

This is the acceptance gate: a preset is disqualified if a capability category
regresses against the baseline by more than ``--max-drop`` (default 0.05, i.e.
one item on a 20-item battery / half an item on a 10-item battery).

    # baseline M0 vs a candidate preset (both produced by run_quality.py)
    python3 quality/compare.py quality/runs/m0-reference/quality.json \
                                quality/runs/candidate/quality.json

    # tolerate 1 item of drift per category (the default), or be strict
    python3 quality/compare.py BASE.json CAND.json --noise 1
    python3 quality/compare.py BASE.json CAND.json --noise 0

    # allow instruction-format noise (exact word counts) but nothing else
    python3 quality/compare.py BASE.json CAND.json --ignore instruction

    # machine-readable comparison for the matrix
    python3 quality/compare.py BASE.json CAND.json --json-diff diff.json

Exit codes: 0 = no regression, 1 = regression detected, 2 = usage/input error.
Stdlib only, fully offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_NOISE = 1.0  # items of drift tolerated per category


def load_run(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("scores file not found: %s" % path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "categories" not in payload:
        raise ValueError("%s does not look like a run_quality.py scores.json (no categories)" % path)
    return payload


def _items_by_id(category: dict) -> dict:
    out = {}
    for item in category.get("items") or []:
        if isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = item
    return out


def compare(baseline: dict, candidate: dict, noise: float = DEFAULT_NOISE,
            max_drop: float | None = None, ignore: list | None = None) -> dict:
    """Compare category scores. Returns a JSON-able diff dict.

    Tolerance: by default ``noise`` items of drift are allowed per category
    (1 item = 1/total of the category score).  Pass ``max_drop`` to use an
    absolute score tolerance instead.
    """
    ignore = set(ignore or [])
    cats_b = baseline.get("categories") or {}
    cats_c = candidate.get("categories") or {}
    rows = []
    regressions = []
    missing = []
    for category in cats_b:
        if category in ignore:
            continue
        base = cats_b[category]
        cand = cats_c.get(category)
        if base.get("skipped"):
            continue
        if cand is None or cand.get("skipped"):
            missing.append(category)
            rows.append({
                "category": category,
                "status": "missing",
                "baseline_score": base.get("score"),
                "candidate_score": None,
                "delta": None,
                "baseline_passed": base.get("passed"),
                "baseline_total": base.get("total"),
                "flipped_pass_to_fail": [],
                "flipped_fail_to_pass": [],
            })
            continue
        base_items = _items_by_id(base)
        cand_items = _items_by_id(cand)
        pass_to_fail = []
        fail_to_pass = []
        for item_id, base_item in base_items.items():
            cand_item = cand_items.get(item_id)
            if cand_item is None:
                continue
            base_ok = float(base_item.get("score", 0)) >= 1.0
            cand_ok = float(cand_item.get("score", 0)) >= 1.0
            if base_ok and not cand_ok:
                pass_to_fail.append(item_id)
            elif not base_ok and cand_ok:
                fail_to_pass.append(item_id)
        base_score = float(base.get("score", 0.0))
        cand_score = float(cand.get("score", 0.0))
        delta = round(cand_score - base_score, 4)
        total = int(base.get("total") or cand.get("total") or 0)
        if max_drop is not None:
            tolerance = abs(float(max_drop))
        elif total > 0:
            tolerance = max(0.0, float(noise)) / total
        else:
            tolerance = 0.05
        tolerance = round(tolerance, 4)
        regressed = delta < -tolerance - 1e-9
        if regressed:
            regressions.append(category)
        rows.append({
            "category": category,
            "status": "regression" if regressed else "ok",
            "baseline_score": round(base_score, 4),
            "candidate_score": round(cand_score, 4),
            "delta": delta,
            "baseline_passed": base.get("passed"),
            "baseline_total": base.get("total"),
            "candidate_passed": cand.get("passed"),
            "candidate_total": cand.get("total"),
            "tolerance": tolerance,
            "flipped_pass_to_fail": sorted(pass_to_fail),
            "flipped_fail_to_pass": sorted(fail_to_pass),
        })
    # Overall score on the categories both runs have in common.
    common = [row["category"] for row in rows if row["status"] != "missing"]
    base_overall = round(sum(float(cats_b[c].get("score", 0)) for c in common) / len(common), 4) if common else None
    cand_overall = round(sum(float(cats_c[c].get("score", 0)) for c in common) / len(common), 4) if common else None
    verdict = "REGRESSION" if regressions else "OK"
    return {
        "baseline": baseline.get("config", {}).get("model_resolved"),
        "candidate": candidate.get("config", {}).get("model_resolved"),
        "baseline_file_mode": (baseline.get("config") or {}).get("mode"),
        "candidate_file_mode": (candidate.get("config") or {}).get("mode"),
        "noise_items": None if max_drop is not None else float(noise),
        "max_drop": None if max_drop is None else abs(float(max_drop)),
        "ignore": sorted(ignore),
        "rows": rows,
        "missing_categories": missing,
        "regressions": regressions,
        "comparable_categories": common,
        "baseline_overall_common": base_overall,
        "candidate_overall_common": cand_overall,
        "delta_overall_common": None if base_overall is None else round(cand_overall - base_overall, 4),
        "verdict": verdict,
    }


def format_table(diff: dict) -> str:
    lines = []
    lines.append("baseline : %s" % (diff.get("baseline") or "?"))
    lines.append("candidate: %s" % (diff.get("candidate") or "?"))
    if diff["max_drop"] is not None:
        rule = "max_drop %.3f" % diff["max_drop"]
    else:
        rule = "noise %.1f item(s)/category" % diff["noise_items"]
    lines.append("tolerance: %s%s" % (rule,
                                      (" | ignore: " + ",".join(diff["ignore"])) if diff["ignore"] else ""))
    lines.append("")
    header = "%-14s %9s %10s %8s %6s %14s %s" % (
        "category", "baseline", "candidate", "delta", "tol", "passed b->c", "pass->fail")
    lines.append(header)
    lines.append("-" * len(header))
    for row in diff["rows"]:
        if row["status"] == "missing":
            lines.append("%-14s %9s %10s %8s %6s %14s %s" % (
                row["category"], row["baseline_score"], "-", "-", "-",
                "%s/%s -> -" % (row["baseline_passed"], row["baseline_total"]), "MISSING"))
            continue
        lines.append("%-14s %9s %10s %+8.4f %6.3f %14s %s" % (
            row["category"], row["baseline_score"], row["candidate_score"], row["delta"],
            row.get("tolerance", 0.0),
            "%s/%s -> %s/%s" % (row["baseline_passed"], row["baseline_total"],
                                row["candidate_passed"], row["candidate_total"]),
            ",".join(row["flipped_pass_to_fail"]) or "-",
        ))
    lines.append("-" * len(header))
    lines.append("%-14s %9s %10s %8s" % (
        "overall(common)", diff["baseline_overall_common"], diff["candidate_overall_common"],
        "%+.4f" % diff["delta_overall_common"] if diff["delta_overall_common"] is not None else "-"))
    lines.append("")
    if diff["missing_categories"]:
        lines.append("missing/skipped categories: %s" % ", ".join(diff["missing_categories"]))
    if diff["regressions"]:
        lines.append("VERDICT: REGRESSION in %s" % ", ".join(diff["regressions"]))
    else:
        rule = ("max_drop %.3f" % diff["max_drop"] if diff["max_drop"] is not None
                else "noise %.1f item(s)/category" % diff["noise_items"])
        lines.append("VERDICT: OK (within %s)" % rule)
    flipped = sum(len(row.get("flipped_pass_to_fail") or []) for row in diff["rows"])
    if flipped:
        lines.append("note: %d item(s) flipped pass->fail (see pass->fail column); "
                     "small counts are within run-to-run noise." % flipped)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two quality scores.json files (acceptance gate).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("baseline", help="scores.json from the reference/baseline preset")
    parser.add_argument("candidate", help="scores.json from the candidate preset")
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE,
                        help="items of drift tolerated per category (1 = one item)")
    parser.add_argument("--max-drop", type=float, default=None,
                        help="absolute score tolerance; overrides --noise")
    parser.add_argument("--ignore", default="",
                        help="comma list of categories to exclude from the verdict")
    parser.add_argument("--require-categories", default="",
                        help="comma list of categories that must be present in both runs")
    parser.add_argument("--json-diff", default=None,
                        help="also write the comparison as JSON to this path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    ignore = [token.strip() for token in args.ignore.split(",") if token.strip()]
    require = [token.strip() for token in args.require_categories.split(",") if token.strip()]
    try:
        baseline = load_run(args.baseline)
        candidate = load_run(args.candidate)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    known = set((baseline.get("categories") or {}).keys()) | set((candidate.get("categories") or {}).keys())
    unknown_ignore = [name for name in ignore if name not in known]
    if unknown_ignore:
        print("error: --ignore names unknown categories: %s (known: %s)"
              % (", ".join(unknown_ignore), ", ".join(sorted(known))), file=sys.stderr)
        return 2
    unknown_require = [name for name in require if name not in known]
    if unknown_require:
        print("error: --require-categories names unknown categories: %s" % ", ".join(unknown_require),
              file=sys.stderr)
        return 2

    diff = compare(baseline, candidate, noise=args.noise, max_drop=args.max_drop, ignore=ignore)
    missing_required = [name for name in require if name in diff["missing_categories"]]
    if missing_required:
        diff["regressions"].extend(missing_required)
        diff["verdict"] = "REGRESSION"

    if not args.quiet:
        print(format_table(diff))
    if args.json_diff:
        path = Path(args.json_diff)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(diff, indent=2), encoding="utf-8")
        if not args.quiet:
            print("\njson diff: %s" % path)
    return 1 if diff["regressions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
