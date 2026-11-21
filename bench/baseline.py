#!/usr/bin/env python3
"""baseline.py — measure every shipped configuration and print one pasteable report.

  ./bench/baseline.py                      # boot each config, measure, tear down
  ./bench/baseline.py --profile quick      # ~10 min total instead of ~40
  ./bench/baseline.py --only az-standard   # one config (repeatable)
  ./bench/baseline.py --attach URL         # measure a server you started yourself, no docker
  ./bench/baseline.py --report DIR         # rebuild the report from results you already have
  ./bench/baseline.py --dry-run            # show the plan, change nothing

The run writes bench/baseline/<run-id>/<config>/summary.json (+ quality.json, meta.json,
server.log) and a single consolidated BASELINE.md at the repo root. The same block is
printed to stdout so you can paste it straight into a conversation.

What it measures per config, and where the number comes from:

  decode      median of the per-category medians (code/json/math/qa), temp 0, thinking off
  C=1..16     total output tokens/s per concurrency level from the suite's concurrency phase
  TTFT/TPOT   p50 of the C=1 level
  prefill     tokens/s at 1k / 8k / 32k prompt targets
  spec        acceptance rate, tokens/step, avg draft depth, accepted tokens per position,
              read from the engine's /metrics deltas (not from the client)
  KV          pool size in tokens and the max-concurrency multiple
  quality     the deterministic gate: math/json/code/instruction/tool/repetition/needle
  boot        seconds from `docker run` to the first healthy /health

Safety: it refuses to start while another model server is running or the port is taken,
because this box serves one model at a time and the numbers are meaningless with two.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402

MATRIX = HERE / "matrix.yaml"
BASE = REPO / "bench" / "baseline"
REPORT = REPO / "BASELINE.md"

# The four shipped configurations, in the order that makes the report read well.
ORDER = ["az-standard", "az-524k-single", "uncensored-standard", "uncensored-524k-single"]


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def g(d, *path, default=None):
    """Nested get that never raises."""
    cur = d
    for k in path:
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list) and isinstance(k, int) and 0 <= k < len(cur):
            cur = cur[k]
        else:
            return default
        if cur is None:
            return default
    return cur


def num(v, fmt="{:.1f}"):
    if v is None:
        return "—"
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def pct(v):
    return "—" if v is None else f"{float(v) * 100:.1f} %"


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def running_servers() -> list[dict]:
    """Containers that look like a vLLM server (anything running `vllm serve`)."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Command}}"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    found = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, image, status = parts[0], parts[1], parts[2]
        try:
            cmd = subprocess.run(["docker", "inspect", name, "--format", "{{join .Config.Cmd \" \"}}"],
                                 capture_output=True, text=True, timeout=10).stdout
        except Exception:
            cmd = ""
        if "serve" in cmd or "vllm" in image:
            found.append({"name": name, "image": image, "status": status, "cmd": cmd.strip()[:120]})
    return found


def port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ──────────────────────────────────────────────────────────────────────────────
# the numbers
# ──────────────────────────────────────────────────────────────────────────────

def extract(name: str, run_dir: Path) -> dict:
    summary = load_json(run_dir / "summary.json") or {}
    meta = load_json(run_dir / "meta.json") or {}
    row = {"name": name, "ok": False, "errors": [], "notes": []}

    if not summary:
        status = load_json(run_dir / "status.json") or {}
        row["errors"].append(status.get("error") or status.get("status") or "no summary.json")
        return row

    row["ok"] = True
    ph = summary.get("phases") or {}
    dec = ph.get("decode") or {}
    conc = ph.get("concurrency") or {}
    pre = ph.get("prefill") or {}
    met = summary.get("metrics") or {}
    ov = met.get("overall") or {}
    srv = summary.get("server") or {}

    # decode: median of the per-category medians
    by_cat = dec.get("by_category") or {}
    cats = {c: (v or {}).get("out_tok_s", {}).get("median") for c, v in by_cat.items()}
    vals = [v for v in cats.values() if isinstance(v, (int, float))]
    row["decode"] = statistics.median(vals) if vals else None
    row["decode_cats"] = cats

    # concurrency ladder
    levels = {}
    for lv in conc.get("levels") or []:
        c = lv.get("c")
        if c is not None:
            levels[int(c)] = {
                "tok_s": lv.get("total_out_tok_s"),
                "per_stream": g(lv, "per_stream_tok_s", "median"),
                "ttft": g(lv, "ttft_ms", "median"),
                "tpot": g(lv, "tpot_ms", "median"),
                "errors": lv.get("n_errors", 0),
            }
    row["levels"] = levels
    row["c1"] = levels.get(1) or {}
    if any(l["errors"] for l in levels.values()):
        row["notes"].append("concurrency errors: " + ", ".join(
            f"C{c}={l['errors']}" for c, l in sorted(levels.items()) if l["errors"]))

    # prefill
    row["prefill"] = {k: g(v, "prefill_tok_s") for k, v in (pre.get("by_target") or {}).items()}

    # speculative decoding
    row["accept"] = ov.get("acceptance_rate")
    row["tps"] = ov.get("tokens_per_step")
    row["depth"] = ov.get("avg_draft_depth")
    row["accepted_per_pos"] = ov.get("accepted_per_position") or {}
    row["spec_config"] = g(meta, "vllm", "speculative_config") or g(srv, "speculative_config")

    # KV
    kv = srv.get("kv_cache") or {}
    row["kv_tokens"] = kv.get("tokens") or g(srv, "kv_cache_tokens")
    row["kv_multiple"] = kv.get("max_concurrency") or g(srv, "kv_cache_max_concurrency")
    row["kv_dtype"] = g(meta, "vllm", "kv_cache_dtype") or g(srv, "kv_cache_dtype")

    # boot + image
    row["boot_s"] = srv.get("boot_seconds")
    row["image"] = srv.get("image")
    row["image_id"] = (srv.get("image_id") or "")[:19]
    row["model"] = srv.get("model")
    row["container"] = srv.get("container")
    row["generated_at"] = summary.get("generated_at")
    row["profile"] = summary.get("profile")
    row["quality_mode"] = g(summary, "suite_args", "quality")

    # quality gate
    q = summary.get("quality") or {}
    scores = q.get("scores") or {}
    overall = scores.get("overall") or {}
    row["quality_score"] = overall.get("score")
    row["quality_cats"] = {c: (v or {}).get("score") for c, v in (scores.get("categories") or {}).items()}
    row["quality_needle"] = (scores.get("categories") or {}).get("long_context", {})
    if q and q.get("status") not in (None, "ok"):
        row["notes"].append(f"quality status={q.get('status')}")

    for e in summary.get("errors") or []:
        row["errors"].append(str(e.get("error"))[:200] if isinstance(e, dict) else str(e)[:200])
    return row


# ──────────────────────────────────────────────────────────────────────────────
# the report
# ──────────────────────────────────────────────────────────────────────────────

def fmt_row(r: dict) -> str:
    lv = r.get("levels") or {}
    def L(c, k):
        return num((lv.get(c) or {}).get(k))
    return (
        f"| {r['name']} | {num(r.get('decode'))} | {L(1,'tok_s')} | {L(4,'tok_s')} | "
        f"{L(8,'tok_s')} | {L(16,'tok_s')} | {num((r.get('c1') or {}).get('ttft'),'{:.0f}')} | "
        f"{num((r.get('c1') or {}).get('tpot'),'{:.1f}')} | "
        f"{num(r.get('prefill',{}).get('8192'),'{:.0f}')} | "
        f"{pct(r.get('accept'))} | {num(r.get('tps'),'{:.2f}')} | "
        f"{num(r.get('depth'),'{:.0f}')} | "
        f"{r['kv_tokens']:,} | {num(r.get('quality_score'),'{:.4f}')} | {num(r.get('boot_s'),'{:.0f}')} |"
    )


def build_report(rows: list[dict], env: dict) -> str:
    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    L = []
    a = L.append

    a("# Baseline")
    a("")
    a(f"- **run**: {env['run_id']}")
    a(f"- **generated**: {env['generated']}")
    a(f"- **box**: {env['box']}")
    a(f"- **image**: `{env['image']}` ({env['image_id']})")
    a(f"- **harness**: suite {env['harness']} · profile {env['profile']} · "
      f"quality {env['quality']}")
    a(f"- **host**: vm.compaction_proactiveness={env['compaction']}, "
      f"{env['mem_free']}G free at start, {env['n_configs']} configs")
    a(f"- **raw**: `{env['results_dir']}/<config>/summary.json`")
    a("")
    a("## Summary")
    a("")
    a("| config | decode | C=1 | C=4 | C=8 | C=16 | TTFT | TPOT | prefill 8k | accept | t/step | depth | KV tokens | quality | boot s |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ok:
        a(fmt_row(r))
    if bad:
        for r in bad:
            a(f"| {r['name']} | FAILED | | | | | | | | | | | | | |")
    a("")
    a("*decode = median of the per-category medians (code/json/math/qa), temperature 0, "
      "thinking off. TTFT/TPOT are p50 at C=1. accept/t/step/depth come from the engine "
      "`/metrics` deltas, not from the client. quality = the deterministic gate, quick or full.*")
    a("")

    if ok:
        a("## Decode by content")
        a("")
        cats = sorted({c for r in ok for c in (r.get("decode_cats") or {})})
        a("| config | " + " | ".join(cats) + " |")
        a("|---" * (len(cats) + 1) + "|")
        for r in ok:
            dc = r.get("decode_cats") or {}
            a(f"| {r['name']} | " + " | ".join(num(dc.get(c)) for c in cats) + " |")
        a("")

        a("## Concurrency ladder (total output tok/s)")
        a("")
        all_c = sorted({c for r in ok for c in (r.get("levels") or {})})
        a("| config | " + " | ".join(f"C={c}" for c in all_c) + " |")
        a("|---" * (len(all_c) + 1) + "|")
        for r in ok:
            lv = r.get("levels") or {}
            a(f"| {r['name']} | " + " | ".join(num((lv.get(c) or {}).get("tok_s")) for c in all_c) + " |")
        a("")

        a("## Accepted tokens per draft position")
        a("")
        a("The depth curve. On this checkpoint it historically dies after position 2 — "
          "anything that says otherwise is noise.")
        a("")
        maxp = 0
        for r in ok:
            aps = r.get("accepted_per_pos") or {}
            for k in aps:
                try:
                    maxp = max(maxp, int(k))
                except (TypeError, ValueError):
                    pass
        if maxp:
            pos = list(range(maxp + 1))
            a("| config | " + " | ".join(f"p{p}" for p in pos) + " |")
            a("|---" * (len(pos) + 1) + "|")
            for r in ok:
                aps = r.get("accepted_per_pos") or {}
                tot = sum(float(v) for v in aps.values() if v is not None) or 1
                cells = []
                for p in pos:
                    v = aps.get(str(p))
                    cells.append(f"{(float(v)/tot*100):.0f}%" if v is not None else "—")
                a(f"| {r['name']} | " + " | ".join(cells) + " |")
            a("")
            a("*percentages are the share of accepted tokens contributed by each position.*")
            a("")

        a("## Prefill (tok/s)")
        a("")
        tgts = sorted({k for r in ok for k in (r.get("prefill") or {})}, key=lambda x: int(x))
        a("| config | " + " | ".join(f"{t} tok" for t in tgts) + " |")
        a("|---" * (len(tgts) + 1) + "|")
        for r in ok:
            pf = r.get("prefill") or {}
            a(f"| {r['name']} | " + " | ".join(num(pf.get(t), "{:.0f}") for t in tgts) + " |")
        a("")

        a("## Quality gate")
        a("")
        qcats = sorted({c for r in ok for c in (r.get("quality_cats") or {})})
        a("| config | " + " | ".join(qcats) + " | overall |")
        a("|---" * (len(qcats) + 2) + "|")
        for r in ok:
            q = r.get("quality_cats") or {}
            a(f"| {r['name']} | " + " | ".join(num(q.get(c), "{:.2f}") for c in qcats)
              + f" | {num(r.get('quality_score'), '{:.4f}')} |")
        a("")

        a("## Configuration actually used")
        a("")
        a("| config | model | KV dtype | spec config |")
        a("|---|---|---|---|")
        for r in ok:
            a(f"| {r['name']} | {r.get('model') or '—'} | {r.get('kv_dtype') or '—'} | "
              f"`{str(r.get('spec_config'))[:70]}` |")
        a("")

    if bad:
        a("## Failures")
        a("")
        for r in bad:
            a(f"- **{r['name']}**: " + "; ".join(r.get("errors") or ["unknown"]))
        a("")

    notes = [(r["name"], n) for r in rows for n in (r.get("notes") or [])]
    if notes:
        a("## Warnings")
        a("")
        for name, n in notes:
            a(f"- **{name}**: {n}")
        a("")

    a("## Reproduce")
    a("")
    a("```bash")
    a(env["reproduce"])
    a(f"./bench/baseline.py --report {env['results_dir']}   # rebuild this file from the raw results")
    a("```")
    a("")
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────────────────
# the runner
# ──────────────────────────────────────────────────────────────────────────────

def host_facts() -> dict:
    box = "DGX Spark GB10"
    try:
        model = subprocess.run(["cat", "/sys/devices/soc0/machine"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        if model:
            box = model
    except Exception:
        pass
    try:
        comp = subprocess.run(["cat", "/proc/sys/vm/compaction_proactiveness"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        comp = "?"
    try:
        mem = subprocess.run(["free", "-g"], capture_output=True, text=True, timeout=5).stdout
        free = [w for w in mem.splitlines() if w.startswith("Mem:")][0].split()[6]
    except Exception:
        free = "?"
    return {"box": box, "compaction": comp, "mem_free": free}


def run_matrix(args, matrix_path: Path, results_dir: Path) -> int:
    cmd = [sys.executable, str(HERE / "run_matrix.py"),
           "--matrix", str(matrix_path),
           "--results-dir", str(results_dir)]
    for n in args.only or []:
        cmd += ["--only", n]
    if args.force:
        cmd.append("--force")
    if args.fail_fast:
        cmd.append("--fail-fast")
    if args.dry_run:
        cmd.append("--dry-run")
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO))


def write_matrix(args, run_id: str) -> Path:
    """bench/matrix.yaml + the baseline overrides (quality on, results dir)."""
    raw = C.load_yaml(MATRIX)
    defaults = dict(raw.get("defaults") or {})
    defaults["results_dir"] = str(BASE / run_id)
    suite = dict(defaults.get("suite") or {})
    suite["quality"] = args.quality
    defaults["suite"] = suite
    raw["defaults"] = defaults
    out = BASE / f"{run_id}.matrix.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    import yaml  # bench/common.py vendors its own minimal loader; use it for the dump
    try:
        out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        # no yaml writer available: fall back to run_matrix's own loader by editing nothing
        out.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--profile", default="full", choices=["quick", "full"],
                    help="suite profile (default full)")
    ap.add_argument("--quality", default="full", choices=["off", "quick", "full"],
                    help="quality gate mode (default full)")
    ap.add_argument("--only", action="append", metavar="NAME")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--attach", metavar="URL", help="measure a server already running at URL")
    ap.add_argument("--report", nargs="?", const="", metavar="DIR",
                    help="rebuild BASELINE.md from an existing results dir")
    args = ap.parse_args(argv)

    results_dir = BASE / args.run_id

    # ---- report only -------------------------------------------------------
    if args.report is not None and args.report != "":
        results_dir = Path(args.report)
        if not results_dir.is_absolute():
            results_dir = REPO / results_dir
        return emit(results_dir, results_dir.name, quiet=False)

    if args.report == "":
        # --report with no argument: newest baseline
        cands = sorted((p for p in BASE.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if not cands:
            print("no baseline results yet — run ./bench/baseline.py first", file=sys.stderr)
            return 2
        return emit(cands[-1], cands[-1].name, quiet=False)

    # ---- preflight ---------------------------------------------------------
    servers = running_servers()
    if args.attach:
        print(f"· attach mode: measuring {args.attach}, no container is started or stopped")
    else:
        if servers:
            print("✗ refusing to start: a model server is already running.", file=sys.stderr)
            for s in servers:
                print(f"    {s['name']}  {s['image']}  {s['status']}", file=sys.stderr)
            print("    This box serves one model at a time and the harness enforces it.\n"
                  f"    Stop it with: ./stop.sh {servers[0]['name']}\n"
                  "    Or measure it directly: ./bench/baseline.py --attach http://127.0.0.1:8000",
                  file=sys.stderr)
            return 2
        if port_open(8000):
            print("✗ port 8000 is already answering. Stop whatever is on it first.", file=sys.stderr)
            return 2

    # ---- run ---------------------------------------------------------------
    if not args.attach:
        matrix_path = write_matrix(args, args.run_id)
        rc = run_matrix(args, matrix_path, results_dir)
        if args.dry_run:
            return rc
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        for name in (args.only or ORDER):
            d = results_dir / name
            d.mkdir(exist_ok=True)
            cmd = [sys.executable, str(HERE / "suite.py"), "--base-url", args.attach,
                   "--run-dir", str(d), "--config-id", name, "--profile", args.profile,
                   "--tag", name, "--quality", args.quality]
            print(" ".join(cmd))
            if not args.dry_run:
                rc = subprocess.call(cmd, cwd=str(REPO))
                if rc and args.fail_fast:
                    break

    if args.dry_run:
        return 0
    return emit(results_dir, args.run_id)


def emit(results_dir: Path, run_id: str, quiet: bool = False) -> int:
    if not results_dir.is_absolute():
        results_dir = REPO / results_dir
    if not results_dir.is_dir():
        print(f"✗ no results at {results_dir}", file=sys.stderr)
        return 2

    names = [p.name for p in sorted(results_dir.iterdir()) if p.is_dir()]
    names.sort(key=lambda n: ORDER.index(n) if n in ORDER else 99)
    if not names:
        print(f"✗ {results_dir} has no config directories", file=sys.stderr)
        return 2

    rows = [extract(n, results_dir / n) for n in names]
    images = sorted({r["image"] for r in rows if r.get("image")})
    profiles = sorted({r["profile"] for r in rows if r.get("profile")})
    qmodes = sorted({r["quality_mode"] for r in rows if r.get("quality_mode")})
    env = {
        "run_id": run_id,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip(),
        "box": host_facts()["box"],
        "image": images[0] if len(images) == 1 else ", ".join(images) or "—",
        "image_id": next((r["image_id"] for r in rows if r.get("image_id")), "—"),
        "harness": "1.0",
        "profile": profiles[0] if len(profiles) == 1 else ", ".join(profiles) or "—",
        "quality": qmodes[0] if len(qmodes) == 1 else (", ".join(qmodes) or "off"),
        "compaction": host_facts()["compaction"],
        "mem_free": host_facts()["mem_free"],
        "n_configs": len(names),
        "results_dir": str(results_dir.relative_to(REPO)) if results_dir.is_relative_to(REPO) else str(results_dir),
        "reproduce": f"./bench/baseline.py --run-id {run_id}   # resume (add --force to re-measure)",
    }
    text = build_report(rows, env)
    REPORT.write_text(text, encoding="utf-8")
    if not quiet:
        print()
        print("=" * 78)
        print(text)
        print("=" * 78)
        print(f"\n✓ written to {REPORT.relative_to(REPO)}")
    return 0 if all(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
