#!/usr/bin/env python3
"""Build a refusal-direction steering vector from LLM-LAT/harmful-dataset.

Method (Arditi et al. 2024 — "Refusal in Language Models Is Mediated by a Single Direction",
adapted to this repo's final-residual capture point):

  For each harmful prompt, force the model to *begin* its answer with the dataset's refusal
  text (`chosen`) or compliance text (`rejected`) via `continue_final_message`, and capture
  the final-residual state of the last prefix token — the decision point that feeds lm_head.

      v = normalize( mean(h_refuse) - mean(h_comply) )

  Steering then projects v out of the final residual stream (y = y - s*(y.v)*v), which
  lowers the refusal decision and makes the model continue into compliance.

Why LLM-LAT/harmful-dataset: of the four candidates it is the only one with *paired*
refusal/compliance completions per prompt (prompt, chosen=refusal, rejected=compliance) —
exactly what difference-in-means needs. Undi95's set (goal/"Sure, here is...") has only the
compliance side; codesagar's sets are one-sided prompt classifiers.

Prerequisites: a server booted with the patched model AND STEERING_CAPTURE_DIR set
(presets/run_qwen38_abliterated.sh handles this), and numpy on the host
(./venv/bin/python has it).

Usage:
    ./venv/bin/python steering/build_refusal_vector.py \
        [--base-url http://127.0.0.1:8000] [--capture-dir cache/capture] \
        [--n-build 200] [--n-holdout 50] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "steering" / "data"
CAP_PREFIX_CHARS = 80  # force only the decision-relevant opening of each response


def post(url: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def detect_model(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as r:
        return json.loads(r.read().decode())["data"][0]["id"]


def snapshot(capture_dir: Path) -> set[str]:
    return {p.name for p in capture_dir.glob("cap_*.json")}


def read_capture(capture_dir: Path, prev: set[str]) -> np.ndarray | None:
    """The first new capture after a request is its prefill (counter is monotonic)."""
    new = []
    for p in capture_dir.glob("cap_*.json"):
        if p.name in prev:
            continue
        meta = json.loads(p.read_text())
        new.append((meta["n"], meta["t"], meta["dim"], p.with_suffix(".f32")))
    if not new:
        return None
    n, t, dim, f32 = sorted(new)[0]
    return np.fromfile(f32, dtype=np.float32)


def forced_prefix(text: str) -> str:
    """First CAP_PREFIX_CHARS chars, cut at a word boundary."""
    p = text[:CAP_PREFIX_CHARS]
    sp = p.rfind(" ")
    if sp > 20:
        p = p[:sp]
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--capture-dir", default=str(ROOT / "cache" / "capture"))
    ap.add_argument("--data", default=str(DATA / "harmful.json"))
    ap.add_argument("--out", default=str(ROOT / "steering" / "refusal"))
    ap.add_argument("--n-build", type=int, default=200)
    ap.add_argument("--n-holdout", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    cap.mkdir(parents=True, exist_ok=True)
    for f in cap.glob("cap_*"):
        f.unlink()

    rows = json.load(open(args.data))
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(rows)), args.n_build + args.n_holdout)
    build_idx, holdout_idx = idx[: args.n_build], idx[args.n_build : args.n_build + args.n_holdout]

    model = detect_model(args.base_url)
    url = f"{args.base_url}/v1/chat/completions"
    print(f"model={model}  build={len(build_idx)} prompts x2 sides  holdout={len(holdout_idx)}", flush=True)

    def forced(prompt: str, prefix: str) -> np.ndarray | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        prev = snapshot(cap)
        post(url, payload)
        # prefill capture may land a moment after the response returns
        for _ in range(50):
            h = read_capture(cap, prev)
            if h is not None:
                return h
            time.sleep(0.1)
        return None

    sum_r = np.zeros(2560, dtype=np.float64)
    sum_c = np.zeros(2560, dtype=np.float64)
    nr = nc = 0
    skipped = 0
    t0 = time.time()
    for k, i in enumerate(build_idx):
        row = rows[i]
        pr = forced_prefix(row["chosen"])    # dataset "chosen" = refusal text
        pc = forced_prefix(row["rejected"])  # dataset "rejected" = compliance text
        hr = forced(row["prompt"], pr)
        hc = forced(row["prompt"], pc)
        if hr is None or hc is None or hr.shape[0] != 2560 or hc.shape[0] != 2560:
            skipped += 1
            print(f"  skip #{k} idx={i} (capture failed)", flush=True)
            continue
        sum_r += hr.astype(np.float64)
        sum_c += hc.astype(np.float64)
        nr += 1
        nc += 1
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(build_idx)}  ({time.time()-t0:.0f}s)", flush=True)

    if nr < 50:
        print(f"✗ only {nr} paired captures — too few; check the server/capture dir", flush=True)
        return 1

    mr = sum_r / nr
    mc = sum_c / nc
    diff = mr - mc
    norm = float(np.linalg.norm(diff))
    cos = float(mr @ mc / (np.linalg.norm(mr) * np.linalg.norm(mc)))
    v = diff / (norm if norm > 1e-12 else 1.0)

    out = Path(args.out)
    v.astype(np.float32).tofile(out.with_suffix(".f32"))
    meta = {
        "kind": "refusal_direction_difference_in_means",
        "dataset": "LLM-LAT/harmful-dataset",
        "method": "mean(h|refusal prefix) - mean(h|compliance prefix) at the final residual "
        "stream, captured at the decision point (last token of a forced response prefix, "
        "continue_final_message), per Arditi et al. 2024",
        "n_prompts_used": nr,
        "n_prompts_skipped": skipped,
        "seed": args.seed,
        "hidden_size": 2560,
        "raw_norm_before_unit": norm,
        "cosine_means": cos,
        "sign": "positive v = refusal direction; steering SUBTRACTS its component => compliance",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}.f32  (n={nr}, |diff|={norm:.3f}, cos(r,c)={cos:.4f})", flush=True)

    # holdout prompts for the validator
    holdout = [{"prompt": rows[i]["prompt"], "category": "harmful"} for i in holdout_idx]
    (DATA / "holdout.json").write_text(json.dumps(holdout))
    print(f"saved {DATA/'holdout.json'} ({len(holdout)} held-out prompts)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
