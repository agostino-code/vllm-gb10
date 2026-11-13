#!/usr/bin/env python3
"""Build a steering vector for the AZ checkpoint from the model's own output head.

The final residual stream (hidden_size = 2560) feeds lm_head. Row t of lm_head is the
direction in that space that raises token t's logit. Averaging rows for "verbose style"
markers and subtracting the average for "concise style" markers gives a single direction
that, projected out of the final hidden state, lowers verbose-style token logits and
relatively raises concise ones (orthogonal projection, DS4-style per docs/ADR/001).

This runs inside the serving image (it has torch + safetensors) with the model snapshot
mounted read-only. CPU only; loads just the lm_head tensor from its shard.

Usage (see steering/build_vector.sh):
    python3 make_steering_vector.py --model /model --out /steering/succinct.pt
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

CONCISE_MARKERS = [
    " briefly",
    " concisely",
    " concise",
    " short",
    " terse",
    " compact",
    " minimal",
    " in short",
    " succinct",
    " one sentence",
]
VERBOSE_MARKERS = [
    " detailed",
    " thoroughly",
    " comprehensive",
    " expansive",
    " in depth",
    " elaborate",
    " extensively",
    " multi-paragraph",
    " thorough",
    " in detail",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--out", default="/steering/succinct.pt")
    ap.add_argument("--meta-out", default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=False, local_files_only=True
    )
    idx = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))
    shard = idx["weight_map"]["lm_head.weight"]
    print(f"lm_head shard: {shard}", flush=True)
    with safe_open(os.path.join(args.model, shard), framework="pt") as f:
        lm_head = f.get_tensor("lm_head.weight")  # [vocab, hidden]
    hidden = int(lm_head.shape[1])
    print(f"lm_head shape {tuple(lm_head.shape)} dtype {lm_head.dtype}", flush=True)

    def token_ids(words):
        ids = []
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
            else:
                print(f"  skip multi-token marker {w!r} -> {t}", flush=True)
        return ids

    concise_ids = token_ids(CONCISE_MARKERS)
    verbose_ids = token_ids(VERBOSE_MARKERS)
    if not concise_ids or not verbose_ids:
        raise SystemExit("no single-token style markers found")

    def mean_rows(ids):
        return lm_head[torch.tensor(ids, dtype=torch.long)].float().mean(0)

    v_concise = mean_rows(concise_ids)
    v_verbose = mean_rows(verbose_ids)

    # Direction to REMOVE: verbose style. Subtracting its projection makes the model less
    # verbose (the ADR formula: y = y - strength * (y @ v) * v, v unit-norm).
    v = v_verbose - v_concise
    v = v / v.norm().clamp_min(1e-8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(v.contiguous(), args.out)
    meta = {
        "kind": "lm_head_style_direction",
        "description": "unit vector: verbose-style direction removed by orthogonal projection "
        "on the final hidden state; positive strength => more concise answers",
        "model": os.path.basename(args.model),
        "hidden_size": hidden,
        "lm_head_shard": shard,
        "concise_markers": CONCISE_MARKERS,
        "verbose_markers": VERBOSE_MARKERS,
        "concise_token_ids": concise_ids,
        "verbose_token_ids": verbose_ids,
        "vector_norm_unscaled": float((v_verbose - v_concise).norm()),
    }
    meta_out = args.meta_out or os.path.splitext(args.out)[0] + ".json"
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved vector {args.out} dim={hidden} norm={float(v.norm()):.6f}", flush=True)
    print(f"saved meta   {meta_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
