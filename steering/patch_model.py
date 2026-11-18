#!/usr/bin/env python3
"""Insert runtime steering (+ optional activation capture) into the v5.4 image's Qwen4Exp model source.

The image has no steering support. Rather than rebuild it, we patch one file and mount it
over the image's copy at launch (see steering/install_patch.sh and presets/run_qwen38_best_steering.sh).

The injection point is the final residual stream of `Qwen4ExpModel.forward`
(`sample_hidden_states`, hidden_size = 2560, the tensor that feeds lm_head):
it applies the orthogonal-projection rule from docs/ADR/001:

    y = y - strength * (y @ v) * v        (v unit-norm, y float32)

This is a captured-by-CUDA-graph tensor op (no Python hooks at replay time), and it does not
touch the MTP drafter (which consumes the pre-mixer multi-stream buffer), so speculative
acceptance and quality are unaffected except through the intended style shift.

When STEERING_CAPTURE_DIR is set, every forward also writes the last row of the
*pre-steering* hidden state to that directory (prefill runs eager, so Python executes):
used by steering/build_refusal_vector.py to collect decision-point activations.

Usage:
    python3 steering/patch_model.py steering/model.py.orig steering/patched_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = "        return sample_hidden_states\n"

CALL = """        # --- runtime steering (steering/patch_model.py) ---
        _maybe_capture_steering(sample_hidden_states)
        sample_hidden_states = _apply_output_steering(sample_hidden_states)
        return sample_hidden_states
"""

HELPER = '''

# ===== runtime steering, added by steering/patch_model.py ==========================
_STEERING_STATE: dict = {}
_CAPTURE_STATE = {"n": 0}


def _maybe_capture_steering(hidden_states):
    """Persist the final-residual last-token state per forward (for direction building).

    Enabled with STEERING_CAPTURE_DIR=<writable dir>. Prefill executes Python (eager),
    so each request's prefill shows up here. Decode runs in CUDA graphs; we skip those
    because CPU<->CUDA copies are forbidden during graph capture.
    """
    import json
    import os

    out_dir = os.environ.get("STEERING_CAPTURE_DIR", "")
    if not out_dir:
        return
    # Skip during CUDA graph capture: CPU<->CUDA copies are not allowed.
    try:
        import torch as _torch
        if _torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        pass
    try:
        n = _CAPTURE_STATE["n"]
        _CAPTURE_STATE["n"] = n + 1
        row = hidden_states[-1].detach().float().cpu()
        t = int(hidden_states.shape[0])
        os.makedirs(out_dir, exist_ok=True)
        # raw float32 + json sidecar: readable without torch (host numpy)
        row.numpy().tofile(f"{out_dir}/cap_{n:08d}.f32")
        with open(f"{out_dir}/cap_{n:08d}.json", "w") as f:
            json.dump({"n": n, "t": t, "dim": int(row.numel())}, f)
        if n == 0:
            print(f"[steering] capture active -> {out_dir} (dim={row.numel()})", flush=True)
    except Exception as e:  # never break serving for capture
        if not _CAPTURE_STATE.get("err"):
            print(f"[steering] capture disabled: {e}", flush=True)
            _CAPTURE_STATE["err"] = True


def _apply_output_steering(hidden_states):
    """Orthogonal-projection steering on the final residual stream (DS4 formula).

    y = y - strength * (y @ v) * v, with v loaded once from $STEERING_VECTOR_PATH.
    Disabled when STEERING_VECTOR_PATH is unset or STEERING_STRENGTH is 0.
    """
    import os

    path = os.environ.get("STEERING_VECTOR_PATH", "")
    if not path:
        return hidden_states
    strength = float(os.environ.get("STEERING_STRENGTH", "1.0") or 0.0)
    if strength == 0.0:
        return hidden_states
    state = _STEERING_STATE
    vec = state.get("vec")
    if vec is None or vec.device != hidden_states.device:
        import torch as _torch

        try:
            loaded = _torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # older torch without weights_only
            loaded = _torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            loaded = loaded.get("vector", loaded)
        vec = loaded.float().flatten()
        width = int(hidden_states.shape[-1])
        if vec.numel() != width:
            raise ValueError(
                f"steering vector dim {vec.numel()} != model hidden size {width}"
            )
        vec = vec / vec.norm().clamp_min(1e-8)
        vec = vec.to(hidden_states.device)
        state["vec"] = vec
        if not state.get("logged"):
            print(
                f"[steering] active on final residual stream: {path} "
                f"dim={width} strength={strength}",
                flush=True,
            )
            state["logged"] = True
    h = hidden_states.float()
    proj = (h @ vec).unsqueeze(-1) * vec
    return (h - strength * proj).to(hidden_states.dtype)
'''


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    text = src.read_text()
    n = text.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"expected exactly 1 anchor in {src}, found {n}")
    text = text.replace(ANCHOR, CALL)
    text = text.rstrip("\n") + "\n" + HELPER
    dst.write_text(text)
    compile(dst.read_text(), str(dst), "exec")  # syntax check
    print(f"patched {src} -> {dst} (syntax OK, +{len(HELPER.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
