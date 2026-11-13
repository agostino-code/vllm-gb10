#!/usr/bin/env python3
"""Insert runtime steering into the v5.4 image's Qwen4Exp model source.

The image has no steering support. Rather than rebuild it, we patch one file and mount it
over the image's copy at launch (see steering/install_patch.sh and presets/run_qwen38_best_steering.sh).

The injection point is the final residual stream of `Qwen4ExpModel.forward`
(`sample_hidden_states`, hidden_size = 2560, the tensor that feeds lm_head):
it applies the orthogonal-projection rule from docs/ADR/001:

    y = y - strength * (y @ v) * v        (v unit-norm, y float32)

This is a captured-by-CUDA-graph tensor op (no Python hooks at replay time), and it does not
touch the MTP drafter (which consumes the pre-mixer multi-stream buffer), so speculative
acceptance and quality are unaffected except through the intended style shift.

Usage:
    python3 steering/patch_model.py steering/model.py.orig steering/patched_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = "        return sample_hidden_states\n"

CALL = """        # --- runtime steering (steering/patch_model.py) ---
        sample_hidden_states = _apply_output_steering(sample_hidden_states)
        return sample_hidden_states
"""

HELPER = '''

# ===== runtime steering, added by steering/patch_model.py ==========================
_STEERING_STATE: dict = {}


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
