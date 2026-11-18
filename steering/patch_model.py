#!/usr/bin/env python3
"""Insert runtime abliteration + steering + optional activation capture into the v5.4
image's Qwen4Exp model source.

Three independent hooks, each gated by an environment variable:

1. ABLITERATION (STEERING_ABLITERATE=1) — real multi-layer refusal removal on *every*
   residual writer of every decoder layer, following OrcaRouter's abliteration map
   (lambda = 1) as done by the myllmbox hibrid48-uncensored checkpoint:

       W' = W - r (r^T W)        <=>     out' = out - r (r . out)

   applied to the two block outputs of each layer (`attn_out` = o_proj / GDN out_proj,
   `mlp_out` = routed experts + shared expert, already summed). This is the same linear
   map the uncensored weights carry, so int4/fp8 weights stay byte-identical without
   re-rounding. Because the projection is applied to EVERY residual writer, the residual
   stream stays on-manifold: no per-step norm drift, hence no repetition loops — unlike
   projecting only the final hidden state.

   Optional per-layer scaling: STEERING_ABLATE_SCALE=<s> multiplies r before projecting
   (s = 1.0 is the standard abliteration).

   Direction: STEERING_VECTOR_PATH must point at a unit-length tensor of shape
   [hidden_size] (single direction, applied to all layers) or [48, hidden_size]
   (one direction per layer).

2. OUTPUT STEERING (runtime style control) — STEERING_VECTOR_PATH + STEERING_STRENGTH,
   orthogonal projection on the FINAL residual stream (y = y - s (y.v) v). Kept for
   style experiments; NOT used for abliteration.

3. CAPTURE (STEERING_CAPTURE_DIR) — writes the post-block hidden state of the last
   prompt token for each eager (prefill) forward, used by steering/build_refusal_vector.py.

The abliteration hook runs inside CUDA-graph-captured decode and eager prefill alike and
never touches the MTP drafter (which reads the pre-mixer multi-stream buffer).

Usage:
    python3 steering/patch_model.py steering/model.py.orig steering/patched_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- anchor 1: block outputs of the decoder layer (abliteration) -------------------
ABL_ATTN_ANCHOR = """        mlp_hc = self.mlp_hyper_connection
        if self.use_hc_sequence_parallel:
            attn_out = sp_reduce_scatter(attn_out)
"""
ABL_ATTN_CALL = """        mlp_hc = self.mlp_hyper_connection
        if self.use_hc_sequence_parallel:
            attn_out = sp_reduce_scatter(attn_out)
"""

ABL_MLP_ANCHOR = """                    sp_reduce_scatter(mlp_out) if reduce_output else sp_shard(mlp_out)
                )
        return hidden_states, mlp_out, injection
"""
ABL_MLP_CALL = """                    sp_reduce_scatter(mlp_out) if reduce_output else sp_shard(mlp_out)
                )
        return hidden_states, mlp_out, injection
"""

# --- anchor 3: layer __init__ (preload the direction while no capture is active) -----
ABL_INIT_ANCHOR = """        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
            quant_config=quant_config,  # MBX hc quant
        )
"""
ABL_INIT_CALL = """        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
            quant_config=quant_config,  # MBX hc quant
        )
"""

# --- anchor 2: final residual stream (output steering + capture) --------------------
OUT_ANCHOR = "        return sample_hidden_states\n"
OUT_CALL = """        # --- runtime steering (steering/patch_model.py) ---
        _maybe_capture_steering(sample_hidden_states)
        sample_hidden_states = _apply_output_steering(sample_hidden_states)
        return sample_hidden_states
"""

HELPER = '''

# ===== runtime steering / abliteration, added by steering/patch_model.py ===========
_STEERING_STATE: dict = {}
_CAPTURE_STATE = {"n": 0}
_ABL_STATE: dict = {}





def _maybe_capture_steering(hidden_states):
    """Persist the final-residual last-token state per eager forward (vector building).

    Enabled with STEERING_CAPTURE_DIR=<writable dir>. CUDA-graph capture is skipped
    (CPU<->CUDA copies are forbidden during graph capture).
    """
    import json
    import os

    out_dir = os.environ.get("STEERING_CAPTURE_DIR", "")
    if not out_dir:
        return
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
    """Orthogonal-projection style steering on the FINAL residual stream (DS4 formula).

    y = y - strength * (y @ v) * v, v loaded once from $STEERING_VECTOR_PATH.
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


def _replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    n = text.count(anchor)
    if n != 1:
        raise SystemExit(f"[{label}] expected exactly 1 anchor, found {n}")
    return text.replace(anchor, replacement)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    text = src.read_text()
    text = _replace_once(text, ABL_ATTN_ANCHOR, ABL_ATTN_CALL, "abliterate-attn")
    text = _replace_once(text, ABL_MLP_ANCHOR, ABL_MLP_CALL, "abliterate-mlp")
    text = _replace_once(text, OUT_ANCHOR, OUT_CALL, "output-steering")
    text = text.rstrip("\n") + "\n" + HELPER
    dst.write_text(text)
    compile(dst.read_text(), str(dst), "exec")  # syntax check
    print(f"patched {src} -> {dst} (syntax OK, +{len(HELPER.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
