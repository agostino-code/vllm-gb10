#!/usr/bin/env python3
"""Patch hook: disable the fused qkv_proj projection for mixed-precision checkpoints.

`myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored` (and other compressed-tensors checkpoints)
store `q_proj`/`k_proj`/`v_proj` under *different* quantization schemes inside the same layer:
12 of the 48 layers carry int8 weights + `weight_scale`, the other 36 are bf16. vLLM's model
loader tries to fuse them into one `qkv_proj` module and then fails with

    ValueError: Found different quantization schemes for the shards of
    language_model.model.layers.3.self_attn.qkv_proj. vLLM requires all to use the same scheme.

Dropping `qkv_proj` from `packed_modules_mapping` keeps the three projections separate, which
is how the weights are actually stored, so the loader maps every tensor 1:1.

Run after steering/patch_model.py:
    python3 steering/patch_unfuse_qkv.py steering/patched_model.py steering/patched_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# The fused entry, exactly as it appears in the image's source.
OLD = '''        "qkv_proj": ["q_proj", "k_proj", "v_proj"] + ([] if (__import__("os").environ.get("VLLM_FP8_HYBRID", "0").lower() in ("1", "true", "yes") or __import__("os").environ.get("MBX_QSA_UNFUSE_INDEXER", "0") == "1") else ["indexer.index_qk_proj"]),  # MBX 27
'''

NEW = '''        # UNFUSED by steering/patch_unfuse_qkv.py: mixed-precision checkpoints keep
        # q_proj / k_proj / v_proj under different schemes per layer, so the fused module
        # cannot be built. The weights are already stored separately.
'''

# The second place that re-adds it for LoRA / hybrid builds.
OLD2 = '''            self.packed_modules_mapping = self.packed_modules_mapping | {
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            }'''


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    text = src.read_text()
    n = text.count(OLD)
    if n != 1:
        raise SystemExit(f"expected exactly 1 fused qkv_proj entry, found {n}")
    text = text.replace(OLD, NEW)
    # Two more places re-add the entry (LoRA/hybrid in ForCausalLM and the conditional-
    # generation class). Unfuse them too.
    old2_indented = """                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
"""
    n2 = text.count(old2_indented)
    text = text.replace(old2_indented, """                # UNFUSED by steering/patch_unfuse_qkv.py
""")
    print(f"  removed {n2} re-add site(s) of qkv_proj")

    # The conditional-generation class copies the entry from the causal-LM class.
    old3 = """        "qkv_proj": Qwen4ExpForCausalLM.packed_modules_mapping["qkv_proj"],
"""
    n3 = text.count(old3)
    text = text.replace(old3, """        # UNFUSED by steering/patch_unfuse_qkv.py (mixed-precision weights)
""")
    print(f"  removed {n3} inherited qkv_proj entry(ies)")
    dst.write_text(text)
    compile(dst.read_text(), str(dst), "exec")
    print(f"unfused qkv_proj in {dst} (syntax OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
