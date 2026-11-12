#!/usr/bin/env python3
"""Generate the draft-lab speculative-decoding recipe matrix (draft-lab/recipes/*.yaml).

One table -> ready-to-launch recipe YAMLs (repo format: server:/env:/vllm:) plus
draft-lab/matrix.yaml.  Every recipe carries the FULL env/vllm section and EXACTLY ONE
speculative-config, so bench/serve.py can launch it directly (it parses recipes itself; nothing is
appended from config.env).  When falling back to ./run.sh, launch with MTP=0 so run.sh does not
append a second --speculative-config.

Run from anywhere:   python3 draft-lab/gen_recipes.py
Validate only:       python3 draft-lab/gen_recipes.py --check
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECIPES = os.path.join(HERE, "recipes")
MATRIX = os.path.join(HERE, "matrix.yaml")
BENCH_MATRIX = os.path.join(HERE, "bench-matrix.yaml")
RUNSH_RECIPES = os.path.join(HERE, "recipes-runsh")
RUNSH_MODEL = "/models/draft-k10"   # run.sh mounts ./models at /models (workspace root is NOT mounted)

MODEL = "azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ"
IMAGE = "myllmbox/qwen38-flash-next-vllm:v5.4"
SNAPSHOT = ("/home/pgx/.cache/huggingface/hub/models--azampatti--Qwen3.8-Flash-Next-FP4-Local-AZ/"
            "snapshots/5d927eabd4e4766a69bf12f7a1387b01d936a78f")
DRAFTS = {
    "k10": os.path.join(ROOT, "draft-lab", "drafts", "qwen38-az-draft-k10"),
    "k5": os.path.join(ROOT, "draft-lab", "drafts", "qwen38-az-draft-k5"),
}
WORKSPACE = ROOT

# --------------------------------------------------------------------------------------------------
# baseline = the measured myllmbox v5.4 reference (M0): self-draft K12, dynamic depth, block
# verification + probabilistic draft sampling, NVFP4 drafter head.  Same as
# recipes/qwen38-flash-solo-optimized.yaml, plus every tested knob written explicitly.
# --------------------------------------------------------------------------------------------------
SERVER = [
    ("image", "str", IMAGE),
    ("docker_args", "str", "--security-opt seccomp=unconfined"),
    ("cpuset", "str", "5-9,15-19"),
    ("model", "str", MODEL),
    ("port", "int", 8000),
    ("models_dir", "str", "./models"),
    ("cache_dir", "str", "./cache"),
    ("patches", "str", ""),
]

BASE_ENV = [
    ("MBX_PLE_MMAP", "1"),
    ("MBX_PLE_MMAP_MODE", "nvme"),
    ("MBX_PLE_MMAP_PREWARM", "0"),
    ("MBX_PLE_ASYNC", "1"),
    ("VLLM_USE_BREAKABLE_CUDAGRAPH", "1"),
    ("MBX_LOADER_PREPARE", "0"),
    ("MBX_LOADER_EMPTY_CACHE", "1"),
    ("MBX_MARLIN_CSF", "1"),
    ("MBX_CSF_VISUAL_BF16", "0"),
    ("MBX_MXFP8_A16", "0"),
    ("MBX_PLE_PREFILL_CKPT", "1"),
    ("VLLM_DISABLED_KERNELS", "FlashInferCutedslMxfp8LinearKernel,"
                               "FlashInferTrtllmMxfp8LinearKernel,"
                               "DeepGemmMxfp8BmmLinearKernel,HummingMxfp8LinearKernel"),
    ("MBX_DEPTH_SYNCFREE", "1"),
    ("MBX_DEPTH_MODE", "dynamic"),
    ("MBX_DEPTH_MIN", "3"),
    ("MBX_DEPTH_WINDOW", "48"),
    ("MBX_DEPTH_PROMOTE", "60,45"),
    ("MBX_DEPTH_DEMOTE", "25,15"),
    ("MBX_DEPTH_CEIL", "[4 -> 8, 6 -> 7, 12 -> 6, 14 -> 5]"),
    ("MBX_DRAFT_HEAD", "1"),
    ("MBX_MTP_DRAFT_SCALE", "1"),
    ("MBX_DRAFT_FP8", "0"),
    ("PYTHONUNBUFFERED", "1"),
    ("VLLM_MARLIN_USE_ATOMIC_ADD", "1"),
    ("TORCH_CUDA_ARCH_LIST", "12.1a"),
    ("FLASHINFER_CUDA_ARCH_LIST", "12.1a"),
    ("TRITON_CACHE_DIR", "/cache/triton"),
    ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    ("MALLOC_MMAP_THRESHOLD_", "65536"),
    ("MALLOC_TRIM_THRESHOLD_", "131072"),
]
BASE_ENV_D = dict(BASE_ENV)

# v5.4 known issue: MBX_DEPTH_MODE=off is NOT bootable. The patched speculator then falls through to
# vLLM's dynamic-SD path, which at CUDA-graph capture raises
#   ValueError: num_speculative_tokens_per_batch_size is required for dynamic speculative decoding
# (observed in Lead runs d-fixed-5 and d-fixed-3, see verification/boot-failure-static-depth.log).
# Bootable "static depth K" = keep the dynamic path and pin it so it can never move:
STATIC_DEPTH_LAW = {
    "MBX_DEPTH_MODE": "dynamic",
    "MBX_DEPTH_WINDOW": "48",
    "MBX_DEPTH_PROMOTE": "101,101",   # unreachable acceptance -> never promote
    "MBX_DEPTH_DEMOTE": "0,0",        # never demote
    "MBX_DEPTH_SYNCFREE": "1",
}


def is_static(v):
    return v["k"] is not None and v["depth"] == "off"


# v5.4 known issue (Lead phase-B runs b-curve-k3 / b-k10-curve-k3): speculative depth < 4 does not
# boot at all; the image's dynamic-SD validation needs num_speculative_tokens_per_batch_size, which it
# only supplies for depth >= 4. Depth 0 (no speculative config) still boots.
MIN_BOOTABLE_SPEC_DEPTH = 4


def is_bootable(v):
    return v["k"] is None or v["k"] >= MIN_BOOTABLE_SPEC_DEPTH

BASE_VLLM = [
    ("served-model-name", "str", MODEL),
    ("engram-config", "json", '{"cpu_offload": false}'),
    ("moe-backend", "str", "marlin"),
    ("kernel-config", "json", '{"linear_backend_per_quant":{"mxfp8":"b12x"}}'),
    ("load-format", "str", "mbx"),
    ("distributed-executor-backend", "str", "mp"),
    ("gpu-memory-utilization", "float", 0.88),
    ("kv-cache-memory", "str", "23000000000"),
    ("kv-cache-dtype", "str", "auto"),
    ("mamba-ssm-cache-dtype", "str", "bfloat16"),
    ("max-model-len", "int", 409600),
    ("hf-overrides", "json",
     '{"text_config":{"max_position_embeddings":409600,"rope_parameters":{"rope_type":"yarn",'
     '"factor":1.5625,"original_max_position_embeddings":262144,"rope_theta":10000000,'
     '"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'),
    ("block-size", "int", 3456),
    ("prefix-match-unit", "int", 64),
    ("compilation-config", "json",
     '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":'
     '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,21,24,25,27,30,32,35,39,40,44,48,49,54,60,'
     '64,70,75,80,90,96]}'),
    ("max-num-seqs", "int", 16),
    ("max-num-batched-tokens", "int", 8192),
    ("enable-prefix-caching", "bool", True),
    ("enable-prompt-tokens-details", "bool", True),
    ("enable-auto-tool-choice", "bool", True),
    ("tool-call-parser", "str", "qwen3_xml"),
    ("reasoning-parser", "str", "qwen3"),
    ("override-generation-config", "json",
     '{"temperature": 0.6, "presence_penalty": 0.1, "top_p": 0.95}'),
    # speculative-config inserted here per variant
    ("async-scheduling", "bool", True),
    ("use-replayssm", "bool", True),
]

REF_NAME = "dynamic-ceiling-12-ref"

# First measured result (Lead matrix run 2026-11-12, bench/results/d-draftk10-spec6). Kept here so the
# generated manifest carries the evidence with the config. Full detail: draft-lab/MEASURED.md.
MEASURED = {
    "draftk10-spec6": {
        "run": "bench/results/d-draftk10-spec6",
        "status": "booted and benchmarked OK",
        "draft_folder_loaded": "drafter reads 3 of 3 files; 4639 tensors, 3 files, 4.1 GB in 4.6 s",
        "acceptance_rate": 0.3396,
        "tokens_per_step": 3.0378,
        "avg_draft_depth": 6.0,
        "accepted_per_position": {"0": 1033, "1": 816, "2": 677, "3": 9, "4": 2, "5": 0},
        "decode_out_tok_s_p50": 56.4,
        "boot_seconds": 350.2,
        "quality_quick": 0.9667,
        "m0_reference": {"acceptance_rate": 0.1697, "tokens_per_step": 3.0359,
                         "avg_draft_depth": 12.0, "decode_out_tok_s_p50": 53.3,
                         "quality_quick": 0.9667},
        "caveat": "acceptance_rate divides by ~K draft tokens, so compare tokens/step + avg_draft_depth "
                  "across K; same-K control (draftk5-spec6 or curve-k6) needed to attribute the 10-expert "
                  "routing effect.",
    },
}

# --------------------------------------------------------------------------------------------------
# variant table
#   drafter: self | k10 | k5; k: num_speculative_tokens (None = MTP off)
#   depth: dynamic | off; head/scale/fp8: MBX_* values; ceil: MBX_DEPTH_CEIL override
# --------------------------------------------------------------------------------------------------
V = []


def variant(name, *, doc, drafter="self", k=12, depth="dynamic", head=1, scale=1, fp8=0,
            rejection="block", draft_sample="probabilistic", ceil=None, warm=False,
            tier=3, priority=None, skip_rerun=False, notes=""):
    V.append(dict(name=name, doc=doc, drafter=drafter, k=k, depth=depth, head=head, scale=scale,
                  fp8=fp8, rejection=rejection, draft_sample=draft_sample, ceil=ceil,
                  warm=warm, tier=tier, priority=priority, skip_rerun=skip_rerun, notes=notes))


# ---- tier 1: the Lead's priority scan (must run without editing anything) ----------------------
variant("dynamic-ceiling-9", tier=1, priority=1, k=9, depth="dynamic",
        doc="dynamic speculative depth with max depth (ceiling) 9 instead of 12; ladder unchanged",
        notes="Lead priority (a). K=9 is the dynamic ceiling: at <=3 concurrent requests the depth "
              "can reach 9; the [4 -> 8, ...] ladder still lowers it under load.")
variant("fixed-5", tier=1, priority=2, k=5, depth="off",
        doc="STATIC depth 5 emulated on the dynamic path: MIN=CEIL=5 ([1 -> 5]), PROMOTE=101,101 "
            "(never promote), DEMOTE=0,0 (never demote)",
        notes="Lead priority (b). MBX_DEPTH_MODE=off is not bootable in v5.4 (ValueError at cudagraph "
              "capture); this is the pinned-ladder equivalent. Compare with curve-k5 (same K, dynamic).")
variant("draftk10-spec6", tier=1, priority=3, drafter="k10", k=6, depth="dynamic",
        doc='draft folder qwen38-az-draft-k10 (drafter top-k 10), K=6, dynamic depth',
        notes="Lead priority (c) and the highest-value hypothesis: the MTP layer was trained at 10 "
              "experts, so the draft folder restores its native routing. Uses \"model\": "
              "draft folder in --speculative-config.")
variant("self-head0", tier=1, priority=4, head=0,
        doc="MBX_DRAFT_HEAD=0: drafter scores with the full bf16 target head (no NVFP4 copy)",
        notes="Lead priority (d). Baseline has MBX_DRAFT_HEAD=1 (0.36 GB NVFP4 copy per rank). "
              "Tests whether the quantized drafter head costs acceptance.")
variant("self-scale2", tier=1, priority=5, scale=2,
        doc="MBX_MTP_DRAFT_SCALE=2: drafter logits sharpened x2 before the accept test",
        notes="Lead priority (e). Exact: the verifier still uses the target's distribution, so "
              "output text should not change; acceptance and tok/s may rise.")
variant("fixed-9", tier=1, priority=6, k=9, depth="off",
        doc="STATIC depth 9 emulated on the dynamic path: MIN=CEIL=9 ([1 -> 9]), PROMOTE=101,101, "
            "DEMOTE=0,0",
        notes="Lead priority (f). Pinned-ladder equivalent of static K=9 (MBX_DEPTH_MODE=off does not "
              "boot in v5.4); pair with dynamic-ceiling-9 for the same max depth, adaptive vs pinned.")

# ---- tier 2: the Lead's second wave -------------------------------------------------------------
variant("self-draftfp8", tier=2, priority=7, fp8=1,
        doc="MBX_DRAFT_FP8=1: fp8 drafter linears (faster drafts, maybe lower acceptance)",
        notes="Lead priority (h). Throughput vs acceptance trade-off for the drafter body, not the "
              "head; baseline drafter linears are bf16.")
variant("draftk10-spec9", tier=2, priority=8, drafter="k10", k=9, depth="dynamic",
        doc='draft folder qwen38-az-draft-k10, K=9, dynamic depth',
        notes="Lead priority: draft-k10 at the dynamic ceiling 9 point.")
variant("draftk5-spec6", tier=2, priority=9, drafter="k5", k=6, depth="dynamic",
        doc='draft folder qwen38-az-draft-k5 (top-k 5 control), K=6, dynamic depth',
        notes="Control: a draft folder identical to the target's own routing (5 experts). It must "
              "behave like the self-draft at the same K; if it does not, the draft-folder plumbing "
              "is wrong.")
variant(REF_NAME, tier=2, priority=10, k=12, depth="dynamic",
        doc="REFERENCE M0: self-draft K=12, dynamic depth, ladder ceiling, head NVFP4, scale 1",
        notes="Already measured by bench-harness as the myllmbox reference (M0). CONTROL: do not "
              "re-run in the scan; use its numbers as the baseline for every acceptance delta.",
        skip_rerun=True)
variant("no-spec", tier=2, priority=11, k=None,
        doc="no speculative decoding at all (token-0 / MTP off point of the curve)",
        notes="Lead priority (j). Emits NO --speculative-config (a literal 0 is not a valid vLLM "
              "value); measures the no-draft floor.")
variant("draftk10-dynamic", tier=2, priority=12, drafter="k10", k=12, depth="dynamic",
        doc="draft folder qwen38-az-draft-k10 with dynamic depth at ceiling K=12",
        notes="The Lead-requested draft-k10 + dynamic-depth variant: 10-expert MTP routing AND "
              "concurrency-adaptive depth. Compare with draftk10-spec6/spec9 and the M0 ref.")

# ---- tier 3: mechanisms and the rest of the token curve ----------------------------------------
variant("self-greedy", tier=3, priority=13, draft_sample="greedy",
        doc="draft_sample_method=greedy (baseline is probabilistic)",
        notes="Lead tier-3 request: greedy vs probabilistic draft sampling (block verification "
              "unchanged). Community mitigation for the MTP degenerate-loop bug.")
variant("self-rejection-standard", tier=3, priority=14, rejection="standard",
        doc="rejection_sample_method=standard (baseline is block/joint verification)",
        notes="Lead tier-3 request: standard vs block verification; block is the repo default "
              "(+2.4pp acceptance with probabilistic draft sampling).")
variant("fixed-3", tier=3, priority=15, k=3, depth="off",
        doc="STATIC depth 3 emulated on the dynamic path: MIN=CEIL=3 ([1 -> 3]), PROMOTE=101,101, "
            "DEMOTE=0,0",
        notes="Lead tier-3 request; old config.env default depth. Pinned-ladder equivalent of static "
              "K=3 (MBX_DEPTH_MODE=off does not boot in v5.4). The Lead's phase-B matrix also has "
              "self-fixed3 / draftk10-fixed3 built with the same depth-law.")
variant("curve-k1", tier=3, k=1, depth="dynamic",
        doc="token curve point: K=1, dynamic depth, self-draft",
        notes="Token curve 0/1/3/4/5/6/9/12: K=1 with the depth ladder able to stay at 1.")
variant("curve-k3", tier=3, k=3, depth="dynamic",
        doc="token curve point: K=3, dynamic depth, self-draft",
        notes="Token curve point (dynamic, ceiling 3).")
variant("curve-k4", tier=3, k=4, depth="dynamic",
        doc="token curve point: K=4, dynamic depth, self-draft",
        notes="Token curve point (dynamic, ceiling 4); block-size 3456 is divisible by 16/64 so no "
              "padding problem.")
variant("curve-k5", tier=3, k=5, depth="dynamic",
        doc="token curve point: K=5, dynamic depth, self-draft",
        notes="Same K as fixed-5 but with the dynamic depth ladder active.")
variant("curve-k6", tier=3, k=6, depth="dynamic",
        doc="token curve point: K=6, dynamic depth, self-draft",
        notes="Token curve point (dynamic, ceiling 6); anchor for the draftk10-spec6 comparison.")
variant("self-scale3", tier=3, scale=3,
        doc="MBX_MTP_DRAFT_SCALE=3 (scale ladder: 1 baseline / 2 tier-1 / 3 here)",
        notes="Completes the scale 1/2/3 axis; scale 2 is tier-1. Sharpening too much can hurt when "
              "the drafter is confident but wrong, so this is exploratory.")
variant("draftk10-head0", tier=3, drafter="k10", head=0,
        doc="draft-k10 with MBX_DRAFT_HEAD=0 (bf16 drafter head)",
        notes="Interaction of the two highest-value axes: 10-expert routing + unquantized head.")
variant("draftk10-greedy", tier=3, drafter="k10", draft_sample="greedy",
        doc="draft-k10 with draft_sample_method=greedy",
        notes="Greedy sampling applied to the 10-expert drafter (loop mitigation check).")
variant("draftk5-spec9", tier=3, drafter="k5", k=9, depth="dynamic",
        doc="draft-k5 control at K=9",
        notes="Control at the dynamic-ceiling-9 point; should match dynamic-ceiling-9.")
variant("draftk10-spec3", tier=3, drafter="k10", k=3, depth="dynamic",
        doc="draft-k10 at low depth K=3",
        notes="Low-depth point of the draft-k10 curve; cheap compute, check whether 10-expert "
              "routing still buys acceptance.")
variant("ceil-flat9", tier=3, ceil="[1 -> 9]",
        doc="K=12 but MBX_DEPTH_CEIL=[1 -> 9]: hard cap depth at 9 for EVERY concurrency",
        notes="MBX_DEPTH_CEIL syntax is 'concurrency-threshold -> max-depth'. The baseline ladder "
              "only caps under load; this cap is flat. Compare with dynamic-ceiling-9 (K=9).")
variant("ceil-aggressive", tier=3, ceil="[8 -> 8, 16 -> 6]",
        doc="K=12, ceiling held high longer: [8 -> 8, 16 -> 6]",
        notes="Keep depth 9-12 up to 7 concurrent requests, 8 at 8-15, 6 at 16+. Exploratory: the "
              "tail latency at high concurrency decides whether this is worth it.")
variant("warmdrafter", tier=3, warm=True,
        doc="MBX_B12X_WARM_DRAFTER=1 (warm the drafter during model load)",
        notes="Low priority (Lead). Value semantics are not documented in the image; check the "
              "boot log for a drafter-warmup line. Does not change acceptance, only time-to-first-"
              "token/boot behaviour.")


# v5.4: depth 1 and 3 are emitted only to document the unreachable curve points; mark them clearly.
for _v in V:
    if _v["k"] is not None and _v["k"] < MIN_BOOTABLE_SPEC_DEPTH:
        _v["doc"] += ("  NOT BOOTABLE on v5.4: the minimum bootable speculative depth is "
                      "%d." % MIN_BOOTABLE_SPEC_DEPTH)
        _v["notes"] += (" DO NOT RUN on v5.4: depth < %d fails at CUDA-graph capture (known issue "
                        "min-bootable-spec-depth-4)." % MIN_BOOTABLE_SPEC_DEPTH)


# --------------------------------------------------------------------------------------------------
# emitters
# --------------------------------------------------------------------------------------------------
def yaml_scalar(value, kind):
    if kind == "bool":
        return "true" if value else "false"
    if kind in ("int", "float"):
        return str(value)
    if kind == "json":
        if "'" in value:
            raise SystemExit(f"JSON scalar contains a single quote: {value}")
        return f"'{value}'"
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./-]+", text):
        return text
    return json.dumps(text)


def spec_dict(v, draft_path=None):
    if v["k"] is None:
        return None
    spec = {"method": "mtp", "num_speculative_tokens": v["k"]}
    if v["drafter"] != "self":
        spec["model"] = draft_path or DRAFTS[v["drafter"]]
    spec["rejection_sample_method"] = v["rejection"]
    spec["draft_sample_method"] = v["draft_sample"]
    return spec


def env_for(v):
    env = dict(BASE_ENV)
    env["MBX_DEPTH_MODE"] = v["depth"]
    if is_static(v):
        # keep the dynamic code path (bootable) but pin depth to K: MIN=CEIL=K, never promote/demote
        env.update(STATIC_DEPTH_LAW)
        env["MBX_DEPTH_MIN"] = str(v["k"])
        env["MBX_DEPTH_CEIL"] = f"[1 -> {v['k']}]"
    env["MBX_DRAFT_HEAD"] = str(v["head"])
    env["MBX_MTP_DRAFT_SCALE"] = str(v["scale"])
    env["MBX_DRAFT_FP8"] = str(v["fp8"])
    if v["ceil"] is not None:
        env["MBX_DEPTH_CEIL"] = v["ceil"]
    if v["warm"]:
        env["MBX_B12X_WARM_DRAFTER"] = "1"
    return env


def axis_text(v):
    parts = []
    if v["k"] is None:
        parts.append("MTP off (no speculative-config)")
    else:
        parts.append(f"K={v['k']}")
        parts.append("depth=static-pinned" if is_static(v)
                     else f"depth={v['depth']}")
        parts.append(f"drafter={v['drafter']}")
        parts.append(f"head={v['head']}")
        parts.append(f"scale={v['scale']}")
        parts.append(f"fp8={v['fp8']}")
        parts.append(f"rejection={v['rejection']}")
        parts.append(f"sample={v['draft_sample']}")
    if v["ceil"] is not None:
        parts.append(f"ceil={v['ceil']}")
    if v["warm"]:
        parts.append("warmdrafter=1")
    return " ".join(parts)


def env_overrides(v):
    base = dict(BASE_ENV)
    env = env_for(v)
    return {k: val for k, val in env.items() if base.get(k) != val}


def render_recipe(v, runsh=False):
    name = v["name"] + ("-models" if runsh else "")
    spec = spec_dict(v, RUNSH_MODEL if runsh else None)
    env = env_for(v)
    header = [
        f"# draft-lab recipe: {name}"
        + ("" if not runsh else "  (run.sh deployment copy, draft folder at /models/draft-k10)")
        + f"  (tier {v['tier']}"
        + (f", priority {v['priority']}" if v["priority"] is not None else "")
        + ")",
        f"# {v['doc']}.",
        f"# Axis: {axis_text(v)}.",
        f"# Reference/control: {REF_NAME}.yaml (self-draft K=12 dynamic = measured M0); the only",
        "# difference from that recipe is the axis above -- every other env/vllm value is identical.",
    ]
    if spec is not None:
        header.append(f"# speculative-config (the ONLY one in this file): {json.dumps(spec, separators=(',', ':'))}")
    else:
        header.append("# speculative-config: OMITTED on purpose -- this config runs without MTP.")
    if runsh:
        header += [
            f"# Drafter: draft folder {RUNSH_MODEL} (host: {DRAFTS['k10']})",
            "# Launch with run.sh; MTP=0 stops run.sh appending a second --speculative-config:",
            f"#   MTP=0 ./run.sh draft-lab/recipes-runsh/{name}.yaml",
            "# run.sh mounts: ./models -> /models, ./cache -> /cache, HF hub at its same absolute path;",
            "# the workspace root is NOT mounted, which is why \"model\" is /models/draft-k10 here.",
            "",
        ]
    else:
        header += [
            "# Drafter: " + ("self-draft (no \"model\" key: the MTP layer is taken from the target)"
                             if v["drafter"] == "self"
                             else f"draft folder {DRAFTS[v['drafter']]}"),
            "# Launch (preferred, bench/serve.py parses this file directly):",
            f"#   bench/serve.py --recipe draft-lab/recipes/{v['name']}.yaml",
            "# run.sh fallback (MTP=0 stops run.sh appending a second --speculative-config):",
            f"#   MTP=0 ./run.sh draft-lab/recipes/{v['name']}.yaml",
            "# Container mounts (bench-harness adds these; same absolute path, read-only):",
            f"#   -v {WORKSPACE}:{WORKSPACE}:ro",
            "#   -v /home/pgx/.cache/huggingface/hub:/home/pgx/.cache/huggingface/hub:ro",
            "",
        ]
    lines = header + ["server:"]
    for key, kind, value in SERVER:
        lines.append(f"  {key}: {yaml_scalar(value, kind)}")
    lines += ["", "env:"]
    for key, _ in BASE_ENV + [("MBX_B12X_WARM_DRAFTER", "")]:
        if key not in env:
            continue
        lines.append(f"  {key}: {json.dumps(env[key])}")
    lines += ["", "vllm:"]
    for key, kind, value in BASE_VLLM:
        if key == "async-scheduling":
            if spec is not None:
                lines.append(f"  speculative-config: {yaml_scalar(json.dumps(spec, separators=(',', ':')), 'json')}")
        lines.append(f"  {key}: {yaml_scalar(value, kind)}")
    return "\n".join(lines) + "\n"


def recipe_path(v):
    return f"draft-lab/recipes/{v['name']}.yaml"


TIER_INTENT = {
    1: "Lead priority scan: run these six first, no edits needed.",
    2: "Second wave: FP8 drafter, draft-k10 at 9, draft-k5 control, M0 reference (do NOT re-run), "
       "no-spec floor, draft-k10 with dynamic depth.",
    3: "Mechanism / token-curve remainder and exploratory depth-ceiling shapes.",
}


def render_matrix():
    """Emit matrix.yaml (PyYAML is available and guarantees valid YAML)."""
    import yaml
    ordered = sorted(V, key=lambda x: (x["tier"], x["priority"] if x["priority"] is not None else 99,
                                       x["name"]))
    configs = []
    for v in ordered:
        spec = spec_dict(v)
        configs.append({
            "name": v["name"],
            "tier": v["tier"],
            "priority": v["priority"],
            "recipe": recipe_path(v),
            "drafter": {"self": "self", "k10": "draft-k10", "k5": "draft-k5"}[v["drafter"]],
            "draft_model": (None if v["drafter"] == "self" else DRAFTS[v["drafter"]]),
            "num_speculative_tokens": v["k"],
            "depth_mode": (None if v["k"] is None
                           else ("static-pinned" if is_static(v) else v["depth"])),
            "pinned_depth": (v["k"] if is_static(v) else None),
            "bootable": is_bootable(v),
            "boot_issue": (None if is_bootable(v) else "min-bootable-spec-depth-4"),
            "speculative_config": (None if spec is None else json.dumps(spec, separators=(',', ':'))),
            "env_overrides": env_overrides(v),
            "axes": axis_text(v),
            "skip_rerun": v["skip_rerun"],
            "measured": MEASURED.get(v["name"]),
            "notes": v["notes"],
        })
    doc = {
        "manifest_version": 1,
        "owner": "draft-lab",
        "model": MODEL,
        "image": IMAGE,
        "snapshot": SNAPSHOT,
        "snapshot_index_sha256": "60e0e2d847ca469105f697ae7721227b07b3bda788cf23c67853cad50b10d71a",
        "baseline": REF_NAME,
        "baseline_note": "M0 is already measured by bench-harness; every acceptance delta is "
                         "against it. CONTROL, do not re-run in the scanning tiers.",
        "mount_contract": {
            "rule": "the served container sees the host filesystem at the same absolute paths, "
                    "read-only; the draft folders use absolute symlinks into the HF hub",
            "workspace_mount": f"-v {WORKSPACE}:{WORKSPACE}:ro",
            "hf_hub_mount": "-v /home/pgx/.cache/huggingface/hub:"
                            "/home/pgx/.cache/huggingface/hub:ro",
            "draft_model_paths": DRAFTS,
        },
        "launch": {
            "preferred": "bench/serve.py --recipe <recipe> (parses the recipe; exactly one "
                         "speculative-config per file)",
            "run_sh_fallback": "MTP=0 ./run.sh <recipe>",
            "duplicate_flag_rule": "never let run.sh append a second --speculative-config: pass "
                                   "MTP=0 when using run.sh",
        },
        "priority_scan": [v["name"] for v in ordered if v["priority"] is not None],
        "tiers": [{"tier": t, "intent": TIER_INTENT[t],
                   "configs": [v["name"] for v in ordered if v["tier"] == t]}
                  for t in (1, 2, 3)],
        "configs": configs,
        "axes": {
            "num_speculative_tokens": [0, 1, 3, 4, 5, 6, 9, 12],
            "depth_mode": ["dynamic", "off"],
            "depth_ceiling": "baseline ladder vs hard/flat caps (ceil-* recipes)",
            "drafter": ["self", "qwen38-az-draft-k10", "qwen38-az-draft-k5"],
            "MBX_DRAFT_HEAD": [0, 1],
            "MBX_MTP_DRAFT_SCALE": [1, 2, 3],
            "rejection_sample_method": ["block", "standard"],
            "draft_sample_method": ["probabilistic", "greedy"],
            "MBX_DRAFT_FP8": [0, 1],
        },
        "known_issues": [
            {
                "id": "static-depth-mode-off-not-bootable",
                "affected_recipes": ["fixed-3", "fixed-5", "fixed-9"],
                "symptom": "MBX_DEPTH_MODE=off with the full MBX_DEPTH_* block fails before /health; the "
                           "container exits at CUDA-graph capture with ValueError: "
                           "num_speculative_tokens_per_batch_size is required for dynamic speculative "
                           "decoding (Lead matrix runs d-fixed-5 and d-fixed-3, 2026-11-12).",
                "root_cause": "disabling the myllmbox depth controller drops the image into vLLM's own "
                              "dynamic-SD schedule, which requires num_speculative_tokens_per_batch_size.",
                "fix": "static depth is emulated on the bootable dynamic path: MBX_DEPTH_MODE=dynamic, "
                       "MBX_DEPTH_MIN=CEIL=K, MBX_DEPTH_CEIL=[1 -> K], MBX_DEPTH_PROMOTE=101,101, "
                       "MBX_DEPTH_DEMOTE=0,0. Applied to fixed-3/fixed-5/fixed-9 in recipes/.",
                "evidence": "draft-lab/verification/boot-failure-static-depth.log",
                "status": "fixed in draft-lab recipes; boot re-verification pending (Lead phase B)",
            },
            {
                "id": "min-bootable-spec-depth-4",
                "affected_recipes": ["curve-k1", "curve-k3", "fixed-3", "draftk10-spec3"],
                "symptom": "speculative depth < 4 never boots on v5.4 -- neither static "
                           "(MBX_DEPTH_MODE=off), nor the pinned ladder, nor plain dynamic depth. It fails "
                           "at CUDA-graph capture with ValueError: num_speculative_tokens_per_batch_size "
                           "is required for dynamic speculative decoding (Lead phase-B runs b-curve-k3 "
                           "and b-k10-curve-k3).",
                "root_cause": "vLLM's dynamic-SD schedule validates num_speculative_tokens_per_batch_size; "
                              "the v5.4 image provides it only for depth >= 4.",
                "fix": "none at config level: minimum bootable speculative depth is 4. The recipes remain "
                       "as documentation of the unreachable curve points but are marked bootable:false "
                       "(matrix.yaml) and enabled:false (bench-matrix.yaml).",
                "evidence": ["bench/results/b-curve-k3/server.log",
                             "bench/results/b-k10-curve-k3/server.log",
                             "draft-lab/verification/boot-failure-min-depth.log"],
                "status": "documented; depth 1/3 unreachable on v5.4",
            },
        ],
        "unavailable": {
            "MBX_DRAFT_HEAD2": "two-stage vocab head: the image ships basis files only for "
                               "Qwen3.8-Flash-Next-FP4-Local and hibrid48, not for AZ -- not used",
            "tools/draft_scale.py": "patches an older module path; v5.4 reads MBX_MTP_DRAFT_SCALE "
                                    "natively, so the recipe env is enough",
        },
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=110,
                          allow_unicode=True)



def render_bench_matrix():
    """Drop-in configs for bench/run_matrix.py --matrix draft-lab/bench-matrix.yaml."""
    import yaml
    ordered = sorted(V, key=lambda x: (x["tier"], x["priority"] if x["priority"] is not None else 99,
                                       x["name"]))
    configs = []
    for index, v in enumerate(ordered):
        entry = {
            "name": "dl-" + v["name"],
            "recipe": recipe_path(v),
            "priority": index + 1,
            "tag": f"draft-lab tier-{v['tier']}",
            "notes": (v["doc"] + (". " + v["notes"] if v["notes"] else "")).strip(),
        }
        if v["tier"] == 3:
            entry["suite_profile"] = "quick"
        if not is_bootable(v):
            entry["enabled"] = False
            entry["notes"] += (" NOT BOOTABLE on v5.4 (min speculative depth %d); disabled."
                               % MIN_BOOTABLE_SPEC_DEPTH)
        if v["skip_rerun"]:
            entry["enabled"] = False
            entry["notes"] += " CONTROL/M0: already measured by bench-harness; leave disabled."
        configs.append(entry)
    doc = {
        "defaults": {
            "results_dir": "bench/results-draft-lab",
            "port": 8000,
            "boot_timeout_s": 1500,
            "per_config_timeout_s": 2400,
            "stop_server": True,
            "suite_profile": "full",
            "suite": {"request_timeout": 300, "retries": 1},
            "env": {},
        },
        "configs": configs,
    }
    head = (
        "# draft-lab drop-in matrix for bench/run_matrix.py (generated by gen_recipes.py)\n"
        "#\n"
        "#   python3 bench/run_matrix.py --matrix draft-lab/bench-matrix.yaml --dry-run\n"
        "#   python3 bench/run_matrix.py --matrix draft-lab/bench-matrix.yaml            # all tiers\n"
        "#   python3 bench/run_matrix.py --matrix draft-lab/bench-matrix.yaml --max-configs 6  # tier 1 only\n"
        "#\n"
        "# Every recipe already carries the full env/vllm section and exactly ONE speculative-config;\n"
        "# bench/serve.py mounts the workspace root and the HF hub at their absolute host paths, so the\n"
        "# draft folders in \"model\": resolve without extra mounts.\n"
        "# Configs run in priority order (tier 1 -> 2 -> 3); the M0 reference is disabled (already measured).\n"
    )
    return head + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=110,
                                 allow_unicode=True)


# --------------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------------
def validate(files, matrix_path):
    import yaml
    problems = []
    checked = 0
    with open(matrix_path) as fh:
        matrix = yaml.safe_load(fh)
    by_name = {c["name"]: c for c in matrix["configs"]}
    for path in files:
        name = os.path.basename(path)[:-5]
        text = open(path).read()
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            problems.append(f"{path}: invalid YAML: {exc}")
            continue
        checked += 1
        if sorted(doc) != ["env", "server", "vllm"]:
            problems.append(f"{path}: sections are {sorted(doc)}")
        spec_lines = [ln for ln in text.splitlines() if ln.strip().startswith("speculative-config:")]
        expect_spec = by_name[name]["speculative_config"]
        if expect_spec is None:
            if spec_lines:
                problems.append(f"{path}: expected NO speculative-config, found {len(spec_lines)}")
        else:
            if len(spec_lines) != 1:
                problems.append(f"{path}: expected exactly 1 speculative-config, "
                                f"found {len(spec_lines)}")
            else:
                got = doc["vllm"].get("speculative-config")
                if isinstance(got, str):
                    got = json.loads(got)  # repo style: the JSON is a quoted YAML scalar
                if got != json.loads(expect_spec):
                    problems.append(f"{path}: speculative-config mismatch: {got}")
            if "MTP" in doc["env"]:
                problems.append(f"{path}: recipe must not export MTP (run.sh would append a flag)")
        if name not in by_name:
            problems.append(f"{path}: not referenced by matrix.yaml")
        if doc["server"].get("model") != MODEL or doc["server"].get("image") != IMAGE:
            problems.append(f"{path}: wrong server model/image")
        # full env section must be identical to the baseline apart from documented overrides
        base = dict(BASE_ENV)
        for key, value in doc["env"].items():
            if key == "MBX_B12X_WARM_DRAFTER":
                continue
            if key not in base:
                problems.append(f"{path}: unknown env key {key}")
        if set(doc["env"]) - {"MBX_B12X_WARM_DRAFTER"} != set(base):
            problems.append(f"{path}: env key set differs from baseline")
    # every matrix config has a file
    for name, c in by_name.items():
        if not os.path.isfile(os.path.join(ROOT, c["recipe"])):
            problems.append(f"matrix: recipe missing for {name}: {c['recipe']}")
    # the bench-runner drop-in must reference the same files and stay ordered
    with open(BENCH_MATRIX) as fh:
        bench = yaml.safe_load(fh)
    bench_configs = bench["configs"]
    if len(bench_configs) != len(files):
        problems.append(f"bench-matrix has {len(bench_configs)} configs, expected {len(files)}")
    names = [c["name"] for c in bench_configs]
    if len(names) != len(set(names)):
        problems.append("bench-matrix: duplicate config names")
    for c in bench_configs:
        if not os.path.isfile(os.path.join(ROOT, c["recipe"])):
            problems.append(f"bench-matrix: missing recipe {c['recipe']}")
        if not c["name"].startswith("dl-"):
            problems.append(f"bench-matrix: name {c['name']} lacks the dl- prefix")
    if [c["priority"] for c in bench_configs] != list(range(1, len(bench_configs) + 1)):
        problems.append("bench-matrix: priorities are not the sequential run order")
    enabled = [c for c in bench_configs if c.get("enabled", True)]
    must_disable = 1 + sum(1 for v in V if not is_bootable(v))  # M0 control + unreachable depths
    if len(enabled) != len(bench_configs) - must_disable:
        problems.append("bench-matrix: expected %d disabled configs (M0 control + non-bootable "
                        "depths), found %d" % (must_disable, len(bench_configs) - len(enabled)))
    by_variant = {v["name"]: v for v in V}
    for c in bench_configs:
        variant_obj = by_variant.get(c["name"][3:])
        if variant_obj is not None and not is_bootable(variant_obj) and c.get("enabled", True):
            problems.append("bench-matrix: non-bootable config %s is not disabled" % c["name"])
    # run.sh deployment copies: draft-k10 recipes re-pointed at /models/draft-k10
    k10 = [v for v in V if v["drafter"] == "k10"]
    runsh_files = sorted(glob.glob(os.path.join(RUNSH_RECIPES, "*-models.yaml")))
    if len(runsh_files) != len(k10):
        problems.append(f"recipes-runsh has {len(runsh_files)} files, expected {len(k10)}")
    for path in runsh_files:
        doc = yaml.safe_load(open(path))
        vllm = doc.get("vllm", {})
        spec = vllm.get("speculative-config")
        if spec is None:
            problems.append(f"{path}: missing speculative-config")
            continue
        spec = json.loads(spec) if isinstance(spec, str) else spec
        if spec.get("model") != RUNSH_MODEL:
            problems.append(f"{path}: model is {spec.get('model')!r}, expected {RUNSH_MODEL}")
        if not path.endswith("-models.yaml"):
            problems.append(f"{path}: expected -models.yaml suffix")
    return checked, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate existing files, write nothing")
    args = ap.parse_args(argv)

    os.makedirs(RECIPES, exist_ok=True)
    files = [os.path.join(RECIPES, v["name"] + ".yaml") for v in V]
    if not args.check:
        for path, v in zip(files, V):
            with open(path, "w") as fh:
                fh.write(render_recipe(v))
        with open(MATRIX, "w") as fh:
            fh.write("# draft-lab draft / speculative-decoding matrix. Generated by gen_recipes.py.\n")
            fh.write(render_matrix())
        with open(BENCH_MATRIX, "w") as fh:
            fh.write(render_bench_matrix())
        os.makedirs(RUNSH_RECIPES, exist_ok=True)
        for v in V:
            if v["drafter"] != "k10":
                continue
            with open(os.path.join(RUNSH_RECIPES, v["name"] + "-models.yaml"), "w") as fh:
                fh.write(render_recipe(v, runsh=True))

    checked, problems = validate(files, MATRIX)
    if problems:
        print("VALIDATION FAILED", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        sys.exit(1)
    print(f"OK {checked} recipes ({sum(1 for v in V if not is_bootable(v))} non-bootable on v5.4) "
          f"+ matrix.yaml + bench-matrix.yaml + "
          f"{sum(1 for v in V if v['drafter'] == 'k10')} recipes-runsh copies validated")
    for tier in (1, 2, 3):
        names = [v["name"] for v in sorted(V, key=lambda x: (x["tier"], x["priority"] or 99,
                                                             x["name"])) if v["tier"] == tier]
        print(f"  tier {tier} ({len(names)}): " + ", ".join(names))


if __name__ == "__main__":
    main()
