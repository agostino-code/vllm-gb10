#!/usr/bin/env python3
"""Generate serving-knob recipe variants from the reference AZ recipe.

The reference is recipes/qwen38-flash-solo-optimized.yaml (the shipped
myllmbox recipe-az settings, K=12 dynamic depth). Each variant here changes
exactly one or two serving knobs so the benchmark matrix can attribute the
difference. Draft/MTP variants are owned by draft-lab and are NOT generated
here.

Usage: python3 matrix/gen_variants.py [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "recipes" / "qwen38-flash-solo-optimized.yaml"
OUT = ROOT / "matrix" / "recipes"

# Each variant: name -> (comment, list of (kind, target, value))
# kind: "vllm" | "env" | "server" | "drop-vllm" | "drop-env" | "drop-server"
VARIANTS: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "az-ref": (
        "Reference preset: shipped myllmbox AZ recipe (K=12 dynamic depth).",
        [],
    ),
    "az-kvfp8-spec7": (
        "fp8 KV doubles the pool (7-19% slower per myllmbox); depth retuned to 7.",
        [
            ("vllm", "kv-cache-dtype", "fp8"),
            (
                "vllm",
                "speculative-config",
                "'{\"method\":\"mtp\",\"num_speculative_tokens\":7,\"rejection_sample_method\":\"block\",\"draft_sample_method\":\"probabilistic\"}'",
            ),
        ],
    ),
    "az-native262k": (
        "Native 262,144-token context: no YaRN hf-overrides, max-model-len 262144.",
        [
            ("vllm", "max-model-len", "262144"),
            ("drop-vllm", "hf-overrides", ""),
        ],
    ),
    "az-seqs8": (
        "8 seats instead of 16 (smaller verify batches, less KV pressure).",
        [("vllm", "max-num-seqs", "8")],
    ),
    "az-batched4096": (
        "Chunked-prefill chunk 4096 instead of 8192 (finer interleaving at concurrency).",
        [("vllm", "max-num-batched-tokens", "4096")],
    ),
    "az-kv26g": (
        "26G KV (v5.4 generic recipe value) instead of 23G.",
        [("vllm", "kv-cache-memory", "\"26000000000\"")],
    ),
    "az-noreplayssm": (
        "Stock GDN verify path (RecoverSSM off): more memory per seat, slower steps.",
        [("vllm", "use-replayssm", "false")],
    ),
    "az-mxfp8a16": (
        "MXFP8 A16 attention kernels on b12x (NVIDIA ships A16 off).",
        [("env", "MBX_MXFP8_A16", "1")],
    ),
    "az-prewarm-ple": (
        "Read the whole n-gram table into page cache at boot (fast first tokens).",
        [("env", "MBX_PLE_MMAP_PREWARM", "1")],
    ),
    "az-cpuset-none": (
        "No cpuset pinning (tests the +2-3% pinning claim on this box).",
        [("drop-server", "cpuset", "")],
    ),
    "az-medium-template": (
        "Medium chat template (azampatti INT4 repo) instead of the model stock template."
        " The template is injected by the launcher (serve.py --chat-template / run.sh CHAT_TEMPLATE),"
        " not by the recipe.",
        [],
    ),
}


def find_line(lines: list[str], section: str, key: str) -> int:
    """Index of `  key:` in the given YAML section, or -1."""
    cur = None
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Za-z_]+):\s*$", line)
        if m:
            cur = m.group(1)
            continue
        if cur == section and re.match(rf"^  {re.escape(key)}:", line):
            return i
    return -1


def find_section_end(lines: list[str], section: str) -> int:
    start = None
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Za-z_]+):\s*$", line)
        if m:
            if start is not None:
                return i
            if m.group(1) == section:
                start = i
    if start is None:
        raise SystemExit(f"section {section} not found")
    return len(lines)


def apply(lines: list[str], kind: str, key: str, value: str) -> list[str]:
    lines = lines[:]
    section = kind.replace("drop-", "")
    idx = find_line(lines, section, key)
    if kind.startswith("drop-"):
        if idx < 0:
            raise SystemExit(f"cannot drop missing {section}.{key}")
        del lines[idx]
        return lines
    if idx < 0:
        # append at the end of the section, preserving indentation
        end = find_section_end(lines, section)
        lines.insert(end, f"  {key}: {value}\n")
    else:
        lines[idx] = f"  {key}: {value}\n"
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify base parsing, write nothing")
    args = ap.parse_args()

    if not BASE.exists():
        sys.exit(f"missing base recipe {BASE}")
    base_lines = BASE.read_text().splitlines(keepends=True)
    if find_line(base_lines, "vllm", "kv-cache-memory") < 0:
        sys.exit("base recipe does not look like the expected AZ recipe")

    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (comment, edits) in VARIANTS.items():
        lines = base_lines[:]
        for kind, key, value in edits:
            lines = apply(lines, kind, key, value)
        text = "".join(lines)
        header = (
            f"# VARIANT: {name}\n"
            f"# {comment}\n"
            f"# Generated by matrix/gen_variants.py from {BASE.relative_to(ROOT)} -- do not hand-edit.\n"
        )
        # replace the original top comment block with our header + original
        out = header + text
        path = OUT / f"{name}.yaml"
        if not args.check:
            path.write_text(out)
        written.append(path.name)

    print(f"base: {BASE.relative_to(ROOT)} ({len(base_lines)} lines)")
    print(f"variants: {len(written)} -> {OUT.relative_to(ROOT)}/")
    for w in written:
        print("  ", w)
    if args.check:
        print("(check mode: nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
