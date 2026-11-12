#!/usr/bin/env python3
"""Build a slim MTP *draft folder* from a Qwen3.8-Flash-Next(-FP4) model snapshot.

Model-card recipe (azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ):
the trunk routes 5 of 512 experts, but the MTP layer keeps NVIDIA's 10-expert drafter.  Serving the
draft at 10 experts while the model routes 5 is done by giving `--speculative-config` a draft folder:

    <draft>/config.json                  = copy of the model config with ONLY
                                           text_config.num_experts_per_tok set (default 10)
    <draft>/model.safetensors.index.json = only mtp.* / lm_head.* / *embed_tokens.weight tensors
    <draft>/tensors/...                  = ABSOLUTE symlinks to the shards those tensors live in
                                           (for this checkpoint: model-0000{1,14,15}-of-00021)
    <draft>/*                            = ABSOLUTE symlinks to the tokenizer/config files

Everything except two small JSON files is a symlink; no weight byte is copied.

Safety contract (this script exists because an earlier revision dumped its filtered index into the
source snapshot and clobbered the real 375,487-tensor index): the source snapshot is READ-ONLY.
The script
  * refuses a destination equal to, inside, or containing the source snapshot,
  * refuses a destination inside the Hugging Face cache ($HF_HOME / $HF_HUB_CACHE),
  * refuses to delete a destination that does not carry its own marker file,
  * asserts every single path it writes resolves inside the destination,
  * fingerprints the source snapshot before and after (sha256 of index+config plus the directory
    listing) and fails if anything there changed,
  * verifies the result offline: every indexed tensor name is present in the shard header it maps to.

usage:
  make_draft_dir.py <snapshot> <draft-dir> [--num-experts-per-tok 10] [--force]
  make_draft_dir.py <snapshot> <draft-dir> --verify-only        # re-check an existing folder

The draft folder can live anywhere (the repo convention is draft-lab/drafts/<name>/).  The served
container must see (a) the HF hub at the same absolute path (run.sh mounts it that way in cache mode)
and (b) this draft folder at its absolute host path, e.g.
  -v /home/pgx/Documenti/vllm-gb10/draft-lab/drafts:/home/pgx/Documenti/vllm-gb10/draft-lab/drafts:ro
then --speculative-config {"method":"mtp",...,"model":"/abs/path/to/draft"}.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time

MARKER = ".make_draft_dir.json"
SCRIPT_VERSION = "2.0"
DRAFT_PREFIXES = ("mtp.", "lm_head.")
EMBED_SUFFIX = "embed_tokens.weight"
# Files of the snapshot that are NOT part of the draft view.
SOURCE_ONLY = {
    "config.json",                  # we write our own copy
    "model.safetensors.index.json",  # we write our own filtered index
    "model_extra_tensors.safetensors",  # unindexed extra files are irrelevant to the drafter
}
# Snapshot housekeeping that must not leak into the draft view (recovery backups, hub metadata).
SKIP_SUFFIXES = (".bak", ".incomplete", ".lock")
SKIP_NAMES = {"README.md"}
MAX_SHARDS = 16  # a sane draft view is a handful of shards; more means the filter is wrong


def die(msg):
    sys.exit(f"make_draft_dir: {msg}")


def inside(child, parent):
    """True if `child` is `parent` or a path below it (both are realpaths)."""
    child, parent = os.path.realpath(child), os.path.realpath(parent)
    return child == parent or child.startswith(parent + os.sep)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_roots():
    """Every directory we must never write under: HF cache roots."""
    roots = set()
    hf_home = os.path.expanduser(os.environ.get("HF_HOME") or "~/.cache/huggingface")
    roots.add(os.path.realpath(hf_home))
    hub = os.environ.get("HF_HUB_CACHE")
    roots.add(os.path.realpath(hub) if hub else os.path.join(os.path.realpath(hf_home), "hub"))
    roots.add(os.path.realpath(os.path.expanduser("~/.cache/huggingface")))
    roots.add(os.path.realpath(os.path.expanduser("~/.cache/huggingface/hub")))
    return sorted(r for r in roots if r and r != "/")


def snapshot_fingerprint(src):
    """Listing + mtime/size of every top-level entry of the snapshot (read-only side-effect check)."""
    out = {}
    with os.scandir(src) as it:
        for e in it:
            st = e.stat(follow_symlinks=False)
            out[e.name] = [st.st_size, st.st_mtime_ns]
    return out


def read_safetensors_header(path):
    """Tensor-name -> metadata from a .safetensors header (no tensor data is read)."""
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
    except (OSError, ValueError, struct.error) as exc:
        die(f"cannot read safetensors header {path}: {exc}")
    header.pop("__metadata__", None)
    return header


def diff_paths(old, new, path=()):
    """Recursively list (path_tuple, old, new) leaf differences between two JSON-able values."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            if key not in old:
                out.append((path + (key,), "<absent>", "<added>"))
            elif key not in new:
                out.append((path + (key,), old[key], "<deleted>"))
            else:
                out.extend(diff_paths(old[key], new[key], path + (key,)))
        return out
    if old != new:
        return [(path, old, new)]
    return []


def load_filtered_keys(index_path):
    """The exact weight_map a draft folder must carry, from the model's real index."""
    with open(index_path) as fh:
        index = json.load(fh)
    wm = index.get("weight_map")
    if not isinstance(wm, dict) or not wm:
        die(f"{index_path} has no weight_map")
    keep = {k: v for k, v in wm.items()
            if k.startswith(DRAFT_PREFIXES) or k.endswith(EMBED_SUFFIX)}
    return index, keep


def classify(keys):
    return {
        "mtp": sum(1 for k in keys if k.startswith("mtp.")),
        "lm_head": sum(1 for k in keys if k.startswith("lm_head.")),
        "embed_tokens": sum(1 for k in keys if k.endswith(EMBED_SUFFIX)),
    }


# --------------------------------------------------------------------------------------------------
# destination guards
# --------------------------------------------------------------------------------------------------
def resolve_destination(src, dst_arg, force):
    src_real = os.path.realpath(src)
    dst = os.path.abspath(dst_arg)
    dst_real = os.path.realpath(dst)

    if dst_real == "/" or dst_real == os.path.realpath(os.path.expanduser("~")):
        die(f"refusing {dst_real}: too broad a destination to own")
    if dst_real == src_real or inside(dst_real, src_real):
        die(f"refusing {dst}: destination is inside the source snapshot {src_real}")
    if inside(src_real, dst_real):
        die(f"refusing {dst}: destination contains the source snapshot {src_real}")
    for root in cache_roots():
        if inside(dst_real, root):
            die(f"refusing {dst}: the Hugging Face cache ({root}) is read-only for this script")
    if os.path.islink(dst):
        die(f"refusing {dst}: destination is a symlink")

    if os.path.lexists(dst):
        if not os.path.isdir(dst):
            die(f"refusing {dst}: exists and is not a directory")
        ours = os.path.isfile(os.path.join(dst, MARKER))
        if not ours and not force:
            die(f"{dst} exists but does not look like a previous draft folder "
                f"(no {MARKER}); pass --force to replace it")
        print(f"[draft-lab] replacing existing destination: {dst}"
              + ("" if ours else "  (--force)"))
        shutil.rmtree(dst)
    return dst, dst_real


def make_writer(dst_real):
    """Return a write helper that asserts the path stays inside the draft folder."""

    def write_text(relpath, text):
        path = os.path.join(dst_real, relpath) if not os.path.isabs(relpath) else relpath
        parent = os.path.realpath(os.path.dirname(path))
        if parent != dst_real and not parent.startswith(dst_real + os.sep):
            die(f"internal error: refusing to write outside the draft folder: {path}")
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def write_json(relpath, obj):
        return write_text(relpath, json.dumps(obj, indent=2) + "\n")

    def link(abs_target, relpath):
        if not os.path.isabs(abs_target):
            die(f"internal error: symlink target must be absolute: {abs_target}")
        path = os.path.join(dst_real, relpath)
        parent = os.path.realpath(os.path.dirname(path))
        if parent != dst_real and not parent.startswith(dst_real + os.sep):
            die(f"internal error: refusing to link outside the draft folder: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.symlink(abs_target, path)
        return path

    return write_text, write_json, link


def safe_shard_rel(rel):
    """weight_map values must be plain relative paths inside the snapshot (no .., not absolute)."""
    if os.path.isabs(rel):
        die(f"weight_map shard path is absolute, refusing: {rel}")
    norm = os.path.normpath(rel)
    if norm.startswith("..") or norm == "." or norm.startswith(os.sep):
        die(f"weight_map shard path escapes the snapshot: {rel}")
    return norm


# --------------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------------
def build(src, dst, dst_real, target_k):
    src_real = os.path.realpath(src)
    for required in ("config.json", "model.safetensors.index.json"):
        if not os.path.isfile(os.path.join(src, required)):
            die(f"{src} is not a model snapshot: missing {required}")

    # ---- read-only fingerprints BEFORE any write
    before_sha = {name: sha256(os.path.join(src, name))
                  for name in ("config.json", "model.safetensors.index.json")}
    before_listing = snapshot_fingerprint(src)

    # ---- ground truth from the model's real index
    _, keep = load_filtered_keys(os.path.join(src, "model.safetensors.index.json"))
    kinds = classify(keep)
    if not kinds["mtp"]:
        die("the snapshot index has no mtp.* tensors -- this checkpoint has no speculative head")
    if not kinds["lm_head"]:
        die("the snapshot index has no lm_head.weight -- a draft folder cannot load without it")
    if not kinds["embed_tokens"]:
        die("the snapshot index has no *embed_tokens.weight tensor -- the drafter needs the "
            "embedding; refusing to build an incomplete draft folder")
    shard_rels = sorted(set(safe_shard_rel(v) for v in keep.values()))
    if len(shard_rels) > MAX_SHARDS:
        die(f"filter selects {len(shard_rels)} shards (> {MAX_SHARDS}); the draft filter looks wrong")

    # ---- config copy: change ONLY num_experts_per_tok (+ the head's own shared-expert width when
    #      the MTP head is shipped as an unindexed extra file, as in the Swift1.5 variant)
    with open(os.path.join(src, "config.json")) as fh:
        src_cfg = json.load(fh)
    cfg = json.loads(json.dumps(src_cfg))
    text = cfg.get("text_config", cfg)
    allowed = {("text_config", "num_experts_per_tok")} if "text_config" in cfg \
        else {("num_experts_per_tok",)}
    old_k = text.get("num_experts_per_tok")
    text["num_experts_per_tok"] = target_k

    extra = os.path.join(src, "model_extra_tensors.safetensors")
    if os.path.isfile(extra):
        rows = {meta["shape"][0]
                for key, meta in read_safetensors_header(extra).items()
                if key.endswith("mlp.shared_expert.gate_proj.weight")}
        if len(rows) == 1:
            width = rows.pop()
            if text.get("shared_expert_intermediate_size") != width:
                print(f"[draft-lab] MTP head shared-expert width "
                      f"{text.get('shared_expert_intermediate_size')} -> {width} "
                      f"(from {os.path.basename(extra)})")
                text["shared_expert_intermediate_size"] = width
                key = "text_config" if "text_config" in cfg else None
                allowed = set(allowed) | {(key, "shared_expert_intermediate_size") if key else
                                          ("shared_expert_intermediate_size",)}
    changed = diff_paths(src_cfg, cfg)
    unexpected = [c for c in changed if c[0] not in allowed]
    if unexpected:
        die(f"internal error: config copy would change unexpected fields: {unexpected}")
    if old_k == target_k:
        print(f"[draft-lab] note: the trunk already routes {target_k} experts -- this folder is a "
              f"control copy (the drafter stays identical to the self-draft)")

    # ---- every indexed tensor must live in the shard the index says it does
    per_shard, header_checks = {}, {}
    for rel in shard_rels:
        shard_path = os.path.join(src_real, rel)
        if not os.path.isfile(shard_path):
            die(f"index points at a shard that is missing from the snapshot: {rel}")
        header = read_safetensors_header(shard_path)
        want = {k for k, v in keep.items() if safe_shard_rel(v) == rel}
        missing = want - set(header)
        if missing:
            die(f"{rel}: {len(missing)} indexed draft tensors are absent from the shard "
                f"(e.g. {sorted(missing)[0]})")
        per_shard[rel] = len(want)
        header_checks[rel] = len(want)

    # ---- create the folder (marker first so a half-built dir is recognised next time)
    os.makedirs(dst, exist_ok=False)
    write_text, write_json, link = make_writer(dst_real)
    marker = {
        "generated_by": "tools/make_draft_dir.py",
        "version": SCRIPT_VERSION,
        "status": "building",
        "source_snapshot": src_real,
        "num_experts_per_tok": target_k,
        "source_index_sha256": before_sha["model.safetensors.index.json"],
        "source_config_sha256": before_sha["config.json"],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(MARKER, marker)

    write_json("config.json", cfg)
    write_json("model.safetensors.index.json", {"metadata": {}, "weight_map": keep})
    for rel in shard_rels:
        link(os.path.join(src_real, rel), rel)  # absolute target: the container sees the hub 1:1

    # tokenizer / generation / template files: absolute symlinks, nothing else
    linked_other = []
    for name in sorted(os.listdir(src)):
        if name in SOURCE_ONLY or name in SKIP_NAMES or name.startswith("."):
            continue
        if name.endswith(SKIP_SUFFIXES):
            continue
        src_path = os.path.join(src_real, name)
        if not os.path.isfile(src_path):
            continue  # tensors/ and any other directory are not linked
        link(src_path, name)
        linked_other.append(name)

    # ---- verification of what we just wrote (before/after checks below)
    report = verify(src_real, dst_real, target_k, keep, shard_rels, kinds, linked_other,
                    marker["source_index_sha256"], marker["source_config_sha256"])

    # ---- side-effect check: the snapshot must be bit-for-bit and listing-for-listing unchanged
    after_sha = {name: sha256(os.path.join(src, name)) for name in before_sha}
    after_listing = snapshot_fingerprint(src)
    if after_sha != before_sha:
        die(f"the source snapshot CHANGED while building ({before_sha} -> {after_sha})")
    if after_listing != before_listing:
        die("the source snapshot directory listing changed while building")
    report["source_unchanged"] = True
    report["source_index_sha256"] = after_sha["model.safetensors.index.json"]
    report["source_config_sha256"] = after_sha["config.json"]

    marker.update({"status": "complete",
                   "indexed_tensors": len(keep),
                   "by_kind": kinds,
                   "shards": shard_rels,
                   "num_speculative_shard_files": len(shard_rels),
                   "completed": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    write_json(MARKER, marker)
    return report


# --------------------------------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------------------------------
def verify(src, dst, target_k, keep, shard_rels, kinds, linked_other,
           expect_index_sha=None, expect_config_sha=None):
    src_real, dst_real = os.path.realpath(src), os.path.realpath(dst)
    problems = []

    def check(cond, msg):
        if not cond:
            problems.append(msg)
        return cond

    # config
    with open(os.path.join(src_real, "config.json")) as fh:
        src_cfg = json.load(fh)
    with open(os.path.join(dst_real, "config.json")) as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config", cfg)
    check(text.get("num_experts_per_tok") == target_k,
          f"config num_experts_per_tok is {text.get('num_experts_per_tok')}, expected {target_k}")
    changed = diff_paths(src_cfg, cfg)
    allowed = {("text_config", "num_experts_per_tok"), ("num_experts_per_tok",),
               ("text_config", "shared_expert_intermediate_size"),
               ("shared_expert_intermediate_size",)}
    bad = [c for c in changed if c[0] not in allowed]
    check(not bad, f"config differs from the model config in unexpected fields: {bad}")
    if text.get("shared_expert_intermediate_size") != \
            src_cfg.get("text_config", src_cfg).get("shared_expert_intermediate_size"):
        check(os.path.isfile(os.path.join(src_real, "model_extra_tensors.safetensors")),
              "shared_expert_intermediate_size changed but there is no head file to justify it")

    # index
    with open(os.path.join(dst_real, "model.safetensors.index.json")) as fh:
        dindex = json.load(fh)
    check(dindex.get("metadata") == {}, "draft index metadata is not {}")
    dwm = dindex.get("weight_map", {})
    check(dwm == keep, "draft index weight_map is not exactly the filtered model weight_map")
    check(len(dwm) == len(keep),
          f"draft index has {len(dwm)} tensors, expected {len(keep)}")
    kinds2 = classify(dwm)
    check(kinds2 == kinds, f"draft index kinds {kinds2} != {kinds}")
    check(all(safe_shard_rel(v) in shard_rels for v in dwm.values()),
          "draft index references shards outside the linked set")
    check(len(shard_rels) == len(set(shard_rels)),
          "duplicate shards in the linked set")

    # every key present in the shard header it points at; symlink resolves to the snapshot file
    for rel in shard_rels:
        path = os.path.join(dst_real, rel)
        if not check(os.path.islink(path), f"{rel} is not a symlink"):
            continue
        target = os.path.realpath(path)
        check(os.path.isfile(target), f"{rel} does not resolve to a file (target {os.readlink(path)})")
        check(target == os.path.join(src_real, rel),
              f"{rel} resolves to {target}, expected {os.path.join(src_real, rel)}")
        header = read_safetensors_header(target)
        want = {k for k, v in keep.items() if safe_shard_rel(v) == rel}
        missing = want - set(header)
        check(not missing, f"{rel}: {len(missing)} indexed tensors missing from the shard header")

    # no unexpected entries in the draft folder
    expected = set(SOURCE_ONLY - {"model_extra_tensors.safetensors"}) | {MARKER, "tensors"} \
        | set(linked_other)
    entries = set(os.listdir(dst_real))
    check(entries == expected, f"unexpected draft folder entries: {sorted(entries ^ expected)}")
    n_draft_shard_files = 0
    if os.path.isdir(os.path.join(dst_real, "tensors")):
        n_draft_shard_files = len(os.listdir(os.path.join(dst_real, "tensors")))
        check(n_draft_shard_files == len(shard_rels),
              f"tensors/ has {n_draft_shard_files} files, expected {len(shard_rels)}")
    for name in linked_other:
        p = os.path.join(dst_real, name)
        check(os.path.islink(p) and os.path.exists(p), f"symlink {name} does not resolve")

    if expect_index_sha:
        check(sha256(os.path.join(src_real, "model.safetensors.index.json")) == expect_index_sha,
              "the model index no longer matches the checksum taken before the build")
    if expect_config_sha:
        check(sha256(os.path.join(src_real, "config.json")) == expect_config_sha,
              "the model config no longer matches the checksum taken before the build")

    if problems:
        die("verification FAILED:\n  - " + "\n  - ".join(problems))

    return {
        "destination": dst_real,
        "num_experts_per_tok": target_k,
        "indexed_tensors": len(keep),
        "by_kind": kinds,
        "shard_rels": list(shard_rels),
        "per_shard": {rel: sum(1 for v in keep.values() if safe_shard_rel(v) == rel)
                      for rel in shard_rels},
        "draft_shard_files": n_draft_shard_files,
        "linked_other": len(linked_other),
        "snapshot_top_entries": len(snapshot_fingerprint(src_real)),
        "checks": "PASS",
    }


def print_report(r, src, dst):
    print(f"[draft-lab] source snapshot   : {src}")
    print(f"[draft-lab] draft folder      : {dst}")
    print(f"[draft-lab] top-k             : {r['num_experts_per_tok']}")
    print(f"[draft-lab] index tensors     : {r['indexed_tensors']}  "
          f"(mtp {r['by_kind']['mtp']}, lm_head {r['by_kind']['lm_head']}, "
          f"embed_tokens {r['by_kind']['embed_tokens']})")
    print(f"[draft-lab] shard files       : {r['draft_shard_files']} "
          f"({', '.join(r['shard_rels'])})")
    for rel, n in r["per_shard"].items():
        print(f"[draft-lab]   {rel}: {n} tensors in header")
    print(f"[draft-lab] other symlinks    : {r['linked_other']}")
    if r.get("source_unchanged"):
        print(f"[draft-lab] source unchanged  : YES  index sha256 {r['source_index_sha256']}")
        print(f"[draft-lab] source unchanged  : YES  config sha256 {r['source_config_sha256']}")
    print(f"[draft-lab] verification      : {r['checks']} (config, index counts, shard headers, links)")
    print(f"OK draft folder ready: {dst}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build (or verify) a slim MTP draft folder for a Qwen3.8-Flash-Next snapshot. "
                    "The source snapshot is strictly read-only.")
    ap.add_argument("snapshot", help="model snapshot directory (read-only)")
    ap.add_argument("draft", help="draft folder to create / verify")
    ap.add_argument("--num-experts-per-tok", type=int, default=10,
                    help="drafter top-k written into the draft config (default 10; use 5 for the "
                         "draft-5 control copy)")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing destination even if it has no draft marker")
    ap.add_argument("--verify-only", action="store_true",
                    help="do not build; re-verify the existing draft folder against the snapshot")
    args = ap.parse_args(argv)

    src = os.path.abspath(args.snapshot.rstrip("/"))
    if not os.path.isdir(src):
        die(f"snapshot directory not found: {src}")
    if not os.path.isfile(os.path.join(src, "model.safetensors.index.json")):
        die(f"{src} is not a model snapshot: no model.safetensors.index.json")

    if args.verify_only:
        dst = os.path.abspath(args.draft)
        if not os.path.isdir(dst):
            die(f"draft folder not found: {dst}")
        _, keep = load_filtered_keys(os.path.join(src, "model.safetensors.index.json"))
        kinds = classify(keep)
        shard_rels = sorted(set(safe_shard_rel(v) for v in keep.values()))
        linked_other = sorted(n for n in os.listdir(dst)
                              if n not in SOURCE_ONLY and n not in (MARKER, "tensors")
                              and os.path.islink(os.path.join(dst, n)))
        report = verify(src, dst, args.num_experts_per_tok, keep, shard_rels, kinds, linked_other)
        print_report(report, os.path.realpath(src), os.path.realpath(dst))
        return

    dst, dst_real = resolve_destination(src, args.draft, args.force)
    report = build(src, dst, dst_real, args.num_experts_per_tok)
    print_report(report, os.path.realpath(src), dst_real)


if __name__ == "__main__":
    main()
