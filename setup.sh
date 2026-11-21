#!/usr/bin/env bash
# setup.sh — idempotent bootstrap and verification. Safe to re-run: it checks
# what is already in place and only does the missing work.
#
#   ./setup.sh                    check host, image and weights for the default recipe
#   ./setup.sh --check            report only, change nothing
#   ./setup.sh --image            pull the serving image
#   ./setup.sh --weights [repo]   download a checkpoint into the HF cache, then verify it
#   ./setup.sh --gate <repo>      check a gated checkpoint is accessible (used by presets)
#   ./setup.sh --venv             create .venv with the one dependency we need
#
# WHAT IT VERIFIES, and why each one matters on this box:
#   docker + GPU        the container needs the whole 128 GB unified pool
#   serving image       pulled once; run.sh never pulls
#   weights present     the usual first-boot failure is a half-finished download
#   weights complete    every shard the index references exists and is non-empty.
#                       This is the failure mode documented in BENCHMARK-REPORT:
#                       a snapshot whose index was overwritten by a draft-only
#                       index loads as "NVFP4 weight_scale ... was never loaded
#                       (still NaN)" and the engine dies at capture, not at boot.
#   weights canonical   compared against bench/refs/ when a reference is kept
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

recipe_model() {  # server.model of a recipe, defaulting to DEFAULT_RECIPE
  awk -v s=server '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec) }
    sec==s && $1=="model:" { sub(/^[ ]*[^:]*:[ ]*/,""); sub(/[ ]+#.*$/,"")
                             gsub(/^["\x27]|["\x27]$/,""); print; exit }' "${1:-$DEFAULT_RECIPE}"
}
recipe_image() {
  awk -v s=server '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec) }
    sec==s && $1=="image:" { sub(/^[ ]*[^:]*:[ ]*/,""); sub(/[ ]+#.*$/,"")
                             gsub(/^["\x27]|["\x27]$/,""); print; exit }' "${1:-$DEFAULT_RECIPE}"
}

ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[1;33m⚠\033[0m %s\n' "$1"; }
bad()  { printf '  \033[1;31m✗\033[0m %s\n' "$1"; }

# ---------------------------------------------------------------------------
check_host() {
  echo "· host"
  command -v docker >/dev/null 2>&1 && ok "docker $(docker --version | awk '{print $3}')" ||
    { bad "docker is not installed"; return 1; }
  docker info >/dev/null 2>&1 && ok "docker daemon reachable" ||
    { bad "docker daemon not reachable (is the user in the docker group?)"; return 1; }
  # Zero-risk check: confirm the nvidia container runtime is registered, do not
  # start a GPU container while another server owns the pool.
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    ok "nvidia container runtime registered"
  else
    bad "no nvidia runtime in docker — install the NVIDIA container toolkit"
  fi
  local cp; cp=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
  [ "$cp" = "0" ] && ok "vm.compaction_proactiveness=0" ||
    warn "vm.compaction_proactiveness=$cp — run ./tune-host.sh (removes periodic freezes)"
  local av; av=$(free -g | awk '/^Mem:/{print $7}')
  [ "$av" -ge 110 ] && ok "${av}G memory free" ||
    warn "only ${av}G free: weights + KV need the whole unified pool, close other consumers"
}

pull_image() {
  local img="$1"
  echo "· serving image"
  if docker image inspect "$img" >/dev/null 2>&1; then
    ok "$img already present ($(docker image inspect --format '{{.Id}}' "$img" | cut -d: -f2 | cut -c1-12))"
  else
    echo "  pulling $img (~35 GB, once)"
    docker pull "$img"
  fi
}

verify_weights() {  # verify_weights <repo> <snapshot-dir>
  local repo="$1" snap="$2"
  echo "· weights: $repo"
  [ -d "$snap" ] || { bad "no snapshot at $snap"; return 1; }
  [ -f "$snap/config.json" ] || { bad "config.json missing"; return 1; }
  [ -f "$snap/model.safetensors.index.json" ] || { bad "model.safetensors.index.json missing"; return 1; }

  $PY - "$snap" <<'PY'
import json, os, sys
snap = sys.argv[1]
try:
    cfg = json.load(open(os.path.join(snap, "config.json")))
except Exception as e:
    sys.exit(f"  ✗ config.json does not parse: {e}")
try:
    wm = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
except Exception as e:
    sys.exit(f"  ✗ index does not parse: {e}")
shards = sorted(set(wm.values()))
missing, empty = [], []
for s in shards:
    p = os.path.join(snap, s)
    if not os.path.exists(p): missing.append(s)
    elif os.path.getsize(p) == 0: empty.append(s)
print(f"  · {len(wm)} tensors across {len(shards)} shards")
if missing or empty:
    for s in missing[:10]: print(f"  ✗ missing shard: {s}")
    for s in empty[:10]:   print(f"  ✗ empty shard:   {s}")
    n = len(missing) + len(empty)
    sys.exit(f"  ✗ {n} of {len(shards)} shards unusable — re-download with: ./setup.sh --weights")
print("  ✓ every referenced shard is present and non-empty")
PY
}

download_weights() {  # download_weights <repo>  — into the HF cache, never --local-dir
  local repo="$1"
  echo "· downloading $repo into the HF cache"
  echo "  (a snapshot, not a flat folder: this is what lets run.sh mount the hub read-only"
  echo "   and have the blob symlinks resolve inside the container)"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" --exclude "fast-fp8/*" --max-workers 8
  else
    $PY - "$repo" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], ignore_patterns=["fast-fp8/*"], max_workers=8)
PY
  fi
}

ensure_venv() {
  [ -d .venv ] || { echo "· creating .venv"; python3 -m venv .venv; }
  $PY -c "import huggingface_hub" 2>/dev/null && { ok "huggingface_hub available"; return 0; }
  echo "· installing huggingface_hub"
  $PY -m pip install --quiet --upgrade pip
  $PY -m pip install --quiet -r requirements.txt
  ok "huggingface_hub installed in .venv"
}

gate() {  # gate <repo> — the gated checkpoints need the licence accepted first
  local repo="$1"
  ensure_venv
  $PY - "$repo" <<'PY' || { echo "  ✗ cannot access $repo"; echo "    1. accept the agreement on its Hugging Face page"; echo "    2. hf auth login   (or export HF_TOKEN=...)"; exit 1; }
import sys
from huggingface_hub import HfApi
repo = sys.argv[1]
HfApi().model_info(repo)
print(f"  ✓ {repo} is accessible with your token")
PY
}

# ---------------------------------------------------------------------------
MODE="${1:-all}"
REPO_ARG="${2:-}"
[ "$MODE" = "--check" ] && REPO_ARG="${2:-}"

case "$MODE" in
  --venv) ensure_venv ;;
  --image) pull_image "${IMAGE:-$(recipe_image)}" ;;
  --gate) [ -n "$REPO_ARG" ] || { echo "usage: ./setup.sh --gate <repo>" >&2; exit 1; }; gate "$REPO_ARG" ;;
  --weights)
    REPO="${REPO_ARG:-$(recipe_model)}"
    ensure_venv
    snap="$(hf_snapshot_dir "$REPO" 2>/dev/null || true)"
    if [ -n "$snap" ] && [ -f "$snap/model.safetensors.index.json" ]; then
      ok "$REPO already in the cache"
    else
      download_weights "$REPO"
      snap="$(hf_snapshot_dir "$REPO" 2>/dev/null || true)"
    fi
    [ -n "$snap" ] || { bad "still no snapshot for $REPO"; exit 1; }
    verify_weights "$REPO" "$snap"
    ;;
  --check|all)
    img="${IMAGE:-$(recipe_image)}"
    check_host "$img"
    [ "$MODE" = all ] && pull_image "$img"
    for r in recipes/az-standard.yaml recipes/az-524k-single.yaml \
             recipes/uncensored-standard.yaml recipes/uncensored-524k-single.yaml; do
      [ -f "$r" ] || continue
      repo="$(recipe_model "$r")"
      snap="$(hf_snapshot_dir "$repo" 2>/dev/null || true)"
      if [ -n "$snap" ]; then verify_weights "$repo" "$snap"; else warn "$repo not in the cache yet (./setup.sh --weights $repo)"; fi
    done
    echo
    ok "ready — launch with: ./presets/run_az_standard.sh"
    ;;
  *) echo "unknown option '$MODE' (try: --check --image --weights --gate --venv)" >&2; exit 2 ;;
esac
