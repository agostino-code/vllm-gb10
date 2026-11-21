#!/usr/bin/env bash
# run.sh — serve one Qwen3.8-Flash-Next checkpoint on one DGX Spark (GB10).
#
#   ./run.sh                              # recipes/az-standard.yaml
#   ./run.sh recipes/az-524k-single.yaml  # your long-context single-stream config
#   ./run.sh <recipe> --dry-run           # print the exact docker argv, change nothing
#
# The recipe is the single source of truth: `server:` for the launch, `env:`
# for the container environment, `vllm:` for the engine flags. run.sh adds
# nothing to the command line beyond the mounts, the port and the model path.
#
# There is deliberately no MTP/CTX/KV_BYTES knob here: every recipe carries
# exactly one --speculative-config, and the old second one was the single
# biggest footgun in this repo (a duplicate at depth <= 3 makes the v5.4 image
# refuse to boot at CUDA-graph capture). Use ./setup.sh to verify the box and
# the weights; use ./check.sh to verify a running server.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env

command -v docker >/dev/null 2>&1 || { echo "✗ docker is required" >&2; exit 1; }

DRY_RUN=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY_RUN=1; done
RECIPE="${1:-$DEFAULT_RECIPE}"
[ "${RECIPE#*/}" = "$RECIPE" ] && [ -f "recipes/$RECIPE" ] && RECIPE="recipes/$RECIPE"
[ -f "$RECIPE" ] || { echo "✗ recipe file '$RECIPE' not found" >&2; exit 1; }

# --- recipe reader ---------------------------------------------------------
# Two-level YAML: a section starts at column 0, keys are indented. Values are
# taken verbatim (so JSON with commas and quotes survives); trailing `#`
# comments outside quotes are stripped.
rkey() {  # rkey <section> <key>
  awk -v s="$1" -v k="$2" '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec) }
    sec==s && $1==k":" {
      sub(/^[ ]*[^:]*:[ ]*/,""); sub(/[ ]+#.*$/,"")
      gsub(/^["\x27]|["\x27]$/,""); print; exit
    }' "$RECIPE"
}
rsection() {  # every "key<TAB>value" of a section
  awk -v s="$1" '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec); next }
    sec==s && $1 ~ /^[A-Za-z0-9_-]+:$/ || (sec==s && /^[ ]+[A-Za-z0-9_-]+:[ ]/) {
      line=$0; sub(/^[ ]+/,"",line)
      key=line; sub(/:.*/,"",key)
      val=line; sub(/^[^:]*:[ ]*/,"",val); sub(/[ ]+#.*$/,"",val)
      gsub(/^["\x27]|["\x27]$/,"",val)
      if (key != "") print key "\t" val
    }' "$RECIPE"
}

IMAGE="${IMAGE:-$(rkey server image)}"
PORT="${PORT:-$(rkey server port)}"
HF_REPO="${MODEL_REPO:-$(rkey server model)}"
CACHE_DIR="${CACHE_DIR:-$(rkey server cache_dir)}"
CPUSET="$(rkey server cpuset)"
DOCKER_ARGS="$(rkey server docker_args)"   # honoured: the PLE table reader needs seccomp=unconfined
: ${IMAGE:=myllmbox/qwen38-flash-next-vllm:v5.4}
: ${PORT:=8000}
: ${CACHE_DIR:=./cache}
[ -n "$HF_REPO" ] || { echo "✗ no model: set server.model in $RECIPE or MODEL_REPO=..." >&2; exit 1; }

# --- resolve the weights ---------------------------------------------------
# The whole point: one read-only mount of the HF hub at its own path, and the
# model served from the snapshot's absolute path. See the note in config.env.
LOCAL_NAME="$(basename "$HF_REPO")"
if [ "$DOWNLOAD_MODE" = cache ]; then
  MODEL_DIR="$(hf_snapshot_dir "$HF_REPO" 2>/dev/null || true)"
  if [ -z "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "· $HF_REPO is not in the HF cache yet — downloading (~1-2 h the first time)"
    if command -v hf >/dev/null 2>&1; then
      hf download "$HF_REPO" --exclude "fast-fp8/*" --max-workers 8
    else
      python3 - "$HF_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], ignore_patterns=["fast-fp8/*"], max_workers=8)
PY
    fi
    MODEL_DIR="$(hf_snapshot_dir "$HF_REPO" 2>/dev/null || true)"
    [ -n "$MODEL_DIR" ] || { echo "✗ download finished but no snapshot found under $HF_HOME/hub" >&2; exit 1; }
  fi
  SERVE_PATH="$MODEL_DIR"
  MOUNTS=(-v "$HF_HUB_MOUNT:$HF_HUB_MOUNT:ro")
else
  MODEL_DIR="$MODELS_DIR/$LOCAL_NAME"
  [ -f "$MODEL_DIR/model.safetensors.index.json" ] || {
    echo "✗ $MODEL_DIR has no model.safetensors.index.json" >&2
    echo "  run ./setup.sh, or use the default DOWNLOAD_MODE=cache" >&2; exit 1; }
  SERVE_PATH="$MODEL_DIR"
  MOUNTS=(-v "$MODEL_DIR:$MODEL_DIR:ro")
fi

mkdir -p "$CACHE_DIR"; CACHE_ABS="$(cd "$CACHE_DIR" && pwd)"
MOUNTS+=(-v "$CACHE_ABS:/cache")

# --- container environment and engine flags --------------------------------
ENVS=()
while IFS=$'\t' read -r k v; do [ -n "$k" ] && ENVS+=(-e "$k=$v"); done < <(rsection env)

# served-model-name is resolved once, from whichever source has it:
# SERVED_NAME (env) > the recipe > the repo id. Emitting it twice is a vLLM
# last-wins footgun, so the key is skipped in the loop and appended after.
SERVED="${SERVED_NAME:-$(rkey vllm served-model-name)}"
[ -n "$SERVED" ] || SERVED="$HF_REPO"

FLAGS=()
while IFS=$'\t' read -r k v; do
  [ "$k" = "served-model-name" ] && continue
  case "$v" in
    true)        FLAGS+=("--$k");;
    false|null|"") ;;
    *)           FLAGS+=("--$k" "$v");;
  esac
done < <(rsection vllm)
FLAGS+=("--served-model-name" "$SERVED")

TPL=()
if [ -n "$CHAT_TEMPLATE" ]; then
  [ -f "$CHAT_TEMPLATE" ] || { echo "✗ CHAT_TEMPLATE '$CHAT_TEMPLATE' is not a readable file" >&2; exit 1; }
  TPL=(--chat-template "$CHAT_TEMPLATE")
fi

# --- preflight -------------------------------------------------------------
cp_now=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
[ "$cp_now" = "0" ] || echo "  ⚠ vm.compaction_proactiveness is $cp_now (want 0): run ./tune-host.sh"
avail=$(free -g | awk '/^Mem:/{print $7}')
echo "  · memory available: ${avail}G   (weights + KV live in the unified pool)"

# --- launch ----------------------------------------------------------------
# Unified-memory insurance (same as bench/serve.py): a hard cgroup cap with no
# container swap turns an over-subscribed KV/chunk into a clean container OOM-
# kill instead of a host-wide freeze; the oom-score bias protects the desktop.
# Override by appending your own -m to DOCKER_EXTRA_ARGS (last flag wins).
RUN=(docker run -d --name "$CONTAINER" --gpus all --ipc=host --shm-size 16g \
     --memory 116g --memory-swap 116g --oom-score-adj 700)
[ -n "$CPUSET" ] && RUN+=(--cpuset-cpus "$CPUSET")
[ -n "$DOCKER_ARGS" ] && RUN+=($DOCKER_ARGS)
[ -n "$DOCKER_EXTRA_ARGS" ] && RUN+=($DOCKER_EXTRA_ARGS)
RUN+=(-p "$PORT:8000" "${MOUNTS[@]}"
     -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
     -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace
     -e VLLM_CACHE_ROOT=/cache/vllm-cache
     "${ENVS[@]}"
     --entrypoint vllm "$IMAGE" serve "$SERVE_PATH" --port 8000 "${FLAGS[@]}")
[ -n "$EXTRA_ARGS" ] && RUN+=($EXTRA_ARGS)

if [ "$DRY_RUN" = 1 ]; then
  printf '%q ' "${RUN[@]}"; printf '\n'
  exit 0
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
echo "· starting $CONTAINER ($IMAGE) on :$PORT from $RECIPE"
echo "  model : $SERVE_PATH"
echo "  recipe: $RECIPE"
"${RUN[@]}" >/dev/null

echo "· streaming engine logs until healthy (Ctrl-C detaches; the server keeps booting)"
docker logs -f "$CONTAINER" 2>&1 &
LOGS=$!
trap 'kill "$LOGS" 2>/dev/null || true' EXIT INT TERM
for _ in $(seq 1 480); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    kill "$LOGS" 2>/dev/null || true; wait "$LOGS" 2>/dev/null || true
    echo
    echo "──────────────────────────────────────────────────────────"
    echo "✓ server booted — OpenAI-compatible API on http://127.0.0.1:$PORT/v1"
    echo "  verify: ./check.sh      monitor: ./view.sh      stop: ./stop.sh"
    echo "──────────────────────────────────────────────────────────"
    exit 0
  fi
  if ! docker ps -q --filter "name=$CONTAINER" | grep -q .; then
    kill "$LOGS" 2>/dev/null || true; wait "$LOGS" 2>/dev/null || true
    echo "✗ container exited — see the engine log above, then: docker logs $CONTAINER" >&2
    exit 1
  fi
  sleep 5
done
echo "✗ not healthy after 40 min — watch with: docker logs -f $CONTAINER" >&2
exit 1
