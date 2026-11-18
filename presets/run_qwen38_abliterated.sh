#!/bin/bash
# ABLITERATED preset (research / red-team / private local use).
#
# Builds a refusal-direction steering vector from LLM-LAT/harmful-dataset and serves the
# best recipe with it projected out of the final residual stream (Arditi et al. 2024):
#   v = normalize( mean(h | forced refusal prefix) - mean(h | forced compliance prefix) )
#   y = y - strength * (y @ v) * v
#
# First run performs the whole pipeline automatically:
#   1. patch the v5.4 image's model source (steering/install_patch.sh)
#   2. boot a CAPTURE server (patched model + STEERING_CAPTURE_DIR, no steering)
#   3. measure the baseline refusal rate on 50 held-out harmful prompts
#   4. collect 400 decision-point activations (200 prompts x forced refusal/compliance)
#   5. build steering/refusal.pt, stop the capture server
#   6. boot the ABLITERATED server (same recipe + refusal vector mounted)
#
# Subsequent runs skip straight to step 6 (vector cached). Delete steering/refusal.* to rebuild.
#
#   ./presets/run_qwen38_abliterated.sh
#   STEERING_STRENGTH=-1.5 ./presets/run_qwen38_abliterated.sh
#   ./presets/run_qwen38_abliterated.sh --rebuild     # force vector rebuild
#   ./presets/run_qwen38_abliterated.sh --stop        # stop the server
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-myllmbox/qwen38-flash-next-vllm:v5.4}"
STRENGTH="${STEERING_STRENGTH:--2.0}"
PYV="${PYV:-./venv/bin/python}"
MODEL_PY=""

[ -x "$PYV" ] || { echo "✗ need $PYV (numpy). Run: ./venv/bin/pip install pyarrow numpy" >&2; exit 1; }

if [ "${1:-}" = "--stop" ]; then exec ./stop.sh; fi
REBUILD=0
for a in "$@"; do [ "$a" = "--rebuild" ] && REBUILD=1; done

rm -f steering/.model_py_path
echo "· installing runtime-steering patch"
./steering/install_patch.sh
MODEL_PY="$(cat steering/.model_py_path)"

export MTP=0
export BACKEND="${BACKEND:-v54}"
export MODEL_REPO="${MODEL_REPO:-azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ}"
export CTX="${CTX:-409600}"
export KV_BYTES="${KV_BYTES:-23g}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-model}"

if [ ! -f steering/refusal.pt ] || [ "$REBUILD" = 1 ]; then
  echo "· building refusal vector (capture server boot + ~400 requests, ~15 min)"
  mkdir -p cache/capture && rm -f cache/capture/cap_*
  export DOCKER_EXTRA_ARGS="-v $PWD/steering/patched_model.py:$MODEL_PY:ro -e STEERING_CAPTURE_DIR=/cache/capture"
  ./run.sh recipes/qwen38-flash-solo-optimized.yaml

  echo "· baseline refusal rate (50 held-out prompts)"
  "$PYV" steering/validate_refusal.py --out /tmp/refusal-base.json || true

  echo "· collecting decision-point activations + building direction"
  "$PYV" steering/build_refusal_vector.py

  echo "· converting vector to torch format"
  docker run --rm --entrypoint python3 -v "$PWD/steering:/steering" "$IMAGE" -c \
    "import numpy, torch; v=torch.from_numpy(numpy.fromfile('/steering/refusal.f32',dtype=numpy.float32)); torch.save(v.contiguous(),'/steering/refusal.pt'); print('saved refusal.pt', tuple(v.shape))"

  ./stop.sh
  sleep 3
fi

echo "· launching ABLITERATED server (strength $STRENGTH)"
export DOCKER_EXTRA_ARGS="-v $PWD/steering/patched_model.py:$MODEL_PY:ro -v $PWD/steering/refusal.pt:/steering/vector.pt:ro"
export STEERING_STRENGTH="$STRENGTH"
exec ./run.sh recipes/qwen38-abliterated.yaml
