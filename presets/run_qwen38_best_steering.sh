#!/bin/bash
# Best preset + runtime steering (concise style) for azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ.
#
# What it does:
#   1. builds the steering vector (steering/succinct.pt) from the model's output head if missing
#   2. extracts the v5.4 image's Qwen4Exp model source and mounts a patched copy that applies
#      orthogonal-projection steering on the final residual stream
#   3. launches the shipped best recipe (K=12 dynamic, bf16 KV 23G, 16 seats) with
#      STEERING_VECTOR_PATH/STEERING_STRENGTH in the container
#
# The stock myllmbox image has no steering support, hence the mounted patched model.py.
# Tune the effect with STEERING_STRENGTH (default 1.0; 0 = off, ~1 = full projection removal,
# try 0.5-3.0). The steering op is inside the CUDA-graph-captured forward, so it adds no
# per-step Python overhead; the MTP drafter is unaffected (it reads the pre-mixer buffer).
#
#   ./presets/run_qwen38_best_steering.sh                 # strength 1.5
#   STEERING_STRENGTH=2.5 ./presets/run_qwen38_best_steering.sh
#   ./presets/run_qwen38_best_steering.sh --dry-run       # (run.sh has no dry-run; use docker logs)
set -euo pipefail
cd "$(dirname "$0")/.."

export MTP=0                       # recipe carries its own single speculative-config
export BACKEND="${BACKEND:-v54}"
export MODEL_REPO="${MODEL_REPO:-azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ}"
export CTX="${CTX:-409600}"
export KV_BYTES="${KV_BYTES:-23g}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-model}"
STRENGTH="${STEERING_STRENGTH:-1.0}"

echo "· building steering vector (CPU, cached)"
./steering/build_vector.sh

echo "· installing runtime-steering patch for $(docker images --format '{{.Repository}}:{{.Tag}}' | grep -m1 'myllmbox/qwen38-flash-next')"
./steering/install_patch.sh
MODEL_PY="$(cat steering/.model_py_path)"

RECIPE="recipes/qwen38-best-steering.yaml"
if [ "$STRENGTH" != "1.0" ]; then
  mkdir -p cache/run
  sed "s/STEERING_STRENGTH: \"1.0\"/STEERING_STRENGTH: \"$STRENGTH\"/" "$RECIPE" \
    > cache/run/qwen38-best-steering.strength.yaml
  RECIPE="cache/run/qwen38-best-steering.strength.yaml"
  echo "· STEERING_STRENGTH=$STRENGTH (temporary recipe: $RECIPE)"
fi

# run.sh passes DOCKER_EXTRA_ARGS to docker run (before the recipe envs).
MOUNTS="-v $PWD/steering/patched_model.py:$MODEL_PY:ro -v $PWD/steering/succinct.pt:/steering/vector.pt:ro"
export DOCKER_EXTRA_ARGS="$MOUNTS ${DOCKER_EXTRA_ARGS:-}"
export STEERING_VECTOR_PATH="$PWD/steering/succinct.pt"
export STEERING_STRENGTH="$STRENGTH"

echo "· launching best preset + steering (strength $STRENGTH)"
exec ./run.sh "$RECIPE"
