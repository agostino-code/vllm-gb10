#!/usr/bin/env bash
# Extract the v5.4 image's Qwen4Exp model source and inject runtime steering.
# Produces steering/patched_model.py + steering/.model_py_path (the image path to mount over).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-myllmbox/qwen38-flash-next-vllm:v5.4}"
MODEL_PY="/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py"

mkdir -p steering
docker run --rm --entrypoint cat "$IMAGE" "$MODEL_PY" > steering/model.py.orig
python3 steering/patch_model.py steering/model.py.orig steering/patched_model.py
printf '%s\n' "$MODEL_PY" > steering/.model_py_path
echo "steering patch ready: steering/patched_model.py -> $MODEL_PY"
