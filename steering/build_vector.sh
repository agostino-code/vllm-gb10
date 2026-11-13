#!/usr/bin/env bash
# Build the concise-style steering vector from the model's output head (CPU only, ~1 min),
# then copy it into ./cache (which run.sh mounts at /cache for the container).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-myllmbox/qwen38-flash-next-vllm:v5.4}"
SNAP="${SNAP:-$(ls -d ~/.cache/huggingface/hub/models--azampatti--Qwen3.8-Flash-Next-FP4-Local-AZ/snapshots/*/ 2>/dev/null | head -1)}"
[ -n "$SNAP" ] && [ -f "$SNAP/model.safetensors.index.json" ] || {
  echo "✗ model snapshot not found (set SNAP=...)" >&2; exit 1; }

mkdir -p steering cache/steering
if [ ! -f steering/succinct.pt ]; then
  docker run --rm --entrypoint python3 \
    -v "$SNAP:/model:ro" -v "$PWD/steering:/steering" \
    -v "$PWD/steering/make_steering_vector.py:/make_steering_vector.py:ro" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "$IMAGE" /make_steering_vector.py --model /model --out /steering/succinct.pt
fi
cp -f steering/succinct.pt cache/steering/succinct.pt
echo "vector ready: cache/steering/succinct.pt (meta: steering/succinct.json)"
