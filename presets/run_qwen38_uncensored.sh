#!/bin/bash
# UNCENSORED preset — NO GUARDRAILS. Research / red-team / private local use only.
#
# Serves myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored (abliterated in the weights) with the
# shipped myllmbox v5.4 best settings. No runtime steering hook, no patched model source: the
# refusal direction is already removed from the checkpoint, so nothing can loop or drift.
#
# One-time setup:
#   1. accept the licence on https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored
#   2. hf auth login           (or: export HF_TOKEN=...)
#   3. ./presets/run_qwen38_uncensored.sh --download     (~105 GB, once)
#
#   ./presets/run_qwen38_uncensored.sh                 # serve
#   ./presets/run_qwen38_uncensored.sh --download      # pre-fetch the weights only
#   ./presets/run_qwen38_uncensored.sh --stop
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored"
PYV="${PYV:-./venv/bin/python}"

case "${1:-}" in
  --stop)     exec ./stop.sh ;;
  --download)
    "$PYV" -c "from huggingface_hub import snapshot_download as d; print(d('$MODEL', allow_patterns=['*.json','*.safetensors','*.jinja','*.txt']))"
    exit 0 ;;
esac

"$PYV" -c "
import sys
from huggingface_hub import HfApi
try:
    HfApi().model_info('$MODEL')
except Exception as e:
    sys.exit(f'✗ cannot access $MODEL: {e}\n  Accept the licence on its HF page and run: hf auth login')
print('· checkpoint accessible')
" || exit 1

export MTP=0
export BACKEND="${BACKEND:-v54}"
export MODEL_REPO="$MODEL"
export CTX="${CTX:-409600}"
export KV_BYTES="${KV_BYTES:-23g}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-model}"

# Weights layout: models/<name>/ with absolute per-file symlinks into the HF cache
# (same convention as models/draft-k10). run.sh mounts ./models and the HF hub, so the
# symlinks resolve inside the container; run.sh still reads config.json from the snapshot.
if [ -d "models/$(basename "$MODEL")" ]; then
  export DOWNLOAD_MODE=local
  # the per-file symlinks point into the HF cache at its host path: mount it at the same
  # path inside the container or they dangle (run.sh only does this in cache mode)
  HFH="${HF_HOME:-$HOME/.cache/huggingface}"
  export DOCKER_EXTRA_ARGS="-v $HFH/hub:$HFH/hub:ro ${DOCKER_EXTRA_ARGS:-}"
  echo "· using models/$(basename "$MODEL") (symlinked to the HF cache)"
fi

# This checkpoint stores q_proj/k_proj/v_proj under mixed schemes (int8 on 12 layers, bf16 on
# the other 36), so vLLM cannot build a fused qkv_proj. UNFUSE_QKV=1 installs ONLY that
# mapping change — no steering hook is mounted (steering is what caused the repetition loops).
echo "· installing the unfused-qkv patch (mixed-precision weights, no steering hook)"
UNFUSE_QKV=1 ./steering/install_patch.sh
MODEL_PY="$(cat steering/.model_py_path)"
export DOCKER_EXTRA_ARGS="-v $PWD/steering/patched_model.py:$MODEL_PY:ro ${DOCKER_EXTRA_ARGS:-}"

echo "· launching UNCENSORED preset (no guardrails)"
exec ./run.sh recipes/qwen38-uncensored.yaml
