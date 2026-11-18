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

echo "· launching UNCENSORED preset (no guardrails)"
exec ./run.sh recipes/qwen38-uncensored.yaml
