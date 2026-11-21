#!/usr/bin/env bash
# presets/run_uncensored_524k.sh — abliterated checkpoint, 524k single stream.
#
# ⚠ NO GUARDRAILS. Research / red-team / private local use only.
#
#   1. accept the licence on
#        https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored
#   2. hf auth login                      (or export HF_TOKEN=...)
#   3. ./presets/run_uncensored_524k.sh --download     (~105 GB, once)
#   4. ./presets/run_uncensored_524k.sh                 # serve
#
# Same long-context tuning as run_az_524k.sh, on the uncensored checkpoint.
# MBX_QSA_UNFUSE_INDEXER=1 is set in the recipe — it is a property of this
# checkpoint (mixed int8/bf16 q,k,v), not of the tuning, and the old
# UNFUSE_QKV model patch that replaced it is what caused the repetition loops.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored"

case "${1:-}" in
  --download) exec ./setup.sh --weights "$MODEL" ;;
esac

./setup.sh --gate "$MODEL"
exec ./run.sh recipes/uncensored-524k-single.yaml "$@"
