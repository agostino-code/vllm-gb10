#!/usr/bin/env bash
# presets/run_uncensored_786k.sh — abliterated checkpoint, 786k single stream.
#
# ⚠ NO GUARDRAILS. Research / red-team / private local use only.
#
#   1. accept the licence on
#        https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored
#   2. hf auth login                      (or export HF_TOKEN=...)
#   3. ./presets/run_uncensored_786k.sh --download     (~105 GB, once)
#   4. ./presets/run_uncensored_786k.sh                 # serve
#
# The uncensored twin of run_az_786k.sh: YaRN 3.0, 786,432 context, one stream.
# Chosen for the single-stream case — at C=1 the uncensored checkpoint matches
# AZ on decode, edges it on quality, and removes refusals; the ~25 % batch
# penalty is a C=8 effect a one-stream config never pays. MBX_QSA_UNFUSE_INDEXER
# =1 is set in the recipe (a property of this checkpoint, not the tuning).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored"
case "${1:-}" in
  --download) exec ./setup.sh --weights "$MODEL" ;;
esac
./setup.sh --gate "$MODEL"
exec ./run.sh recipes/uncensored-786k-single.yaml "$@"
