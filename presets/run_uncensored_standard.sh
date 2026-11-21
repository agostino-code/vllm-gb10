#!/usr/bin/env bash
# presets/run_uncensored_standard.sh — abliterated checkpoint, shipped settings.
#
# ⚠ NO GUARDRAILS. Research / red-team / private local use only, at your own
# risk and your own moderation.
#
#   1. accept the licence on
#        https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored
#   2. hf auth login                      (or export HF_TOKEN=...)
#   3. ./presets/run_uncensored_standard.sh --download     (~105 GB, once)
#   4. ./presets/run_uncensored_standard.sh                # serve
#
#   ./presets/run_uncensored_standard.sh --download   fetch the weights only
#   ./presets/run_uncensored_standard.sh --dry-run    show the docker argv
#
# The weights come from the Hugging Face cache exactly like the AZ checkpoint:
# one read-only mount of the hub, served from the snapshot's absolute path.
# There is no models/ folder and no symlink farm to keep in sync.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored"

case "${1:-}" in
  --download) exec ./setup.sh --weights "$MODEL" ;;
esac

./setup.sh --gate "$MODEL"
exec ./run.sh recipes/uncensored-standard.yaml "$@"
