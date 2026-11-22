#!/usr/bin/env bash
# presets/run_uncensored_256k_fleet.sh — abliterated, 4-5 seats x 256k native.
#
# ⚠ NO GUARDRAILS. Research / red-team / private local use only.
#
#   1. accept the licence on
#        https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored
#   2. hf auth login                      (or export HF_TOKEN=...)
#   3. ./presets/run_uncensored_256k_fleet.sh --download   (~105 GB, once)
#   4. ./presets/run_uncensored_256k_fleet.sh               # serve
#
# The uncensored twin of az-256k-fleet: 4-5 seats at native 262,144, fp8 KV.
# CAVEAT: this shape carries the 2.6 s concurrent-admission barrier (fp8 +
# 262144) AND the uncensored ~25 % batch penalty — both bite at C=N. Right for
# a FEW long uncensored agents (<=2 concurrent: neither applies); for a wide
# uncensored fleet use a bf16 extended-length shape. Details in the recipe.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored"
case "${1:-}" in
  --download) exec ./setup.sh --weights "$MODEL" ;;
esac
./setup.sh --gate "$MODEL"
exec ./run.sh recipes/uncensored-256k-fleet.yaml "$@"
