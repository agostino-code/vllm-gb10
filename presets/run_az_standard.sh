#!/usr/bin/env bash
# presets/run_az_standard.sh — the best measured configuration for the AZ checkpoint.
#
#   ./presets/run_az_standard.sh              # serve
#   ./presets/run_az_standard.sh --dry-run     # show the exact docker argv
#
# Thin on purpose: everything lives in recipes/az-standard.yaml, and run.sh
# reads that file. This launcher exists so the recommended path is one command
# and so the measured settings cannot be drifted by a typo in a shell export.
#
# Overrides you may want without editing anything:
#   PORT=8010 ./presets/run_az_standard.sh
#   IMAGE=myllmbox/qwen38-flash-next-vllm:v5.4 ./presets/run_az_standard.sh
#   SERVED_NAME=my-model ./presets/run_az_standard.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./run.sh recipes/az-standard.yaml "$@"
