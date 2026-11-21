#!/usr/bin/env bash
# presets/run_az_524k.sh — AZ checkpoint, 524k context, single stream.
#
#   ./presets/run_az_524k.sh              # serve
#   ./presets/run_az_524k.sh --dry-run     # show the exact docker argv
#
# Your tuned configuration: one agent gets the whole KV pool and the whole
# draft budget. Same image, kernels and drafter as run_az_standard.sh; the
# differences are documented inline in recipes/az-524k-single.yaml.
#
# If your client is pinned to a specific model name, pass it:
#   SERVED_NAME=Qwen/Qwen3.8-Flash-Next ./presets/run_az_524k.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./run.sh recipes/az-524k-single.yaml "$@"
