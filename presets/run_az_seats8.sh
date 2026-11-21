#!/usr/bin/env bash
# presets/run_az_seats8.sh — AZ checkpoint, 8 sub-agent seats, 94k each.
#
#   ./presets/run_az_seats8.sh              # serve
#   ./presets/run_az_seats8.sh --dry-run     # show the exact docker argv
#
# The fleet configuration: 8 concurrent agents, 93,866 KV tokens each (every
# realistic agent session fits without eviction), 335 ms TTFT at C=8. One
# change from run_az_standard.sh: 8 seats instead of 16. Details in
# recipes/az-standard-seats8.yaml. For 256k per agent use az-256k-fleet.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./run.sh recipes/az-standard-seats8.yaml "$@"
