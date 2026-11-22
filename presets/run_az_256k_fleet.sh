#!/usr/bin/env bash
# presets/run_az_256k_fleet.sh — AZ checkpoint, 4-5 seats x 256k native.
#
#   ./presets/run_az_256k_fleet.sh              # serve
#   ./presets/run_az_256k_fleet.sh --dry-run     # show the exact docker argv
#
# The 256k-per-agent fleet: 4-5 seats at the native 262,144 (no YaRN, recall
# verified at 259k), fp8 KV to fit them in 23 GB. Carries the 2.6 s
# concurrent-admission barrier (it is fp8 AND 262144 — see the recipe) so it is
# right for a few long-running agents (<=2 concurrent: 106 ms, no barrier), not
# a wide lockstep fleet. For a barrier-free wide fleet use run_az_seats8.sh.
# Details in recipes/az-256k-fleet.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./run.sh recipes/az-256k-fleet.yaml "$@"
