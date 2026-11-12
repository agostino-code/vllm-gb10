#!/bin/bash
# Alternative preset: K=4 speculative depth (half the draft compute of the shipped K=12).
#
# Measured throughput-equal to the shipped preset within run-to-run noise (controlled sustained
# C=8: 198.5 vs 198.5 tok/s), slightly behind on pure-code content (67.6 vs 70.1 tok/s) and
# slightly ahead on one A/B pair. It is NOT faster; use it if you want lower draft compute/power.
# See BENCHMARK-REPORT.md §4.2 and §7.2.
set -euo pipefail
cd "$(dirname "$0")/.."
export MTP=0
export BACKEND="${BACKEND:-v54}"
export MODEL_REPO="${MODEL_REPO:-azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ}"
export CTX="${CTX:-409600}"
export KV_BYTES="${KV_BYTES:-23g}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-model}"
exec ./run.sh recipes/qwen38-k4.yaml "$@"
