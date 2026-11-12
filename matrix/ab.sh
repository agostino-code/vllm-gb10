#!/usr/bin/env bash
# Controlled back-to-back A/B/A: shipped K=12 vs winner K=4 on the same quick protocol under
# the same host conditions, with the reference repeated to estimate drift.
set -euo pipefail
cd "$(dirname "$0")/.."
R=bench/results
run() { # name recipe
  local name="$1" recipe="$2"
  mkdir -p "$R/$name"
  python3 bench/serve.py up --recipe "$recipe" --run-id "$name" --boot-timeout 1500
  python3 bench/suite.py --base-url http://127.0.0.1:8000 --run-dir "$R/$name" \
    --config-id "$name" --tag ab --profile quick --concurrency 1,2,4,8 \
    --request-timeout 300 --retries 1
  python3 bench/serve.py down --name "bench-$(basename "$recipe" .yaml)" || true
}
run ab-m0-a recipes/qwen38-flash-solo-optimized.yaml
run ab-k4 draft-lab/recipes/curve-k4.yaml
run ab-m0-b recipes/qwen38-flash-solo-optimized.yaml
echo "AB DONE"
