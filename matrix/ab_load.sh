#!/usr/bin/env bash
# Decisive controlled A/B/A under sustained load: shipped K=12 vs K=4.
# Every config: boot -> quick decode suite -> 60 s at C=8 -> 60 s at C=16 -> stop.
# The reference is run first and last to estimate drift; sustained load averages are far less
# noisy than the short waves in the quick profile.
set -euo pipefail
cd "$(dirname "$0")/.."
R=bench/results
run() { # name recipe
  local name="$1" recipe="$2"
  mkdir -p "$R/$name"
  python3 bench/serve.py up --recipe "$recipe" --run-id "$name" --boot-timeout 1500
  python3 bench/suite.py --base-url http://127.0.0.1:8000 --run-dir "$R/$name" \
    --config-id "$name" --tag ab2 --profile quick --concurrency 1,2,4,8 \
    --request-timeout 300 --retries 1
  python3 matrix/soak.py --base http://127.0.0.1:8000 --seconds 60 --concurrency 8 \
    --out "$R/$name/soak-c8.json" || true
  python3 matrix/soak.py --base http://127.0.0.1:8000 --seconds 60 --concurrency 16 \
    --out "$R/$name/soak-c16.json" || true
  python3 bench/serve.py down --name "bench-$(basename "$recipe" .yaml)" || true
}
run ab2-m0-a recipes/qwen38-flash-solo-optimized.yaml
run ab2-k4 draft-lab/recipes/curve-k4.yaml
run ab2-m0-b recipes/qwen38-flash-solo-optimized.yaml
echo "AB2 DONE"
