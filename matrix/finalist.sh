#!/usr/bin/env bash
# Finalist validation for the measured winner (d-curve-k4: K=4 dynamic, self-draft).
# Runs: boot -> full suite + full quality -> temperature 0.6 quick suite -> 7-min C=8 soak -> stop.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=http://127.0.0.1:8000
R=bench/results
mkdir -p "$R/finalist-curve-k4-full" "$R/finalist-curve-k4-temp06" "$R/finalist-curve-k4-soak"

echo "=== [1/5] boot d-curve-k4 (K=4) ==="
python3 bench/serve.py up --recipe draft-lab/recipes/curve-k4.yaml \
  --run-id finalist-curve-k4 --boot-timeout 1500

echo "=== [2/5] full suite + full quality ==="
python3 bench/suite.py --base-url "$BASE" --run-dir "$R/finalist-curve-k4-full" \
  --config-id finalist-curve-k4-full --tag finalist --profile full \
  --include-16 --long-smoke --quality full --quality-timeout 1200 \
  --concurrency 1,2,4,8 --request-timeout 300 --retries 1

echo "=== [3/5] temperature 0.6 quick suite (acceptance at sampled decoding) ==="
python3 bench/suite.py --base-url "$BASE" --run-dir "$R/finalist-curve-k4-temp06" \
  --config-id finalist-curve-k4-temp06 --tag finalist-temp06 --profile quick \
  --temperature 0.6 --concurrency 1,2,4,8 --include-16 --request-timeout 300 --retries 1

echo "=== [4/5] 7-minute C=8 soak ==="
python3 matrix/soak.py --base "$BASE" --seconds 420 --concurrency 8 \
  --out "$R/finalist-curve-k4-soak/soak.json"

echo "=== [5/5] stop server ==="
python3 bench/serve.py down --name bench-curve-k4 || true
echo "FINALIST DONE"
