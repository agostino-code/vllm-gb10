#!/usr/bin/env bash
# Clean full-suite reference rerun for the shipped recipe (K=12 dynamic) so the winner's
# full-profile numbers can be compared apples-to-apples (the earlier m0-reference-full run
# was measurably contaminated by a concurrent 90 s probe).
set -euo pipefail
cd "$(dirname "$0")/.."
R=bench/results/m0-reference-clean-full
mkdir -p "$R"
python3 bench/serve.py up --recipe recipes/qwen38-flash-solo-optimized.yaml --run-id m0-reference-clean-full --boot-timeout 1500
python3 bench/suite.py --base-url http://127.0.0.1:8000 --run-dir "$R" \
  --config-id m0-reference-clean-full --tag reference --profile full \
  --include-16 --long-smoke --quality off --concurrency 1,2,4,8 \
  --request-timeout 300 --retries 1
python3 bench/serve.py down --name bench-qwen38-flash-solo-optimized || true
echo "M0 CLEAN FULL DONE"
