#!/usr/bin/env bash
# Boot one recipe, run the quick suite, stop. Used for depth-curve point checks.
# Usage: matrix/quickcheck.sh <recipe-path> <run-id> [extra suite args...]
set -euo pipefail
cd "$(dirname "$0")/.."
RECIPE="${1:?recipe}"; NAME="${2:?run-id}"; shift 2 || true
mkdir -p "bench/results/$NAME"
python3 bench/serve.py up --recipe "$RECIPE" --run-id "$NAME" --boot-timeout 1500
python3 bench/suite.py --base-url http://127.0.0.1:8000 --run-dir "bench/results/$NAME" \
  --config-id "$NAME" --tag finalist-curve --profile quick \
  --concurrency 1,2,4,8 --request-timeout 300 --retries 1 "$@"
python3 bench/serve.py down --name "bench-$(basename "$RECIPE" .yaml)" || true
