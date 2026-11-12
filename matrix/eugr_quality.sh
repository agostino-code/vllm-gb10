#!/usr/bin/env bash
# eugr b12x finalist validation: boot the b12x base config and run the quality quick battery
# (same battery as M0) so the stack comparison includes answer quality, not only throughput.
# Run only when the GPU is free; the main matrix must not be running.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p quality/runs/eugr-b12x-base

echo "=== boot eugr b12x base (MTP=4, fp8 KV, AUTOTUNE=0) ==="
cd eugr-b12x
AUTOTUNE=0 ./launch.sh --wait
cd ..

echo "=== quality quick on eugr b12x base ==="
python3 quality/run_quality.py --base-url http://127.0.0.1:8000/v1 \
  --mode quick --label eugr-b12x-base \
  --output quality/runs/eugr-b12x-base/quality.json

echo "=== stop eugr ==="
cd eugr-b12x && ./launch.sh --stop
cd ..
echo "EUGR QUALITY DONE"
