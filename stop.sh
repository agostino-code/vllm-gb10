#!/usr/bin/env bash
# stop.sh — stop and remove the serving container. Weights and caches stay.
#
#   ./stop.sh              stop the container named by CONTAINER (default qwen38-flash-next)
#   ./stop.sh <name>       stop a specific container
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env

if [ $# -ge 1 ]; then
  NAME="$1"
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "✓ stopped $NAME" || echo "· $NAME was not running"
  exit 0
fi

for NAME in "${CONTAINER:-qwen38-flash-next}" "${CONTAINER:-qwen38-flash-next}-proxy"; do
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "✓ stopped $NAME" || true
done
echo "· server stopped (weights and caches intact)"
