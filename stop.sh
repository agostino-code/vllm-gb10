#!/usr/bin/env bash
# stop.sh — Stop and remove the serve container
set -euo pipefail
docker rm -f qwen38-flash-next-proxy >/dev/null 2>&1 || true
docker rm -f qwen38-flash-next >/dev/null 2>&1 && echo "✓ stopped qwen38-flash-next" || true
docker rm -f qwen38-flash >/dev/null 2>&1 && echo "✓ stopped qwen38-flash" || true
echo "· server stopped (weights and caches remain intact)"
