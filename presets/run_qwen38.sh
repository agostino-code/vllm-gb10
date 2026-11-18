#!/bin/bash
# Best measured preset for azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ (2026-11-12 preset study).
#
# Result in one line: the SHIPPED myllmbox v5.4 recipe is already the best configuration we
# measured on this box. The 2026-11-12 study tested 20+ draft/KV/template/depth variants and
# none beat it beyond run-to-run noise (controlled sustained C=8: 198.5 tok/s for the shipped
# preset vs 198.5 for the K=4 variant; C=16 287-292 vs 301). See BENCHMARK-REPORT.md.
#
# This launcher fixes the one configuration footgun we found: run.sh appends a SECOND
# --speculative-config from config.env when MTP != 0, and a K=3 duplicate makes the v5.4 image
# refuse to boot ("min bootable speculative depth is 4"). MTP=0 keeps the recipe's single
# speculative-config (K=12 dynamic) authoritative.
set -euo pipefail
cd "$(dirname "$0")/.."

export MTP=0
export BACKEND="${BACKEND:-v54}"
export MODEL_REPO="${MODEL_REPO:-azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ}"
export CTX="${CTX:-409600}"
export KV_BYTES="${KV_BYTES:-23g}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-model}"   # stock template: medium measured no gain for this checkpoint

exec ./run.sh recipes/qwen38-flash-solo-optimized.yaml "$@"
