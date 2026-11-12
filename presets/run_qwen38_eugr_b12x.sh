#!/bin/bash
# Preset: Qwen3.8-Flash-Next-FP4-Local-AZ on the eugr B12X stack.
#
# The eugr stack is NOT launched through ./run.sh (run.sh parses the myllmbox
# recipe schema); it has its own self-contained launcher:
#   eugr-b12x/launch.sh      (plain docker run, model mounted ro from HF cache)
#   eugr-b12x/run_config.sh  (boot -> validate -> bench/suite.py -> stop)
#
# Usage examples:
#   ./presets/run_qwen38_eugr_b12x.sh                 # base: MTP=4, fp8 KV, util 0.75
#   MTP=6 ./presets/run_qwen38_eugr_b12x.sh           # MTP depth 6
#   KV_DTYPE=auto GPU_UTIL=0.80 ./presets/run_qwen38_eugr_b12x.sh   # bf16 KV
#   AUTOTUNE=0 ./presets/run_qwen38_eugr_b12x.sh --dry-run          # show docker argv
#   ./presets/run_qwen38_eugr_b12x.sh --stop                          # stop server
#
# Notes:
# - AUTOTUNE defaults to 0 here: the b12x kernel autotune search costs ~55-65 min
#   on first boot (see eugr-b12x/README.md). Use AUTOTUNE=1 only for a tuned boot.
# - MTP depth is capped at 7 by this image (MTP 9 fails at engine start).
# - One model server at a time; this script only manages container
#   qwen38-flash-next-b12x and refuses to touch the myllmbox container.
cd "$(dirname "$0")/.." || exit 1
export AUTOTUNE="${AUTOTUNE:-0}"
export MTP="${MTP:-4}"
export KV_DTYPE="${KV_DTYPE:-fp8}"
export GPU_UTIL="${GPU_UTIL:-0.75}"
exec ./eugr-b12x/launch.sh "$@"
