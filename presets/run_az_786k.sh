#!/usr/bin/env bash
# presets/run_az_786k.sh — AZ checkpoint, 786k context (YaRN 3.0), single stream.
#
#   ./presets/run_az_786k.sh              # serve
#   ./presets/run_az_786k.sh --dry-run     # show the exact docker argv
#
# The widest context measured on this box: 786,432 tokens, YaRN factor 3.0,
# 778k needle verified. One stream only (the 26 GB KV pool is 1.14x a full
# request). A full 780k prompt takes ~6 min to first token. Details in
# recipes/az-786k-single.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./run.sh recipes/az-786k-single.yaml "$@"
