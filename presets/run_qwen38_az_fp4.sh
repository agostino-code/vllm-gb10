#!/bin/bash
# Preset: Qwen3.8-Flash-Next-FP4-Local-AZ (Installed Model)
export QUANT=fp4
export RECIPE=recipes/qwen3.8-flash-next-nvfp4-solo.yaml
# Override modello se necessario
export MODEL_REPO="azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ"

echo "--- Launching Qwen3.8-Flash-Next-FP4-Local-AZ ---"
./tune-host.sh
./run.sh $RECIPE
