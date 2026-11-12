#!/bin/bash
# Preset: Qwen3.8-Flash-FP4 (High Performance)
export QUANT=fp4
export RECIPE=recipes/qwen3.8-flash-next-nvfp4-solo.yaml

echo "--- Launching Qwen3.8 FP4 Performance Preset ---"
./tune-host.sh
./run.sh $RECIPE
