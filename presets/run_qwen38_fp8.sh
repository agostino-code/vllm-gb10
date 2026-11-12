#!/bin/bash
# Preset: Qwen-FP8 (Quality Focus)
export QUANT=fp8
export RECIPE=recipes/qwen3.5-122b-fp8.yaml

echo "--- Launching Qwen3.8 FP8 Precision Preset ---"
./tune-host.sh
./run.sh $RECIPE
