# vLLM GB10-Optimized Stack

This repository provides a high-performance vLLM deployment stack optimized for the DGX Spark (GB10) architecture. 

## Features
- **Dynamic KV Cache:** Automatically allocates maximum available GPU VRAM.
- **Modular Steering:** Apply arbitrary steering vectors at runtime using orthogonal projection (DS4-style).
- **Ready for Research:** Includes evaluation tools to measure steering effectiveness.

## Setup
1. Configure settings in `config.env`.
2. Ensure you have the vLLM stack image.
3. Launch with `BACKEND=v54 ./serve.sh`.

## Steering
Generate vectors using `steering/generate_vector.py`.
Apply via env:
```bash
export STEERING_VECTOR_PATH="/workspace/steering/your_vector.pt"
export STEERING_STRENGTH="1.0"
BACKEND=v54 ./serve.sh
```
