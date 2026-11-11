# vLLM GB10-Optimized Steering Stack

A high-performance deployment stack for Qwen3.8-Flash models on DGX Spark (GB10) hardware.

## Features
- **Dynamic KV Cache:** Auto-scales based on available GPU VRAM.
- **Modular Runtime Steering:** Apply arbitrary steering vectors at runtime using orthogonal projection (DS4-style).
- **Portable Architecture:** Uses Git submodules for core dependencies, ensuring environment consistency.
- **Ready for Research:** Includes evaluation tools to measure steering effectiveness with standard prompt suites.

## Setup
1. Clone the repository: `git clone --recursive <your-repo-url>`
2. Initialize environment: `./setup.sh`
3. Launch server: `BACKEND=v54 ./serve.sh`

## Steering
Generate vectors with `steering/generate_vector.py` and apply them via environment variables:
```bash
export STEERING_VECTOR_PATH="/workspace/steering/your_vector.pt"
export STEERING_STRENGTH="1.0"
BACKEND=v54 ./serve.sh
```
