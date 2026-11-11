# vLLM GB10-Optimized Steering Stack

A high-performance deployment stack for Qwen3.8-Flash models on DGX Spark (GB10) hardware.

## Features
- **Dynamic KV Cache:** Auto-scales based on available GPU VRAM.
- **Modular Runtime Steering:** Apply arbitrary steering vectors at runtime using orthogonal projection (DS4-style).
- **Portable Architecture:** Uses Git submodules for core dependencies, ensuring environment consistency.
- **Production-Ready Toolkit:** Includes benchmarking, tuning, and diagnostic tools.

## Setup
1. Clone: `git clone --recursive <your-repo-url>`
2. Setup Environment: `./setup.sh`
3. Launch: `BACKEND=v54 ./run.sh recipe-az.yaml`

## Tools
- **`tune-host.sh`**: Optimized host tuning for DGX Spark (disable compaction, etc).
- **`bench.sh`**: Multi-stream benchmarking suite (QA, JSON, Code, LongCode).
- **`check.sh`**: Server health and configuration diagnostics.
- **`run.sh`**: Universal recipe runner.
- **`steering/`**: Tools for vector generation and evaluation (`eval_steering.py`).
- **`tools/`**: Helper scripts for speculative decoding and MTP head management.

## Steering
Generate vectors with `steering/generate_vector.py` and apply them via environment variables:
```bash
export STEERING_VECTOR_PATH="/workspace/steering/your_vector.pt"
export STEERING_STRENGTH="1.0"
BACKEND=v54 ./run.sh recipe-az.yaml
```
EOF
