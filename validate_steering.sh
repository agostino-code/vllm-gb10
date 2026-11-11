#!/bin/bash
# validate_steering.sh
# Esegue la suite di test completa e calcola il punteggio di conformità

echo "--- Steering Compliance Pipeline ---"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/$TIMESTAMP"
mkdir -p "$OUT_DIR"

# 1. Test di Qualità (Baseline)
echo "[1/3] Running Quality Baseline (no steering)..."
./steering/eval_steering.py --prompts examples/benchmarks/quality_bench.jsonl --out "$OUT_DIR/baseline.json"

# 2. Test di Steering (se attivo)
if [ -n "${STEERING_VECTOR_PATH:-}" ]; then
    echo "[2/3] Running Steering Evaluation (Succinctness check)..."
    ./steering/eval_steering.py --prompts examples/steering_pairs/succinct.txt --out "$OUT_DIR/steering_succinct.json"
    
    echo "[3/3] Running Red-Teaming Evaluation (Refusal check)..."
    ./steering/eval_steering.py --prompts examples/red_teaming/refusal_eval.txt --out "$OUT_DIR/steering_refusal.json"
else
    echo "Skipping steering tests (STEERING_VECTOR_PATH not set)."
fi

echo "Pipeline completata. Risultati salvati in: $OUT_DIR"
