#!/usr/bin/env bash
# Finalist phase: complete the depth curve (self K=5, K=6), pick the winner among K=4/5/6
# by measured single-stream decode, then validate the winner end-to-end:
# full suite (C=16 + 128k needle) + full quality battery + temperature-0.6 acceptance check
# + 7-minute C=8 soak.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=http://127.0.0.1:8000
R=bench/results
mkdir -p "$R/finalist-curve-k5" "$R/finalist-curve-k6" \
         "$R/finalist-winner-full" "$R/finalist-winner-temp06" "$R/finalist-winner-soak"

echo "=== depth curve point: self K=5 ==="
python3 bench/serve.py up --recipe draft-lab/recipes/curve-k5.yaml --run-id finalist-curve-k5 --boot-timeout 1500
python3 bench/suite.py --base-url "$BASE" --run-dir "$R/finalist-curve-k5" --config-id finalist-curve-k5 \
  --tag finalist-curve --profile quick --concurrency 1,2,4,8 --request-timeout 300 --retries 1
python3 bench/serve.py down --name bench-curve-k5 || true

echo "=== depth curve point: self K=6 ==="
python3 bench/serve.py up --recipe draft-lab/recipes/curve-k6.yaml --run-id finalist-curve-k6 --boot-timeout 1500
python3 bench/suite.py --base-url "$BASE" --run-dir "$R/finalist-curve-k6" --config-id finalist-curve-k6 \
  --tag finalist-curve --profile quick --concurrency 1,2,4,8 --request-timeout 300 --retries 1
python3 bench/serve.py down --name bench-curve-k6 || true

echo "=== pick winner among K=4/5/6 ==="
WINNER=$(python3 - <<'PY'
import json, glob, os
cands = []
for name, recipe in (("d-curve-k4", "draft-lab/recipes/curve-k4.yaml"),
                     ("finalist-curve-k5", "draft-lab/recipes/curve-k5.yaml"),
                     ("finalist-curve-k6", "draft-lab/recipes/curve-k6.yaml")):
    p = f"bench/results/{name}/summary.json"
    try:
        d = json.load(open(p))
        dec = d["phases"]["decode"]["out_tok_s"]["median"]
        c8 = next((l["total_out_tok_s"] for l in d["phases"]["concurrency"]["levels"] if l["c"] == 8), 0)
        cands.append((name, recipe, dec, c8))
    except FileNotFoundError:
        pass
for c in sorted(cands, key=lambda x: -x[2]):
    print(f"  candidate {c[0]}: decode={c[2]:.2f} C8={c[3]:.1f}", flush=True)
best = max(cands, key=lambda x: x[2])
print(best[1])
PY
)
WINNER=$(echo "$WINNER" | tail -1)
echo "WINNER RECIPE: $WINNER"
echo "$WINNER" > "$R/finalist-winner.recipe"

echo "=== full validation: boot winner ($WINNER) ==="
python3 bench/serve.py up --recipe "$WINNER" --run-id finalist-winner --boot-timeout 1500

echo "=== full suite + full quality ==="
python3 bench/suite.py --base-url "$BASE" --run-dir "$R/finalist-winner-full" \
  --config-id finalist-winner-full --tag finalist-winner --profile full \
  --include-16 --long-smoke --quality full --quality-timeout 1500 \
  --concurrency 1,2,4,8 --request-timeout 300 --retries 1

echo "=== temperature 0.6 quick suite ==="
python3 bench/suite.py --base-url "$BASE" --run-dir "$R/finalist-winner-temp06" \
  --config-id finalist-winner-temp06 --tag finalist-winner-temp06 --profile quick \
  --temperature 0.6 --concurrency 1,2,4,8 --request-timeout 300 --retries 1

echo "=== 7-minute C=8 soak ==="
python3 matrix/soak.py --base "$BASE" --seconds 420 --concurrency 8 \
  --out "$R/finalist-winner-soak/soak.json"

echo "=== stop ==="
NAME=$(python3 -c "import sys,pathlib; print('bench-'+pathlib.Path('$WINNER').stem)")
python3 bench/serve.py down --name "$NAME" || true
echo "FINALIST PHASE DONE — winner: $WINNER"
