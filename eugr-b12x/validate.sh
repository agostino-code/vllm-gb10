#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eugr-b12x/validate.sh
#
# Post-boot validation for the eugr b12x server started by ./launch.sh.
# Read-only: it only issues HTTP calls (health, models, one chat completion,
# metrics) and writes evidence under eugr-b12x/runs/.
#
# Usage: ./validate.sh            # against 127.0.0.1:8000
#        PORT=8001 ./validate.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${RUNS_DIR:-$HERE/runs}"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:$PORT"
MODEL="${SERVED_NAME:-azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ}"
MAX_TOKENS="${MAX_TOKENS:-192}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUNS_DIR"

say() { printf '%s\n' "$*"; }

say "=== health ==="
curl -sf -m 10 "$BASE/health" -o "$RUNS_DIR/${STAMP}-health.txt" && echo "health: HTTP 200 (body $(stat -c %s "$RUNS_DIR/${STAMP}-health.txt") bytes)"

say "=== /v1/models ==="
curl -sf -m 10 "$BASE/v1/models" | tee "$RUNS_DIR/${STAMP}-models.json" | python3 -c '
import json,sys
d=json.load(sys.stdin)
for m in d.get("data", []):
    print("served:", m.get("id"), "| max_model_len:", m.get("max_model_len"))
'

say "=== boot log head (container start -> engine args) ==="
# NOTE: avoid `docker logs | head` (SIGPIPE + pipefail aborts the script).
docker logs qwen38-flash-next-b12x > "$RUNS_DIR/${STAMP}-server.log" 2>&1 || true
head -40 "$RUNS_DIR/${STAMP}-server.log" > "$RUNS_DIR/${STAMP}-boot-head.txt" || true
grep -E 'vLLM version|modelopt|load format|b12x|PLE|eager|kv-cache|gpu_memory|speculative|MTP|graph' \
  "$RUNS_DIR/${STAMP}-boot-head.txt" | head -25 || true

say "=== one chat completion (max_tokens=$MAX_TOKENS) ==="
PROMPT='Explain in three short sentences why a 100B-parameter model with NVFP4 weights fits on a single 128 GB DGX Spark GB10, and name the two memory costs besides the weights.'
REQ="$(mktemp)"
python3 - "$MODEL" "$MAX_TOKENS" "$PROMPT" > "$REQ" <<'PY'
import json, sys
model, max_tokens, prompt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0,
}))
PY

T0=$(date +%s.%N)
HTTP_CODE=$(curl -s -o "$RUNS_DIR/${STAMP}-completion.json" -w '%{http_code}' \
  -m 300 -H 'Content-Type: application/json' --data @"$REQ" "$BASE/v1/chat/completions")
T1=$(date +%s.%N)
rm -f "$REQ"
ELAPSED=$(python3 -c "print(f'{$T1-$T0:.3f}')")
say "http_code=$HTTP_CODE  wall_seconds=$ELAPSED"

if [[ "$HTTP_CODE" != "200" ]]; then
  say "!! completion failed; body saved to $RUNS_DIR/${STAMP}-completion.json"
else
  python3 - "$RUNS_DIR/${STAMP}-completion.json" "$ELAPSED" <<'PY'
import json, sys
path, elapsed = sys.argv[1], float(sys.argv[2])
d = json.load(open(path))
usage = d.get("usage", {})
msg = d["choices"][0]["message"]
text = msg.get("content") or msg.get("reasoning_content") or ""
completion_tokens = usage.get("completion_tokens", 0)
prompt_tokens = usage.get("prompt_tokens", 0)
print(f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}")
print(f"output_tok_per_s={completion_tokens/elapsed:.2f} (incl. prefill, single request)")
print(f"total_tok_per_s={(prompt_tokens+completion_tokens)/elapsed:.2f}")
print("--- completion text ---")
print(text[:800])
PY
fi

say "=== /metrics snapshot ==="
curl -sf -m 20 "$BASE/metrics" -o "$RUNS_DIR/${STAMP}-metrics.txt" || true
say "saved: $RUNS_DIR/${STAMP}-metrics.txt ($(wc -l < "$RUNS_DIR/${STAMP}-metrics.txt" 2>/dev/null || echo 0) lines)"
grep -E '^vllm:(num_requests|gpu_cache_usage|prompt_tokens_total|generation_tokens_total|time_to_first_token|request_success|spec_decode|num_preemptions|e2e_request_latency)' \
  "$RUNS_DIR/${STAMP}-metrics.txt" | grep -v '^#' | head -30 || true

say
say "=== leftovers to include in the report ==="
say "  health      : $RUNS_DIR/${STAMP}-health.txt"
say "  models      : $RUNS_DIR/${STAMP}-models.json"
say "  completion  : $RUNS_DIR/${STAMP}-completion.json"
say "  metrics     : $RUNS_DIR/${STAMP}-metrics.txt"
say "  boot log    : runs/<stamp>-boot.log (from launch.sh --wait)"
say "  inspect     : runs/<stamp>-docker-inspect.json (from launch.sh --wait)"
say "  wall time   : runs/<stamp>-meta.txt (boot_wall_seconds)"
