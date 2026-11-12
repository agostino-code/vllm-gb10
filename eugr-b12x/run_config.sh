#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eugr-b12x/run_config.sh
#
# One benchmark config end-to-end inside an authorized GPU window:
#   boot (launch.sh --replace --wait)  ->  validate.sh  ->  bench/suite.py quick
#   ->  stop (launch.sh --stop)
#
# Run one config at a time; only one model server may own the GPU.
#
# Usage:
#   CFG_ID=eugr-base MTP=4 KV_DTYPE=fp8 GPU_UTIL=0.75 ./run_config.sh
#   KEEP_UP=1 ... ./run_config.sh        # leave the server running afterwards
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

CFG_ID="${CFG_ID:?set CFG_ID, e.g. eugr-base}"
MTP="${MTP:-4}"
KV_DTYPE="${KV_DTYPE:-fp8}"
GPU_UTIL="${GPU_UTIL:-0.75}"
SEQS="${SEQS:-8}"
BLOCK="${BLOCK:-16}"
MAXLEN="${MAXLEN:-262144}"
KEEP_UP="${KEEP_UP:-0}"
PROFILE="${PROFILE:-quick}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_ID="${RUN_ID:-eugr-b12x-$CFG_ID}"
SUITE_DIR="${SUITE_DIR:-$PROJECT/bench/results/$RUN_ID}"

export MTP KV_DTYPE GPU_UTIL SEQS BLOCK MAXLEN AUTOTUNE

echo "=================== $CFG_ID ==================="
echo "MTP=$MTP KV_DTYPE=$KV_DTYPE GPU_UTIL=$GPU_UTIL SEQS=$SEQS BLOCK=$BLOCK MAXLEN=$MAXLEN AUTOTUNE=$AUTOTUNE"

# --- guard: GPU must be ours (no other model server) -------------------------
if docker ps --format '{{.Names}}' | grep -qx 'qwen38-flash-next'; then
  echo "error: myllmbox container qwen38-flash-next is still running; refusing to launch" >&2
  exit 1
fi
if docker ps --format '{{.Names}}' | grep -qx 'qwen38-flash-next-b12x' && [[ "${REPLACE:-1}" != "1" ]]; then
  echo "error: $CFG_ID target container already running; use REPLACE=1" >&2
  exit 1
fi

T_START=$(date +%s)

# --- boot -------------------------------------------------------------------
"$HERE/launch.sh" --replace --wait
T_HEALTH=$(date +%s)
BOOT_LOG="$(ls -1t "$HERE"/runs/*-boot.log 2>/dev/null | head -1 || true)"
META="$(ls -1t "$HERE"/runs/*-meta.txt 2>/dev/null | head -1 || true)"

# --- explicit API evidence (health/models/one completion/metrics) ------------
"$HERE/validate.sh"

# --- standardized bench suite (bench-harness harness) -----------------------
if [[ -f "$PROJECT/bench/serve.py" ]]; then
  echo "=== bench/serve.py adopt (records launch manifest; no lifecycle touch) ==="
  python3 "$PROJECT/bench/serve.py" adopt --name qwen38-flash-next-b12x --port 8000 \
    --run-id "$RUN_ID" --config-id "$RUN_ID" \
    || echo "warning: adopt failed (suite can still run; manifest/bootstrap fields missing)" >&2
fi
if [[ -f "$PROJECT/bench/suite.py" ]]; then
  echo "=== bench/suite.py --profile $PROFILE --concurrency 1,2,4,8 --config-id $RUN_ID ==="
  SUITE_ARGS=(--base-url "$BASE_URL" --profile "$PROFILE" --concurrency 1,2,4,8
              --config-id "$RUN_ID" --run-dir "$SUITE_DIR" --tag "$RUN_ID" --quality off)
  [[ -n "$BOOT_LOG" ]] && SUITE_ARGS+=(--server-log "$BOOT_LOG")
  python3 "$PROJECT/bench/suite.py" "${SUITE_ARGS[@]}" || \
    echo "warning: suite rc=$? (partial results may still be written to $SUITE_DIR)" >&2
else
  echo "bench/suite.py not found; skipping suite (validate.sh evidence only)"
fi

T_END=$(date +%s)
{
  echo "config_id=$CFG_ID"
  echo "mtp=$MTP"
  echo "autotune=$AUTOTUNE"
  echo "kv_dtype=$KV_DTYPE"
  echo "gpu_util=$GPU_UTIL"
  echo "max_num_seqs=$SEQS"
  echo "block_size=$BLOCK"
  echo "boot_and_health_seconds=$((T_HEALTH-T_START))"
  echo "total_seconds=$((T_END-T_START))"
  echo "boot_log=$BOOT_LOG"
  echo "meta=$META"
  echo "run_id=$RUN_ID"
  echo "suite_dir=$SUITE_DIR"
} | tee "$HERE/runs/$CFG_ID-summary.txt"

# --- keep or stop ------------------------------------------------------------
if [[ "$KEEP_UP" == "1" ]]; then
  echo "KEEP_UP=1: server left running (container qwen38-flash-next-b12x)"
else
  "$HERE/launch.sh" --stop
fi
echo "=================== $CFG_ID done ==================="
