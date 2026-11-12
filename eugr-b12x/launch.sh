#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eugr-b12x/launch.sh
#
# Self-contained launcher for azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ on the
# eugr B12X stack (image: eugr/spark-vllm-b12x:latest) on one DGX Spark / GB10.
#
# Plain `docker run` (no eugr tooling) so the exact argv is reproducible and can
# be diffed/copied. Model weights are bind-mounted READ-ONLY from the local HF
# cache snapshot: nothing is downloaded and nothing is written to the HF cache.
#
# Knobs are environment variables (all overridable):
#   MTP=4              speculative tokens, fixed draft depth (0 disables MTP)
#   KV_DTYPE=fp8       fp8 | auto   (auto == bf16 KV for this bf16 model)
#   GPU_UTIL=0.75      --gpu-memory-utilization
#   ALLOC_FRAC=GPU_UTIL  PYTORCH_ALLOC_CONF=per_process_memory_fraction:<frac>
#   SEQS=8             --max-num-seqs
#   BLOCK=16           --block-size
#   MAXLEN=262144      --max-model-len
#   BATCHED=4096       --max-num-batched-tokens
#   PORT=8000          host/container port (--network host)
#   NAME=qwen38-flash-next-b12x
#   IMAGE=eugr/spark-vllm-b12x:latest
#   MODEL_SNAP=<path>  override the auto-detected HF snapshot directory
#   EXTRA_ARGS=""      extra vLLM args appended verbatim
#   WAIT=1             poll /health, save boot log + docker inspect under runs/
#   BOOT_TIMEOUT=2400  seconds to wait for /health when WAIT=1
#   DRY_RUN=1          print the command, change nothing
#
# Usage:
#   ./launch.sh --dry-run            # show the exact docker run argv
#   ./launch.sh --wait               # start and wait until /health (validation)
#   ./launch.sh                      # start detached
#   ./launch.sh --stop               # remove our container only
#   ./launch.sh --replace --wait     # replace a previous container of ours
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

IMAGE="${IMAGE:-eugr/spark-vllm-b12x:latest}"
NAME="${NAME:-qwen38-flash-next-b12x}"
PORT="${PORT:-8000}"
MTP="${MTP:-4}"
KV_DTYPE="${KV_DTYPE:-fp8}"
GPU_UTIL="${GPU_UTIL:-0.75}"
ALLOC_FRAC="${ALLOC_FRAC:-$GPU_UTIL}"
SEQS="${SEQS:-8}"
BLOCK="${BLOCK:-16}"
MAXLEN="${MAXLEN:-262144}"
BATCHED="${BATCHED:-4096}"
SERVED_NAME="${SERVED_NAME:-azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ}"
REPO="${REPO:-azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ}"
MODEL_LOCAL_NAME="${MODEL_LOCAL_NAME:-Qwen3.8-Flash-Next-FP4-Local-AZ}"
CACHE_DIR="${CACHE_DIR:-$HERE/cache}"
RUNS_DIR="${RUNS_DIR:-$HERE/runs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
WAIT="${WAIT:-0}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-2400}"
DRY_RUN="${DRY_RUN:-0}"
REPLACE="${REPLACE:-0}"
AUTOTUNE="${AUTOTUNE:-1}"
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

ACTION="start"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --wait)    WAIT=1 ;;
    --no-wait) WAIT=0 ;;
    --stop)    ACTION="stop" ;;
    --replace) REPLACE=1 ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg (see --help)" >&2; exit 2 ;;
  esac
done

if [[ "$ACTION" == "stop" ]]; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker rm -f "$NAME"
    echo "stopped/removed container: $NAME"
  else
    echo "container '$NAME' not running"
  fi
  exit 0
fi

# Never let this launcher touch the myllmbox container.
if [[ "$NAME" == "qwen38-flash-next" || "$NAME" == *myllmbox* ]]; then
  echo "refusing to manage container named '$NAME' (myllmbox-owned)" >&2
  exit 1
fi

# --- Locate the model snapshot in the local HF cache (read-only source) -------
find_snapshot() {
  local hub="$HF_HOME_DIR/hub/models--${REPO//\//--}/snapshots"
  [[ -d "$hub" ]] || return 1
  # Prefer snapshots that look complete; newest mtime first.
  local d best=""
  while IFS= read -r d; do
    [[ -f "$d/config.json" && -f "$d/model.safetensors.index.json" ]] || continue
    best="$d"
    break
  done < <(ls -1dt "$hub"/*/ 2>/dev/null)
  [[ -n "$best" ]] && printf '%s\n' "${best%/}"
}

if [[ -n "${MODEL_SNAP:-}" ]]; then
  SNAP="$MODEL_SNAP"
else
  SNAP="$(find_snapshot || true)"
fi
if [[ -z "$SNAP" || ! -f "$SNAP/config.json" ]]; then
  echo "error: could not locate the $REPO snapshot under $HF_HOME_DIR/hub" >&2
  echo "       set MODEL_SNAP=/path/to/snapshot to override" >&2
  exit 1
fi
[[ -f "$SNAP/model.safetensors.index.json" ]] || {
  echo "error: $SNAP/model.safetensors.index.json is missing" >&2; exit 1; }
[[ -d "$SNAP/tensors" ]] || {
  echo "error: $SNAP/tensors/ (21 shards) is missing" >&2; exit 1; }
if find "$SNAP/tensors" -maxdepth 1 -name '*.safetensors' -size 0 | grep -q .; then
  echo "error: found zero-byte shards under $SNAP/tensors" >&2; exit 1
fi

# --- Workspace-local writable caches (nothing outside the workspace) ----------
mkdir -p "$CACHE_DIR"/{b12x/compile,vllm,flashinfer,flashinfer-workspace,triton,tilelang,torchinductor}
mkdir -p "$RUNS_DIR"

# --- Build argv ---------------------------------------------------------------
DOCKER_ARGS=(
  --gpus all
  --ipc=host
  --privileged
  --network host
  -d
  --name "$NAME"
  --entrypoint vllm
  -v "$SNAP:/models/$MODEL_LOCAL_NAME:ro"
  # Writable caches live inside the workspace (mirrors eugr launch-cluster.sh
  # mount points, but sourced from eugr-b12x/cache/ instead of ~/.cache).
  -v "$CACHE_DIR/vllm:/root/.cache/vllm"
  -v "$CACHE_DIR/flashinfer:/root/.cache/flashinfer"
  -v "$CACHE_DIR/b12x:/root/.cache/b12x"
  -v "$CACHE_DIR/triton:/root/.triton"
  -v "$CACHE_DIR/tilelang:/root/.tilelang"
  -v "$CACHE_DIR/torchinductor:/root/.cache/torchinductor"
  -v "$CACHE_DIR/flashinfer-workspace:/cache/flashinfer-workspace"
)

ENV_ARGS=(
  -e "HF_HUB_OFFLINE=1"
  -e "TRANSFORMERS_OFFLINE=1"
  -e "FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace"
  # PLE n-gram table: disk-backed io_uring row reads straight from the shards.
  -e "VLLM_PLE_TABLE_MEMORY=disk"
  -e "B12X_DISK_BACKEND=io_uring"
  # b12x / model knobs (parity with eugr's qwen3.8-flash-next-nvfp4-solo recipe)
  -e "PYTORCH_ALLOC_CONF=per_process_memory_fraction:${ALLOC_FRAC}"
  -e "CUTE_DSL_ARCH=sm_121a"
  -e "SAFETENSORS_FAST_GPU=1"
  -e "VLLM_WORKER_MULTIPROC_METHOD=spawn"
  -e "VLLM_SSM_CONV_STATE_LAYOUT=DS"
  -e "VLLM_USE_AOT_COMPILE=1"
  -e "VLLM_USE_MEGA_AOT_ARTIFACT=1"
  -e "VLLM_USE_V2_MODEL_RUNNER=1"
  -e "B12X_POLICY_MODE=auto"
  -e "VLLM_MXFP8_LM_HEAD=1"
  -e "VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13"
  -e "VLLM_B12X_MOE_FP4_FORCE_A16=1"
  -e "B12X_W4A16_A4_PREFILL=1"
)

VLLM_ARGS=(
  serve "/models/$MODEL_LOCAL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --served-model-name "$SERVED_NAME"
  --tensor-parallel-size 1
  --pipeline-parallel-size 1
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype "$KV_DTYPE"
  --quantization modelopt_mixed
  --block-size "$BLOCK"
  --load-format b12x
  --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS"
  --max-num-batched-tokens "$BATCHED"
  --gdn-decode-kernel b12x
  --linear-backend b12x
  --moe-backend b12x
  --no-enable-flashinfer-autotune
  --mm-encoder-tp-mode data
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --compilation-config '{"pass_config":{"fuse_act_quant":true}}'
  --gpu-memory-utilization "$GPU_UTIL"
)
# b12x kernel autotuning: 241-unit search on first boot (~60 min on GB10).
# AUTOTUNE=0 disables the search via KernelConfig but keeps mandatory
# kernel preparation, so boots are minutes, not an hour. Same setting must be
# used for every config in a comparison (it slightly changes kernel choices).
if [[ "$AUTOTUNE" == "0" ]]; then
  VLLM_ARGS+=(--kernel-config '{"enable_b12x_autotune": false}')
fi

# Fixed-depth MTP only: this image has no MBX_DEPTH_* / dynamic draft depth.
if [[ "$MTP" != "0" ]]; then
  VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  VLLM_ARGS+=($EXTRA_ARGS)
fi

CMD=("${DOCKER_ARGS[@]}" "${ENV_ARGS[@]}" "$IMAGE" "${VLLM_ARGS[@]}")

echo "image       : $IMAGE"
echo "model snap  : $SNAP  (mounted :ro)"
echo "container   : $NAME   network=host port=$PORT"
echo "config      : MTP=$MTP KV_DTYPE=$KV_DTYPE GPU_UTIL=$GPU_UTIL ALLOC_FRAC=$ALLOC_FRAC SEQS=$SEQS BLOCK=$BLOCK MAXLEN=$MAXLEN BATCHED=$BATCHED AUTOTUNE=$AUTOTUNE"
echo "cache       : $CACHE_DIR -> /cache"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "=== DRY RUN — docker argv ==="
  printf '%q ' "${CMD[@]}"
  echo
  echo "=== end (nothing executed) ==="
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  if [[ "$REPLACE" == "1" ]]; then
    docker rm -f "$NAME" >/dev/null
  else
    echo "container '$NAME' already exists; rerun with --replace or --stop" >&2
    exit 1
  fi
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
printf '%q ' "${CMD[@]}" > "$RUNS_DIR/${STAMP}-cmd.txt"
echo

echo "starting $NAME ..."
T0=$(date +%s)
docker run "${CMD[@]}" >/dev/null
echo "container started (id: $(docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null | cut -c1-12))"

if [[ "$WAIT" != "1" ]]; then
  echo "detached. logs: docker logs -f $NAME   stop: $0 --stop"
  exit 0
fi

echo "waiting for http://127.0.0.1:$PORT/health (timeout ${BOOT_TIMEOUT}s) ..."
HEALTH=0
for _ in $(seq 1 "$BOOT_TIMEOUT"); do
  if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTH=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited before becoming healthy" >&2
    docker logs "$NAME" > "$RUNS_DIR/${STAMP}-boot.log" 2>&1 || true
    docker inspect "$NAME" > "$RUNS_DIR/${STAMP}-docker-inspect.json" 2>/dev/null || true
    echo "boot log saved: $RUNS_DIR/${STAMP}-boot.log" >&2
    tail -40 "$RUNS_DIR/${STAMP}-boot.log" >&2 || true
    exit 1
  fi
  sleep 1
done

T1=$(date +%s)
docker inspect "$NAME" > "$RUNS_DIR/${STAMP}-docker-inspect.json" 2>/dev/null || true
docker logs "$NAME" > "$RUNS_DIR/${STAMP}-boot.log" 2>&1 || true
{
  echo "stamp=$STAMP"
  echo "healthy=$HEALTH"
  echo "boot_wall_seconds=$((T1-T0))"
  echo "image=$IMAGE"
  echo "name=$NAME port=$PORT"
  echo "model_snap=$SNAP"
  echo "mtp=$MTP kv_dtype=$KV_DTYPE gpu_util=$GPU_UTIL alloc_frac=$ALLOC_FRAC seqs=$SEQS block=$BLOCK maxlen=$MAXLEN batched=$BATCHED autotune=$AUTOTUNE"
} > "$RUNS_DIR/${STAMP}-meta.txt"

if [[ "$HEALTH" == "1" ]]; then
  echo "healthy after $((T1-T0))s"
  echo "artifacts: $RUNS_DIR/${STAMP}-{boot.log,docker-inspect.json,cmd.txt,meta.txt}"
else
  echo "NOT healthy after ${BOOT_TIMEOUT}s — see  docker logs $NAME" >&2
  exit 1
fi
