#!/usr/bin/env bash
# run.sh — Run Qwen3.8-Flash-Next on ONE DGX Spark / GB10 with vLLM 0.32 (v5.4 stack)
#
# Configured via YAML (default: qwen38-flash-solo-optimized.yaml)
# Usage:
#   ./run.sh                  # runs qwen38-flash-solo-optimized.yaml
#   ./run.sh qwen38-flash-cluster-optimized.yaml      # runs qwen38-flash-cluster-optimized.yaml
set -euo pipefail
export DOWNLOAD_MODE="${DOWNLOAD_MODE:-cache}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MODELS_DIR="./models"
cd "$(dirname "$0")"; source ./config.env

# Dynamic KV Cache Calculation
KV_BYTES=$(get_kv_cache_bytes)
echo "KV Cache dinamica calcolata: $KV_BYTES bytes"

# Parsing Steering (if provided via ENV or flag)
if [ -n "${STEERING_KEY:-}" ]; then
    STEERING_DATA=$(python3 -c "import json; d=json.load(open('steering/index.json')); print(f'{d[\"$STEERING_KEY\"][\"path\"]}|{d[\"$STEERING_KEY\"][\"strength\"]}')" 2>/dev/null || echo "")
    if [ -n "$STEERING_DATA" ]; then
        export STEERING_VECTOR_PATH=$(echo $STEERING_DATA | cut -d'|' -f1)
        export STEERING_STRENGTH=$(echo $STEERING_DATA | cut -d'|' -f2)
        echo "Steering attivo: $STEERING_KEY (path: $STEERING_VECTOR_PATH, strength: $STEERING_STRENGTH)"
    else
        echo "Errore: Steering key '$STEERING_KEY' non trovata in steering/index.json"
        exit 1
    fi
fi

DOWNLOAD_MODE="${DOWNLOAD_MODE:-cache}"
export DOWNLOAD_MODE
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

set -euo pipefail
cd "$(dirname "$0")"; source ./config.env

command -v docker >/dev/null || { echo "docker is required"; exit 1; }
RECIPE="${1:-$DEFAULT_RECIPE}"
[ -f "$RECIPE" ] || { echo "✗ recipe file '$RECIPE' not found"; exit 1; }

# Helper to read two-level YAML sections
rkey() {  # rkey <section> <key>
  awk -v s="$1" -v k="$2" '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec) }
    sec==s && $1==k":" {
      sub(/^[ ]*[^:]*:[ ]*/,""); sub(/[ ]+#.*$/,"")
      gsub(/^["\x27]|["\x27]$/,""); print; exit
    }' "$RECIPE"
}
rsection() {  # all key/value lines of a section, "key<TAB>value"
  awk -v s="$1" '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec); next }
    sec==s && $1 ~ /^[A-Za-z0-9_-]+:$/ || (sec==s && /^[ ]+[A-Za-z0-9_-]+:[ ]/) {
      line=$0; sub(/^[ ]+/,"",line)
      key=line; sub(/:.*/,"",key)
      val=line; sub(/^[^:]*:[ ]*/,"",val); sub(/[ ]+#.*$/,"",val)
      gsub(/^["\x27]|["\x27]$/,"",val)
      if (key != "") print key "\t" val
    }' "$RECIPE"
}

IMAGE="${IMAGE:-$(rkey server image)}"
# Force override if recipe image is the default one that doesn't exist
if [ "$IMAGE" = "qwen38-flash-dgx:a5b-int4" ]; then
    IMAGE="myllmbox/qwen38-flash-next-vllm:v5.4"
fi
PORT="${PORT:-$(rkey server port)}"
HF_REPO="${MODEL_REPO:-$(rkey server model)}"
MODELS_DIR="${MODELS_DIR:-$(rkey server models_dir)}"
CACHE_DIR="${CACHE_DIR:-$(rkey server cache_dir)}"
CPUSET="$(rkey server cpuset)"
DOCKER_ARGS="$(rkey server docker_args)"
NAME="qwen38-flash-next"

: ${IMAGE:=myllmbox/qwen38-flash-next-vllm:v5.4}
: ${PORT:=8000}
: ${MODELS_DIR:=./models}
: ${CACHE_DIR:=./cache}

mkdir -p "$MODELS_DIR" "$CACHE_DIR"
MODELS_ABS="$(cd "$MODELS_DIR" && pwd)"
CACHE_ABS="$(cd "$CACHE_DIR" && pwd)"
LOCAL_NAME="$(basename "$HF_REPO")"
MODEL_DIR="$MODELS_ABS/$LOCAL_NAME"
if [ "$DOWNLOAD_MODE" = "cache" ]; then
    CACHE_MODEL_DIR="$(hf_snapshot_dir "$HF_REPO" 2>/dev/null || true)"
    if [ -n "$CACHE_MODEL_DIR" ] && [ -f "$CACHE_MODEL_DIR/model.safetensors.index.json" ]; then
        MODEL_DIR="$CACHE_MODEL_DIR"
    fi
fi

# Check weights
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ] || ( [ "$DOWNLOAD_MODE" != "cache" ] && [ -n "$(find "$MODEL_DIR/.cache" -name '*.incomplete' -print -quit 2>/dev/null)" ] ); then
  echo "· Weights for $HF_REPO not complete in $MODEL_DIR"
  echo "  Downloading..."
  if command -v hf >/dev/null; then
    hf download "$HF_REPO" --local-dir "$MODEL_DIR" --exclude "fast-fp8/*" --max-workers 8
  else
    docker run --rm -v "$MODELS_ABS:/dl" --entrypoint python3 "$IMAGE" \
      -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_REPO', local_dir='/dl/$LOCAL_NAME', ignore_patterns=['fast-fp8/*'])"
  fi
fi

# Assemble docker env & vLLM flags
ENVS=()
while IFS=$'\t' read -r k v; do [ -n "$k" ] && ENVS+=(-e "$k=$v"); done < <(rsection env)

# Cache mounts
MOUNTS=(-v "$MODELS_ABS:/models" -v "$CACHE_ABS:/cache")
if [ "$DOWNLOAD_MODE" = "cache" ]; then
    MOUNTS+=(-v "$HF_HOME/hub":"$HF_HOME/hub":ro)
fi
MOUNTS+=(-v "$(readlink -f "$MODEL_DIR"):/models/$LOCAL_NAME")

# Support INT4-AutoRound if selected
if [ -d "$MODEL_DIR/ple-table" ]; then
  ENVS+=(-e "MBX_PLE_FP8_DIR=/models/$LOCAL_NAME/ple-table" -e "MBX_PLE_NVME_DIR=/cache/ple-nvme-fp8")
  python3 - "$MODEL_DIR" "/models/$LOCAL_NAME" "$CACHE_ABS/draft-k10" <<'PY' || { echo "✗ could not build draft-k10 folder"; exit 1; }
import json, os, shutil, struct, sys
snap, csnap, out = sys.argv[1:4]
wm = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
keep = {k: v for k, v in wm.items() if k.startswith(("mtp.", "lm_head.")) or k.endswith("embed_tokens.weight")}
if os.path.lexists(out):
    shutil.rmtree(out)
os.makedirs(out)
for f in sorted(os.listdir(snap)):
    if f in ("config.json", "model.safetensors.index.json", "ple-table", "fast-fp8", ".cache") or \
       (f.endswith(".safetensors") and f not in set(keep.values())):
        continue
    os.symlink(os.path.join(csnap, f), os.path.join(out, f))
json.dump({"metadata": {}, "weight_map": keep}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
cfg = json.load(open(os.path.join(snap, "config.json"))); t = cfg.get("text_config", cfg)
t["num_experts_per_tok"] = 10
with open(os.path.join(snap, "model_extra_tensors.safetensors"), "rb") as fh:
    h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
w = {v["shape"][0] for k, v in h.items() if k.endswith("mlp.shared_expert.gate_proj.weight")}
if len(w) == 1:
    t["shared_expert_intermediate_size"] = w.pop()
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
PY
  echo "· INT4-AutoRound detected: fp8 table configured"
fi

FLAGS=()
while IFS=$'\t' read -r k v; do
  case "$v" in
    true)        FLAGS+=("--$k");;
    false|null|"") ;;
    *)           FLAGS+=("--$k" "$v");;
  esac
done < <(rsection vllm)

# Host compaction check
cp_now=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
if [ "$cp_now" != "0" ]; then
  echo "  ⚠ vm.compaction_proactiveness is $cp_now (recommended: 0). Run ./tune-host.sh to prevent memory freezes."
fi

# Chat template handling
TPL=(); TPL_MOUNT=(); TPL_NOTE=""
[ "$CHAT_TEMPLATE" = model ] && CHAT_TEMPLATE=""
if [ -n "$CHAT_TEMPLATE" ]; then
  TPL_FILE="$CHAT_TEMPLATE"
  if [ "${CHAT_TEMPLATE#*/}" = "$CHAT_TEMPLATE" ] && [ ! -f "$CHAT_TEMPLATE" ]; then
    for cand in "$MODEL_DIR/${CHAT_TEMPLATE}_chat_template.jinja" "$MODEL_DIR/$CHAT_TEMPLATE"; do
      [ -f "$cand" ] && { TPL_FILE="$cand"; break; }
    done
  fi
  if [ -f "$TPL_FILE" ]; then
    TPL_MOUNT=(-v "$TPL_FILE:/chat_template.jinja:ro"); TPL=(--chat-template /chat_template.jinja)
    TPL_NOTE=", template $(basename "$TPL_FILE")"
  else
    echo "  ⚠ CHAT_TEMPLATE '$CHAT_TEMPLATE' not found, skipping template"
  fi
fi

# MTP / Speculative decoding
SPEC=(); DRAFT_NOTE=""
if [ "${MTP:-0}" != 0 ]; then
  # Use self-draft (same model as target) to avoid quantization issues with MTP layers
  # This works like the original MTP=12 config
  RS=""; [ "${REJECTION_SAMPLE:-standard}" != standard ] && RS=",\"rejection_sample_method\":\"$REJECTION_SAMPLE\""
  [ "${DRAFT_SAMPLE:-greedy}" != greedy ] && RS="$RS,\"draft_sample_method\":\"$DRAFT_SAMPLE\""
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}$RS}")
  DRAFT_NOTE=" (MTP=${MTP}, self-draft)"
fi

docker rm -f "$NAME" "$NAME-proxy" >/dev/null 2>&1 || true

# Memory check
avail=$(free -g | awk '/^Mem:/{print $7}')
echo "  · memory available: ${avail}G"

# Evict page cache for safetensors before launch to prevent false OOM
echo "  · evicting safetensors from page cache..."
find "$MODEL_DIR" -type f -name "*.safetensors" -exec dd if={} iflag=nocache count=0 status=none \; 2>/dev/null || true

echo "· starting $NAME ($IMAGE) on port :$PORT with recipe $RECIPE"
docker run -d --name "$NAME" --gpus all --ipc=host \
  ${CPUSET:+--cpuset-cpus "$CPUSET"} \
  ${DOCKER_EXTRA_ARGS} \
  -p "$PORT:8000" \
  "${MOUNTS[@]}" "${TPL_MOUNT[@]}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace \
  -e VLLM_CACHE_ROOT=/cache/vllm-cache \
  "${ENVS[@]}" \
  --entrypoint vllm "$IMAGE" serve "/models/$LOCAL_NAME" --port 8000 "${FLAGS[@]}" "${TPL[@]}" "${SPEC[@]}" $EXTRA_ARGS >/dev/null

echo "· streaming engine logs until healthy (Ctrl-C detaches; the server continues booting)"
docker logs -f "$NAME" 2>&1 &
LOGS=$!
trap 'kill "$LOGS" 2>/dev/null || true' EXIT INT TERM
for i in $(seq 1 480); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    kill "$LOGS" 2>/dev/null || true; wait "$LOGS" 2>/dev/null || true
    echo
    echo "──────────────────────────────────────────────────────────"
    echo "✓ server booted — OpenAI-compatible API is live"
    echo "    endpoint : http://127.0.0.1:$PORT/v1"
    echo "    monitor  : ./view.sh"
    echo "    stop     : ./stop.sh"
    echo "──────────────────────────────────────────────────────────"
    exit 0
  fi
  if ! docker ps -q --filter "name=$NAME" | grep -q .; then
    kill "$LOGS" 2>/dev/null || true; wait "$LOGS" 2>/dev/null || true
    echo "✗ container exited — see logs above"
    exit 1
  fi
  sleep 5
done
echo "✗ not healthy after 40 min — watch: docker logs -f $NAME"
exit 1
