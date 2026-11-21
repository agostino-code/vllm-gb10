#!/usr/bin/env bash
# check.sh — what is actually being served, and does it match the recipe?
#
#   ./check.sh              health, served model, context, experts, one prompt
#   ./check.sh --flags      the engine flags the running container was started with
#   ./check.sh --env        the container environment (MBX_* knobs)
#   ./check.sh --penalty    did the image parse override-generation-config?
#   ./check.sh --spec       run tools/mtp_drafter_check.sh (about 1-2 min of traffic)
#
# Everything here is read from the live server and the live container, never
# from a local variable that may have drifted. That is the point: the old
# check.sh printed "draft: depth 0" from config.env even when the recipe was
# running K=12.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env

NAME="${CONTAINER:-qwen38-flash-next}"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:$PORT"

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "✗ no container named '$NAME' is running (start one with ./presets/run_az_standard.sh)" >&2
  exit 1
fi
curl -sf -m 5 "$BASE/health" >/dev/null || { echo "✗ $BASE/health is not answering (still loading? ./view.sh)" >&2; exit 1; }

IMAGE=$(docker inspect "$NAME" --format '{{.Config.Image}}')
SERVE_PATH=$(docker inspect "$NAME" --format '{{join .Config.Cmd " "}}' | awk '{for(i=1;i<=NF;i++) if($i=="serve") print $(i+1)}')

case "${1:-}" in
  --flags)
    echo "· engine flags ($NAME, $IMAGE)"
    docker inspect "$NAME" --format '{{json .Config.Cmd}}' | python3 -c "
import json,sys
c=json.load(sys.stdin); i=0
while i < len(c):
    a=c[i]
    if a.startswith('--'):
        v=c[i+1] if i+1<len(c) and not c[i+1].startswith('--') else True
        print(f'  {a}: {v if v is not True else \"\"}')
        i += 1 if v is True else 2
    else: i += 1"
    exit 0 ;;
  --env)
    echo "· container environment ($NAME)"
    docker inspect "$NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' |
      grep -E '^(MBX_|VLLM_|TORCH_|FLASHINFER_|TRITON_|PYTORCH_|MALLOC_)' | sort | sed 's/^/  /'
    exit 0 ;;
  --penalty)
    echo "· did the image parse --override-generation-config? (engine log, first boot)"
    docker logs "$NAME" 2>&1 | grep -iE 'override_generation_config|override-generation-config' | head -3
    echo
    echo "  NOTE: this proves the flag reached the engine config. Whether"
    echo "  presence_penalty is actually applied is a separate question: stock"
    echo "  vLLM ignores it in --override-generation-config without a patch."
    echo "  To settle it, compare a request that omits presence_penalty against"
    echo "  one that sends it explicitly, at temperature 0.6."
    exit 0 ;;
  --spec)
    exec sh tools/mtp_drafter_check.sh "$BASE" ;;
esac

echo "· container : $NAME   image $IMAGE"
echo "  mounts    : $(docker inspect "$NAME" --format '{{range .Mounts}}{{.Source}}=>{{.Destination}} {{end}}')"
echo "  serving   : $SERVE_PATH"
echo

python3 - "$BASE" "$NAME" <<'PY'
import json, sys, urllib.request
base, name = sys.argv[1], sys.argv[2]
def get(p):
    with urllib.request.urlopen(base + p, timeout=10) as r:
        return json.load(r)
d = get("/v1/models")["data"][0]
print("· served model")
print(f"    id          : {d['id']}")
print(f"    root        : {d.get('root')}")
print(f"    max context : {d.get('max_model_len')} tokens")

# The config the engine actually loaded, read out of the container.
import subprocess
path = subprocess.run(["docker","inspect",name,"--format","{{join .Config.Cmd \" \"}}"],
                      capture_output=True, text=True).stdout.split()
model = path[path.index("serve")+1] if "serve" in path else None
if model:
    cp = subprocess.run(["docker","exec",name,"cat",f"{model}/config.json"],
                        capture_output=True, text=True)
    if cp.returncode == 0:
        try:
            t = json.loads(cp.stdout).get("text_config", json.loads(cp.stdout))
            print(f"    architecture: {t.get('architectures',['?'])[0] if isinstance(t.get('architectures'),list) else t.get('architectures','?')}")
            print(f"    experts     : {t.get('num_experts_per_tok','?')} of {t.get('num_experts','?')} routed per token")
            print(f"    shared exp. : {t.get('shared_expert_intermediate_size','?')}")
            print(f"    layers      : {t.get('num_hidden_layers','?')}   hidden {t.get('hidden_size','?')}")
        except Exception as e:
            print(f"    (config unreadable: {e})")

print()
print("· test prompt")
body = json.dumps({
    "model": d["id"],
    "messages": [{"role": "user", "content": "Reply with exactly the word READY."}],
    "max_tokens": 30,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
try:
    r = json.load(urllib.request.urlopen(req, timeout=60))
    if "choices" in r:
        m = r["choices"][0]["message"]
        print(f"    {m.get('content') or m.get('reasoning') or '(empty)'}")
        u = r.get("usage", {})
        print(f"    tokens: {u.get('prompt_tokens')} prompt / {u.get('completion_tokens')} completion")
    else:
        print(f"    no answer: {str(r.get('error') or r)[:200]}")
except Exception as e:
    print(f"    request failed: {e}")
PY

echo
echo "· speculative decoding (from /metrics, since the last boot)"
docker logs "$NAME" 2>&1 | grep -oE 'Speculative decisions: [0-9.]+, Draft tokens: [0-9.]+, Accepted tokens: [0-9.]+' | tail -1 || echo "  (no spec counters logged yet)"
