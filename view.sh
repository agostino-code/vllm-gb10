#!/usr/bin/env bash
# view.sh — Live stats and log monitor for the running serve
set -euo pipefail
cd "$(dirname "$0")"

source ./config.env
NAME="qwen38-flash-next"
docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "^$NAME$" || NAME="qwen38-flash"

S="$(docker ps --filter "name=$NAME" --format '{{.Status}}')"
[ -n "$S" ] || { echo "· Server container is not running (start it with ./run.sh $DEFAULT_RECIPE)"; exit 1; }

PORT=8000
if [ -f "recipes/$DEFAULT_RECIPE" ]; then
  PORT="$(awk '$1=="port:"{print $2; exit}' "recipes/$DEFAULT_RECIPE" 2>/dev/null || echo 8000)"
fi

H="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "offline")"
echo "· $NAME: $S — health HTTP $H — API http://127.0.0.1:${PORT}/v1   (Ctrl-C detaches)"
exec docker logs -f --tail 30 "$NAME"
