#!/usr/bin/env bash
# view.sh — follow the engine log and show where the server is.
#
#   ./view.sh
#
# The port is read from the running container's own binding, not from a recipe
# that may not be the one that was launched.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env

NAME="${CONTAINER:-qwen38-flash-next}"
docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
  echo "· '$NAME' is not running. Start one with ./presets/run_az_standard.sh" >&2; exit 1; }

# The container's own binding wins over config.env's PORT default.
BOUND="$(docker port "$NAME" 8000/tcp 2>/dev/null | head -1 | awk -F: '{print $NF}')"
PORT="${BOUND:-$PORT}"
H="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo offline)"
echo "· $NAME  $(docker ps --filter name=^${NAME}$ --format '{{.Status}}')  health $H  API http://127.0.0.1:$PORT/v1"
echo "  (Ctrl-C detaches; the server keeps running)"
exec docker logs -f --tail 40 "$NAME"
