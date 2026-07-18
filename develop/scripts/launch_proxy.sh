#!/usr/bin/env bash
# Launch the toy disagg-prefill proxy.
# Usage: launch_proxy.sh [gdr|cpu]   (transport arg ignored, kept for symmetry)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}"

echo "[$(date +%H:%M:%S)] Starting proxy on port $PROXY_PORT"

python "$VLLM_ROOT/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --port "$PROXY_PORT" \
  --prefiller-hosts localhost \
  --prefiller-ports "$PORT_P" \
  --decoder-hosts localhost \
  --decoder-ports "$PORT_D" \
  > "$LOG_DIR/proxy.log" 2>&1 &

echo $! > "$PID_DIR/proxy.pid"
echo "Proxy PID=$(cat "$PID_DIR/proxy.pid"), log=$LOG_DIR/proxy.log"
