#!/usr/bin/env bash
# Launch the sglang PD router.
# Usage: launch_router.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

[[ "$STRATEGY" == "tp8" ]] && { echo "STRATEGY=tp8 is baseline, no router"; exit 0; }

echo "[$(date +%H:%M:%S)] Starting router: port=$ROUTER_PORT -> prefill=$PORT_P decode=$PORT_D"

exec "$PYTHON" -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:$PORT_P" \
  --decode  "http://127.0.0.1:$PORT_D" \
  --host 0.0.0.0 --port "$ROUTER_PORT" \
  > "$LOG_DIR/router.log" 2>&1 &
echo $! > "$PID_DIR/router.pid"
echo "Router PID=$(cat "$PID_DIR/router.pid"), log=$LOG_DIR/router.log"
