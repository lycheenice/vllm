#!/usr/bin/env bash
# Stop P/D/proxy by pid files, then fallback pkill scoped to our ports/model.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh"

for name in proxy decode prefill; do
  pf="$PID_DIR/$name.pid"
  if [[ -f "$pf" ]]; then
    pid="$(cat "$pf" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (pid=$pid)"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
  fi
done

# Fallback: any lingering vllm serve / toy_proxy bound to our ports/model.
pkill -f "toy_proxy_server.py --port $PROXY_PORT" 2>/dev/null || true
pkill -f "vllm serve $MODEL_PATH --port $PORT_P" 2>/dev/null || true
pkill -f "vllm serve $MODEL_PATH --port $PORT_D" 2>/dev/null || true

echo "Stopped. (verify with: $SCRIPT_DIR/status.sh)"
