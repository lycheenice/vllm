#!/usr/bin/env bash
# Stop prefill / decode / router by PID files.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

kill_pid() {
  local f="$1" name="$2"
  if [[ -f "$f" ]]; then
    local pid; pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "[stop] $name pid=$pid"
    fi
    rm -f "$f"
  fi
}

kill_pid "$PID_DIR/router.pid"  router
kill_pid "$PID_DIR/decode.pid"  decode
kill_pid "$PID_DIR/prefill.pid" prefill
echo "stopped (strategy=$STRATEGY transport=$TRANSPORT)"
