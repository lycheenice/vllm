#!/usr/bin/env bash
# Quick status of the sglang PD stack.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

echo "=== strategy=$STRATEGY transport=$TRANSPORT ==="
for name in prefill decode router; do
  f="$PID_DIR/$name.pid"
  if [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; then
    echo "[up]   $name pid=$(cat "$f")"
  else
    echo "[down] $name"
  fi
done

echo "=== ports ==="
for p in "$PORT_P" "$PORT_D" "$ROUTER_PORT" "$BASELINE_PORT"; do
  if curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1; then
    echo "[up] port $p"
  else
    echo "[--] port $p"
  fi
done

echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true

echo "=== log tails ==="
for f in prefill decode router; do
  log="$LOG_DIR/$f.log"
  [[ -f "$log" ]] && { echo "--- $f.log ---"; tail -n 5 "$log"; }
done
