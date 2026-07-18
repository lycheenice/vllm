#!/usr/bin/env bash
# Snapshot: pids, http health, GPU, recent log tails.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh"

echo "=== PIDs ==="
for name in prefill decode proxy; do
  pf="$PID_DIR/$name.pid"
  if [[ -f "$pf" ]]; then
    pid="$(cat "$pf" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      printf "  %-8s pid=%-8s ALIVE\n" "$name" "$pid"
    else
      printf "  %-8s pid=%-8s DEAD\n" "$name" "$pid"
    fi
  else
    printf "  %-8s (no pid file)\n" "$name"
  fi
done

echo "=== Health ==="
for pair in "prefill:$PORT_P" "decode:$PORT_D" "proxy:$PROXY_PORT"; do
  name="${pair%%:*}"; port="${pair##*:}"
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null || echo 000)"
  printf "  %-8s :%-5s HTTP %s\n" "$name" "$port" "$code"
done
# proxy also exposes /healthcheck with instance counts (toy_proxy_server.py:274)
echo "  proxy    /healthcheck: $(curl -s "http://127.0.0.1:$PROXY_PORT/healthcheck" 2>/dev/null || echo '{unreachable}')"

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"

echo "=== recent log tails (last 3 lines each) ==="
for f in prefill decode proxy; do
  lf="$LOG_DIR/$f.log"
  if [[ -f "$lf" ]]; then
    echo "--- $f.log ---"
    tail -n 3 "$lf"
  fi
done
