#!/usr/bin/env bash
# One-shot: start P -> health -> D -> health -> proxy -> print endpoints.
# Usage: run_pd.sh [gdr|cpu]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}"

wait_health() {
  local port="$1" name="$2"
  echo "[$(date +%H:%M:%S)] Waiting for $name on http://127.0.0.1:$port/health ..."
  for _ in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] $name is healthy"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: $name did not become healthy within timeout" >&2
  return 1
}

"$SCRIPT_DIR/launch_prefill.sh" "$TRANSPORT"
wait_health "$PORT_P" "prefill" || { echo "Prefill failed health check, see $LOG_DIR/prefill.log" >&2; exit 1; }

"$SCRIPT_DIR/launch_decode.sh" "$TRANSPORT"
wait_health "$PORT_D" "decode" || { echo "Decode failed health check, see $LOG_DIR/decode.log" >&2; exit 1; }

"$SCRIPT_DIR/launch_proxy.sh"
sleep 2

cat <<EOF
============================================================
P/D + proxy are up (transport=$TRANSPORT, kv_buffer_device=$KV_BUFFER_DEVICE).
  Prefill : http://127.0.0.1:$PORT_P  (GPUs $P_GPUS, TP=$TP)
  Decode  : http://127.0.0.1:$PORT_D  (GPUs $D_GPUS, TP=$TP)
  Proxy   : http://127.0.0.1:$PROXY_PORT  <-- send client/bench requests here
  Health  : curl http://127.0.0.1:$PROXY_PORT/healthcheck
  Logs    : $LOG_DIR/{prefill,decode,proxy}.log
  PIDs    : $PID_DIR/{prefill,decode,proxy}.pid
Stop with: $SCRIPT_DIR/stop_pd.sh
============================================================
EOF
