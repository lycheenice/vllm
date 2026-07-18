#!/usr/bin/env bash
# Launch the sglang Prefill instance.
# Usage: launch_prefill.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

[[ "$STRATEGY" == "tp8" ]] && { echo "STRATEGY=tp8 is baseline, no prefill"; exit 0; }

echo "[$(date +%H:%M:%S)] Starting Prefill: GPUs=$P_GPUS port=$PORT_P strategy=$STRATEGY transport=$TRANSPORT tp=$P_TP"

CUDA_VISIBLE_DEVICES="$P_GPUS" \
exec "$PYTHON" -m sglang.launch_server \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend "$DISAGG_BACKEND" \
  --tp-size "$P_TP" \
  --port "$PORT_P" \
  "${COMMON_ARGS[@]}" \
  ${QUANT[@]+"${QUANT[@]}"} \
  > "$LOG_DIR/prefill.log" 2>&1 &
echo $! > "$PID_DIR/prefill.pid"
echo "Prefill PID=$(cat "$PID_DIR/prefill.pid"), log=$LOG_DIR/prefill.log"
