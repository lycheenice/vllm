#!/usr/bin/env bash
# Launch the sglang Decode instance.
# Usage: launch_decode.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

[[ "$STRATEGY" == "tp8" ]] && { echo "STRATEGY=tp8 is baseline, no decode"; exit 0; }

D_ARGS=(--disaggregation-mode decode --disaggregation-transfer-backend "$DISAGG_BACKEND"
  --tp-size "$D_TP" --port "$PORT_D")

[[ "$DP_ATTN" == "1" ]] && D_ARGS+=(--dp-size "$D_DP" --enable-dp-attention)
[[ -n "$MOE_A2A_BACKEND" ]] && D_ARGS+=(--moe-a2a-backend "$MOE_A2A_BACKEND")

echo "[$(date +%H:%M:%S)] Starting Decode: GPUs=$D_GPUS port=$PORT_D strategy=$STRATEGY transport=$TRANSPORT tp=$D_TP dp=$D_DP dp_atten=$DP_ATTN moe_a2a=${MOE_A2A_BACKEND:-none} staging=$STAGING"

CUDA_VISIBLE_DEVICES="$D_GPUS" \
exec "$PYTHON" -m sglang.launch_server \
  "${D_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  ${QUANT[@]+"${QUANT[@]}"} \
  > "$LOG_DIR/decode.log" 2>&1 &
echo $! > "$PID_DIR/decode.pid"
echo "Decode PID=$(cat "$PID_DIR/decode.pid"), log=$LOG_DIR/decode.log"
