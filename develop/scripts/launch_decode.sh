#!/usr/bin/env bash
# Launch the Decode (kv_consumer) vLLM instance.
# Usage: launch_decode.sh [gdr|cpu]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}"

KV_CONFIG_D='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"'"$KV_BUFFER_DEVICE"'","kv_load_failure_policy":"fail"}'

EXTRA=()
[[ "$ENFORCE_EAGER" == "1" ]] && EXTRA+=(--enforce-eager)

echo "[$(date +%H:%M:%S)] Starting Decode: GPUs=$D_GPUS port=$PORT_D transport=$TRANSPORT kv_buffer_device=$KV_BUFFER_DEVICE"

CUDA_VISIBLE_DEVICES="$D_GPUS" \
VLLM_KV_CACHE_LAYOUT=HND \
UCX_NET_DEVICES=all \
UCX_TLS="$UCX_TLS" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_PORT_D" \
vllm serve "$MODEL_PATH" \
  --port "$PORT_D" \
  --tensor-parallel-size "$TP" \
  --block-size "$BLOCK_SIZE" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --kv-transfer-config "$KV_CONFIG_D" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  > "$LOG_DIR/decode.log" 2>&1 &

echo $! > "$PID_DIR/decode.pid"
echo "Decode PID=$(cat "$PID_DIR/decode.pid"), log=$LOG_DIR/decode.log"
