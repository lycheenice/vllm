#!/usr/bin/env bash
# Launch MiniMax-M2.5 PD stack with specified config (A/B/C)
# Usage: run_pd.sh <config> [start|stop]
#   config = A (basic), B (chunked8192+nocb), C (B+hicache)
set -uo pipefail
IMG=br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129
MODEL_VOL=/data1/models/MiniMax-M2.5:/data1/models/MiniMax-M2.5
SCRIPT_DIR=/opt/sglang-minimax-pd
CONFIG="${1:-B}"
ACTION="${2:-start}"

PF_SCRIPT="start-prefill-${CONFIG}.sh"
DC_SCRIPT="start-decode-${CONFIG}.sh"
# Config A uses default router; B/C use nocb router
if [ "$CONFIG" = "A" ]; then
  RT_SCRIPT="start-router-default.sh"
else
  RT_SCRIPT="start-router-nocb.sh"
fi

wait_health() { local port="$1" name="$2" max="${3:-120}"; for i in $(seq 1 "$max"); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "[$(date +%H:%M:%S)] $name healthy"; return 0; }; sleep 3; done; echo "ERR $name"; return 1; }

case "$ACTION" in
start)
  docker rm -f mm-router mm-decode mm-prefill 2>/dev/null || true
  docker run -d --name mm-prefill --network host --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX -e UCX_TLS=cuda_ipc,cuda_copy,tcp -e UCX_NET_DEVICES=all \
    -e SGLANG_DISAGGREGATION_QUEUE_SIZE=8 -e SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12 -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 \
    --cap-add SYS_NICE --cap-add IPC_LOCK --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size 32gb \
    -v "$MODEL_VOL" -v "$SCRIPT_DIR/$PF_SCRIPT:/start.sh" -v "$SCRIPT_DIR/logs/prefill:/root" --restart unless-stopped "$IMG" /start.sh
  wait_health 8001 prefill 120 || exit 1
  docker run -d --name mm-decode --network host --gpus all -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
    -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX -e UCX_TLS=cuda_ipc,cuda_copy,tcp -e UCX_NET_DEVICES=all \
    -e SGLANG_DISAGGREGATION_QUEUE_SIZE=8 -e SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12 \
    -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 \
    --cap-add SYS_NICE --cap-add IPC_LOCK --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size 32gb \
    -v "$MODEL_VOL" -v "$SCRIPT_DIR/$DC_SCRIPT:/start.sh" -v "$SCRIPT_DIR/logs/decode:/root" --restart unless-stopped "$IMG" /start.sh
  wait_health 8002 decode 120 || exit 1
  docker run -d --name mm-router --network host -v "$SCRIPT_DIR/$RT_SCRIPT:/start.sh" --restart unless-stopped "$IMG" /start.sh
  echo "MiniMax PD Config $CONFIG up: router :8000"
  ;;
stop) docker update --restart=no mm-router mm-decode mm-prefill 2>/dev/null; docker rm -f mm-router mm-decode mm-prefill 2>/dev/null; echo stopped;;
*) echo "Usage: $0 <A|B|C> [start|stop]"; exit 1;;
esac
