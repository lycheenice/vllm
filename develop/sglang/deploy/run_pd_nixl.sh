#!/usr/bin/env bash
# Launch sglang PD disaggregation (NIXL) on h200-2: prefill(GPU0-3) + decode(GPU4-7) + router
# Usage: bash run_pd_nixl.sh [start|stop]
set -euo pipefail
IMG=br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129
MODEL_VOL=/data1/GLM-5.2-W4AFP8:/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8
SCRIPT_DIR=/opt/sglang-glm-pd

case "${1:-start}" in
start)
  echo "[$(date +%H:%M:%S)] Stopping old containers..."
  docker rm -f pd-prefill pd-decode pd-router 2>/dev/null || true
  docker rm -f sglang-glm-smg-1 2>/dev/null || true

  echo "[$(date +%H:%M:%S)] Starting prefill (GPU 0-3, TP4, NIXL)..."
  docker run -d --name pd-prefill \
    --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
    -e UCX_TLS=cuda_ipc,cuda_copy,tcp \
    -e UCX_NET_DEVICES=all \
    --ipc=host \
    --cap-add SYS_NICE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --shm-size 32gb \
    -v "$MODEL_VOL" \
    -v "$SCRIPT_DIR/start-prefill.sh:/start.sh" \
    -v "$SCRIPT_DIR/logs/prefill:/root" \
    -p 8001:8001 \
    "$IMG" /start.sh

  echo "[$(date +%H:%M:%S)] Waiting prefill health..."
  for i in $(seq 1 300); do
    if docker exec pd-prefill curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] prefill healthy"; break
    fi
    sleep 5
  done

  echo "[$(date +%H:%M:%S)] Starting decode (GPU 4-7, TP4, NIXL)..."
  docker run -d --name pd-decode \
    --gpus all -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
    -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
    -e UCX_TLS=cuda_ipc,cuda_copy,tcp \
    -e UCX_NET_DEVICES=all \
    --ipc=host \
    --cap-add SYS_NICE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --shm-size 32gb \
    -v "$MODEL_VOL" \
    -v "$SCRIPT_DIR/start-decode.sh:/start.sh" \
    -v "$SCRIPT_DIR/logs/decode:/root" \
    -p 8002:8002 \
    "$IMG" /start.sh

  echo "[$(date +%H:%M:%S)] Waiting decode health..."
  for i in $(seq 1 300); do
    if docker exec pd-decode curl -sf http://127.0.0.1:8002/health >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] decode healthy"; break
    fi
    sleep 5
  done

  echo "[$(date +%H:%M:%S)] Starting router..."
  docker run -d --name pd-router \
    --network host \
    -v "$SCRIPT_DIR/start-router.sh:/start.sh" \
    "$IMG" /start.sh

  sleep 3
  echo "============================================"
  echo "PD (NIXL) stack up:"
  echo "  router  : http://h200-2:8000  (send requests here)"
  echo "  prefill : http://h200-2:8001  (GPU 0-3)"
  echo "  decode  : http://h200-2:8002  (GPU 4-7)"
  echo "  logs    : $SCRIPT_DIR/logs/{prefill,decode}/"
  echo "  stop    : bash $0 stop"
  echo "============================================"
  ;;
stop)
  echo "Stopping PD containers..."
  docker rm -f pd-router pd-decode pd-prefill 2>/dev/null || true
  echo "Stopped."
  ;;
status)
  docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'pd-' || echo "no pd containers"
  ;;
*)
  echo "Usage: $0 [start|stop|status]"; exit 1;;
esac
