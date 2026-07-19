#!/usr/bin/env bash
# Launch sglang PD disaggregation (Mooncake + INTRA_NODE_NVLINK) on h200-2
# Usage: bash run_pd_mooncake.sh [start|stop]
set -euo pipefail
IMG=br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129
MODEL_VOL=/data1/GLM-5.2-W4AFP8:/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8
SCRIPT_DIR=/opt/sglang-glm-pd

case "${1:-start}" in
start)
  echo "[$(date +%H:%M:%S)] Stopping old containers..."
  docker rm -f pd-prefill-mc pd-decode-mc pd-router-mc 2>/dev/null || true
  docker rm -f sglang-glm-smg-1 2>/dev/null || true

  echo "[$(date +%H:%M:%S)] Starting prefill (GPU 0-3, TP4, Mooncake+NVLink)..."
  docker run -d --name pd-prefill-mc \
    --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK \
    -e MC_INTRANODE_NVLINK=true \
    --ipc=host \
    --cap-add SYS_NICE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --shm-size 32gb \
    -v "$MODEL_VOL" \
    -v "$SCRIPT_DIR/start-prefill-mooncake.sh:/start.sh" \
    -v "$SCRIPT_DIR/logs/prefill-mc:/root" \
    -p 8001:8001 \
    "$IMG" /start.sh

  echo "[$(date +%H:%M:%S)] Starting decode (GPU 4-7, TP4, Mooncake+NVLink)..."
  docker run -d --name pd-decode-mc \
    --gpus all -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
    -e SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK \
    -e MC_INTRANODE_NVLINK=true \
    --ipc=host \
    --cap-add SYS_NICE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --shm-size 32gb \
    -v "$MODEL_VOL" \
    -v "$SCRIPT_DIR/start-decode-mooncake.sh:/start.sh" \
    -v "$SCRIPT_DIR/logs/decode-mc:/root" \
    -p 8002:8002 \
    "$IMG" /start.sh

  echo "[$(date +%H:%M:%S)] Waiting for both to be healthy (max 20 min each)..."
  for name in pd-prefill-mc pd-decode-mc; do
    port=8001; [[ "$name" == "pd-decode-mc" ]] && port=8002
    echo "Waiting $name on :$port..."
    for i in $(seq 1 240); do
      if docker exec "$name" curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] $name healthy"; break
      fi
      sleep 5
    done
  done

  echo "[$(date +%H:%M:%S)] Starting router..."
  docker run -d --name pd-router-mc \
    --network host \
    -v "$SCRIPT_DIR/start-router.sh:/start.sh" \
    "$IMG" /start.sh

  sleep 3
  echo "============================================"
  echo "PD (Mooncake+NVLink) stack up:"
  echo "  router  : http://h200-2:8000"
  echo "  prefill : http://h200-2:8001  (GPU 0-3)"
  echo "  decode  : http://h200-2:8002  (GPU 4-7)"
  echo "============================================"
  ;;
stop)
  docker rm -f pd-router-mc pd-decode-mc pd-prefill-mc 2>/dev/null || true
  echo "Mooncake PD stopped."
  ;;
*)
  echo "Usage: $0 [start|stop]"; exit 1;;
esac
