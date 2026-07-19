#!/usr/bin/env bash
# Launch sglang PD disaggregation (NIXL) on h200-2 with host networking.
# prefill(GPU 0-3, port 8001, bootstrap 8998) + decode(GPU 4-7, port 8002, bootstrap 8999) + router(port 8000)
# Usage: bash run_pd_nixl.sh [start|stop]
set -uo pipefail
IMG=br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129
MODEL_VOL=/data1/GLM-5.2-W4AFP8:/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8
SCRIPT_DIR=/opt/sglang-glm-pd

wait_health() {
  local port="$1" name="$2" max="${3:-300}"
  for i in $(seq 1 "$max"); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] $name healthy on :$port"; return 0
    fi
    sleep 5
  done
  echo "ERROR: $name not ready within $((max*5))s" >&2; return 1
}

case "${1:-start}" in
start)
  echo "[$(date +%H:%M:%S)] Stopping old containers..."
  docker rm -f pd-router pd-decode pd-prefill 2>/dev/null || true
  docker rm -f sglang-glm-smg-1 2>/dev/null || true

  echo "[$(date +%H:%M:%S)] Starting prefill (GPU 0-3, TP4, NIXL, host net)..."
  docker run -d --name pd-prefill \
    --network host \
    --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
    -e UCX_TLS=cuda_ipc,cuda_copy,tcp \
    -e UCX_NET_DEVICES=all \
    --cap-add SYS_NICE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --shm-size 32gb \
    -v "$MODEL_VOL" \
    -v "$SCRIPT_DIR/start-prefill.sh:/start.sh" \
    -v "$SCRIPT_DIR/logs/prefill:/root" \
    --restart unless-stopped \
    "$IMG" /start.sh

  echo "[$(date +%H:%M:%S)] Waiting prefill health (DeepGEMM JIT cache reuse, expect ~2 min)..."
  wait_health 8001 prefill 360 || { docker logs --tail 30 pd-prefill; exit 1; }

  echo "[$(date +%H:%M:%S)] Starting decode (GPU 4-7, TP4, NIXL, host net, bootstrap 8999)..."
  docker run -d --name pd-decode \
    --network host \
    --gpus all -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
    -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
    -e UCX_TLS=cuda_ipc,cuda_copy,tcp \
    -e UCX_NET_DEVICES=all \
    --cap-add SYS_NICE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --shm-size 32gb \
    -v "$MODEL_VOL" \
    -v "$SCRIPT_DIR/start-decode.sh:/start.sh" \
    -v "$SCRIPT_DIR/logs/decode:/root" \
    --restart unless-stopped \
    "$IMG" /start.sh

  echo "[$(date +%H:%M:%S)] Waiting decode health..."
  wait_health 8002 decode 360 || { docker logs --tail 30 pd-decode; exit 1; }

  echo "[$(date +%H:%M:%S)] Starting PD router on :8000..."
  docker run -d --name pd-router \
    --network host \
    -v "$SCRIPT_DIR/start-router.sh:/start.sh" \
    --restart unless-stopped \
    "$IMG" /start.sh
  sleep 3

  echo "============================================"
  echo "PD (NIXL host-net) stack up:"
  echo "  router  : http://h200-2:8000  <-- benchmark here"
  echo "  prefill : http://h200-2:8001  (GPU 0-3, bootstrap 8998)"
  echo "  decode  : http://h200-2:8002  (GPU 4-7, bootstrap 8999)"
  echo "  stop    : bash $0 stop"
  echo "============================================"
  ;;
stop)
  docker rm -f pd-router pd-decode pd-prefill 2>/dev/null || true
  echo "Stopped PD stack."
  ;;
status)
  docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'pd-' || echo "no pd containers"
  ;;
*)
  echo "Usage: $0 [start|stop|status]"; exit 1;;
esac
