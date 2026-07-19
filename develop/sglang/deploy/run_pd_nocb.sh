#!/usr/bin/env bash
# Launch sglang PD P1.2: exp1 opt config + router --disable-circuit-breaker
set -uo pipefail
IMG=br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129
MODEL_VOL=/data1/GLM-5.2-W4AFP8:/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8
SCRIPT_DIR=/opt/sglang-glm-pd
wait_health() { local port="$1" name="$2" max="${3:-600}"; for i in $(seq 1 "$max"); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "[$(date +%H:%M:%S)] $name healthy"; return 0; }; sleep 5; done; echo "ERR $name"; return 1; }
case "${1:-start}" in
start)
  docker rm -f pd-router pd-decode pd-prefill 2>/dev/null || true
  docker run -d --name pd-prefill --network host --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX -e UCX_TLS=cuda_ipc,cuda_copy,tcp -e UCX_NET_DEVICES=all \
    -e SGLANG_DISAGGREGATION_QUEUE_SIZE=8 -e SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12 -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 \
    --cap-add SYS_NICE --cap-add IPC_LOCK --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size 32gb \
    -v "$MODEL_VOL" -v "$SCRIPT_DIR/start-prefill-opt.sh:/start.sh" -v "$SCRIPT_DIR/logs/prefill:/root" --restart unless-stopped "$IMG" /start.sh
  wait_health 8001 prefill 600 || exit 1
  docker run -d --name pd-decode --network host --gpus all -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
    -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX -e UCX_TLS=cuda_ipc,cuda_copy,tcp -e UCX_NET_DEVICES=all \
    -e SGLANG_DISAGGREGATION_QUEUE_SIZE=8 -e SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12 \
    -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 \
    --cap-add SYS_NICE --cap-add IPC_LOCK --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size 32gb \
    -v "$MODEL_VOL" -v "$SCRIPT_DIR/start-decode-opt.sh:/start.sh" -v "$SCRIPT_DIR/logs/decode:/root" --restart unless-stopped "$IMG" /start.sh
  wait_health 8002 decode 600 || exit 1
  docker run -d --name pd-router --network host -v "$SCRIPT_DIR/start-router-nocb.sh:/start.sh" --restart unless-stopped "$IMG" /start.sh
  echo "PD-nocb up: router :8000 (circuit breaker disabled)"
  ;;
stop) docker update --restart=no pd-router pd-decode pd-prefill 2>/dev/null; docker rm -f pd-router pd-decode pd-prefill 2>/dev/null; echo stopped;;
*) echo "Usage: $0 [start|stop]"; exit 1;;
esac
