#!/usr/bin/env bash
# Launch MiniMax-M2.5 TP8 baseline (non-PD)
set -uo pipefail
IMG=br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129
MODEL_VOL=/data1/models/MiniMax-M2.5:/data1/models/MiniMax-M2.5
SCRIPT_DIR=/opt/sglang-minimax-pd
wait_health() { local port="$1" name="$2" max="${3:-120}"; for i in $(seq 1 "$max"); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "[$(date +%H:%M:%S)] $name healthy"; return 0; }; sleep 3; done; echo "ERR $name"; return 1; }
case "${1:-start}" in
start)
  docker rm -f minimax-baseline 2>/dev/null || true
  docker run -d --name minimax-baseline --network host --gpus all \
    --cap-add SYS_NICE --cap-add IPC_LOCK --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size 32gb \
    -v "$MODEL_VOL" -v "$SCRIPT_DIR/start-baseline-tp8.sh:/start.sh" --restart unless-stopped "$IMG" /start.sh
  wait_health 8000 baseline 120 || exit 1
  echo "MiniMax TP8 baseline up: :8000"
  ;;
stop) docker update --restart=no minimax-baseline 2>/dev/null; docker rm -f minimax-baseline 2>/dev/null; echo stopped;;
*) echo "Usage: $0 [start|stop]"; exit 1;;
esac
