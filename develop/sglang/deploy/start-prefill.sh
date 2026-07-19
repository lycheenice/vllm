#!/bin/bash
# PD prefill worker (NIXL backend, TP4, GPU 0-3)
set -ex
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export UCX_TLS=cuda_ipc,cuda_copy,tcp
export UCX_NET_DEVICES=all
export UCX_TLS=cuda_ipc,cuda_copy,tcp
exec python3 -m sglang.launch_server \
    --model /mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8 \
    --served-model-name glm \
    --trust-remote-code \
    --port 8001 \
    --host 0.0.0.0 \
    --context-len 300000 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --schedule-policy fcfs \
    --enable-metrics \
    --enable-cache-report \
    --tp-size 4 \
    --chunked-prefill-size 32768 \
    --max-running-requests 64 --max-queued-requests 512 \
    --mem-fraction-static 0.85 \
    --watchdog-timeout 1800 \
    --kv-cache-dtype fp8_e4m3 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl
