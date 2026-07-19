#!/bin/bash
# PD decode worker (NIXL, TP4, GPU 4-7) — P1.1: + L2 hicache ratio=10
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export UCX_TLS=cuda_ipc,cuda_copy,tcp
export UCX_NET_DEVICES=all
export SGLANG_DISAGGREGATION_QUEUE_SIZE=8
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
exec python3 -m sglang.launch_server \
    --model /mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8 \
    --served-model-name glm \
    --trust-remote-code \
    --port 8002 \
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
    --cuda-graph-max-bs 128 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-decode-enable-radix-cache \
    --enable-hierarchical-cache \
    --hicache-ratio 3 \
    --hicache-io-backend direct
