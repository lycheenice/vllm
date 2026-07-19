#!/bin/bash
# MiniMax-M2.5 PD prefill Config C (P1.3 + hicache ratio=3)
# MiniMax 标准 MHA, hicache 应兼容 (无 MLA element_size=656 问题)
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export UCX_TLS=cuda_ipc,cuda_copy,tcp
export UCX_NET_DEVICES=all
export SGLANG_DISAGGREGATION_QUEUE_SIZE=8
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
exec python3 -m sglang.launch_server \
    --model /data1/models/MiniMax-M2.5 \
    --served-model-name minimax \
    --trust-remote-code \
    --port 8001 \
    --host 0.0.0.0 \
    --context-len 100000 \
    --schedule-policy fcfs \
    --enable-metrics \
    --enable-cache-report \
    --tp-size 4 \
    --chunked-prefill-size 8192 \
    --max-running-requests 64 --max-queued-requests 512 \
    --mem-fraction-static 0.85 \
    --watchdog-timeout 1800 \
    --kv-cache-dtype fp8_e4m3 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port 8998 \
    --enable-hierarchical-cache \
    --hicache-ratio 3 \
    --hicache-io-backend direct
