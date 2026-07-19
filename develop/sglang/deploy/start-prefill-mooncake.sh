#!/bin/bash
# PD prefill worker (Mooncake backend, TP4, GPU 0-3)
# NVLink intra-node optimization for single-machine KV transfer
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export MC_INTRANODE_NVLINK=true
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
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device all
