#!/bin/bash
# MiniMax-M2.5 TP8 baseline (non-PD, single server, 8 GPUs mixed prefill+decode)
# GPU 0-7, TP8, no disaggregation
exec python3 -m sglang.launch_server \
    --model /data1/models/MiniMax-M2.5 \
    --served-model-name minimax \
    --trust-remote-code \
    --port 8000 \
    --host 0.0.0.0 \
    --context-len 100000 \
    --schedule-policy fcfs \
    --enable-metrics \
    --enable-cache-report \
    --tp-size 8 \
    --chunked-prefill-size 8192 \
    --max-running-requests 128 --max-queued-requests 512 \
    --mem-fraction-static 0.90 \
    --watchdog-timeout 1800 \
    --kv-cache-dtype fp8_e4m3 \
    --cuda-graph-max-bs 256
