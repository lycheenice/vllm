#!/bin/bash
# PD router (sglang_router with --pd-disaggregation)
set -ex
exec python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:8001 \
    --decode http://127.0.0.1:8002 \
    --host 0.0.0.0 --port 8000
