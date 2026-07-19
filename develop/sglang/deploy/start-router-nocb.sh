#!/bin/bash
# PD router — P1.2: disable circuit breaker + lenient timeouts
set -ex
exec python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:8001 \
    --decode http://127.0.0.1:8002 \
    --host 0.0.0.0 --port 8000 \
    --disable-circuit-breaker \
    --request-timeout-secs 600 \
    --queue-timeout-secs 600 \
    --health-failure-threshold 10 \
    --health-success-threshold 2
