#!/usr/bin/env bash
# MiniMax-M2.5 bench runner: runs ramp_test.py for all configs
# Usage: run_bench.sh <config_name> <endpoint_port> [levels] [trials]
# config_name: baseline | pd-A | pd-B | pd-C
# endpoint_port: 8000 (baseline direct) or 8000 (PD via router)
# levels: default "1,4,16,32,64"
# trials: default 5
set -uo pipefail
IMG=br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129
CASE="${1:?Usage: $0 <config_name> [levels] [trials]}"
LEVELS="${2:-1,4,16,32,64}"
TRIALS="${3:-5}"

echo "=== MiniMax bench: case=$CASE levels=$LEVELS trials=$TRIALS ==="
docker rm -f mm-bench 2>/dev/null || true
docker run -d --name mm-bench --rm --network host \
  -v /home/lychee/mycode/kvcache-benchmarks:/bench -w /bench \
  "$IMG" \
  python3 scripts/ramp_test.py \
    --dataset data/codex_swebenchpro_traces/codex_swebenchpro.json \
    --endpoint http://127.0.0.1:8000/v1 \
    --model minimax --api-key EMPTY \
    --server sglang --case "$CASE" \
    --levels "$LEVELS" \
    --trials-per-user "$TRIALS" \
    --max-tokens 4096
echo "bench started, monitor: docker logs -f mm-bench"
