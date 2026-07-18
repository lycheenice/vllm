#!/usr/bin/env bash
# Benchmark via router (PD) or baseline (tp8).
# Usage: bench.sh [strategy] [transport] [--baseline]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

if [[ "${3:-}" == "--baseline" ]]; then
  URL="http://127.0.0.1:$BASELINE_PORT"
  TAG="baseline-tp8"
else
  URL="http://127.0.0.1:$ROUTER_PORT"
  TAG="pd-$STRATEGY-$TRANSPORT"
fi

OUT="$LOG_DIR/bench-${TAG}-$(date +%H%M%S).json"
echo "[bench] url=$URL tag=$TAG out=$OUT"

# sglang bench_serving 参数以安装后 --help 为准；下列为常见字段。
"$PYTHON" -m sglang.bench.serving \
  --url "$URL" \
  --model "$MODEL_PATH" \
  --num-prompts "${NUM_PROMPTS:-256}" \
  --request-rate "${REQUEST_RATE:-INF}" \
  > "$OUT" 2>&1 || echo "[bench] non-zero exit, see $OUT"

echo "bench done: $OUT"
