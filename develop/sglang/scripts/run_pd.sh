#!/usr/bin/env bash
# One-shot: prefill -> health -> decode -> health -> router -> health.
# Usage: run_pd.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
STRATEGY="${1:-tp4dp2}"; TRANSPORT="${2:-nixl}"

wait_health() {
  local url="$1" name="$2" tries="${3:-60}"
  for i in $(seq 1 "$tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "[ok] $name healthy ($i tries)"; return 0
    fi
    sleep 2
  done
  echo "[FAIL] $name not healthy after $tries tries"; return 1
}

"$SCRIPT_DIR/launch_prefill.sh" "$STRATEGY" "$TRANSPORT"
wait_health "http://127.0.0.1:30000/health" prefill 90 || exit 1

"$SCRIPT_DIR/launch_decode.sh" "$STRATEGY" "$TRANSPORT"
wait_health "http://127.0.0.1:30001/health" decode 90 || exit 1

"$SCRIPT_DIR/launch_router.sh" "$STRATEGY" "$TRANSPORT"
wait_health "http://127.0.0.1:8000/health" router 60 || exit 1

echo "PD stack up: router http://127.0.0.1:8000 (strategy=$STRATEGY transport=$TRANSPORT)"
