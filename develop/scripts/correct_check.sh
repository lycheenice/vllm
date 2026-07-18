#!/usr/bin/env bash
# Correctness: diff PD-via-proxy output vs a standalone baseline instance.
# Prerequisite: a baseline `vllm serve` (no --kv-transfer-config) on $BASELINE_PORT.
# Usage: correct_check.sh ["your prompt"]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh"

PROMPT="${1:-Once upon a time in a galaxy far far away, there lived a quiet engineer who}"
OUT_DIR="$LOG_DIR/correctness"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

call() {
  local port="$1" tag="$2"
  curl -s "http://127.0.0.1:$port/v1/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{"model":"'"$MODEL_PATH"'","prompt":"'"$PROMPT"'","max_tokens":32,"temperature":0,"stream":false}' \
    | python -c 'import sys,json
try:
    print(json.load(sys.stdin)["choices"][0]["text"])
except Exception as e:
    print("ERROR: %s :: %s" % (e, sys.stdin.read()))' \
    > "$OUT_DIR/${tag}_${STAMP}.txt"
  echo "[${tag}] saved -> $OUT_DIR/${tag}_${STAMP}.txt ($(wc -c < "$OUT_DIR/${tag}_${STAMP}.txt") bytes)"
}

call "$PROXY_PORT"    "pd"
call "$BASELINE_PORT" "baseline"

echo "----- diff (baseline vs pd) -----"
if diff -u "$OUT_DIR/baseline_${STAMP}.txt" "$OUT_DIR/pd_${STAMP}.txt"; then
  echo "PASS: PD output matches baseline (greedy, max_tokens=32)."
else
  echo "WARN: outputs differ (for greedy + healthy P/D path they should match)."
fi
