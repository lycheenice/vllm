#!/usr/bin/env bash
# Run vllm bench serve against the proxy.
# Usage: bench.sh [gdr|cpu]
# Env overrides: IN (random-input-len) OUT (random-output-len)
#                N  (num-prompts)       B (burstiness)        R (request-rate)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}"

IN="${IN:-7500}"
OUT="${OUT:-200}"
N="${N:-50}"
B="${B:-1.0}"
R="${R:-2}"

LABEL="pd_${TRANSPORT}_${N}n_${IN}in_${OUT}out"

echo "[$(date +%H:%M:%S)] bench -> http://127.0.0.1:$PROXY_PORT  label=$LABEL"
echo "  input_len=$IN output_len=$OUT num_prompts=$N burstiness=$B request_rate=$R"

vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port "$PROXY_PORT" \
  --model "$MODEL_PATH" \
  --dataset-name random \
  --random-input-len "$IN" \
  --random-output-len "$OUT" \
  --num-prompts "$N" \
  --burstiness "$B" \
  --request-rate "$R" \
  --ignore-eos \
  --save-result \
  --result-dir "$LOG_DIR" \
  --result-filename "${LABEL}.json"

echo "Result saved to $LOG_DIR/${LABEL}.json"
