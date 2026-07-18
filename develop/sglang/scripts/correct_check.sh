#!/usr/bin/env bash
# Correctness diff: PD (router) vs baseline (tp8).
# Usage: correct_check.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

PROMPT="${PROMPT:-用一句话解释什么是 Prefill/Decode 分离。}"
TMP="$LOG_DIR/correct_$$"
PAYLOAD="$TMP.payload"

# 用 python 生成合法 JSON，避免 shell 转义
"$PYTHON" - "$PROMPT" "$MODEL_PATH" "$PAYLOAD" <<'PYEOF'
import json, sys
prompt, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
with open(out, "w") as f:
    json.dump({"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 64}, f, ensure_ascii=False)
PYEOF

call_chat() {  # url out
  curl -sf "$1/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary "@$PAYLOAD" \
    | "$PYTHON" -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' > "$2"
}

call_chat "http://127.0.0.1:$ROUTER_PORT"   "$TMP.pd"
call_chat "http://127.0.0.1:$BASELINE_PORT" "$TMP.base"

echo "--- PD ---";  cat "$TMP.pd"
echo "--- BASE ---"; cat "$TMP.base"
if diff -u "$TMP.base" "$TMP.pd" > "$TMP.diff"; then
  echo "[ok] PD == BASELINE (greedy)"
else
  echo "[warn] diff (容许量化精度差异):"; cat "$TMP.diff"
fi
rm -f "$TMP.pd" "$TMP.base" "$TMP.diff" "$PAYLOAD"
