#!/usr/bin/env bash
# develop/experiments/verify-kvrole/verify.sh
# ★在 h200-2 本机执行★。一键验证 kv_role 语义对 KV 传输的影响。
#
# 流程:
#   1) 起单实例 TP4 golden(无 PD),对固定 prompt 集取 greedy 输出作为标准答案。
#   2) 起 PD(KV_ROLE_MODE=pc),取同一 prompt 集输出 + 抓传输证据,停。
#   3) 起 PD(KV_ROLE_MODE=both),同上,停。
#   4) 判定:pc/both 的输出是否与 golden 一致 + 各自的 KV 传输证据计数。
#
# 判据(见 README「判据」):
#   - 输出与 golden 一致  => 该模式下 PD 端到端语义正确(必要条件)。
#   - 传输证据计数 > 0     => 该模式下 KV 确实在 P/D 间传输(而非 decode 端重算)。
#   - 若 pc 证据=0 但 both 证据>0 => 印证"v0.25.0 应使用 kv_both"的猜想,应改所有 PD 实验默认。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../../common/common.env"
source "$HERE/config.env"
source "$COMMON_DIR/lib.sh"

OUT="$HERE/results/verify_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
GOLDEN_PORT="${GOLDEN_PORT:-8041}"
EVIDENCE_KEYS="${EVIDENCE_KEYS:-nixl|xfer|transfer|handshake|remote|recv|save_kv|load_kv|producer|consumer|kv_both}"
PROMPTS=(
  "Explain in one sentence why the sky is blue."
  "Write a Python function that returns the nth Fibonacci number."
  "Summarize the plot of Romeo and Juliet in two sentences."
  "What is the time complexity of quicksort in the average case?"
  "Translate 'good morning, how are you' into French."
)

gen_outputs() {   # $1=port $2=outfile
  local port="$1" out="$2"; : > "$out"
  for p in "${PROMPTS[@]}"; do
    curl -s --max-time 120 "http://127.0.0.1:$port/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_PATH\",\"prompt\":\"$p\",\"max_tokens\":64,\"temperature\":0}" \
      | python3 -c 'import sys,json;
try: print(json.load(sys.stdin)["choices"][0]["text"].strip())
except Exception as e: print("ERR:%s"%e)' >> "$out"
    echo "----" >> "$out"
  done
}

count_evidence() {   # $1=tag  -> 统计当前 PD 日志里的传输证据行数
  local tag="$1" logs="$HERE/results/logs"
  local n=$(grep -ihcE "$EVIDENCE_KEYS" "$logs/prefill.log" "$logs/decode.log" 2>/dev/null | awk '{s+=$1} END{print s+0}')
  echo "$n" > "$OUT/${tag}_evidence_count.txt"
  grep -ihE "$EVIDENCE_KEYS" "$logs/prefill.log" "$logs/decode.log" 2>/dev/null | tail -40 > "$OUT/${tag}_evidence.log" || true
  echo "$n"
}

# --- 1) golden(单实例 TP4)---
log "起 golden 单实例 (TP4, GPU $P_GPUS, port $GOLDEN_PORT)"
docker rm -f verify-kvrole-golden >/dev/null 2>&1 || true
docker run -d --name verify-kvrole-golden --network host --ipc=host --shm-size=32g \
  --gpus "\"device=$P_GPUS\"" \
  -v "$MODEL_DIR_HOST:$MODEL_DIR_HOST" -v "$REPO_ON_EXEC:$REPO_ON_EXEC" \
  "${COMMON_DOCKER_ENV[@]}" "$IMAGE" \
  "$MODEL_PATH" --port "$GOLDEN_PORT" --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$UTIL" --max-model-len "$MAX_MODEL_LEN" --trust-remote-code \
    --enable-prefix-caching --enforce-eager >/dev/null
( docker logs -f verify-kvrole-golden >"$OUT/golden.log" 2>&1 & )
waited=0; until curl -sf "http://127.0.0.1:$GOLDEN_PORT/health" >/dev/null 2>&1; do
  sleep 5; waited=$((waited+5)); (( waited>HEALTH_TIMEOUT )) && die "golden 未就绪"; done
log "golden 就绪,取标准答案"
gen_outputs "$GOLDEN_PORT" "$OUT/golden.txt"
docker rm -f verify-kvrole-golden >/dev/null 2>&1 || true

# --- 2)/3) 两种 kv_role 模式 ---
for mode in pc both; do
  log "=== KV_ROLE_MODE=$mode: 起 PD ==="
  KV_ROLE_MODE="$mode" bash "$HERE/run.sh"
  gen_outputs "$PROXY_PORT" "$OUT/$mode.txt"
  ev=$(count_evidence "$mode")
  log "KV_ROLE_MODE=$mode: 传输证据行数=$ev"
  bash "$HERE/stop.sh"
  sleep 5
done

# --- 4) 判定 ---
{
  echo "# kv_role 验证结果  ($(basename "$OUT"))"
  echo
  for mode in pc both; do
    if diff -q "$OUT/golden.txt" "$OUT/$mode.txt" >/dev/null 2>&1; then eq=一致; else eq=不一致; fi
    echo "- KV_ROLE_MODE=$mode: 输出与 golden **$eq**,传输证据计数=$(cat "$OUT/${mode}_evidence_count.txt" 2>/dev/null)"
  done
  echo
  echo "判据:输出一致=语义正确(必要条件);证据>0=KV 真传输。"
  echo "若 pc 证据≈0 而 both>0 => 应把所有 PD 实验默认改为 kv_both。"
} | tee "$OUT/VERDICT.md"
log "完成。详见 $OUT/VERDICT.md"
