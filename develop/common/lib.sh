#!/usr/bin/env bash
# develop/common/lib.sh — 公共 shell 函数,被 serve_pd.sh / bench_ramp.sh source。
set -u

log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# 在 h200-2 上等某端口 /health 就绪(通过 EXEC_SSH curl localhost)。
wait_health_remote() {
  local port="$1" name="$2" timeout="${3:-$HEALTH_TIMEOUT}"
  log "等待 $name 健康 (h200-2:$port/health, 超时 ${timeout}s) ..."
  local waited=0
  while (( waited < timeout )); do
    if $EXEC_SSH "curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1"; then
      log "$name 已健康"; return 0
    fi
    sleep 5; waited=$((waited+5))
  done
  die "$name 在 ${timeout}s 内未就绪"
}

# 冒烟:经端点发一条 greedy 请求,打印前若干字符。返回非 0 表示失败。
smoke_remote() {
  local port="$1"
  log "冒烟请求 -> h200-2:$port/v1/completions"
  $EXEC_SSH "curl -s --max-time 180 http://127.0.0.1:$port/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"$MODEL_PATH\",\"prompt\":\"The quick brown fox\",\"max_tokens\":16,\"temperature\":0}'" \
    | head -c 400
  echo
}
