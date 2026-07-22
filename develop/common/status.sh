#!/usr/bin/env bash
# develop/common/status.sh — 状态快照。★在 h200-2 本机执行★。
# 用法: bash status.sh <EXP_NAME> [port1 port2 ...]
set -uo pipefail
EXP_NAME="${1:?用法: status.sh <EXP_NAME> [ports...]}"; shift || true
echo "=== 容器 ($EXP_NAME) ==="
docker ps --filter "name=$EXP_NAME" --format '  {{.Names}}\t{{.Status}}' || true
echo "=== Health ==="
for p in "$@"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$p/health" 2>/dev/null || echo 000)
  printf "  :%-6s HTTP %s\n" "$p" "$code"
done
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
