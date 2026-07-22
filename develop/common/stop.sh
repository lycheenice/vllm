#!/usr/bin/env bash
# develop/common/stop.sh — 停某实验的全部容器。★在 h200-2 本机执行★。
# 用法: bash stop.sh <EXP_NAME>
set -uo pipefail
EXP_NAME="${1:?用法: stop.sh <EXP_NAME>}"
for tag in decode prefill single; do
  name="${EXP_NAME}-${tag}"
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    echo "停止并删除 $name"
    docker exec "$name" pkill -f "proxy_server.py" 2>/dev/null || true
    docker rm -f "$name" >/dev/null 2>&1 || true
  fi
done
echo "已清理 $EXP_NAME。验证: docker ps ; nvidia-smi"
