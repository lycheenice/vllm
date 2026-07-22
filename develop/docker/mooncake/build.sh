#!/usr/bin/env bash
# 构建 mooncake 派生镜像。★在 h200-6 执行★,★纯 CPU★(docker build 不加 --gpus)。
# 用法: bash build.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TAG="${TAG:-docker.1ms.run/vllm/vllm-openai:v0.25.0-mooncake}"

echo "[build] $TAG  (CPU-only, 不占 GPU)"
docker build -t "$TAG" "$HERE"

echo "[build] 完成。运行时验证(不加 --gpus,纯 CPU):"
echo "  docker run --rm --network host $TAG python -c 'import mooncake.engine; print(\"mooncake.engine OK\")'"
