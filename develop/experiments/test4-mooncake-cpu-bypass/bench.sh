#!/usr/bin/env bash
# 发压。★从 a100-2 / 监控端执行★(内部 ssh h200-6 跑 ramp,结果回拉本目录 results/):
#   bash bench.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../../common/common.env"
source "$HERE/config.env"
source "$COMMON_DIR/bench_ramp.sh"
