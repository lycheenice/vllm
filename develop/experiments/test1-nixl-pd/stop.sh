#!/usr/bin/env bash
# 停服务。★在 h200-2 本机执行★:  bash stop.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../../common/common.env"; source "$HERE/config.env"
bash "$COMMON_DIR/stop.sh" "$EXP_NAME"
