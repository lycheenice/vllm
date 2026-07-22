#!/usr/bin/env bash
# 起服务。★在 h200-2 本机执行★:  bash run.sh
# 或从 a100-2:  ssh -l root h200-2 "bash <此脚本绝对路径>"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../../common/common.env"
source "$HERE/config.env"
source "$COMMON_DIR/serve_pd.sh"
