#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../../common/common.env"; source "$HERE/config.env"; bash "$COMMON_DIR/stop.sh" "$EXP_NAME"
