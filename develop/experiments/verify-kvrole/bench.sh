#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../../common/common.env"; source "$HERE/config.env"; source "$COMMON_DIR/bench_ramp.sh"
