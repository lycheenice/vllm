#!/usr/bin/env bash
# Watch aria2 download progress for MiniMax-M2.5.
# Usage: watch_dl.sh [interval_sec]
set -uo pipefail
DEV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LOG="$DEV_ROOT/logs/aria2_m25.log"
STDOUT="$DEV_ROOT/logs/aria2_m25.stdout"
DEST="/data1/models/MiniMax-M2.5"
INTERVAL="${1:-5}"

PIDS="$(pgrep -f 'aria2c.*minimax_m2.5_aria2.list' 2>/dev/null || true)"
if [[ -z "$PIDS" ]]; then
  echo "aria2c for MiniMax-M2.5 is NOT running."
  echo "Start it with:"
  echo "  setsid nohup $DEV_ROOT/scripts/dl_m25.sh > \"$STDOUT\" 2>&1 < /dev/null &"
  exit 0
fi

echo "Watching aria2 (pid=$(echo "$PIDS" | head -1)). Ctrl+C to stop. interval=${INTERVAL}s"
while true; do
  clear
  echo "=== $(date) ==="
  echo "--- downloaded files under $DEST ---"
  ls -lh "$DEST" 2>/dev/null | tail -n 25 || echo "(destination not yet created)"
  echo "--- last 8 lines of aria2 log ($LOG) ---"
  tail -n 8 "$LOG" 2>/dev/null || echo "(no log yet)"
  sleep "$INTERVAL"
done
