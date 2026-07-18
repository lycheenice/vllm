#!/usr/bin/env bash
# Download MiniMax-M2.5 from hf-mirror (resume-capable).
# Run: setsid nohup develop/scripts/dl_m25.sh > develop/logs/aria2_m25.stdout 2>&1 < /dev/null &
set -u
cd /home/lychee/mycode/vllm
exec aria2c \
  --input-file=develop/minimax_m2.5_aria2.list \
  --continue=true \
  --max-connection-per-server=16 \
  --split=16 \
  --min-split-size=10M \
  --max-concurrent-downloads=6 \
  --file-allocation=none \
  --console-log-level=warn \
  --summary-interval=60 \
  --download-result=hide \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --log=develop/logs/aria2_m25.log \
  --log-level=warn
