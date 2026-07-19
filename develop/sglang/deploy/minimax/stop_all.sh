#!/usr/bin/env bash
# MiniMax-M2.5 stop all containers (baseline + PD + bench)
set -uo pipefail
docker update --restart=no minimax-baseline mm-router mm-decode mm-prefill mm-bench 2>/dev/null
docker rm -f minimax-baseline mm-router mm-decode mm-prefill mm-bench 2>/dev/null
echo "all minimax containers stopped"
