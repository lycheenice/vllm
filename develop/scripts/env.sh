#!/usr/bin/env bash
# Common environment for H200 single-node 4+4 NIXL P/D disaggregated validation.
# Sourced by other scripts:  source "$(dirname "$0")/env.sh" [gdr|cpu]

set -u

# --- Paths (scripts live in develop/scripts/, dev root is one level up) ---
SCRIPT_DIR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEV_ROOT="$(cd "$SCRIPT_DIR_ENV/.." && pwd -P)"
VLLM_ROOT="$(cd "$DEV_ROOT/.." && pwd -P)"
LOG_DIR="$DEV_ROOT/logs"
RUN_DIR="$DEV_ROOT/run"
PID_DIR="$RUN_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

# --- Model (downloaded by scripts/dl_m25.sh into /data1/models/MiniMax-M2.5) ---
MODEL_PATH="${MODEL_PATH:-/data1/models/MiniMax-M2.5}"

# --- Ports ---
PORT_P="${PORT_P:-8100}"           # Prefill vLLM OpenAI server
PORT_D="${PORT_D:-8200}"           # Decode  vLLM OpenAI server
PROXY_PORT="${PROXY_PORT:-8000}"   # toy_proxy_server (toy_proxy_server.py:92 default 8000)
SIDE_PORT_P="${SIDE_PORT_P:-5600}" # NIXL side channel, prefill engine
SIDE_PORT_D="${SIDE_PORT_D:-5601}" # NIXL side channel, decode  engine
BASELINE_PORT="${BASELINE_PORT:-8300}"  # standalone baseline for correctness diff

# --- GPU partition (4+4 on a single 8-GPU H200 node) ---
P_GPUS="${P_GPUS:-0,1,2,3}"
D_GPUS="${D_GPUS:-4,5,6,7}"
TP="${TP:-4}"

# --- Scheduling ---
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
UTIL="${UTIL:-0.90}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"   # 1 => --enforce-eager, 0 => off

# --- Transport mapping: gdr (GPU direct) vs cpu (D2H/H2D staged) ---
# First positional arg wins; otherwise env TRANSPORT; default gdr.
TRANSPORT="${1:-${TRANSPORT:-gdr}}"
case "$TRANSPORT" in
  gdr)
    KV_BUFFER_DEVICE="cuda"
    UCX_TLS="cuda_ipc,cuda_copy,tcp"
    ;;
  cpu)
    KV_BUFFER_DEVICE="cpu"
    UCX_TLS="cuda_ipc,cuda_copy,tcp"
    ;;
  *)
    echo "ERROR: unknown TRANSPORT='$TRANSPORT' (expected: gdr|cpu)" >&2
    exit 1
    ;;
esac

export UCX_TLS
export MODEL_PATH VLLM_ROOT DEV_ROOT LOG_DIR RUN_DIR PID_DIR
export PORT_P PORT_D PROXY_PORT SIDE_PORT_P SIDE_PORT_D BASELINE_PORT
export P_GPUS D_GPUS TP MAX_MODEL_LEN UTIL BLOCK_SIZE ENFORCE_EAGER TRANSPORT KV_BUFFER_DEVICE
