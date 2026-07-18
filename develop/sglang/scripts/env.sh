#!/usr/bin/env bash
# Common environment for sglang single-node 4+4 PD disaggregation.
# Sourced by other scripts:  source "$(dirname "$0")/env.sh" [strategy] [transport]
set -u

# --- Paths (scripts live in develop/sglang/scripts/, sglang dev root one level up) ---
SCRIPT_DIR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SGLANG_DEV_ROOT="$(cd "$SCRIPT_DIR_ENV/.." && pwd -P)"
LOG_DIR="$SGLANG_DEV_ROOT/logs"
PID_DIR="$SGLANG_DEV_ROOT/run/pids"
VENV_DIR="$SGLANG_DEV_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"
mkdir -p "$LOG_DIR" "$PID_DIR"

# --- Model (blocked: /data1/GLM-5.2-W4AFP8 is 640 root:root until root chmod) ---
MODEL_PATH="${MODEL_PATH:-/data1/GLM-5.2-W4AFP8}"

# --- Ports ---
PORT_P="${PORT_P:-30000}"          # Prefill launch_server
PORT_D="${PORT_D:-30001}"          # Decode  launch_server
ROUTER_PORT="${ROUTER_PORT:-8000}" # sglang_router (对外)
BASELINE_PORT="${BASELINE_PORT:-30002}" # 单实例 TP8 基线

# --- GPU partition (4+4 on a single 8-GPU H200 node) ---
P_GPUS="${P_GPUS:-0,1,2,3}"
D_GPUS="${D_GPUS:-4,5,6,7}"

# --- Scheduling ---
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
UTIL="${UTIL:-0.90}"

# --- Strategy + Transport (first two positional args) ---
STRATEGY="${1:-${STRATEGY:-tp4dp2}}"
TRANSPORT="${2:-${TRANSPORT:-nixl}}"

# Strategy -> P/D TP/DP, dp-attention, staging, moe-a2a wish
case "$STRATEGY" in
  tp4tp4) P_TP=4; P_DP=1; D_TP=4; D_DP=1; DP_ATTN=0; STAGING=0; MOE_A2A_WISH=0 ;;
  tp4dp2) P_TP=4; P_DP=1; D_TP=2; D_DP=2; DP_ATTN=1; STAGING=1; MOE_A2A_WISH=1 ;;
  tp4dp4) P_TP=4; P_DP=1; D_TP=1; D_DP=4; DP_ATTN=1; STAGING=1; MOE_A2A_WISH=1 ;;
  tp8)    P_TP=8; P_DP=1; D_TP=0; D_DP=0; DP_ATTN=0; STAGING=0; MOE_A2A_WISH=0 ;;
  *)
    echo "ERROR: unknown STRATEGY='$STRATEGY' (expected: tp4tp4|tp4dp2|tp4dp4|tp8)" >&2
    exit 1 ;;
esac

case "$TRANSPORT" in
  nixl)
    DISAGG_BACKEND="nixl"
    export SGLANG_DISAGGREGATION_NIXL_BACKEND="${SGLANG_DISAGGREGATION_NIXL_BACKEND:-UCX}" ;;
  mooncake)
    DISAGG_BACKEND="mooncake"
    export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
    export MC_INTRANODE_NVLINK=true ;;
  *)
    echo "ERROR: unknown TRANSPORT='$TRANSPORT' (expected: nixl|mooncake)" >&2
    exit 1 ;;
esac

# --- MOE a2a: only emit --moe-a2a-backend when wish AND confirmed enabled ---
# Flip MOE_A2A_ENABLED=1 only after config.json confirms GLM-5.2 is MoE.
MOE_A2A_ENABLED="${MOE_A2A_ENABLED:-0}"
if [[ "$MOE_A2A_WISH" == "1" && "$MOE_A2A_ENABLED" == "1" ]]; then
  MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-deepep}"
else
  MOE_A2A_BACKEND=""
fi

# --- Staging buffer env (only takes effect for non-MLA; auto-bypass on homogeneous TP) ---
if [[ "$STAGING" == "1" ]]; then
  export SGLANG_DISAGG_STAGING_BUFFER=1
  export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB="${SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB:-128}"
  export SGLANG_DISAGG_STAGING_POOL_SIZE_MB="${SGLANG_DISAGG_STAGING_POOL_SIZE_MB:-8192}"
else
  export SGLANG_DISAGG_STAGING_BUFFER=0
fi

# --- Disaggregation tunables (shared by nixl/mooncake) ---
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-300}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-300}"
export SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL="${SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL:-5}"
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE="${SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE:-2}"

# --- Quantization args: fill after config.json confirmation ---
# Example: QUANT=(--quantization w4afp8 --kv-cache-dtype fp8)
QUANT=()

# --- Common launch_server args shared by P and D ---
COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --host 127.0.0.1
  --max-model-len "$MAX_MODEL_LEN"
  --mem-fraction-static "$UTIL"
)

export MODEL_PATH LOG_DIR PID_DIR VENV_DIR PYTHON
export PORT_P PORT_D ROUTER_PORT BASELINE_PORT P_GPUS D_GPUS
export STRATEGY TRANSPORT DISAGG_BACKEND
export P_TP P_DP D_TP D_DP DP_ATTN STAGING MOE_A2A_BACKEND
export MAX_MODEL_LEN UTIL COMMON_ARGS QUANT
