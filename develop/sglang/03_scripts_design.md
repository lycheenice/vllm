# sglang 4+4 PD 分离脚本设计

> 配套文档：`01_sglang_pd_research.md`（机制）、`02_h200_glm_pd_experiment_design.md`（实验矩阵）。
> 本文给出 `develop/sglang/scripts/` 下各脚本的设计与可直接复制的 bash 代码块。
> 脚本与 `develop/scripts/`（vllm 那套）独立，不混用；sglang 与 vllm 是不同框架，本分支不放 vllm 代码。
> 凡依赖 GLM-5.2 架构确认的参数（`--quantization` / `--moe-a2a-backend` / `--max-model-len` 上限）以变量预留，待 root 授权读取 `config.json` 后回填。

## 目录

- [1. 概述](#1-概述)
- [2. 脚本清单](#2-脚本清单)
- [3. 各脚本设计](#3-各脚本设计)
  - [3.1 env.sh](#31-envsh)
  - [3.2 launch_prefill.sh](#32-launch_prefillsh)
  - [3.3 launch_decode.sh](#33-launch_decodesh)
  - [3.4 launch_router.sh](#34-launch_routersh)
  - [3.5 run_pd.sh](#35-run_pdsh)
  - [3.6 stop_pd.sh](#36-stop_pdsh)
  - [3.7 bench.sh](#37-benchsh)
  - [3.8 correct_check.sh](#38-correct_checksh)
  - [3.9 status.sh](#39-statussh)
  - [3.10 install_sglang.sh](#310-install_sglangsh)
- [4. 参数映射表（strategy -> 启动参数）](#4-参数映射表strategy---启动参数)
- [5. 配置参考：完整启动命令示例](#5-配置参考完整启动命令示例)
- [6. 健康检查与日志关键字](#6-健康检查与日志关键字)
- [7. 与 vllm scripts 的差异说明](#7-与-vllm-scripts-的差异说明)

---

## 1. 概述

- 脚本目录：`develop/sglang/scripts/`，与 `develop/scripts/`（vllm）并列、互不依赖。
- 运行目录：`develop/sglang/run/pids/`（PID）、`develop/sglang/logs/`（日志）。
- venv：`develop/sglang/.venv/`，由 `install_sglang.sh` 创建。
- 参数化两维：
  - `strategy` ∈ `{tp4tp4, tp4dp2, tp4dp4, tp8}`，决定 P/D 的 TP/DP/DP-attention/staging 组合。
  - `transport` ∈ `{nixl, mooncake}`，决定 `--disaggregation-transfer-backend` 与对应 env。
- 默认 `strategy=tp4dp2 transport=nixl`（TP4DPA2 主路径）。
- 用法约定：`./<script>.sh [strategy] [transport]`，env.sh 接收前两个位置参并导出。

---

## 2. 脚本清单

| 脚本 | 作用 | 关键依赖 |
| - | - | - |
| `env.sh` | 路径/端口/GPU/strategy/transport/sglang env 统一导出 | 被其它脚本 source |
| `launch_prefill.sh` | 启动 P 实例（port 30000） | env.sh |
| `launch_decode.sh` | 启动 D 实例（port 30001, GPU 4-7） | env.sh |
| `launch_router.sh` | 启动 sglang_router（port 8000） | env.sh |
| `run_pd.sh` | 一键 P -> 健康检查 -> D -> 健康检查 -> router | 上述四个 |
| `stop_pd.sh` | 按 PID 停 P/D/router | run/pids/ |
| `bench.sh` | 跑 sglang bench_serving | env.sh, router 已起 |
| `correct_check.sh` | 正确性比对（router vs 基线） | router 已起 |
| `status.sh` | 查进程/端口/日志尾部/显存 | run/pids/, logs/ |
| `install_sglang.sh` | 在 develop/sglang/ 建 venv 装 sglang | 需网络 |

---

## 3. 各脚本设计

### 3.1 env.sh

统一环境入口。接收 `[strategy] [transport]` 两个位置参，导出所有变量。

```bash
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
```

> 说明：`--mem-fraction-static` 是 sglang 的显存占用比例参数（对应 vllm 的 `--gpu-memory-utilization`），确切参数名以安装后 `python -m sglang.launch_server --help` 为准，必要时回填。`QUANT` 数组留空，待 root 授权后从 `config.json` / README / sglang 支持列表确定 `--quantization` 名后填入。

### 3.2 launch_prefill.sh

启动 P 实例。P 侧始终为纯 TP，不开 DP attention。

```bash
#!/usr/bin/env bash
# Launch the sglang Prefill instance.
# Usage: launch_prefill.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

[[ "$STRATEGY" == "tp8" ]] && { echo "STRATEGY=tp8 is baseline, no prefill"; exit 0; }

echo "[$(date +%H:%M:%S)] Starting Prefill: GPUs=$P_GPUS port=$PORT_P strategy=$STRATEGY transport=$TRANSPORT tp=$P_TP"

CUDA_VISIBLE_DEVICES="$P_GPUS" \
exec "$PYTHON" -m sglang.launch_server \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend "$DISAGG_BACKEND" \
  --tp-size "$P_TP" \
  --port "$PORT_P" \
  "${COMMON_ARGS[@]}" \
  ${QUANT[@]+"${QUANT[@]}"} \
  > "$LOG_DIR/prefill.log" 2>&1 &
echo $! > "$PID_DIR/prefill.pid"
echo "Prefill PID=$(cat "$PID_DIR/prefill.pid"), log=$LOG_DIR/prefill.log"
```

### 3.3 launch_decode.sh

启动 D 实例。按 strategy 决定 `--tp-size` / `--dp-size` / `--enable-dp-attention` / `--moe-a2a-backend`。

```bash
#!/usr/bin/env bash
# Launch the sglang Decode instance.
# Usage: launch_decode.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

[[ "$STRATEGY" == "tp8" ]] && { echo "STRATEGY=tp8 is baseline, no decode"; exit 0; }

D_ARGS=(--disaggregation-mode decode --disaggregation-transfer-backend "$DISAGG_BACKEND"
  --tp-size "$D_TP" --port "$PORT_D")

[[ "$DP_ATTN" == "1" ]] && D_ARGS+=(--dp-size "$D_DP" --enable-dp-attention)
[[ -n "$MOE_A2A_BACKEND" ]] && D_ARGS+=(--moe-a2a-backend "$MOE_A2A_BACKEND")

echo "[$(date +%H:%M:%S)] Starting Decode: GPUs=$D_GPUS port=$PORT_D strategy=$STRATEGY transport=$TRANSPORT tp=$D_TP dp=$D_DP dp_atten=$DP_ATTN moe_a2a=${MOE_A2A_BACKEND:-none} staging=$STAGING"

CUDA_VISIBLE_DEVICES="$D_GPUS" \
exec "$PYTHON" -m sglang.launch_server \
  "${D_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  ${QUANT[@]+"${QUANT[@]}"} \
  > "$LOG_DIR/decode.log" 2>&1 &
echo $! > "$PID_DIR/decode.pid"
echo "Decode PID=$(cat "$PID_DIR/decode.pid"), log=$LOG_DIR/decode.log"
```

> `--base-gpu-id` 不在此设：已用 `CUDA_VISIBLE_DEVICES=4,5,6,7` 把 D 的可见集裁到 4 张卡，`launch_server` 在该可见集内从 0 起编号即可。若改用 `--base-gpu-id 4` 方式，则不要同时设 `CUDA_VISIBLE_DEVICES`。

### 3.4 launch_router.sh

启动 sglang_router，串联 P 与 D。

```bash
#!/usr/bin/env bash
# Launch the sglang PD router.
# Usage: launch_router.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

[[ "$STRATEGY" == "tp8" ]] && { echo "STRATEGY=tp8 is baseline, no router"; exit 0; }

echo "[$(date +%H:%M:%S)] Starting router: port=$ROUTER_PORT -> prefill=$PORT_P decode=$PORT_D"

exec "$PYTHON" -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:$PORT_P" \
  --decode  "http://127.0.0.1:$PORT_D" \
  --host 0.0.0.0 --port "$ROUTER_PORT" \
  > "$LOG_DIR/router.log" 2>&1 &
echo $! > "$PID_DIR/router.pid"
echo "Router PID=$(cat "$PID_DIR/router.pid"), log=$LOG_DIR/router.log"
```

### 3.5 run_pd.sh

一键拉起：P -> 等 health -> D -> 等 health -> router -> 等 health。

```bash
#!/usr/bin/env bash
# One-shot: prefill -> health -> decode -> health -> router -> health.
# Usage: run_pd.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
STRATEGY="${1:-tp4dp2}"; TRANSPORT="${2:-nixl}"

wait_health() {
  local url="$1" name="$2" tries="${3:-60}"
  for i in $(seq 1 "$tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "[ok] $name healthy ($i tries)"; return 0
    fi
    sleep 2
  done
  echo "[FAIL] $name not healthy after $tries tries"; return 1
}

"$SCRIPT_DIR/launch_prefill.sh" "$STRATEGY" "$TRANSPORT"
wait_health "http://127.0.0.1:30000/health" prefill 90 || exit 1

"$SCRIPT_DIR/launch_decode.sh" "$STRATEGY" "$TRANSPORT"
wait_health "http://127.0.0.1:30001/health" decode 90 || exit 1

"$SCRIPT_DIR/launch_router.sh" "$STRATEGY" "$TRANSPORT"
wait_health "http://127.0.0.1:8000/health" router 60 || exit 1

echo "PD stack up: router http://127.0.0.1:8000 (strategy=$STRATEGY transport=$TRANSPORT)"
```

> health 端点 `/health` 以安装后 sglang 版本为准；若 router 健康端点不同（如 `/health_ready`），回填此处。

### 3.6 stop_pd.sh

按 PID 停 P/D/router。

```bash
#!/usr/bin/env bash
# Stop prefill / decode / router by PID files.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

kill_pid() {
  local f="$1" name="$2"
  if [[ -f "$f" ]]; then
    local pid; pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "[stop] $name pid=$pid"
    fi
    rm -f "$f"
  fi
}

kill_pid "$PID_DIR/router.pid"  router
kill_pid "$PID_DIR/decode.pid"  decode
kill_pid "$PID_DIR/prefill.pid" prefill
echo "stopped (strategy=$STRATEGY transport=$TRANSPORT)"
```

### 3.7 bench.sh

跑 sglang 自带 bench_serving，指标口径尽量对齐 vllm bench_serve 以便横向对比。

```bash
#!/usr/bin/env bash
# Benchmark via router (PD) or baseline (tp8).
# Usage: bench.sh [strategy] [transport] [--baseline]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

if [[ "${3:-}" == "--baseline" ]]; then
  URL="http://127.0.0.1:$BASELINE_PORT"
  TAG="baseline-tp8"
else
  URL="http://127.0.0.1:$ROUTER_PORT"
  TAG="pd-$STRATEGY-$TRANSPORT"
fi

OUT="$LOG_DIR/bench-${TAG}-$(date +%H%M%S).json"
echo "[bench] url=$URL tag=$TAG out=$OUT"

# sglang bench_serving 参数以安装后 --help 为准；下列为常见字段。
"$PYTHON" -m sglang.bench.serving \
  --url "$URL" \
  --model "$MODEL_PATH" \
  --num-prompts "${NUM_PROMPTS:-256}" \
  --request-rate "${REQUEST_RATE:-INF}" \
  > "$OUT" 2>&1 || echo "[bench] non-zero exit, see $OUT"

echo "bench done: $OUT"
```

> bench 子参数（如 `--payload-file` / `--dataset-name`）待 sglang 版本确认后回填；`NUM_PROMPTS` / `REQUEST_RATE` 通过环境变量调。若需与 vllm bench_serve 同口径，可改用 vllm 的 `benchmark_serving.py` 打 `--url`，但本分支优先用 sglang 自带 bench。

### 3.8 correct_check.sh

正确性比对：同一 prompt 经 router（PD）与基线（TP8）输出一致（greedy）。payload 写到临时文件以避免 shell 引号转义。

```bash
#!/usr/bin/env bash
# Correctness diff: PD (router) vs baseline (tp8).
# Usage: correct_check.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

PROMPT="${PROMPT:-用一句话解释什么是 Prefill/Decode 分离。}"
TMP="$LOG_DIR/correct_$$"
PAYLOAD="$TMP.payload"

# 用 python 生成合法 JSON，避免 shell 转义
"$PYTHON" - "$PROMPT" "$MODEL_PATH" "$PAYLOAD" <<'PYEOF'
import json, sys
prompt, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
with open(out, "w") as f:
    json.dump({"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 64}, f, ensure_ascii=False)
PYEOF

call_chat() {  # url out
  curl -sf "$1/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary "@$PAYLOAD" \
    | "$PYTHON" -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' > "$2"
}

call_chat "http://127.0.0.1:$ROUTER_PORT"   "$TMP.pd"
call_chat "http://127.0.0.1:$BASELINE_PORT" "$TMP.base"

echo "--- PD ---";  cat "$TMP.pd"
echo "--- BASE ---"; cat "$TMP.base"
if diff -u "$TMP.base" "$TMP.pd" > "$TMP.diff"; then
  echo "[ok] PD == BASELINE (greedy)"
else
  echo "[warn] diff (容许量化精度差异):"; cat "$TMP.diff"
fi
rm -f "$TMP.pd" "$TMP.base" "$TMP.diff" "$PAYLOAD"
```

> 基线 TP8 需另起（手动 `python -m sglang.launch_server --tp-size 8 --port 30002 ...`），见第 5 节配置参考。W4AFP8 量化下 PD 与基线应同量化口径；若基线换精度，diff 仅供参考。

### 3.9 status.sh

查看进程、端口、日志尾部、显存。

```bash
#!/usr/bin/env bash
# Quick status of the sglang PD stack.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

echo "=== strategy=$STRATEGY transport=$TRANSPORT ==="
for name in prefill decode router; do
  f="$PID_DIR/$name.pid"
  if [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; then
    echo "[up]   $name pid=$(cat "$f")"
  else
    echo "[down] $name"
  fi
done

echo "=== ports ==="
for p in "$PORT_P" "$PORT_D" "$ROUTER_PORT" "$BASELINE_PORT"; do
  if curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1; then
    echo "[up] port $p"
  else
    echo "[--] port $p"
  fi
done

echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true

echo "=== log tails ==="
for f in prefill decode router; do
  log="$LOG_DIR/$f.log"
  [[ -f "$log" ]] && { echo "--- $f.log ---"; tail -n 5 "$log"; }
done
```

### 3.10 install_sglang.sh

在 `develop/sglang/` 建 venv 并装 sglang + nixl + mooncake。**需网络。** 不动系统 python。

```bash
#!/usr/bin/env bash
# Create venv under develop/sglang/.venv and install sglang + transfer engines.
# Requires network. Does NOT touch system python.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SGLANG_DEV_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
VENV_DIR="$SGLANG_DEV_ROOT/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "[install] creating venv at $VENV_DIR"
uv venv --python 3.12 "$VENV_DIR"
PY="$VENV_DIR/bin/python"

echo "[install] sglang[all] + nixl + mooncake-transfer-engine (network required)"
uv pip install --python "$PY" "sglang[all]"
uv pip install --python "$PY" nixl mooncake-transfer-engine

echo "[install] versions:"
"$PY" -m sglang.launch_server --version || true
"$PY" -c "import nixl; print('nixl ok', nixl.__file__)" || echo "nixl import failed (may need source build with ucx_path)"
"$PY" -c "import mooncake; print('mooncake ok')" || echo "mooncake import failed"

echo "[install] done. activate: source $VENV_DIR/bin/activate"
```

> 若 `nixl` pip 包在目标平台不可用，需从源码编译（带 `ucx_path`）。`sglang[all]` 的具体 extra 以 PyPI 版本为准。本脚本只装软件，不修改 vllm 代码。

---

## 4. 参数映射表（strategy -> 启动参数）

| strategy | P `--tp-size` | P `--enable-dp-attention` | D `--tp-size` | D `--dp-size` | D `--enable-dp-attention` | `--moe-a2a-backend` | staging env | 备注 |
| - | - | - | - | - | - | - | - | - |
| tp4tp4 | 4 | 否 | 4 | - | 否 | -（待 MoE 确认） | STAGING=0（bypass） | 同构对照 |
| tp4dp2 | 4 | 否 | 2 | 2 | 是 | deepep（`MOE_A2A_ENABLED=1` 时） | STAGING=1 | TP4DPA2 主路径 |
| tp4dp4 | 4 | 否 | 1 | 4 | 是 | deepep（同上） | STAGING=1 | 异构 4x |
| tp8 | 8 | 否 | - | - | - | - | - | 单实例基线，无 PD |

P/D 共用 `--disaggregation-transfer-backend <nixl|mooncake>`、`--model-path`、`--trust-remote-code`、`--max-model-len`、`--mem-fraction-static`、`QUANT`（待回填）。

transport -> env 映射：

| transport | `--disaggregation-transfer-backend` | 额外 env |
| - | - | - |
| nixl | nixl | `SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX` |
| mooncake | mooncake | `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK` + `MC_INTRANODE_NVLINK=true` |

---

## 5. 配置参考：完整启动命令示例

> 以下命令均假设已在 `develop/sglang/` 下 `source .venv/bin/activate`，且 GLM-5.2 已可读。`QUANT` 待回填，示例中省略量化参数。

### 5.1 strategy=tp4tp4 + transport=nixl（同构对照 C）

P：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 SGLANG_DISAGG_STAGING_BUFFER=0 \
python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-W4AFP8 --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --tp-size 4 --port 30000 --host 127.0.0.1 \
  --max-model-len 32768 --mem-fraction-static 0.90
```

D：
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_DISAGG_STAGING_BUFFER=0 \
python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-W4AFP8 --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --tp-size 4 --port 30001 --host 127.0.0.1 \
  --max-model-len 32768 --mem-fraction-static 0.90
```

router：
```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

### 5.2 strategy=tp4dp2 + transport=nixl（TP4DPA2 主路径 A）

> 前提：GLM-5.2 为 non-MLA 且 MoE（`MOE_A2A_ENABLED=1`）。若非 MoE，去掉 `--moe-a2a-backend deepep` 并确认 sglang 允许 dense 模型开 `--enable-dp-attention`。

P（不变，TP4）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_DISAGG_STAGING_BUFFER=1 \
SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=128 \
python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-W4AFP8 --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --tp-size 4 --port 30000 --host 127.0.0.1 \
  --max-model-len 32768 --mem-fraction-static 0.90
```

D（TP2×DP2 + DP attention + staging）：
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
SGLANG_DISAGG_STAGING_BUFFER=1 \
SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=128 \
SGLANG_DISAGG_STAGING_POOL_SIZE_MB=8192 \
python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-W4AFP8 --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --tp-size 2 --dp-size 2 --enable-dp-attention \
  --moe-a2a-backend deepep \
  --port 30001 --host 127.0.0.1 \
  --max-model-len 32768 --mem-fraction-static 0.90
```

router 同 5.1。

### 5.3 strategy=tp4dp2 + transport=mooncake（对照 B）

P 与 D 命令同 5.2，但去掉 NIXL/staging env 行，替换 transport 与 Mooncake NVLink env：

```bash
# env 通用前缀（P 与 D 都加）
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export MC_INTRANODE_NVLINK=true
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=128
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=8192  # D 侧

# P
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-W4AFP8 --trust-remote-code \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --tp-size 4 --port 30000 --host 127.0.0.1 \
  --max-model-len 32768 --mem-fraction-static 0.90

# D
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-W4AFP8 --trust-remote-code \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tp-size 2 --dp-size 2 --enable-dp-attention --moe-a2a-backend deepep \
  --port 30001 --host 127.0.0.1 \
  --max-model-len 32768 --mem-fraction-static 0.90
```

> IB 设备参数 `--disaggregation-ib-device mlx5_0` 仅多机/IB 场景需要，单机 NVLink 用上面 env 即可，故本例省略。

### 5.4 基线 tp8（无 PD）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-W4AFP8 --trust-remote-code \
  --tp-size 8 --port 30002 --host 127.0.0.1 \
  --max-model-len 32768 --mem-fraction-static 0.90
```

---

## 6. 健康检查与日志关键字

### 6.1 健康检查端点

| 组件 | URL | 期望 |
| - | - | - |
| Prefill | `http://127.0.0.1:30000/health` | 200 |
| Decode | `http://127.0.0.1:30001/health` | 200 |
| router | `http://127.0.0.1:8000/health` | 200（端点名待 sglang 版本确认，可能为 `/health_ready`） |

### 6.2 日志关键字

| 日志 | 关键字（grep） | 含义 |
| - | - | - |
| `prefill.log` | `The server is fired up` / `Application startup complete` | P 就绪 |
| `prefill.log` | `disaggregation` / `prefill mode` | 进入 PD prefill 角色 |
| `decode.log` | `decode mode` / `bootstrap` / `recv` | D 进入 decode 角色、KV 接收链路建立 |
| `decode.log` | `heartbeat` / `waiting` | 心跳与等待 KV（见 `WAITING_TIMEOUT`） |
| `router.log` | `prefill` + `decode` + `up` | router 发现 P/D 在线 |
| 各日志 | `staging buffer` / `staging pool` | staging 启用日志（仅非同构 + non-MLA 时生效） |
| 各日志 | `error` / `Traceback` / `OOM` / `CUDA error` | 故障 |

### 6.3 常见问题排查

- D 侧一直 `waiting`：核对 P 是否就绪、`BOOTSTRAP_TIMEOUT` 是否够、transport env 是否对。
- staging 未生效：确认 strategy 为异构（tp4dp2/tp4dp4）、`SGLANG_DISAGG_STAGING_BUFFER=1`、且模型为 non-MLA（MLA 下 staging 被禁用）。
- OOM：降 `--mem-fraction-static` 或 `--max-model-len`；D_TP1×DP4 单卡 KV 全量时优先回退 tp4dp2。
- `--moe-a2a-backend deepep` 报错：GLM-5.2 可能非 MoE，设 `MOE_A2A_ENABLED=0` 或去掉该参数。

---

## 7. 与 vllm scripts 的差异说明

| 维度 | `develop/scripts/`（vllm） | `develop/sglang/scripts/`（本套） |
| - | - | - |
| 启动命令 | `vllm serve` | `python -m sglang.launch_server` |
| 角色声明 | `--kv-transfer-config` JSON（`kv_role`） | `--disaggregation-mode prefill\|decode` |
| 后端选择 | JSON `kv_connector` 名（NixlConnector / MooncakeConnector） | `--disaggregation-transfer-backend nixl\|mooncake` |
| 编排 | 外部 `toy_proxy_server`（launch_proxy.sh） | 内置 `python -m sglang_router.launch_router --pd-disaggregation` |
| CPU 转发 | `kv_buffer_device=cpu/gdr`（env.sh TRANSPORT 映射） | 无对应（sglang NIXL 走 UCX/LIBFABRIC RDMA，无 host buffer 文档） |
| side channel | `VLLM_NIXL_SIDE_CHANNEL_PORT`（5600/5601） | 无独立 side channel 端口，后端自管 |
| 异构 TP | `compute_tp_mapping` 自动按头切分（含 MLA） | `SGLANG_DISAGG_STAGING_BUFFER` 显式开关（仅 non-MLA） |
| DP attention | 不在本分支 vllm 方案内 | `--enable-dp-attention --dp-size --moe-a2a-backend` |
| 端口 | P 8100 / D 8200 / proxy 8000 | P 30000 / D 30001 / router 8000 / baseline 30002 |
| venv | vllm 主 venv | `develop/sglang/.venv`（install_sglang.sh 独立建） |

> 两套脚本完全独立、目录分离，不互相 source、不共享 env。本分支只探索 sglang，不修改 vllm 源码也不复用其脚本。

