# sglang 4+4 PD 分离脚本设计（同构 TP4+4）

> 配套文档：`01_sglang_pd_research.md`（机制）、`02_h200_glm_pd_experiment_design.md`（实验矩阵）。
> 本文给出 `develop/sglang/scripts/` 下各脚本的设计与可直接复制的 bash 代码块。
> 策略已收敛为 **TP4+4 同构 PD 分离**：GLM-5.2 是 MLA（`kv_lora_rank=512`），GPU Staging Buffer 不可用，异构 TP（tp4dp2/tp4dp4）已移除，`STAGING=0`/`DP_ATTN=0` 固定，无 `--moe-a2a-backend`。
> 脚本由 h200-2 实测的 `/opt/sglang-glm/start.sh` 派生：保留 `--served-model-name glm --tool-call-parser glm47 --reasoning-parser glm45 --context-len 300000 --kv-cache-dtype fp8_e4m3` 等业务参数，加 PD 三件套。
> 脚本与 `develop/scripts/`（vllm 那套）独立，不混用；sglang 与 vllm 是不同框架，本分支不放 vllm 代码。

## 目录

- [1. 概述](#1-概述)
- [2. 脚本清单](#2-脚本清单)
- [3. 各脚本设计](#3-各脚本设计)
  - [3.1 env.sh](#31-envsh)
  - [3.2 launch_prefill.sh](#32-launch_prefillsh)
  - [3.3 launch_decode.sh](#33-launch_decodesh)
  - [3.4 launch_router.sh](#34-launch_routersh)
  - [3.5 launch_baseline.sh](#35-launch_baselinesh)
  - [3.6 run_pd.sh](#36-run_pdsh)
  - [3.7 stop_pd.sh](#37-stop_pdsh)
  - [3.8 bench.sh](#38-benchsh)
  - [3.9 correct_check.sh](#39-correct_checksh)
  - [3.10 status.sh](#310-statussh)
  - [3.11 install_sglang.sh](#311-install_sglangsh)
- [4. 参数映射表（strategy -> 启动参数）](#4-参数映射表strategy---启动参数)
- [5. 配置参考：完整启动命令示例](#5-配置参考完整启动命令示例)
- [6. 健康检查与日志关键字](#6-健康检查与日志关键字)
- [7. 容器内执行说明](#7-容器内执行说明)
- [8. 与 vllm scripts 的差异说明](#8-与-vllm-scripts-的差异说明)

---

## 1. 概述

- 脚本目录：`develop/sglang/scripts/`，与 `develop/scripts/`（vllm）并列、互不依赖。
- 运行目录：`develop/sglang/run/pids/`（PID）、`develop/sglang/logs/`（日志）。
- 参数化两维：
  - `strategy` ∈ `{tp4tp4, tp4, tp8}`。`tp4tp4` = PD 同构 TP4+4（主路径，默认）；`tp4`/`tp8` = 单实例基线（无 PD）。异构策略已移除。
  - `transport` ∈ `{nixl, mooncake}`，决定 `--disaggregation-transfer-backend` 与对应 env（基线 strategy 忽略 transport）。
- 默认 `strategy=tp4tp4 transport=nixl`。
- GLM-5.2 MLA 约束内置：`STAGING=0`、`DP_ATTN=0` 固定，无 `--moe-a2a-backend`（同构 TP4 不开 DP attention）。
- 用法约定：`./<script>.sh [strategy] [transport]`，env.sh 接收前两个位置参并导出。
- sglang 跑在容器镜像内（见第 7 节）；`PYTHON` 默认 `python3`（容器内），宿主 venv 用法见 `install_sglang.sh`。

---

## 2. 脚本清单

| 脚本 | 作用 | 关键依赖 |
| - | - | - |
| `env.sh` | 路径/端口/GPU/strategy/transport/sglang env 统一导出 | 被其它脚本 source |
| `launch_prefill.sh` | 启动 P 实例（port 30000，TP4） | env.sh |
| `launch_decode.sh` | 启动 D 实例（port 30001，GPU 4-7，TP4） | env.sh |
| `launch_router.sh` | 启动 sglang_router（port 8000，`--pd-disaggregation`） | env.sh |
| `launch_baseline.sh` | 启动单实例基线（`tp4` 或 `tp8`，port 30002，无 PD） | env.sh |
| `run_pd.sh` | 一键 P -> 健康检查 -> D -> 健康检查 -> router | 上述四个 |
| `stop_pd.sh` | 按 PID 停 P/D/router/baseline | run/pids/ |
| `bench.sh` | 跑 sglang bench_serving（输入 7500/输出 200） | env.sh, router/baseline 已起 |
| `correct_check.sh` | 正确性比对（router vs 基线，greedy） | router + baseline 已起 |
| `status.sh` | 查进程/端口/日志尾部/显存 | run/pids/, logs/ |
| `install_sglang.sh` | 容器内或新 venv 补装 sglang/nixl/mooncake | 需网络（本环境已有容器，通常无需） |

---

## 3. 各脚本设计

### 3.1 env.sh

统一环境入口。接收 `[strategy] [transport]` 两个位置参，导出所有变量。

```bash
#!/usr/bin/env bash
# Common environment for sglang single-node homogeneous TP4+4 PD disaggregation.
# GLM-5.2 is MLA (kv_lora_rank=512) -> staging buffer unavailable, heterogeneous TP abandoned.
# Sourced by other scripts:  source "$(dirname "$0")/env.sh" [strategy] [transport]
set -u

# --- Paths (scripts live in develop/sglang/scripts/, dev root one level up) ---
SCRIPT_DIR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SGLANG_DEV_ROOT="$(cd "$SCRIPT_DIR_ENV/.." && pwd -P)"
LOG_DIR="$SGLANG_DEV_ROOT/logs"
PID_DIR="$SGLANG_DEV_ROOT/run/pids"
# sglang runs in container image by default; override PYTHON=.venv/bin/python on host.
PYTHON="${PYTHON:-python3}"
mkdir -p "$LOG_DIR" "$PID_DIR"

# --- Model (container path; host /data1/GLM-5.2-W4AFP8 maps to this via docker-compose) ---
MODEL_PATH="${MODEL_PATH:-/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8}"

# --- Ports ---
PORT_P="${PORT_P:-30000}"          # Prefill launch_server
PORT_D="${PORT_D:-30001}"          # Decode  launch_server
ROUTER_PORT="${ROUTER_PORT:-8000}" # sglang_router (对外)
BASELINE_PORT="${BASELINE_PORT:-30002}" # 单实例基线 (tp4/tp8)

# --- GPU partition (4+4 on a single 8-GPU H200 node) ---
P_GPUS="${P_GPUS:-0,1,2,3}"
D_GPUS="${D_GPUS:-4,5,6,7}"

# --- Scheduling (from /opt/sglang-glm/start.sh measured on h200-2) ---
CONTEXT_LEN="${CONTEXT_LEN:-300000}"
MEM_FRAC_P="${MEM_FRAC_P:-0.85}"
MEM_FRAC_D="${MEM_FRAC_D:-0.85}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-32768}"
MAX_RUNNING="${MAX_RUNNING:-64}"
MAX_QUEUED="${MAX_QUEUED:-512}"
WATCHDOG="${WATCHDOG:-1800}"
CG_MAX_BS="${CG_MAX_BS:-128}"

# --- Strategy + Transport (first two positional args) ---
STRATEGY="${1:-${STRATEGY:-tp4tp4}}"
TRANSPORT="${2:-${TRANSPORT:-nixl}}"

case "$STRATEGY" in
  tp4tp4) P_TP=4; D_TP=4; IS_BASELINE=0 ;;   # PD homogeneous TP4+4 (main path)
  tp4)    P_TP=4; D_TP=0; IS_BASELINE=1 ;;   # baseline single-instance TP4
  tp8)    P_TP=8; D_TP=0; IS_BASELINE=1 ;;   # baseline single-instance TP8
  *)
    echo "ERROR: unknown STRATEGY='$STRATEGY' (expected: tp4tp4|tp4|tp8)" >&2
    exit 1 ;;
esac

# GLM-5.2 is MLA (kv_lora_rank=512) -> staging buffer unavailable, heterogeneous TP abandoned.
# DP_ATTN fixed 0 (homogeneous TP4, no --enable-dp-attention).
# No --moe-a2a-backend: pure TP4 uses sglang default MoE-TP path (no dp-attention -> no deepep).
DP_ATTN=0
STAGING=0   # MLA: never enable SGLANG_DISAGG_STAGING_BUFFER

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

# staging explicitly OFF (MLA unavailable; homogeneous TP bypasses anyway)
export SGLANG_DISAGG_STAGING_BUFFER=0

# --- Disaggregation tunables (shared by nixl/mooncake) ---
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-300}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-300}"
export SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL="${SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL:-5}"
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE="${SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE:-2}"

# --- Common launch_server args (derived from /opt/sglang-glm/start.sh; business params kept) ---
# Quantization auto-detected from config.json (quant_method=w4afp8); no --quantization needed.
COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name glm
  --trust-remote-code
  --host 0.0.0.0
  --context-len "$CONTEXT_LEN"
  --tool-call-parser glm47
  --reasoning-parser glm45
  --schedule-policy fcfs
  --enable-metrics --enable-cache-report
  --chunked-prefill-size "$CHUNKED_PREFILL"
  --max-running-requests "$MAX_RUNNING" --max-queued-requests "$MAX_QUEUED"
  --watchdog-timeout "$WATCHDOG"
  --cuda-graph-max-bs "$CG_MAX_BS"
  --kv-cache-dtype fp8_e4m3
)

export MODEL_PATH LOG_DIR PID_DIR PYTHON
export PORT_P PORT_D ROUTER_PORT BASELINE_PORT P_GPUS D_GPUS
export STRATEGY TRANSPORT DISAGG_BACKEND P_TP D_TP IS_BASELINE DP_ATTN STAGING
export CONTEXT_LEN MEM_FRAC_P MEM_FRAC_D CHUNKED_PREFILL MAX_RUNNING MAX_QUEUED
export WATCHDOG CG_MAX_BS COMMON_ARGS
```

> 说明：
> - `--mem-fraction-static` 是 sglang 显存占用比例（对应 vllm `--gpu-memory-utilization`）。P 用 `MEM_FRAC_P`、D 用 `MEM_FRAC_D`，在各自 launch 脚本中按 `--mem-fraction-static` 传入。D 侧如需留更多激活余量可设 `MEM_FRAC_D=0.80`。
> - 量化由 `config.json` 的 `quant_method=w4afp8` 自动识别，不需要 `--quantization`（与现网 start.sh 一致）。
> - 异构策略 `tp4dp2`/`tp4dp4` 已移除；若误传会报错退出。`STAGING`/`DP_ATTN` 字段保留但固定为 0，仅作自文档化（MLA 不可用）。

### 3.2 launch_prefill.sh

启动 P 实例。P 侧始终为纯 TP4，不开 DP attention，不开 EAGLE（P 侧通常不投机）。

```bash
#!/usr/bin/env bash
# Launch the sglang Prefill instance (homogeneous TP4, PD disaggregation).
# Usage: launch_prefill.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

[[ "$IS_BASELINE" == "1" ]] && { echo "STRATEGY=$STRATEGY is baseline, no prefill"; exit 0; }

echo "[$(date +%H:%M:%S)] Starting Prefill: GPUs=$P_GPUS port=$PORT_P strategy=$STRATEGY transport=$TRANSPORT tp=$P_TP"

CUDA_VISIBLE_DEVICES="$P_GPUS" \
exec "$PYTHON" -m sglang.launch_server \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend "$DISAGG_BACKEND" \
  --tp-size "$P_TP" \
  --port "$PORT_P" \
  --mem-fraction-static "$MEM_FRAC_P" \
  "${COMMON_ARGS[@]}" \
  > "$LOG_DIR/prefill.log" 2>&1 &
echo $! > "$PID_DIR/prefill.pid"
echo "Prefill PID=$(cat "$PID_DIR/prefill.pid"), log=$LOG_DIR/prefill.log"
```

### 3.3 launch_decode.sh

启动 D 实例。与 P 同构 TP4，KV head 一一对应。EAGLE 默认关闭（PD 下可用性待实测，见调研/实验设计文档）。

```bash
#!/usr/bin/env bash
# Launch the sglang Decode instance (homogeneous TP4, PD disaggregation).
# Usage: launch_decode.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

[[ "$IS_BASELINE" == "1" ]] && { echo "STRATEGY=$STRATEGY is baseline, no decode"; exit 0; }

echo "[$(date +%H:%M:%S)] Starting Decode: GPUs=$D_GPUS port=$PORT_D strategy=$STRATEGY transport=$TRANSPORT tp=$D_TP"

CUDA_VISIBLE_DEVICES="$D_GPUS" \
exec "$PYTHON" -m sglang.launch_server \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend "$DISAGG_BACKEND" \
  --tp-size "$D_TP" \
  --port "$PORT_D" \
  --mem-fraction-static "$MEM_FRAC_D" \
  "${COMMON_ARGS[@]}" \
  > "$LOG_DIR/decode.log" 2>&1 &
echo $! > "$PID_DIR/decode.pid"
echo "Decode PID=$(cat "$PID_DIR/decode.pid"), log=$LOG_DIR/decode.log"
```

> `--base-gpu-id` 不在此设：已用 `CUDA_VISIBLE_DEVICES=4,5,6,7` 把 D 的可见集裁到 4 张卡，`launch_server` 在该可见集内从 0 起编号即可。若改用 `--base-gpu-id 4` 方式，则不要同时设 `CUDA_VISIBLE_DEVICES`。

### 3.4 launch_router.sh

启动 sglang_router，PD 模式串联 P 与 D。

```bash
#!/usr/bin/env bash
# Launch the sglang PD router (--pd-disaggregation mode).
# Usage: launch_router.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

[[ "$IS_BASELINE" == "1" ]] && { echo "STRATEGY=$STRATEGY is baseline, no router"; exit 0; }

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

> 与现网 `/opt/sglang-glm/start-smg.sh`（非 PD 的 DP-aware `cache_aware` router，port 18080/29000）不同：本方案 router 用 `--pd-disaggregation` 显式 PD 模式，对外 port 8000。

### 3.5 launch_baseline.sh

启动单实例基线（`tp4` 或 `tp8`，无 PD）。用于与 PD 做「同配置不分离」对照（实验 C/D）。

```bash
#!/usr/bin/env bash
# Launch a single-instance baseline (no PD): tp4 or tp8 on port 30002.
# Usage: launch_baseline.sh [tp4|tp8]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-tp4}" "${2:-nixl}"

[[ "$IS_BASELINE" != "1" ]] && { echo "STRATEGY=$STRATEGY is not a baseline (use tp4|tp8)"; exit 1; }

# tp4 baseline uses GPU 0-3; tp8 baseline uses all 8.
if [[ "$STRATEGY" == "tp4" ]]; then
  BASE_GPUS="${BASE_GPUS:-0,1,2,3}"
else
  BASE_GPUS="${BASE_GPUS:-0,1,2,3,4,5,6,7}"
fi

echo "[$(date +%H:%M:%S)] Starting baseline: strategy=$STRATEGY GPUs=$BASE_GPUS tp=$P_TP port=$BASELINE_PORT"

CUDA_VISIBLE_DEVICES="$BASE_GPUS" \
exec "$PYTHON" -m sglang.launch_server \
  --tp-size "$P_TP" \
  --port "$BASELINE_PORT" \
  --mem-fraction-static "$MEM_FRAC_P" \
  "${COMMON_ARGS[@]}" \
  > "$LOG_DIR/baseline.log" 2>&1 &
echo $! > "$PID_DIR/baseline.pid"
echo "Baseline PID=$(cat "$PID_DIR/baseline.pid"), log=$LOG_DIR/baseline.log"
```

> 基线 `tp4` 与 PD 的 P/D 单实例同配置（同 `COMMON_ARGS`、同 TP4、同 `--kv-cache-dtype fp8_e4m3`），是最公平的「PD vs 不 PD」对照。可选 `tp8` 作量级参考；也可直接复用现网 DP4×TP2 8 卡实例（调研文档 7.2），但口径不同（开了 DP attention + EAGLE + hicache）。

### 3.6 run_pd.sh

一键拉起：P -> 等 health -> D -> 等 health -> router -> 等 health。

```bash
#!/usr/bin/env bash
# One-shot: prefill -> health -> decode -> health -> router -> health.
# Usage: run_pd.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
STRATEGY="${1:-tp4tp4}"; TRANSPORT="${2:-nixl}"

wait_health() {
  local url="$1" name="$2" tries="${3:-90}"
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

> health 端点 `/health` 以容器内 sglang 版本为准；若 router 健康端点不同（如 `/health_ready`），回填此处。GLM-5.2 W4AFP8 400GB 权重加载较慢（TP4 每卡 100GB），prefill 健康检查 `tries=90`（约 3 分钟）可能不够，按需调大。

### 3.7 stop_pd.sh

按 PID 停 P/D/router/baseline。

```bash
#!/usr/bin/env bash
# Stop prefill / decode / router / baseline by PID files.
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

kill_pid "$PID_DIR/router.pid"   router
kill_pid "$PID_DIR/decode.pid"   decode
kill_pid "$PID_DIR/prefill.pid"  prefill
kill_pid "$PID_DIR/baseline.pid" baseline
echo "stopped (strategy=$STRATEGY transport=$TRANSPORT)"
```

### 3.8 bench.sh

跑 sglang 自带 bench_serving，负载固定为输入 7500 / 输出 200，`NUM_PROMPTS` 默认 50（可设 100）。指标口径对齐 vllm bench_serve（TTFT/ITAT/吞吐）以便横向对比。

```bash
#!/usr/bin/env bash
# Benchmark via router (PD) or baseline (tp4/tp8).
# Usage: bench.sh [strategy] [transport] [--baseline]
#        NUM_PROMPTS=100 ./bench.sh tp4tp4 nixl
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

if [[ "${3:-}" == "--baseline" ]]; then
  URL="http://127.0.0.1:$BASELINE_PORT"
  TAG="baseline-$STRATEGY"
else
  URL="http://127.0.0.1:$ROUTER_PORT"
  TAG="pd-$STRATEGY-$TRANSPORT"
fi

OUT="$LOG_DIR/bench-${TAG}-n${NUM_PROMPTS:-50}-$(date +%H%M%S).json"
echo "[bench] url=$URL tag=$TAG num_prompts=${NUM_PROMPTS:-50} out=$OUT"

# Load: input 7500 / output 200. Exact arg names per sglang version (--help to confirm).
"$PYTHON" -m sglang.bench.serving \
  --url "$URL" \
  --model-name glm \
  --dataset-name random \
  --random-input-len 7500 --random-output-len 200 \
  --num-prompts "${NUM_PROMPTS:-50}" \
  --request-rate "${REQUEST_RATE:-INF}" \
  > "$OUT" 2>&1 || echo "[bench] non-zero exit, see $OUT"

echo "bench done: $OUT"
```

> bench 子参数（`--random-input-len` / `--random-output-len` / `--model-name`）以容器内 sglang `bench.serving --help` 为准。`--model-name glm` 对应 `--served-model-name glm`。若需与 vllm bench_serve 同口径，可改用 vllm `benchmark_serving.py --url` 打同一端口，但本分支优先用 sglang 自带 bench。

### 3.9 correct_check.sh

正确性比对：同一 prompt 经 router（PD）与基线（30002）输出一致（greedy）。`model` 字段用 served-model-name `glm`。payload 用 python 生成以避免 shell 转义。

```bash
#!/usr/bin/env bash
# Correctness diff: PD (router) vs baseline (tp4/tp8).
# Usage: correct_check.sh [strategy] [transport]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

PROMPT="${PROMPT:-用一句话解释什么是 Prefill/Decode 分离。}"
TMP="$LOG_DIR/correct_$$"
PAYLOAD="$TMP.payload"

# 用 python 生成合法 JSON，model 字段用 served-model-name "glm"
"$PYTHON" - "$PROMPT" "$PAYLOAD" <<'PYEOF'
import json, sys
prompt, out = sys.argv[1], sys.argv[2]
with open(out, "w") as f:
    json.dump({"model": "glm",
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

> 基线需另起（`./launch_baseline.sh tp4`，见 3.5）。PD 与基线同 W4AFP8 量化口径、同 `--kv-cache-dtype fp8_e4m3`，greedy 下应一致；若有 diff 疑为 KV 搬移丢页或 MLA latent 映射问题（见实验设计文档 8.1）。

### 3.10 status.sh

查看进程、端口、日志尾部、显存。

```bash
#!/usr/bin/env bash
# Quick status of the sglang PD stack + baseline.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}" "${2:-}"

echo "=== strategy=$STRATEGY transport=$TRANSPORT staging=$STAGING dp_attn=$DP_ATTN ==="
for name in prefill decode router baseline; do
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
for f in prefill decode router baseline; do
  log="$LOG_DIR/$f.log"
  [[ -f "$log" ]] && { echo "--- $f.log ---"; tail -n 5 "$log"; }
done
```

### 3.11 install_sglang.sh

> 本环境（h200-2）已有 sglang 容器（`lmsysorg/sglang:v0.5.15.post1-cu129`，commit `0b3bb0c`），**通常无需安装**。本脚本用于在容器内补装 nixl/mooncake，或在宿主建独立 venv 装 sglang。不动系统 python。

```bash
#!/usr/bin/env bash
# Install/repair sglang + transfer engines inside a container or a fresh host venv.
# NOTE: h200-2 already has the sglang image (v0.5.15.post1-cu129), so this is optional.
# Requires network. Does NOT touch system python.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SGLANG_DEV_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
VENV_DIR="${VENV_DIR:-$SGLANG_DEV_ROOT/.venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "[install] creating venv at $VENV_DIR (skip if running inside the sglang container; there use python3 directly)"
uv venv --python 3.12 "$VENV_DIR" 2>/dev/null || true
PY="${PYTHON:-$VENV_DIR/bin/python}"

echo "[install] sglang[all] + nixl + mooncake-transfer-engine (network required)"
uv pip install --python "$PY" "sglang[all]" 2>/dev/null || echo "[install] sglang may already be present in container"
uv pip install --python "$PY" nixl mooncake-transfer-engine 2>/dev/null || \
  echo "[install] nixl/mooncake pip install failed (may need source build with ucx_path)"

echo "[install] versions:"
"$PY" -m sglang.launch_server --version || true
"$PY" -c "import nixl; print('nixl ok', nixl.__file__)" || echo "nixl import failed (source build: set ucx_path)"
"$PY" -c "import mooncake; print('mooncake ok')" || echo "mooncake import failed"

echo "[install] done. In sglang container: PYTHON=python3 (no venv needed). On host: source $VENV_DIR/bin/activate"
```

> 容器内通常已含 sglang 本体；若 NIXL/Mooncake backend 缺失，仅需 `uv pip install nixl mooncake-transfer-engine`（指向容器内 python3）。若 `nixl` pip 包在目标平台不可用，需从源码编译（带 `ucx_path`）。本脚本只装软件，不修改 vllm 代码。

---

## 4. 参数映射表（strategy -> 启动参数）

| strategy | P `--tp-size` | P `--disaggregation-mode` | D `--tp-size` | D `--disaggregation-mode` | `--enable-dp-attention` | `--moe-a2a-backend` | staging env | 备注 |
| - | - | - | - | - | - | - | - | - |
| tp4tp4 | 4 | prefill | 4 | decode | 否 | 无 | `STAGING=0` 固定 | PD 同构主路径 |
| tp4 | 4 | （无 PD） | - | - | 否 | 无 | - | 基线单实例 TP4 |
| tp8 | 8 | （无 PD） | - | - | 否 | 无 | - | 基线单实例 TP8 |

P/D 共用 `COMMON_ARGS`（含 `--model-path`、`--served-model-name glm`、`--trust-remote-code`、`--context-len 300000`、`--tool-call-parser glm47`、`--reasoning-parser glm45`、`--kv-cache-dtype fp8_e4m3`、`--mem-fraction-static`、`--enable-metrics` 等，由 `start.sh` 派生）。

transport -> env 映射：

| transport | `--disaggregation-transfer-backend` | 额外 env |
| - | - | - |
| nixl | nixl | `SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX` |
| mooncake | mooncake | `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK` + `MC_INTRANODE_NVLINK=true` |

> 异构策略 `tp4dp2`/`tp4dp4` 已移除（GLM-5.2 MLA，staging 不可用，见调研文档 5.5）。`STAGING`/`DP_ATTN` 字段在 env.sh 保留但固定为 0。

---

## 5. 配置参考：完整启动命令示例

> 以下命令由现网 `/opt/sglang-glm/start.sh` 派生为 PD 版本。假设在 sglang 容器内执行（`PYTHON=python3`，`--model-path` 用容器路径）；容器内执行方式见第 7 节。量化由 `config.json` 自动识别，不传 `--quantization`。

### 5.1 strategy=tp4tp4 + transport=nixl（同构 PD 主路径 A）

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGG_STAGING_BUFFER=0
MODEL=/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8

# Prefill (GPU 0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path $MODEL --served-model-name glm --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
  --tp-size 4 \
  --context-len 300000 --tool-call-parser glm47 --reasoning-parser glm45 \
  --schedule-policy fcfs --enable-metrics --enable-cache-report \
  --chunked-prefill-size 32768 \
  --max-running-requests 64 --max-queued-requests 512 \
  --mem-fraction-static 0.85 --watchdog-timeout 1800 \
  --cuda-graph-max-bs 128 --kv-cache-dtype fp8_e4m3

# Decode (GPU 4,5,6,7)
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server \
  --model-path $MODEL --served-model-name glm --trust-remote-code \
  --host 0.0.0.0 --port 30001 \
  --disaggregation-mode decode --disaggregation-transfer-backend nixl \
  --tp-size 4 \
  --context-len 300000 --tool-call-parser glm47 --reasoning-parser glm45 \
  --schedule-policy fcfs --enable-metrics --enable-cache-report \
  --chunked-prefill-size 32768 \
  --max-running-requests 64 --max-queued-requests 512 \
  --mem-fraction-static 0.85 --watchdog-timeout 1800 \
  --cuda-graph-max-bs 128 --kv-cache-dtype fp8_e4m3

# Router (PD mode)
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

### 5.2 strategy=tp4tp4 + transport=mooncake（对照 B）

命令同 5.1，但替换 transport 与 Mooncake NVLink env（去掉 NIXL env）：

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export MC_INTRANODE_NVLINK=true
export SGLANG_DISAGG_STAGING_BUFFER=0
MODEL=/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8

# Prefill (GPU 0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path $MODEL --served-model-name glm --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --tp-size 4 \
  --context-len 300000 --tool-call-parser glm47 --reasoning-parser glm45 \
  --schedule-policy fcfs --enable-metrics --enable-cache-report \
  --chunked-prefill-size 32768 \
  --max-running-requests 64 --max-queued-requests 512 \
  --mem-fraction-static 0.85 --watchdog-timeout 1800 \
  --cuda-graph-max-bs 128 --kv-cache-dtype fp8_e4m3

# Decode (GPU 4,5,6,7)
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server \
  --model-path $MODEL --served-model-name glm --trust-remote-code \
  --host 0.0.0.0 --port 30001 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tp-size 4 \
  --context-len 300000 --tool-call-parser glm47 --reasoning-parser glm45 \
  --schedule-policy fcfs --enable-metrics --enable-cache-report \
  --chunked-prefill-size 32768 \
  --max-running-requests 64 --max-queued-requests 512 \
  --mem-fraction-static 0.85 --watchdog-timeout 1800 \
  --cuda-graph-max-bs 128 --kv-cache-dtype fp8_e4m3

# Router 同 5.1
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

> IB 设备参数 `--disaggregation-ib-device mlx5_0` 仅多机/IB 场景需要；单机 NVLink 用上面 env 即可，故本例省略。

### 5.3 基线 tp4 / tp8（无 PD，对照 C/D）

```bash
MODEL=/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8

# 基线 TP4（GPU 0-3），主基线
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path $MODEL --served-model-name glm --trust-remote-code \
  --host 0.0.0.0 --port 30002 \
  --tp-size 4 \
  --context-len 300000 --tool-call-parser glm47 --reasoning-parser glm45 \
  --schedule-policy fcfs --enable-metrics --enable-cache-report \
  --chunked-prefill-size 32768 \
  --max-running-requests 64 --max-queued-requests 512 \
  --mem-fraction-static 0.85 --watchdog-timeout 1800 \
  --cuda-graph-max-bs 128 --kv-cache-dtype fp8_e4m3

# 基线 TP8（GPU 0-7，可选量级参考）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
  --model-path $MODEL --served-model-name glm --trust-remote-code \
  --host 0.0.0.0 --port 30002 \
  --tp-size 8 \
  --context-len 300000 --tool-call-parser glm47 --reasoning-parser glm45 \
  --schedule-policy fcfs --enable-metrics --enable-cache-report \
  --chunked-prefill-size 32768 \
  --max-running-requests 64 --max-queued-requests 512 \
  --mem-fraction-static 0.85 --watchdog-timeout 1800 \
  --cuda-graph-max-bs 128 --kv-cache-dtype fp8_e4m3
```

---

## 6. 健康检查与日志关键字

### 6.1 健康检查端点

| 组件 | URL | 期望 |
| - | - | - |
| Prefill | `http://127.0.0.1:30000/health` | 200 |
| Decode | `http://127.0.0.1:30001/health` | 200 |
| router | `http://127.0.0.1:8000/health` | 200（端点名以 sglang 版本为准，可能为 `/health_ready`） |
| 基线 | `http://127.0.0.1:30002/health` | 200 |

### 6.2 日志关键字

| 日志 | 关键字（grep） | 含义 |
| - | - | - |
| `prefill.log` | `The server is fired up` / `Application startup complete` | P 就绪 |
| `prefill.log` | `disaggregation` / `prefill mode` | 进入 PD prefill 角色 |
| `decode.log` | `decode mode` / `bootstrap` / `recv` | D 进入 decode 角色、KV 接收链路建立 |
| `decode.log` | `heartbeat` / `waiting` | 心跳与等待 KV（见 `WAITING_TIMEOUT`） |
| `router.log` | `prefill` + `decode` + `up` | router 发现 P/D 在线 |
| 各日志 | `staging buffer` / `staging pool` | 不应出现（本方案 `STAGING=0`；若出现说明 env 误设） |
| 各日志 | `error` / `Traceback` / `OOM` / `CUDA error` | 故障 |
| 各日志 | `nixl` / `mooncake` / `cuda_ipc` / `NVLink` | 传输路径（确认走 NVLink 而非 TCP 回退） |

### 6.3 常见问题排查

- D 侧一直 `waiting`：核对 P 是否就绪、`BOOTSTRAP_TIMEOUT` 是否够、transport env 是否对、`--disaggregation-transfer-backend` 两侧是否一致。
- OOM：D 侧降 `MEM_FRAC_D`（0.85→0.80）或降 `--max-running-requests`；权重 TP4 每卡 100GB，剩余约 41GB，MLA KV 占比极小，OOM 多为激活/碎片。
- KV 搬移走 TCP 回退：核对 `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK` / `MC_INTRANODE_NVLINK` / UCX_TLS，日志确认 `cuda_ipc`/`NVLink`。
- MLA 相关报错：sglang PD 对 MLA 支持度若不完整（见实验设计文档 8.1），日志可能出现 latent/KV shape 相关错误，记录现象回退基线。

---

## 7. 容器内执行说明

sglang 跑在镜像 `lmsysorg/sglang:v0.5.15.post1-cu129` 内（commit `0b3bb0c`），宿主无 sglang。脚本有两种执行方式：起独立容器（推荐，P/D/router 各一），或在现有容器内 `docker exec` 起多进程。模型挂载沿用现网映射：宿主 `/data1/GLM-5.2-W4AFP8` ↔ 容器 `/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8`。

### 7.1 方式一：docker run 独立容器（推荐）

P、D、router 各起一个容器，均用 `--network host`（单机 localhost 互通，端口 30000/30001/8000 落在宿主），`--gpus all` + 容器内 `CUDA_VISIBLE_DEVICES` 切分。

镜像变量：

```bash
IMAGE=br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129
SCRIPTS_HOST=/home/lychee/mycode/vllm/develop/sglang/scripts
MODEL_HOST=/data1/GLM-5.2-W4AFP8
MODEL_CNT=/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8
```

Prefill 容器（GPU 0-3，NIXL）：

```bash
docker run -d --name sglang-pd-p --network host --gpus all \
  --shm-size 32gb --cap-add SYS_NICE --cap-add IPC_LOCK \
  -v $MODEL_HOST:$MODEL_CNT:ro \
  -v $SCRIPTS_HOST:/sglang-scripts:ro \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
  -e SGLANG_DISAGG_STAGING_BUFFER=0 \
  $IMAGE \
  bash -c "cd /sglang-scripts && ./launch_prefill.sh tp4tp4 nixl"
```

Decode 容器（GPU 4-7，NIXL）：

```bash
docker run -d --name sglang-pd-d --network host --gpus all \
  --shm-size 32gb --cap-add SYS_NICE --cap-add IPC_LOCK \
  -v $MODEL_HOST:$MODEL_CNT:ro \
  -v $SCRIPTS_HOST:/sglang-scripts:ro \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
  -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
  -e SGLANG_DISAGG_STAGING_BUFFER=0 \
  $IMAGE \
  bash -c "cd /sglang-scripts && ./launch_decode.sh tp4tp4 nixl"
```

Router 容器（无需 GPU，复用镜像里的 python）：

```bash
docker run -d --name sglang-pd-router --network host \
  -v $SCRIPTS_HOST:/sglang-scripts:ro \
  $IMAGE \
  bash -c "cd /sglang-scripts && ./launch_router.sh tp4tp4 nixl"
```

> Mooncake 版：把 P/D 容器的 `-e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX` 换成 `-e SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK -e MC_INTRANODE_NVLINK=true`，启动参传 `mooncake`。基线容器：用 `./launch_baseline.sh tp4`，`--gpus all` + `-e CUDA_VISIBLE_DEVICES=0,1,2,3`。

### 7.2 方式二：现有容器内 docker exec 多进程

复用现有 `sglang-glm-*` 容器（若还在），在其中起 P/D 两个 `launch_server` 进程 + 一个 router 进程，靠 `CUDA_VISIBLE_DEVICES` 切分 8 卡，`--network host` 已由容器提供：

```bash
docker exec -d -e CUDA_VISIBLE_DEVICES=0,1,2,3 sglang-glm-smg-1 \
  bash -c "cd /sglang-scripts && ./launch_prefill.sh tp4tp4 nixl"
docker exec -d -e CUDA_VISIBLE_DEVICES=4,5,6,7 sglang-glm-smg-1 \
  bash -c "cd /sglang-scripts && ./launch_decode.sh tp4tp4 nixl"
docker exec -d sglang-glm-smg-1 \
  bash -c "cd /sglang-scripts && ./launch_router.sh tp4tp4 nixl"
```

> 注意：现有 `sglang-glm-smg-1` 容器里 router（`start-smg.sh`，非 PD 模式）可能已占 18080/29000；本方案 router 用 8000，不冲突。但若现有 sglang 实例（`start.sh`，port 8001）仍在跑并占 GPU 0-7，需先停现网服务再起 PD，否则 GPU 冲突。日志/PID 落在容器内 `/sglang-scripts/../logs`（即宿主 `develop/sglang/logs`，通过挂载可见）。

### 7.3 日志与 PID 落点

- 方式一：脚本挂载为 `/sglang-scripts`，其 `LOG_DIR`/`PID_DIR` 解析为容器内 `/sglang-scripts/../logs`、`/sglang-scripts/../run/pids`，对应宿主 `develop/sglang/logs`、`develop/sglang/run/pids`（只读挂载需改为可写挂载脚本目录，或将 logs/pids 挂出独立可写卷）。
- 实操建议：把 `develop/sglang/scripts` 与 `develop/sglang/logs`、`develop/sglang/run` 都以可写挂载进容器，便于宿主直接看日志与 PID。

---

## 8. 与 vllm scripts 的差异说明

`develop/scripts/env.sh`（vllm 那套，实测于本仓库）与 `develop/sglang/scripts/env.sh`（本套）对照：

| 维度 | `develop/scripts/`（vllm） | `develop/sglang/scripts/`（本套） |
| - | - | - |
| 启动命令 | `vllm serve` | `python -m sglang.launch_server` |
| 角色声明 | `--kv-transfer-config` JSON（`kv_role`） | `--disaggregation-mode prefill\|decode` |
| 后端选择 | JSON `kv_connector` 名（NixlConnector / MooncakeConnector） | `--disaggregation-transfer-backend nixl\|mooncake` |
| 编排 | 外部 `toy_proxy_server`（`launch_proxy.sh`） | 内置 `python -m sglang_router.launch_router --pd-disaggregation` |
| CPU 转发 | `kv_buffer_device=cpu/gdr`（`env.sh:40-54` TRANSPORT 映射） | 无对应（sglang NIXL 走 UCX/LIBFABRIC RDMA，无 host buffer） |
| side channel | `SIDE_PORT_P/D` 5600/5601（`env.sh:23-24`） | 无独立 side channel 端口，后端自管 |
| 异构 TP | `compute_tp_mapping` 自动按头切分（含 MLA 分支 `tp_mapping.py:79-84`） | `SGLANG_DISAGG_STAGING_BUFFER` 仅 non-MLA；本方案 MLA 固定 `STAGING=0` 走同构 |
| DP attention | 不在本分支 vllm 方案内 | 本方案不开（同构 TP4，`DP_ATTN=0`） |
| 模型 | MiniMax-M2.5（`env.sh:17`） | GLM-5.2-W4AFP8（容器路径，MLA MoE） |
| 端口 | P 8100 / D 8200 / proxy 8000 / baseline 8300（`env.sh:20-25`） | P 30000 / D 30001 / router 8000 / baseline 30002 |
| venv | vllm 主 venv | sglang 容器内 `python3`（或 `develop/sglang/.venv`） |
| strategy 维度 | `gdr\|cpu`（transport）单一 | `tp4tp4\|tp4\|tp8`（strategy）× `nixl\|mooncake`（transport） |

> 两套脚本完全独立、目录分离，不互相 source、不共享 env。本分支只探索 sglang，不修改 vllm 源码也不复用其脚本。vllm 侧的 `compute_tp_mapping`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:65`）与 `kv_buffer_device`（`vllm/config/kv_transfer.py:33`）仅作能力对比引用，说明「vLLM NIXL 对 MLA/异构 TP 有专门处理，sglang 侧需实测」，详见调研文档第 5、8 章。
