# H200 单机 4+4 NIXL PD 分离验证脚本设计

## 目录

- [1. 概述](#1-概述)
- [2. 脚本清单](#2-脚本清单)
- [3. 脚本详细设计](#3-脚本详细设计)
  - [3.1 env.sh](#31-envsh)
  - [3.2 launch_prefill.sh](#32-launch_prefillsh)
  - [3.3 launch_decode.sh](#33-launch_decodesh)
  - [3.4 launch_proxy.sh](#34-launch_proxysh)
  - [3.5 run_pd.sh](#35-run_pdsh)
  - [3.6 stop_pd.sh](#36-stop_pdsh)
  - [3.7 bench.sh](#37-benchsh)
  - [3.8 correct_check.sh](#38-correct_checksh)
  - [3.9 status.sh](#39-statussh)
  - [3.10 watch_dl.sh](#310-watch_dlsh)
- [4. 关键参数映射表](#4-关键参数映射表)
- [5. 配置文件参考](#5-配置文件参考)
- [6. 健康检查与日志说明](#6-健康检查与日志说明)
- [7. 与仓库现有脚本的关系](#7-与仓库现有脚本的关系)

---

## 1. 概述

本文档设计一套用于 **H200 单机 8 GPU、4+4 NIXL PD 分离（disaggregated prefill/decode）验证** 的 bash
脚本，统一放置在 `develop/scripts/` 下。目标：

- 单机 8 卡按 4+4 切分：GPU `0,1,2,3` 跑 Prefill（`kv_producer`），`4,5,6,7` 跑 Decode（`kv_consumer`）。
- 通过 `toy_proxy_server.py` 把客户端请求路由到 P，再把 P 返回的 `kv_transfer_params` 透传给 D 完成解码。
- 支持通过 `TRANSPORT=gdr|cpu` 一键切换 KV 传输缓冲方式（GPU 直传 vs CPU 中转），便于在相同硬件上对比性能。
- 脚本可执行、可复用、参数化；后台运行、pid 受控、日志落盘、健康检查、一键启停。

模型使用本地路径 `/data1/models/MiniMax-M2.5`（由 `develop/scripts/dl_m25.sh` 通过 aria2 下载得到）。
该模型在 vLLM 中对应 `MiniMaxM2ForCausalLM`（`vllm/model_executor/models/minimax_m2.py:432`），需通过
`--trust-remote-code` 加载其 `config.json` 中的自定义代码。

目录布局约定：

```
develop/
├── 03_scripts_design.md        # 本设计文档
├── minimax_m2.5_aria2.list     # aria2 下载列表
├── logs/                       # 运行日志（prefill.log/decode.log/proxy.log/...）
│   ├── aria2_m25.log
│   └── aria2_m25.stdout
├── run/pids/                   # pid 文件目录
└── scripts/                    # 本文档设计的脚本集
    ├── env.sh
    ├── launch_prefill.sh
    ├── launch_decode.sh
    ├── launch_proxy.sh
    ├── run_pd.sh
    ├── stop_pd.sh
    ├── bench.sh
    ├── correct_check.sh
    ├── status.sh
    ├── watch_dl.sh
    └── dl_m25.sh               # 已存在的下载脚本
```

> 本文档只给出脚本设计与可直接复制使用的代码块内容，暂不在文件系统创建这些脚本。

---

## 2. 脚本清单

| 文件名 | 职责 | 依赖 |
|--------|------|------|
| `env.sh` | 公共环境变量、路径、端口、GPU 划分、`TRANSPORT`→`kv_buffer_device`/`UCX_TLS` 映射 | bash |
| `launch_prefill.sh` | 启动 Prefill 实例（`kv_producer`），参数化 transport | `env.sh`、`vllm serve` |
| `launch_decode.sh` | 启动 Decode 实例（`kv_consumer`），参数化 transport | `env.sh`、`vllm serve` |
| `launch_proxy.sh` | 启动 `toy_proxy_server.py` | `env.sh`、`python` |
| `run_pd.sh` | 一键启动 P→健康检查→D→健康检查→proxy，打印访问端点 | 上述 3 个 launch 脚本、`curl` |
| `stop_pd.sh` | 按 pid 文件 kill，兜底 `pkill` 清理 | `env.sh` |
| `bench.sh` | 运行 `vllm bench serve`，参数化 input/output/num_prompts 等 | `env.sh`、`vllm bench` |
| `correct_check.sh` | 正确性对比：相同 prompt 经 proxy（PD）与基线单实例输出 diff | `env.sh`、`curl`、`python`、基线实例 |
| `status.sh` | 状态快照：pid 存活、端口健康、GPU 占用、日志尾 | `env.sh`、`curl`、`nvidia-smi` |
| `watch_dl.sh` | 监控 aria2 下载进度（`MiniMax-M2.5` 文件、aria2 日志尾） | bash、`ls` |

---

## 3. 脚本详细设计

### 3.1 env.sh

公共环境与路径，被其它脚本 `source`。第一个位置参数为 `TRANSPORT`（`gdr` 或 `cpu`，默认 `gdr`），
内部将其映射到 `KV_BUFFER_DEVICE`（写入 `--kv-transfer-config`）与 `UCX_TLS`（环境变量）。

设计要点与代码核对：

- `kv_buffer_device` 字段定义见 `vllm/config/kv_transfer.py:33-35`，默认值由平台设备类型工厂给出，
  显式取值范围 `cuda`/`cpu`/`xpu`。
- `kv_load_failure_policy` 默认即 `"fail"`（`vllm/config/kv_transfer.py:69`），文档中显式写出以便对齐
  `docs/features/nixl_connector_usage.md:84` 的官方样例。
- `VLLM_NIXL_SIDE_CHANNEL_HOST`/`PORT` 环境变量默认值见 `vllm/envs.py:202-203`（`localhost` / `5600`）。
  单机部署 host 保持默认 `localhost` 即可，只需为 P、D 两个 engine 分别指定不同 `SIDE_CHANNEL_PORT`。
- `UCX_TLS=cuda_ipc,cuda_copy,tcp` 取自 `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_vllm_launcher.sh:31`。

```bash
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
```

### 3.2 launch_prefill.sh

启动 Prefill（`kv_producer`）实例。关键设计：

- `--kv-transfer-config` 的 JSON 按 `KV_BUFFER_DEVICE` 动态拼接，等价于
  `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh:53-59` 的 `KV_CONFIG_P` 构造逻辑，
  但显式带上 `kv_buffer_device` 与 `kv_load_failure_policy` 两个字段。
- 环境变量前缀（`VLLM_KV_CACHE_LAYOUT`、`UCX_NET_DEVICES`、`UCX_TLS`、`VLLM_NIXL_SIDE_CHANNEL_PORT`）
  与 `run_accuracy_test.sh:155-164` 保持一致；P 侧 `VLLM_KV_CACHE_LAYOUT=HND`（NixlConnector 默认布局，
  见 `docs/features/nixl_connector_compatibility.md:94`）。
- `--tensor-parallel-size $TP`、`--trust-remote-code`（MiniMax-M2.5 需要）、`--enforce-eager` 受
  `ENFORCE_EAGER` 控制（参考 `run_accuracy_test.sh:79,165-167`）。
- 进程后台运行，PID 写入 `develop/run/pids/prefill.pid`，stdout/stderr 重定向到
  `develop/logs/prefill.log`。

```bash
#!/usr/bin/env bash
# Launch the Prefill (kv_producer) vLLM instance.
# Usage: launch_prefill.sh [gdr|cpu]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}"

KV_CONFIG_P='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"'"$KV_BUFFER_DEVICE"'","kv_load_failure_policy":"fail"}'

EXTRA=()
[[ "$ENFORCE_EAGER" == "1" ]] && EXTRA+=(--enforce-eager)

echo "[$(date +%H:%M:%S)] Starting Prefill: GPUs=$P_GPUS port=$PORT_P transport=$TRANSPORT kv_buffer_device=$KV_BUFFER_DEVICE"

CUDA_VISIBLE_DEVICES="$P_GPUS" \
VLLM_KV_CACHE_LAYOUT=HND \
UCX_NET_DEVICES=all \
UCX_TLS="$UCX_TLS" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_PORT_P" \
vllm serve "$MODEL_PATH" \
  --port "$PORT_P" \
  --tensor-parallel-size "$TP" \
  --block-size "$BLOCK_SIZE" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --kv-transfer-config "$KV_CONFIG_P" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  > "$LOG_DIR/prefill.log" 2>&1 &

echo $! > "$PID_DIR/prefill.pid"
echo "Prefill PID=$(cat "$PID_DIR/prefill.pid"), log=$LOG_DIR/prefill.log"
```

### 3.3 launch_decode.sh

启动 Decode（`kv_consumer`）实例。与 `launch_prefill.sh` 对称，差异点：

- `kv_role` 为 `kv_consumer`；
- GPU 集合为 `D_GPUS`、端口 `PORT_D`、side channel `SIDE_PORT_D`；
- 其余参数（`--tensor-parallel-size`、`--trust-remote-code`、`--max-model-len`、transport 映射）与 P 一致，
  保证 P、D 模型/dtype/attention backend/cache_dtype 完全相同（兼容性哈希要求，见
  `docs/features/nixl_connector_compatibility.md:74-82`）。

```bash
#!/usr/bin/env bash
# Launch the Decode (kv_consumer) vLLM instance.
# Usage: launch_decode.sh [gdr|cpu]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}"

KV_CONFIG_D='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"'"$KV_BUFFER_DEVICE"'","kv_load_failure_policy":"fail"}'

EXTRA=()
[[ "$ENFORCE_EAGER" == "1" ]] && EXTRA+=(--enforce-eager)

echo "[$(date +%H:%M:%S)] Starting Decode: GPUs=$D_GPUS port=$PORT_D transport=$TRANSPORT kv_buffer_device=$KV_BUFFER_DEVICE"

CUDA_VISIBLE_DEVICES="$D_GPUS" \
VLLM_KV_CACHE_LAYOUT=HND \
UCX_NET_DEVICES=all \
UCX_TLS="$UCX_TLS" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_PORT_D" \
vllm serve "$MODEL_PATH" \
  --port "$PORT_D" \
  --tensor-parallel-size "$TP" \
  --block-size "$BLOCK_SIZE" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --kv-transfer-config "$KV_CONFIG_D" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  > "$LOG_DIR/decode.log" 2>&1 &

echo $! > "$PID_DIR/decode.pid"
echo "Decode PID=$(cat "$PID_DIR/decode.pid"), log=$LOG_DIR/decode.log"
```

### 3.4 launch_proxy.sh

启动 `toy_proxy_server.py`。参数名校对自 `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:92-114`：

- `--port`（int，默认 `8000`）
- `--prefiller-hosts`（`nargs="+"`，默认 `["localhost"]`）
- `--prefiller-ports`（`nargs="+"`，int，默认 `[8100]`）
- `--decoder-hosts`（`nargs="+"`，默认 `["localhost"]`）
- `--decoder-ports`（`nargs="+"`，int，默认 `[8200]`）

`toy_proxy_server.py` 的工作机制：收到 `/v1/completions` 或 `/v1/chat/completions` 后，先把请求
轮询发给某个 prefiller，并把 `max_tokens` 强制为 1（`toy_proxy_server.py:171`），P 返回
`kv_transfer_params`（含远端 block id / host / port），proxy 再把该透传信息附加到原请求转发给 decoder
完成真正的流式解码（`toy_proxy_server.py:219-251`）。它还提供 `/healthcheck` 端点
（`toy_proxy_server.py:274-281`）。

```bash
#!/usr/bin/env bash
# Launch the toy disagg-prefill proxy.
# Usage: launch_proxy.sh [gdr|cpu]   (transport arg ignored, kept for symmetry)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}"

echo "[$(date +%H:%M:%S)] Starting proxy on port $PROXY_PORT"

python "$VLLM_ROOT/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --port "$PROXY_PORT" \
  --prefiller-hosts localhost \
  --prefiller-ports "$PORT_P" \
  --decoder-hosts localhost \
  --decoder-ports "$PORT_D" \
  > "$LOG_DIR/proxy.log" 2>&1 &

echo $! > "$PID_DIR/proxy.pid"
echo "Proxy PID=$(cat "$PID_DIR/proxy.pid"), log=$LOG_DIR/proxy.log"
```

### 3.5 run_pd.sh

一键启动 P → 等健康 → D → 等健康 → proxy，最后打印访问端点。健康检查使用 vLLM 的 `GET /health`
（`vllm/entrypoints/serve/instrumentator/health.py:22`），轻量且无需请求体；参考脚本
`run_accuracy_test.sh:95-101` 使用的是 `/v1/completions` 轮询，二者均可。启动顺序与
`run_accuracy_test.sh:249-276` 一致（先起 P/D 再起 proxy）。

```bash
#!/usr/bin/env bash
# One-shot: start P -> health -> D -> health -> proxy -> print endpoints.
# Usage: run_pd.sh [gdr|cpu]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}"

wait_health() {
  local port="$1" name="$2"
  echo "[$(date +%H:%M:%S)] Waiting for $name on http://127.0.0.1:$port/health ..."
  for _ in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] $name is healthy"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: $name did not become healthy within timeout" >&2
  return 1
}

"$SCRIPT_DIR/launch_prefill.sh" "$TRANSPORT"
wait_health "$PORT_P" "prefill" || { echo "Prefill failed health check, see $LOG_DIR/prefill.log" >&2; exit 1; }

"$SCRIPT_DIR/launch_decode.sh" "$TRANSPORT"
wait_health "$PORT_D" "decode" || { echo "Decode failed health check, see $LOG_DIR/decode.log" >&2; exit 1; }

"$SCRIPT_DIR/launch_proxy.sh"
sleep 2

cat <<EOF
============================================================
P/D + proxy are up (transport=$TRANSPORT, kv_buffer_device=$KV_BUFFER_DEVICE).
  Prefill : http://127.0.0.1:$PORT_P  (GPUs $P_GPUS, TP=$TP)
  Decode  : http://127.0.0.1:$PORT_D  (GPUs $D_GPUS, TP=$TP)
  Proxy   : http://127.0.0.1:$PROXY_PORT  <-- send client/bench requests here
  Health  : curl http://127.0.0.1:$PROXY_PORT/healthcheck
  Logs    : $LOG_DIR/{prefill,decode,proxy}.log
  PIDs    : $PID_DIR/{prefill,decode,proxy}.pid
Stop with: $SCRIPT_DIR/stop_pd.sh
============================================================
EOF
```

### 3.6 stop_pd.sh

按 pid 文件精确 kill，再做基于命令行的兜底 `pkill` 清理（与 `run_accuracy_test.sh:104-108` 的
`pkill -f "vllm serve"` 思路一致，但收窄到本脚本使用的端口/模型，避免误杀）。

```bash
#!/usr/bin/env bash
# Stop P/D/proxy by pid files, then fallback pkill scoped to our ports/model.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh"

for name in proxy decode prefill; do
  pf="$PID_DIR/$name.pid"
  if [[ -f "$pf" ]]; then
    pid="$(cat "$pf" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (pid=$pid)"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
  fi
done

# Fallback: any lingering vllm serve / toy_proxy bound to our ports/model.
pkill -f "toy_proxy_server.py --port $PROXY_PORT" 2>/dev/null || true
pkill -f "vllm serve $MODEL_PATH --port $PORT_P" 2>/dev/null || true
pkill -f "vllm serve $MODEL_PATH --port $PORT_D" 2>/dev/null || true

echo "Stopped. (verify with: $SCRIPT_DIR/status.sh)"
```

### 3.7 bench.sh

运行 `vllm bench serve` 打到 proxy 端口。CLI 入口与参数校对自
`vllm/benchmarks/serve.py:10-18`（用法示例）、`:1488-1502`（`--backend/--host/--port`）、
`:1587-1607`（`--request-rate/--burstiness`），以及 `vllm/benchmarks/datasets/datasets.py:1609-1617`
（`--num-prompts/--dataset-name`）和 `:1931-1937`（`--random-input-len/--random-output-len`）。
`vllm` 是合法 backend（`vllm/benchmarks/lib/endpoint_request_func.py:874`）。

所有数值参数支持环境变量覆盖，默认值针对 MiniMax-M2.5 中等长度 prefill 验证场景。

```bash
#!/usr/bin/env bash
# Run vllm bench serve against the proxy.
# Usage: bench.sh [gdr|cpu]
# Env overrides: IN (random-input-len) OUT (random-output-len)
#                N  (num-prompts)       B (burstiness)        R (request-rate)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh" "${1:-}"

IN="${IN:-7500}"
OUT="${OUT:-200}"
N="${N:-50}"
B="${B:-1.0}"
R="${R:-2}"

LABEL="pd_${TRANSPORT}_${N}n_${IN}in_${OUT}out"

echo "[$(date +%H:%M:%S)] bench -> http://127.0.0.1:$PROXY_PORT  label=$LABEL"
echo "  input_len=$IN output_len=$OUT num_prompts=$N burstiness=$B request_rate=$R"

vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port "$PROXY_PORT" \
  --model "$MODEL_PATH" \
  --dataset-name random \
  --random-input-len "$IN" \
  --random-output-len "$OUT" \
  --num-prompts "$N" \
  --burstiness "$B" \
  --request-rate "$R" \
  --ignore-eos \
  --save-result \
  --result-dir "$LOG_DIR" \
  --result-filename "${LABEL}.json"

echo "Result saved to $LOG_DIR/${LABEL}.json"
```

### 3.8 correct_check.sh

正确性对比：用同一 prompt、`temperature=0`（贪心）、固定 `max_tokens` 分别请求 proxy（PD 路径）与一台
**基线单实例**（`$BASELINE_PORT`，需另行以无 `--kv-transfer-config` 的普通 `vllm serve` 启动），然后
`diff` 两边的解码文本。因为 P 侧只算 1 个 token（`toy_proxy_server.py:171` 把 P 的 `max_tokens` 置 1），
真正的解码在 D 侧完成，贪心语义下 PD 输出应与基线一致。

```bash
#!/usr/bin/env bash
# Correctness: diff PD-via-proxy output vs a standalone baseline instance.
# Prerequisite: a baseline `vllm serve` (no --kv-transfer-config) on $BASELINE_PORT.
# Usage: correct_check.sh ["your prompt"]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh"

PROMPT="${1:-Once upon a time in a galaxy far far away, there lived a quiet engineer who}"
OUT_DIR="$LOG_DIR/correctness"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

call() {
  local port="$1" tag="$2"
  curl -s "http://127.0.0.1:$port/v1/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{"model":"'"$MODEL_PATH"'","prompt":"'"$PROMPT"'","max_tokens":32,"temperature":0,"stream":false}' \
    | python -c 'import sys,json
try:
    print(json.load(sys.stdin)["choices"][0]["text"])
except Exception as e:
    print("ERROR: %s :: %s" % (e, sys.stdin.read()))' \
    > "$OUT_DIR/${tag}_${STAMP}.txt"
  echo "[${tag}] saved -> $OUT_DIR/${tag}_${STAMP}.txt ($(wc -c < "$OUT_DIR/${tag}_${STAMP}.txt") bytes)"
}

call "$PROXY_PORT"    "pd"
call "$BASELINE_PORT" "baseline"

echo "----- diff (baseline vs pd) -----"
if diff -u "$OUT_DIR/baseline_${STAMP}.txt" "$OUT_DIR/pd_${STAMP}.txt"; then
  echo "PASS: PD output matches baseline (greedy, max_tokens=32)."
else
  echo "WARN: outputs differ (for greedy + healthy P/D path they should match)."
fi
```

### 3.9 status.sh

状态快照：pid 存活、端口 HTTP 健康码、GPU 占用、各日志最后几行。

```bash
#!/usr/bin/env bash
# Snapshot: pids, http health, GPU, recent log tails.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh"

echo "=== PIDs ==="
for name in prefill decode proxy; do
  pf="$PID_DIR/$name.pid"
  if [[ -f "$pf" ]]; then
    pid="$(cat "$pf" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      printf "  %-8s pid=%-8s ALIVE\n" "$name" "$pid"
    else
      printf "  %-8s pid=%-8s DEAD\n" "$name" "$pid"
    fi
  else
    printf "  %-8s (no pid file)\n" "$name"
  fi
done

echo "=== Health ==="
for pair in "prefill:$PORT_P" "decode:$PORT_D" "proxy:$PROXY_PORT"; do
  name="${pair%%:*}"; port="${pair##*:}"
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null || echo 000)"
  printf "  %-8s :%-5s HTTP %s\n" "$name" "$port" "$code"
done
# proxy also exposes /healthcheck with instance counts (toy_proxy_server.py:274)
echo "  proxy    /healthcheck: $(curl -s "http://127.0.0.1:$PROXY_PORT/healthcheck" 2>/dev/null || echo '{unreachable}')"

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"

echo "=== recent log tails (last 3 lines each) ==="
for f in prefill decode proxy; do
  lf="$LOG_DIR/$f.log"
  if [[ -f "$lf" ]]; then
    echo "--- $f.log ---"
    tail -n 3 "$lf"
  fi
done
```

### 3.10 watch_dl.sh

监控 `dl_m25.sh` 的 aria2 下载进度（`develop/logs/aria2_m25.log` 与目标目录文件）。`dl_m25.sh` 是
已有脚本，其启动方式见文件头注释：`setsid nohup develop/scripts/dl_m25.sh > develop/logs/aria2_m25.stdout 2>&1 < /dev/null &`。

```bash
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
```

---

## 4. 关键参数映射表

`TRANSPORT` 是本套脚本的顶层切换开关，由 `env.sh` 映射到 `--kv-transfer-config` 的
`kv_buffer_device` 字段与 `UCX_TLS` 环境变量。

| `TRANSPORT` | `kv_buffer_device` | `UCX_TLS` | 说明 |
|-------------|--------------------|-----------|------|
| `gdr` | `cuda` | `cuda_ipc,cuda_copy,tcp` | KV buffer 留在显存，GPU 直传；同机跨进程走 `cuda_ipc`，跨机兜底 `tcp`。延迟最低的路径。 |
| `cpu` | `cpu` | `cuda_ipc,cuda_copy,tcp` | KV buffer 落主机内存：发送侧 D2H 拷出、接收侧 H2D 拷入；网络/IPC 段仍可走 `cuda_ipc`/`tcp`，仅 buffer 位于 host。用于对照显存压力与中转开销。 |

判定依据：

- `kv_buffer_device` 字段见 `vllm/config/kv_transfer.py:33-35`，取值 `cuda`/`cpu`/`xpu`。
- 是否使用 host buffer 由 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:353-369` 决定：
  在 GPU 平台上 `use_host_buffer = (kv_buffer_device == "cpu")`，`host_xfer_buffers` 用于"设备显存无法直接
  注册到 NIXL"时的中转。
- `UCX_TLS=cuda_ipc,cuda_copy,tcp` 取自 `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_vllm_launcher.sh:31,47`；
  NIXL 单机官方样例（`docs/features/nixl_connector_usage.md:76-100`）使用 `UCX_NET_DEVICES=all` 但未显式
  设 `UCX_TLS`，本设计沿用 lmcache launcher 的显式 TLS 列表以保证单机 P2P 走 `cuda_ipc`。

---

## 5. 配置文件参考

下面给出 P、D 在两种 transport 下的完整 `--kv-transfer-config` JSON。脚本中由 `env.sh` +
`launch_prefill.sh`/`launch_decode.sh` 动态拼接生成，等价于以下字面量。字段含义：

- `kv_connector`：固定 `NixlConnector`。
- `kv_role`：P 侧 `kv_producer`，D 侧 `kv_consumer`（`vllm/config/kv_transfer.py:11-13,41-43`）。
- `kv_buffer_device`：`cuda` 或 `cpu`（`vllm/config/kv_transfer.py:33-35`）。
- `kv_load_failure_policy`：`fail`，加载失败立即失败（默认值，`vllm/config/kv_transfer.py:69-72`）。

### 5.1 gdr（kv_buffer_device=cuda）

Prefill（producer）：

```json
{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","kv_load_failure_policy":"fail"}
```

Decode（consumer）：

```json
{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda","kv_load_failure_policy":"fail"}
```

### 5.2 cpu（kv_buffer_device=cpu）

Prefill（producer）：

```json
{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cpu","kv_load_failure_policy":"fail"}
```

Decode（consumer）：

```json
{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cpu","kv_load_failure_policy":"fail"}
```

> 注：`run_accuracy_test.sh:53-59` 在 `cuda` 分支会省略 `kv_buffer_device`（依赖默认工厂返回 `cuda`），
> 在非 `cuda` 分支才显式写入。本设计在两种分支都显式写入，便于日志可读与 transport 对齐，行为等价。

---

## 6. 健康检查与日志说明

### 6.1 健康检查端点

| 端点 | 来源 | 用途 |
|------|------|------|
| `GET /health` | vLLM 服务端（`vllm/entrypoints/serve/instrumentator/health.py:22`） | `run_pd.sh` 的 `wait_health` 轮询，返回 200 即就绪 |
| `GET /v1/completions` | vLLM OpenAI server | `run_accuracy_test.sh:95-101` 使用的就绪探测（需请求体，较重） |
| `GET /healthcheck` | `toy_proxy_server.py:274-281` | proxy 自检，返回 `{status, prefill_instances, decode_instances}` |
| `GET /v1/models` | vLLM OpenAI server | `status.sh` 可选，确认模型已加载 |

`run_pd.sh` 选用 `GET /health`：轻量、无需请求体、不消耗 KV 资源。超时上限 600 次 × 2s = 1200s，
与参考脚本的 `timeout 1200`（`run_accuracy_test.sh:97`）量级一致。

### 6.2 日志关键字

日志统一落在 `develop/logs/`，常用排查关键字：

| 关键字 / 模式 | 出现位置 | 含义 |
|---------------|----------|------|
| `NIXL handshake` / `handshake` | `prefill.log`、`decode.log`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py:64-67`、`push_worker.py:434-467`） | P 与 D 通过 side channel 交换元数据；出现 `Push handshake to D ... done` / handshake 完成回调表示链路建立 |
| `NIXL compatibility hash` | `prefill.log`、`decode.log`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py:114,130`） | 握手时校验 P/D 版本、模型、dtype、KV heads、head size、层数、attention backend、cache_dtype 是否一致（`docs/features/nixl_connector_compatibility.md:74-82`） |
| `Avg xfer time (ms)` / `P90 xfer time (ms)` / `Throughput (MB/s)` | `decode.log`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py:91-125`） | KV 传输指标，由 `get_kv_connector_stats` 周期性 reduce 并经 CLI logging 输出 |
| `Num successful transfers` | 同上（`stats.py:92,117`） | 区间内成功传输次数 |
| `kv_load_failure_policy` / `load_errors` | `decode.log` | 加载失败计数，配合 `fail` 策略定位坏块 |
| `ERROR` / `Traceback` | 所有日志 | 通用错误筛查 |
| `Initialized N prefill clients and N decode clients` | `proxy.log`（`toy_proxy_server.py:70-73`） | proxy 启动完成、客户端池就绪 |

快速过滤示例：

```bash
# handshake 是否完成
grep -nE "handshake|compatibility hash" develop/logs/prefill.log develop/logs/decode.log

# KV 传输吞吐与延迟
grep -nE "Avg xfer time|P90 xfer time|Throughput \(MB/s\)|Num successful" develop/logs/decode.log

# 错误
grep -nE "ERROR|Traceback|load_errors|Failed handshake" develop/logs/*.log
```

---

## 7. 与仓库现有脚本的关系

本套脚本在思路上参考仓库内两份现有脚本，但针对 **单机 4+4 H200** 与 **transport 一键切换** 做了参数化收敛：

| 参考脚本 | 借鉴点 | 本设计的差异 |
|----------|--------|--------------|
| `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh` | `KV_CONFIG_P`/`KV_CONFIG_D` 按 `kv_buffer_device` 拼接（`:53-59`）；`ENFORCE_EAGER` 开关（`:79`）；`wait_for_server` 健康轮询（`:95-101`）；`VLLM_KV_CACHE_LAYOUT`/`UCX_NET_DEVICES`/`VLLM_NIXL_SIDE_CHANNEL_PORT` env 前缀（`:155-164`）；`pkill -f "vllm serve"` 清理（`:104-108`）；P→D→proxy 启动顺序（`:249-276`） | 原脚本面向 CI 多模型批量正确性测试，GPU 自动分配、proxy 端口固定 `8192`、无 transport 概念；本设计固定 4+4 GPU 划分、参数化 `TRANSPORT=gdr\|cpu`、proxy 端口与 P/D 端口集中到 `env.sh`、加入 pid 管理与一键启停 |
| `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_vllm_launcher.sh` | `UCX_TLS=cuda_ipc,cuda_copy,tcp`（`:31,47`）；`CUDA_VISIBLE_DEVICES` 单卡隔离；`--enforce-eager`；8100/8200 端口约定 | 原脚本只起单个 prefiller/decoder（LMCache connector，非 NIXL），需手动分两次调用；本设计改用 NixlConnector、TP=4、4 卡一组，并由 `run_pd.sh` 串联 |
| `docs/features/nixl_connector_usage.md:76-100` | NIXL producer/consumer 官方启动样例（`CUDA_VISIBLE_DEVICES`、`UCX_NET_DEVICES=all`、`VLLM_NIXL_SIDE_CHANNEL_PORT=5600/5601`、`--kv-transfer-config` JSON、`--enforce-eager`） | 官方样例为 TP=1 单卡 toy 模型；本设计扩展到 TP=4 MiniMax-M2.5，并显式注入 `kv_buffer_device` 以支持 transport 切换 |
| `docs/features/nixl_connector_compatibility.md:74-82` | 兼容性哈希约束（模型/dtype/attention backend/cache_dtype 必须一致；TP、block-size 可异构） | 指导 `launch_prefill.sh`/`launch_decode.sh` 保持 P/D 模型与布局一致，仅 GPU 集与 role 不同 |

环境变量对照：`VLLM_NIXL_SIDE_CHANNEL_HOST`/`PORT` 默认值与用法见 `vllm/envs.py:202-203`，单机部署
host 保持 `localhost`，P、D 各占一个 side channel port（`5600`/`5601`）。`vllm/envs.py:204-226` 的
`VLLM_MOONCAKE_*` 系列仅用于 Mooncake connector，本设计不涉及。
