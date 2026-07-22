#!/usr/bin/env bash
# develop/common/serve_pd.sh
# 通用起服务逻辑。★在 h200-2 本机执行★(脚本经 /ceph 可见,docker 操作本机)。
# 由各实验 experiments/<exp>/run.sh 调用:先 source config.env,再 source 本文件。
#
# config.env 需 export 的变量:
#   EXP_NAME              实验名(容器名 / 结果目录前缀)
#   MODE                  single | pd
#   CONNECTOR             none | nixl | mooncake
#   BYPASS                none | cpu           (KV 是否走 CPU 中转)
#   TP                    张量并行度
#   GPUS                  single 模式的 GPU 列表,如 "0,1,2,3,4,5,6,7"
#   P_GPUS / D_GPUS       pd 模式的 P / D GPU 列表
#   PORT                  single 模式服务端口
#   PORT_P / PORT_D       pd 模式 P / D 端口
#   PROXY_PORT            pd 模式 proxy 端口
#   SIDE_PORT_P/SIDE_PORT_D  nixl 侧信道端口
#   BOOTSTRAP_PORT        mooncake bootstrap 端口
#   VLLM_CODE_OVERRIDE    (可选)1 => bind-mount 本实验 code/vllm 覆盖镜像内 vllm
#   VLLM_PKG_PATH         (可选)容器内 vllm 包路径,配合上一项
#   EXTRA_SERVE_ARGS      (可选)追加给 vllm serve 的参数字符串
set -euo pipefail
source "$COMMON_DIR/lib.sh"

RESULT_DIR="$DEV_ROOT/experiments/$EXP_NAME/results"
LOG_DIR="$RESULT_DIR/logs"; mkdir -p "$LOG_DIR"
CODE_DIR="$DEV_ROOT/experiments/$EXP_NAME/code"

# --- 构造 kv-transfer-config(按 connector × bypass)---
kv_config() {   # $1=role: kv_producer|kv_consumer
  local role="$1"
  # KV_ROLE_MODE=both -> 统一用 kv_both(v0.25.0 官方集成测试口径);默认 pc 保持不变。
  [[ "${KV_ROLE_MODE:-pc}" == "both" ]] && role="kv_both"
  case "$CONNECTOR" in
    none) echo "" ;;
    nixl)
      local dev="cuda"; [[ "$BYPASS" == "cpu" ]] && dev="cpu"
      echo '{"kv_connector":"NixlConnector","kv_role":"'"$role"'","kv_buffer_device":"'"$dev"'","kv_load_failure_policy":"fail"}'
      ;;
    mooncake)
      # test2: 上游 MooncakeConnector(P2P)。test4 的 CPU 绕行在 code/ 里自定义,
      # 通过 VLLM_CODE_OVERRIDE 覆盖后仍复用此配置(kv_role 不变)。
      # 单机 P/D:CPU 端已验证 rdma + MC_GID_INDEX=3(见下方 env)可通(数据实测送达);
      # tcp 路径此前失败,故默认 rdma(MOONCAKE_PROTOCOL 可覆盖)。device_name 留空自动选。
      echo '{"kv_connector":"MooncakeConnector","kv_role":"'"$role"'","kv_connector_extra_config":{"mooncake_protocol":"'"${MOONCAKE_PROTOCOL:-rdma}"'","device_name":"'"${MOONCAKE_DEVICE:-}"'"}}'
      ;;
    *) die "未知 CONNECTOR=$CONNECTOR" ;;
  esac
}

# --- 起单个 vllm serve 容器 ---
launch_instance() {   # $1=tag(prefill/decode/single) $2=gpus $3=port $4=kv_json $5=role
  local tag="$1" gpus="$2" port="$3" kv_json="$4" role="$5"
  local name="${EXP_NAME}-${tag}"
  log "启动 $name: GPUs=$gpus port=$port TP=$TP connector=$CONNECTOR bypass=$BYPASS"

  local d=( docker run -d --name "$name" --network host --ipc=host --shm-size=32g
            --gpus "\"device=$gpus\""
            -v "$MODEL_DIR_HOST:$MODEL_DIR_HOST"
            -v "$REPO_ON_EXEC:$REPO_ON_EXEC"
            "${COMMON_DOCKER_ENV[@]}" )

  # connector 专属环境
  if [[ "$CONNECTOR" == "nixl" ]]; then
    local side=$SIDE_PORT_P; [[ "$role" == "kv_consumer" ]] && side=$SIDE_PORT_D
    d+=( -e VLLM_KV_CACHE_LAYOUT=HND -e UCX_NET_DEVICES=all
         -e UCX_TLS=cuda_ipc,cuda_copy,tcp -e VLLM_NIXL_SIDE_CHANNEL_PORT="$side" )
  elif [[ "$CONNECTOR" == "mooncake" ]]; then
    d+=( -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$BOOTSTRAP_PORT" )
    # mooncake TransferEngine 走 RDMA/RoCE:需把 host 的 IB 字符设备透传进容器
    # (仅挂 /sys 不够,topology 探测 uverbs 需 /dev/infiniband)+ IPC_LOCK 供 pinned mem 注册。
    d+=( --cap-add=IPC_LOCK --ulimit memlock=-1:-1 )
    # RoCEv2 GID index:h200-2 各 mlx5 的 RoCEv2 GID 在 index 3(RoCEv1 在 0/2)。mooncake 不会
    # 自动选,不设会报 "GID is NULL / GID -1 / No available RNIC"。CPU 端实测 MC_GID_INDEX=3 时
    # initialize 返回 0(mlx5_1/mlx5_bond_0/auto 均通)。用 `show_gids` 确认目标机的 v2 index。
    d+=( -e MC_GID_INDEX="${MC_GID_INDEX:-3}" )
    if [[ -d /dev/infiniband ]]; then
      for ibdev in /dev/infiniband/*; do d+=( --device "$ibdev" ); done
    fi
  fi

  # 代码覆盖(test4:用本实验 code/vllm 替换镜像内 vllm)
  if [[ "${VLLM_CODE_OVERRIDE:-0}" == "1" ]]; then
    [[ -n "${VLLM_PKG_PATH:-}" ]] || die "VLLM_CODE_OVERRIDE=1 需同时设 VLLM_PKG_PATH"
    [[ -d "$CODE_DIR/vllm" ]] || die "未找到 $CODE_DIR/vllm(代码覆盖目录)"
    d+=( -v "$CODE_DIR/vllm:$VLLM_PKG_PATH" )
    log "  [code override] $CODE_DIR/vllm -> $VLLM_PKG_PATH"
  fi

  d+=( "$IMAGE" "$MODEL_PATH" --port "$port"
       --tensor-parallel-size "$TP" --block-size "$BLOCK_SIZE"
       --gpu-memory-utilization "$UTIL" --max-model-len "$MAX_MODEL_LEN" --trust-remote-code )
  [[ "$ENABLE_PREFIX_CACHING" == "1" ]] && d+=( --enable-prefix-caching )
  [[ "$ENFORCE_EAGER"        == "1" ]] && d+=( --enforce-eager )
  [[ -n "$kv_json" ]] && d+=( --kv-transfer-config "$kv_json" )
  [[ -n "${EXTRA_SERVE_ARGS:-}" ]] && d+=( ${EXTRA_SERVE_ARGS} )

  "${d[@]}"
  # 把容器日志实时抽到 /ceph,供 a100-2 直接 tail
  ( docker logs -f "$name" >"$LOG_DIR/$tag.log" 2>&1 & echo $! >"$LOG_DIR/$tag.logpid" )
}

# --- 起 proxy ---
launch_proxy() {
  local name="${EXP_NAME}-decode"   # 借 decode 容器的 vllm python 环境跑 proxy
  log "在 $name 内启动 proxy(port=$PROXY_PORT)"
  if [[ "$CONNECTOR" == "nixl" ]]; then
    docker exec -d "$name" bash -lc \
      "python3 $REPO_ON_EXEC/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
         --port $PROXY_PORT --host 0.0.0.0 --prefiller-hosts localhost --prefiller-ports $PORT_P \
         --decoder-hosts localhost --decoder-ports $PORT_D \
         >$LOG_DIR/proxy.log 2>&1"
  elif [[ "$CONNECTOR" == "mooncake" ]]; then
    docker exec -d "$name" bash -lc \
      "python3 $REPO_ON_EXEC/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py \
         --prefill http://0.0.0.0:$PORT_P $BOOTSTRAP_PORT --decode http://0.0.0.0:$PORT_D \
         --port $PROXY_PORT --host 0.0.0.0 >$LOG_DIR/proxy.log 2>&1"
  fi
}

# --- 主流程 ---
main() {
  if [[ "$MODE" == "single" ]]; then
    launch_instance single "$GPUS" "$PORT" "$(kv_config kv_producer)" kv_producer
    wait_health_remote "$PORT" "single" 2>/dev/null || wait_local_health "$PORT" single
    smoke_local "$PORT"
    log "single 就绪: http://$EXEC_IP:$PORT (发压端点)"
  else
    launch_instance prefill "$P_GPUS" "$PORT_P" "$(kv_config kv_producer)" kv_producer
    wait_local_health "$PORT_P" prefill
    launch_instance decode  "$D_GPUS" "$PORT_D" "$(kv_config kv_consumer)" kv_consumer
    wait_local_health "$PORT_D" decode
    launch_proxy
    sleep 4
    smoke_local "$PROXY_PORT"
    log "PD 就绪: proxy=http://$EXEC_IP:$PROXY_PORT (发压端点)  P=$PORT_P D=$PORT_D"
  fi
  cat <<EOF
============================================================
[$EXP_NAME] 已启动 (mode=$MODE connector=$CONNECTOR bypass=$BYPASS)
  发压端点 : http://$EXEC_IP:$([[ $MODE == single ]] && echo $PORT || echo $PROXY_PORT)/v1
  日志     : $LOG_DIR/*.log  (a100-2 可直接 tail)
  停止     : bash $DEV_ROOT/common/stop.sh $EXP_NAME
============================================================
EOF
}

# 本机 health(serve 在 h200-2 本机跑时用 localhost)
wait_local_health() {
  local port="$1" name="$2" waited=0
  log "等待 $name 健康 (localhost:$port/health, 超时 ${HEALTH_TIMEOUT}s) ..."
  while (( waited < HEALTH_TIMEOUT )); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { log "$name 已健康"; return 0; }
    sleep 5; waited=$((waited+5))
  done
  die "$name 未在 ${HEALTH_TIMEOUT}s 内就绪,见 $LOG_DIR/$name.log"
}
smoke_local() {
  local port="$1"
  log "冒烟 -> localhost:$port/v1/completions"
  curl -s --max-time 180 "http://127.0.0.1:$port/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_PATH\",\"prompt\":\"The quick brown fox\",\"max_tokens\":16,\"temperature\":0}" \
    | head -c 400; echo
}

main "$@"
