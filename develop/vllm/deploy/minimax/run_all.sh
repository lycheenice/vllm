#!/usr/bin/env bash
# MiniMax-M2.5 + vLLM PD: master orchestrator for all phases
# Usage: ./run_all.sh <phase> [action]
#   phase: baseline | pd-basic | pd-bidir | stop | bench
#   action: start | stop | bench (default: start)
set -uo pipefail

VLLM_IMG=docker.1ms.run/vllm/vllm-openai:v0.25.0
MODEL_PATH=/data1/models/MiniMax-M2.5
MODEL_NAME=MiniMax-M2.5
BENCH_DIR=/home/lychee/mycode/kvcache-benchmarks
SCRIPT_DIR=/opt/vllm-minimax-pd
RESULTS_DIR=/opt/vllm-minimax-pd/results

wait_health() {
  local port="$1" name="$2" max="${3:-300}"
  for i in $(seq 1 "$max"); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] $name healthy (port $port)"
      return 0
    fi
    sleep 3
  done
  echo "ERR: $name not healthy after ${max}x3s"
  return 1
}

smoke_test() {
  local port="$1" model="$2"
  echo "[$(date +%H:%M:%S)] Smoke test on :$port ..."
  local resp
  resp=$(curl -sf -m 60 "http://127.0.0.1:$port/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}" 2>&1)
  if echo "$resp" | grep -q "choices"; then
    echo "  OK: $(echo "$resp" | head -c 200)"
  else
    echo "  FAIL: $resp"
    return 1
  fi
}

# Common vLLM serve args
COMMON_ARGS="--trust-remote-code --max-model-len 100000 --gpu-memory-utilization 0.90 --max-num-seqs 128 --enable-prefix-caching --kv-cache-dtype fp8 --enable-expert-parallel"

# ─── Phase 1: TP8 Baseline ───────────────────────────────────────────
start_baseline() {
  echo "=== Phase 1: TP8 Baseline ==="
  docker rm -f minimax-baseline 2>/dev/null || true
  docker run -d --name minimax-baseline --network host --gpus all \
    --shm-size 16g \
    -v "$MODEL_PATH:/models/MiniMax-M2.5" \
    -v "$RESULTS_DIR/baseline/logs:/logs" \
    --restart unless-stopped \
    "$VLLM_IMG" \
    --model /models/MiniMax-M2.5 \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 8 \
    --port 8000 \
    $COMMON_ARGS
  wait_health 8000 "baseline" 300 || { echo "FAIL baseline"; exit 1; }
  smoke_test 8000 "$MODEL_NAME" || { echo "FAIL smoke"; docker logs minimax-baseline --tail 50; exit 1; }
  echo "Phase 1 baseline ready on :8000"
}

# ─── Phase 2: PD Basic (no bidirectional) ────────────────────────────
start_pd_basic() {
  echo "=== Phase 2: PD Basic (no bidir) ==="
  docker rm -f minimax-prefill minimax-decode minimax-proxy 2>/dev/null || true
  KV_CFG_P='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"kv_lease_duration":60}}'
  KV_CFG_D='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

  docker run -d --name minimax-prefill --network host --gpus all \
    --shm-size 16g -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    -e UCX_TLS=cuda_ipc,cuda_copy,tcp -e UCX_NET_DEVICES=all \
    -v "$MODEL_PATH:/models/MiniMax-M2.5" \
    --restart unless-stopped \
    "$VLLM_IMG" \
    --model /models/MiniMax-M2.5 \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 4 \
    --port 8100 \
    --kv-transfer-config "$KV_CFG_P" \
    $COMMON_ARGS
  wait_health 8100 "prefill" 300 || { echo "FAIL prefill"; docker logs minimax-prefill --tail 50; exit 1; }

  docker run -d --name minimax-decode --network host --gpus all \
    --shm-size 16g -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
    -e UCX_TLS=cuda_ipc,cuda_copy,tcp -e UCX_NET_DEVICES=all \
    -v "$MODEL_PATH:/models/MiniMax-M2.5" \
    --restart unless-stopped \
    "$VLLM_IMG" \
    --model /models/MiniMax-M2.5 \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 4 \
    --port 8200 \
    --kv-transfer-config "$KV_CFG_D" \
    $COMMON_ARGS
  wait_health 8200 "decode" 300 || { echo "FAIL decode"; docker logs minimax-decode --tail 50; exit 1; }

  # Proxy: disagg_proxy_demo.py
  docker run -d --name minimax-proxy --network host \
    -v "$SCRIPT_DIR/disagg_proxy_demo.py:/proxy.py:ro" \
    --restart unless-stopped \
    --entrypoint python3 \
    "$VLLM_IMG" /proxy.py \
    --model "$MODEL_NAME" \
    --prefill localhost:8100 --decode localhost:8200 \
    --port 8000
  sleep 3
  smoke_test 8000 "$MODEL_NAME" || { echo "FAIL proxy smoke"; docker logs minimax-proxy --tail 30; exit 1; }
  echo "Phase 2 PD-basic ready: proxy :8000, P :8100, D :8200"
}

# ─── Phase 3: PD + Bidirectional ─────────────────────────────────────
start_pd_bidir() {
  echo "=== Phase 3: PD + Bidirectional ==="
  docker rm -f minimax-prefill minimax-decode minimax-proxy 2>/dev/null || true
  KV_CFG_P='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"bidirectional_kv_xfer":true,"kv_lease_duration":60}}'
  KV_CFG_D='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"bidirectional_kv_xfer":true}}'

  docker run -d --name minimax-prefill --network host --gpus all \
    --shm-size 16g -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    -e UCX_TLS=cuda_ipc,cuda_copy,tcp -e UCX_NET_DEVICES=all \
    -v "$MODEL_PATH:/models/MiniMax-M2.5" \
    --restart unless-stopped \
    "$VLLM_IMG" \
    --model /models/MiniMax-M2.5 \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 4 \
    --port 8100 \
    --kv-transfer-config "$KV_CFG_P" \
    $COMMON_ARGS
  wait_health 8100 "prefill" 300 || { echo "FAIL prefill"; docker logs minimax-prefill --tail 50; exit 1; }

  docker run -d --name minimax-decode --network host --gpus all \
    --shm-size 16g -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
    -e UCX_TLS=cuda_ipc,cuda_copy,tcp -e UCX_NET_DEVICES=all \
    -v "$MODEL_PATH:/models/MiniMax-M2.5" \
    --restart unless-stopped \
    "$VLLM_IMG" \
    --model /models/MiniMax-M2.5 \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 4 \
    --port 8200 \
    --kv-transfer-config "$KV_CFG_D" \
    $COMMON_ARGS
  wait_health 8200 "decode" 300 || { echo "FAIL decode"; docker logs minimax-decode --tail 50; exit 1; }

  # Proxy: multiturn with auto conversation_id
  docker run -d --name minimax-proxy --network host \
    -v "$SCRIPT_DIR/disagg_proxy_multiturn_autoid.py:/proxy.py:ro" \
    --restart unless-stopped \
    --entrypoint python3 \
    "$VLLM_IMG" /proxy.py \
    --host 0.0.0.0 --port 8000 \
    --prefiller-host localhost --prefiller-port 8100 \
    --decoder-host localhost --decoder-port 8200
  sleep 3
  smoke_test 8000 "$MODEL_NAME" || { echo "FAIL proxy smoke"; docker logs minimax-proxy --tail 30; exit 1; }
  echo "Phase 3 PD-bidir ready: proxy :8000, P :8100, D :8200"
}

# ─── Bench runner ────────────────────────────────────────────────────
run_bench() {
  local case_name="${1:?Usage: run_bench <case_name> [levels]}"
  local levels="${2:-1,4,16,32,64}"
  local trials="${3:-5}"
  echo "=== Bench: case=$case_name levels=$levels trials=$trials ==="
  docker rm -f mm-bench 2>/dev/null || true
  docker run -d --name mm-bench --rm --network host \
    -v "$BENCH_DIR:/bench" -w /bench \
    --entrypoint python3 \
    "$VLLM_IMG" scripts/ramp_test.py \
      --dataset data/codex_swebenchpro_traces/codex_swebenchpro.json \
      --endpoint http://127.0.0.1:8000/v1 \
      --model "$MODEL_NAME" --api-key EMPTY \
      --server vllm --case "$case_name" \
      --levels "$levels" \
      --trials-per-user "$trials" \
      --max-tokens 4096
  echo "bench started: docker logs -f mm-bench"
}

# ─── Stop all ────────────────────────────────────────────────────────
stop_all() {
  docker rm -f minimax-baseline minimax-prefill minimax-decode minimax-proxy mm-bench 2>/dev/null || true
  echo "all stopped"
}

# ─── Main ────────────────────────────────────────────────────────────
PHASE="${1:-}"
ACTION="${2:-start}"
case "$PHASE" in
  baseline)   start_baseline ;;
  pd-basic)   start_pd_basic ;;
  pd-bidir)   start_pd_bidir ;;
  stop)       stop_all ;;
  bench)      run_bench "$ACTION" "${3:-}" "${4:-}" ;;
  *) echo "Usage: $0 {baseline|pd-basic|pd-bidir|stop|bench} [args]"; exit 1 ;;
esac
