#!/usr/bin/env bash
# develop/common/bench_ramp.sh
# 统一发压。★从 a100-2 / 监控端执行★:内部 ssh 到 h200-6 跑 kvcache-benchmarks
# ramp_test.py(打 h200-2 的服务端口),跑完把结果目录拉回本地 /ceph 实验 results/。
#
# 由 experiments/<exp>/bench.sh 调用(先 source config.env,再 source 本文件),
# 需 export: EXP_NAME, 以及 single 模式 PORT 或 pd 模式 PROXY_PORT。
#
# 统一口径(common.env): LEVELS=1,4,8,16  MAX_TURNS=4  TRIALS_PER_USER=4  MAX_TOKENS=256
set -euo pipefail
source "$COMMON_DIR/lib.sh"

BENCH_PORT="${BENCH_PORT:-${PROXY_PORT:-${PORT:-}}}"
[[ -n "$BENCH_PORT" ]] || die "未确定发压端口(PROXY_PORT / PORT)"
ENDPOINT="http://$EXEC_IP:$BENCH_PORT/v1"

RESULT_DIR="$DEV_ROOT/experiments/$EXP_NAME/results"; mkdir -p "$RESULT_DIR"
STAMP="$($BENCH_SSH date +%Y%m%d_%H%M%S)"
REMOTE_OUT="$KVBENCH_DIR/results/${EXP_NAME}_${STAMP}"
KVBENCH_PY="${KVBENCH_PY:-python3}"

log "发压: case=$EXP_NAME endpoint=$ENDPOINT"
log "  levels=$LEVELS max-turns=$MAX_TURNS trials/user=$TRIALS_PER_USER max-tokens=$MAX_TOKENS"
log "  远端输出(h200-6): $REMOTE_OUT"

# 先探活,避免空跑
$BENCH_SSH "curl -sf $ENDPOINT/models -H 'Authorization: Bearer EMPTY' >/dev/null" \
  || die "从 h200-6 无法访问 $ENDPOINT(检查 h200-2 服务是否起 / 端口是否放通)"

$BENCH_SSH "cd $KVBENCH_DIR && $KVBENCH_PY scripts/ramp_test.py \
  --dataset '$DATASET' \
  --endpoint '$ENDPOINT' \
  --model '$MODEL_PATH' \
  --api-key EMPTY \
  --server vllm --case '$EXP_NAME' \
  --levels '$LEVELS' --max-turns $MAX_TURNS \
  --trials-per-user $TRIALS_PER_USER --max-tokens $MAX_TOKENS \
  --shuffle-seed $SHUFFLE_SEED \
  --output-dir '$REMOTE_OUT'"

log "拉回结果 -> $RESULT_DIR/$(basename "$REMOTE_OUT")"
$BENCH_SSH "tar czf - -C '$(dirname "$REMOTE_OUT")' '$(basename "$REMOTE_OUT")'" | tar xzf - -C "$RESULT_DIR/"
log "完成。summary: $RESULT_DIR/$(basename "$REMOTE_OUT")/summary.json"
