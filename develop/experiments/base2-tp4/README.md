# base2-tp4

单实例 **4 卡 TP4** 基线(**只用 GPU 0-3,另 4 张空闲**)。

## 配置
关键参数见 [`config.env`](./config.env)。公共默认见 [`../../common/common.env`](../../common/common.env)。
模型 MiniMax-M2.5;vLLM **v0.25.0**(docker `vllm-openai:v0.25.0`);执行机 **h200-2**。

## 代码改动
无。使用上游 vLLM,不带 `--kv-transfer-config`。

## 运行(前置:h200-2 的 8 张 H200 空闲)
| 步骤 | 在哪执行 | 命令 |
|------|---------|------|
| 起服务 | h200-2 本机 | `bash run.sh` |
| 发压   | a100-2 / 监控端 | `bash bench.sh` |
| 状态   | h200-2 本机 | `bash ../../common/status.sh base2-tp4 8081` |
| 日志   | a100-2(经 /ceph) | `tail -f results/logs/*.log` |
| 停止   | h200-2 本机 | `bash stop.sh` |

从 a100-2 一键起服务:`ssh -l root h200-2 "bash /ceph/User/E01223/mycode/vllm/develop/experiments/base2-tp4/run.sh"`

## 发压口径(统一)
并发 `1,4,8,16` · `--max-turns 4` · `--trials-per-user 4` · `--max-tokens 256`
(改 common.env 或 `LEVELS=.. bash bench.sh` 覆盖)。目标每组 <10min,首跑用 C=1 校准时长。

## 关注指标 / 预期
- 与 base1 对比 = 纯 **TP8→TP4 算力减半** 的代价。
- 与 test1 对比:PD 的 D 节点也是 TP4,可据此把 PD 的差异拆成'算力'与'分离开销'两部分。

## 结果归档
`results/base2-tp4_<时间戳>/summary.json`(bench.sh 自动从 h200-6 拉回)。日志在 `results/logs/`。

## 执行记录
> 跑完填:日期 / 镜像 tag / 关键发现 / 踩坑 / 与基线对比结论。

**2026-07-21 夜** · `vllm-openai:v0.25.0` · MAX_MODEL_LEN=65536 · 结果 `results/base2-tp4_20260721_214626/`。
- TP4 无需 EP(gate/up 分片 384÷128=3 ✓),直接起。GPU 0-3,4-7 空闲。
- **结果**(C=1/4/8/16):吞吐 14/55/109/192 tok/s,E2E 17.1/16.4/16.8/19.4s,TPOT 59~77ms。
- **对比 base1**:C=16 192 vs 238 tok/s → **TP8→TP4 算力代价 −19%**(非线性减半,该负载受 context/延迟约束)。
- 作为 PD 方案(同 TP4 算力)的公平分母。跨实验对比见 `../../results_report/PD_COMPARISON.md`。
