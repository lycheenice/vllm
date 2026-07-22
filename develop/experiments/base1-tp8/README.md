# base1-tp8

单实例 **8 卡 TP8** 基线,无 PD 分离。所有 PD 方案的性能对照上界。

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
| 状态   | h200-2 本机 | `bash ../../common/status.sh base1-tp8 8080` |
| 日志   | a100-2(经 /ceph) | `tail -f results/logs/*.log` |
| 停止   | h200-2 本机 | `bash stop.sh` |

从 a100-2 一键起服务:`ssh -l root h200-2 "bash /ceph/User/E01223/mycode/vllm/develop/experiments/base1-tp8/run.sh"`

## 发压口径(统一)
并发 `1,4,8,16` · `--max-turns 4` · `--trials-per-user 4` · `--max-tokens 256`
(改 common.env 或 `LEVELS=.. bash bench.sh` 覆盖)。目标每组 <10min,首跑用 C=1 校准时长。

## 关注指标 / 预期
- 记录 TTFT / TPOT / 吞吐(tok/s)/ 错误率 随并发 1→16 的曲线,作为其余实验的分母。
- 多轮场景下 prefix cache 命中率(预期高,单实例内 KV 复用)。

## 结果归档
`results/base1-tp8_<时间戳>/summary.json`(bench.sh 自动从 h200-6 拉回)。日志在 `results/logs/`。

## 执行记录
> 跑完填:日期 / 镜像 tag / 关键发现 / 踩坑 / 与基线对比结论。

**2026-07-21 夜** · 镜像 `vllm-openai:v0.25.0` · 结果 `results/base1-tp8_20260721_212609/`(65536)。
- **踩坑**:纯 TP8 崩(FP8 blockquant:每卡 expert gate/up=192,192%128≠0,`fp8.py create_weights ValueError`)。
  解:加 `--enable-expert-parallel`(EXTRA_SERVE_ARGS)→ experts 整块分 rank 不切 gate/up,attention 仍 TP8。
- **踩坑**:32768 下 codex 多轮累积超长,~15% HTTP400;改 MAX_MODEL_LEN=65536(模型支持 196608)后清零。
- **结果**(C=1/4/8/16):吞吐 18/70/131/238 tok/s,E2E 12.0/13.3/13.9/15.0s,TTFT p50 280~380ms,TPOT 53~61ms。
- **结论**:整机上界。吞吐随并发线性扩展、E2E 稳定,余量充足。全面优于任何 TP4/PD 方案。
- 跨实验对比见 `../../results_report/PD_COMPARISON.md`。
