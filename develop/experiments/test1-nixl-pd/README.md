# test1-nixl-pd

**TP4+TP4 PD 分离,NixlConnector,KV 走 GPU 直传**(NVLink/cuda_ipc)。PD 主基线。

## 配置
关键参数见 [`config.env`](./config.env)。公共默认见 [`../../common/common.env`](../../common/common.env)。
模型 MiniMax-M2.5;vLLM **v0.25.0**(docker `vllm-openai:v0.25.0`);执行机 **h200-2**。

## 代码改动
无。上游 NixlConnector,`kv_buffer_device=cuda`。

## 运行(前置:h200-2 的 8 张 H200 空闲)
| 步骤 | 在哪执行 | 命令 |
|------|---------|------|
| 起服务 | h200-2 本机 | `bash run.sh` |
| 发压   | a100-2 / 监控端 | `bash bench.sh` |
| 状态   | h200-2 本机 | `bash ../../common/status.sh test1-nixl-pd 8100 8200 8000` |
| 日志   | a100-2(经 /ceph) | `tail -f results/logs/*.log` |
| 停止   | h200-2 本机 | `bash stop.sh` |

从 a100-2 一键起服务:`ssh -l root h200-2 "bash /ceph/User/E01223/mycode/vllm/develop/experiments/test1-nixl-pd/run.sh"`

## 发压口径(统一)
并发 `1,4,8,16` · `--max-turns 4` · `--trials-per-user 4` · `--max-tokens 256`
(改 common.env 或 `LEVELS=.. bash bench.sh` 覆盖)。目标每组 <10min,首跑用 C=1 校准时长。

## 关注指标 / 预期
- 对比 base1/base2:复现并量化 PD 分离的净收益/净损失。
- 关注 TTFT(含 P→D KV 传输)、C=16 时 TPOT、错误率。
- ⚠ `kv_role` 目前用 producer/consumer;既有分析怀疑 v0.25.0 应为 `kv_both`,列为待验证项。

## 结果归档
`results/test1-nixl-pd_<时间戳>/summary.json`(bench.sh 自动从 h200-6 拉回)。日志在 `results/logs/`。

## 执行记录
> 跑完填:日期 / 镜像 tag / 关键发现 / 踩坑 / 与基线对比结论。

**2026-07-21 夜** · `vllm-openai:v0.25.0` · kv_role=pc(verify-kvrole 实测 pc 有效)· 结果 `results/test1-nixl-pd_20260721_224311/`。
- **踩坑(影响所有 PD)**:serve_pd.sh proxy 用 `python`(容器只有 `python3`)→ proxy 不启动、冒烟卡死;
  且 toy_proxy 默认绑 127.0.0.1、发压端打不到。已修:`python3` + `--host 0.0.0.0`。
- **结果**(C=1/4/8/16):吞吐 12/47/83/118 tok/s,E2E 18.5/19.6/22.0/29.9s,TTFT p50 4190→15637ms,TPOT ~60-63ms(平稳)。
- **对比 base2(同 TP4 算力)**:C=16 118 vs 192 tok/s → **PD 分离本身开销 −38%**;TTFT 爆炸(15637 vs 489ms,~32×)。
- TPOT 是 PD 唯一优势(专用 decode,平稳 63ms vs base2 77ms)。跨实验对比见 `../../results_report/PD_COMPARISON.md`。
