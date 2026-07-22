# test3-nixl-cpu-bypass

**TP4+TP4 PD 分离,NixlConnector + CPU 绕行**(`kv_buffer_device=cpu`,D2H→传输→H2D)。

## 配置
关键参数见 [`config.env`](./config.env)。公共默认见 [`../../common/common.env`](../../common/common.env)。
模型 MiniMax-M2.5;vLLM **v0.25.0**(docker `vllm-openai:v0.25.0`);执行机 **h200-2**。

## 代码改动
**无需改代码**(既有能力):BYPASS=cpu 即把 kv_buffer_device 设为 cpu,等价既有 scripts 的 TRANSPORT=cpu。

## 运行(前置:h200-2 的 8 张 H200 空闲)
| 步骤 | 在哪执行 | 命令 |
|------|---------|------|
| 起服务 | h200-2 本机 | `bash run.sh` |
| 发压   | a100-2 / 监控端 | `bash bench.sh` |
| 状态   | h200-2 本机 | `bash ../../common/status.sh test3-nixl-cpu-bypass 8120 8220 8020` |
| 日志   | a100-2(经 /ceph) | `tail -f results/logs/*.log` |
| 停止   | h200-2 本机 | `bash stop.sh` |

从 a100-2 一键起服务:`ssh -l root h200-2 "bash /ceph/User/E01223/mycode/vllm/develop/experiments/test3-nixl-cpu-bypass/run.sh"`

## 发压口径(统一)
并发 `1,4,8,16` · `--max-turns 4` · `--trials-per-user 4` · `--max-tokens 256`
(改 common.env 或 `LEVELS=.. bash bench.sh` 覆盖)。目标每组 <10min,首跑用 C=1 校准时长。

## 关注指标 / 预期
- 对比 test1(GPU 直传):量化 CPU 中转的额外 D2H/H2D 延迟。
- 验证 CPU 侧 KV 缓冲容量优势(宿主内存远大于显存)在高并发/长 context 下是否换来稳定性。

## 结果归档
`results/test3-nixl-cpu-bypass_<时间戳>/summary.json`(bench.sh 自动从 h200-6 拉回)。日志在 `results/logs/`。

## 执行记录
> 跑完填:日期 / 镜像 tag / 关键发现 / 踩坑 / 与基线对比结论。

**2026-07-21 夜** · `vllm-openai:v0.25.0` · kv_buffer_device=cpu · 结果 `results/test3-nixl-cpu-bypass_20260721_231532/`。
- decode 日志实测 **KV xfer ~1.1s/次**(CPU 中转 D2H→NIXL→H2D)。
- **结果**(C=1/4/8/16):吞吐 12/46/80/127 tok/s,E2E 19.4/19.6/22.3/27.5s,TTFT p50 5227→6913ms,TPOT 61~85ms。
- **对比 test1(GPU 直传)**:C=16 127 vs 118 tok/s → **CPU 中转 ≈ GPU 直传(+8%,噪声内)**。
- **结论**:KV 传输方式不是瓶颈——1.1s/次相对 20-30s 多轮 E2E 占比极低;PD 分离的结构性开销才是主因。
  跨实验对比见 `../../results_report/PD_COMPARISON.md`。
