# test4-mooncake-cpu-bypass

**TP4+TP4 PD 分离,MooncakeConnector + CPU 绕行**。

## 配置
关键参数见 [`config.env`](./config.env)。公共默认见 [`../../common/common.env`](../../common/common.env)。
模型 MiniMax-M2.5;vLLM **v0.25.0**(docker `vllm-openai:v0.25.0`);执行机 **h200-2**。

## 代码改动
**需要开发**(上游无此开关)。设计与占位见 [`code/README.md`](./code/README.md);
开发产物放 `code/vllm/`,`config.env` 置 `VLLM_CODE_OVERRIDE=1` 后经 bind-mount 生效。

## 运行(前置:h200-2 的 8 张 H200 空闲)
| 步骤 | 在哪执行 | 命令 |
|------|---------|------|
| 起服务 | h200-2 本机 | `bash run.sh` |
| 发压   | a100-2 / 监控端 | `bash bench.sh` |
| 状态   | h200-2 本机 | `bash ../../common/status.sh test4-mooncake-cpu-bypass 8130 8230 8030` |
| 日志   | a100-2(经 /ceph) | `tail -f results/logs/*.log` |
| 停止   | h200-2 本机 | `bash stop.sh` |

从 a100-2 一键起服务:`ssh -l root h200-2 "bash /ceph/User/E01223/mycode/vllm/develop/experiments/test4-mooncake-cpu-bypass/run.sh"`

## 发压口径(统一)
并发 `1,4,8,16` · `--max-turns 4` · `--trials-per-user 4` · `--max-tokens 256`
(改 common.env 或 `LEVELS=.. bash bench.sh` 覆盖)。目标每组 <10min,首跑用 C=1 校准时长。

## 关注指标 / 预期
- 目标:验证 mooncake 经 CPU(DRAM)中转/共享池的可行性与收益,与 test3(nixl+cpu)横向对比。
- 里程碑:先跑通(能起 + 冒烟通过),再谈性能。

## 结果归档
`results/test4-mooncake-cpu-bypass_<时间戳>/summary.json`(bench.sh 自动从 h200-6 拉回)。日志在 `results/logs/`。
## 状态:🚧 待开发,当前不可运行
`VLLM_CODE_OVERRIDE=0` 时 run.sh 会按 test2 的 P2P 路径起(等价 mooncake 直传);
真正的 CPU 绕行需完成 code/ 开发并置 1。

## 执行记录
> 跑完填:日期 / 镜像 tag / 关键发现 / 踩坑 / 与基线对比结论。
