# test2-mooncake-pd

**TP4+TP4 PD 分离,MooncakeConnector(P2P),KV 直传**。与 test1 只差 connector 实现。

## 配置
关键参数见 [`config.env`](./config.env)。公共默认见 [`../../common/common.env`](../../common/common.env)。
模型 MiniMax-M2.5;vLLM **v0.25.0**(docker `vllm-openai:v0.25.0`);执行机 **h200-2**。

## 代码改动
无 vLLM 源码改动,但 **需容器内安装 `mooncake.engine`**。

## 运行(前置:h200-2 的 8 张 H200 空闲)
| 步骤 | 在哪执行 | 命令 |
|------|---------|------|
| 起服务 | h200-2 本机 | `bash run.sh` |
| 发压   | a100-2 / 监控端 | `bash bench.sh` |
| 状态   | h200-2 本机 | `bash ../../common/status.sh test2-mooncake-pd 8110 8210 8010` |
| 日志   | a100-2(经 /ceph) | `tail -f results/logs/*.log` |
| 停止   | h200-2 本机 | `bash stop.sh` |

从 a100-2 一键起服务:`ssh -l root h200-2 "bash /ceph/User/E01223/mycode/vllm/develop/experiments/test2-mooncake-pd/run.sh"`

## 发压口径(统一)
并发 `1,4,8,16` · `--max-turns 4` · `--trials-per-user 4` · `--max-tokens 256`
(改 common.env 或 `LEVELS=.. bash bench.sh` 覆盖)。目标每组 <10min,首跑用 C=1 校准时长。

## 关注指标 / 预期
- 与 test1 横向对比:nixl vs mooncake 的传输实现在同硬件上的性能差异。
- 首要关注能否稳定起来 + bootstrap 握手是否成功。

## 结果归档
`results/test2-mooncake-pd_<时间戳>/summary.json`(bench.sh 自动从 h200-6 拉回)。日志在 `results/logs/`。
## 前置依赖
镜像默认无 `mooncake.engine`。起服务前在容器内(或做一个带 mooncake 的派生镜像):
```bash
docker exec <container> pip install mooncake-transfer-engine
docker exec <container> python -c 'import mooncake.engine; print("ok")'
```
建议固化成派生镜像 `vllm-openai:v0.25.0-mooncake` 以便复用(在 h200-6 构建后传 h200-2)。

## 执行记录
> 跑完填:日期 / 镜像 tag / 关键发现 / 踩坑 / 与基线对比结论。
