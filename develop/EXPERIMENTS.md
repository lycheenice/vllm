# MiniMax-M2.5 vLLM PD 分离实验课题 —— 总纲

**日期**: 2026-07-21 · **模型**: MiniMax-M2.5(w8a8 FP8, 62 层, GQA 48/8, 256 experts top-8)
**框架**: vLLM **v0.25.0**(镜像 `docker.1ms.run/vllm/vllm-openai:v0.25.0`)· **执行机**: h200-2(8×H200)

> 长期课题。本文件是入口:实验矩阵、统一口径、拓扑分工、运行手册、归档规范。
> 每个实验的细节在 `experiments/<exp>/README.md`。既有性能分析见
> [`vllm_pd_analysis_and_optimization_20260721.md`](./vllm_pd_analysis_and_optimization_20260721.md)。

---

## 1. 实验矩阵

| 实验 | GPU 切分 | connector | KV 路径 | 代码改动 | 目的 |
|------|---------|-----------|---------|---------|------|
| **base1-tp8** | 单实例 TP8 (0-7) | 无 | — | 无 | 上界基线 |
| **base2-tp4** | 单实例 TP4 (0-3,余空闲) | 无 | — | 无 | 隔离"算力减半" |
| **test1-nixl-pd** | P TP4 + D TP4 | NixlConnector | GPU 直传 | 无 | PD 主基线 |
| **test2-mooncake-pd** | P TP4 + D TP4 | MooncakeConnector | GPU 直传 | 无(需装 mooncake.engine) | nixl vs mooncake |
| **test3-nixl-cpu-bypass** | P TP4 + D TP4 | NixlConnector | **CPU 中转** | 无(既有 `kv_buffer_device=cpu`) | 量化 CPU 绕行开销 |
| **test4-mooncake-cpu-bypass** | P TP4 + D TP4 | MooncakeConnector | **CPU 中转** | **需开发** | mooncake CPU 绕行 |

对比逻辑:`base1 − base2` = 纯 TP8→TP4 算力代价;`test1 − base2` = PD 分离本身开销;
`test3 − test1` = CPU 中转 vs GPU 直传;`test4 ↔ test3` = 两种 connector 的 CPU 绕行横向对比。

## 2. 统一发压口径(用户指定)

由 `common/common.env` 固化,全实验一致,便于横向对比:

| 项 | 值 |
|----|----|
| 工具 | kvcache-benchmarks `ramp_test.py`(h200-6) |
| 数据集 | `codex_swebenchpro.json`(610 traces) |
| 并发 sweep | `--levels 1,4,8,16` |
| 轮次 | `--max-turns 4` |
| 每用户 trials | `--trials-per-user 4`(单组时长主旋钮) |
| 输出上限 | `--max-tokens 256` |
| 目标 | 每并发组 **<10min**;首跑先用 C=1 校准,必要时下调 trials/user 或 max-tokens |

## 3. 拓扑与执行/同步分工

| 机器 | 角色 | 说明 |
|------|------|------|
| **a100-2**(当前机) | 撰写 + 监控 | `/ceph/.../vllm/develop` 权威副本;Claude 在此;可 `ssh -l root` 两台 h200 |
| **h200-2** | 执行 | 8×H200;挂 `/ceph`,容器 **bind-mount `/ceph` 仓库** 直接执行(代码/proxy/脚本免二次同步);模型 `/data1/models/MiniMax-M2.5` |
| **h200-6** | 发压 + 拉镜像 | kvcache-benchmarks 在此,打 h200-2 内网 IP `10.118.89.32`;无 `/ceph` |

**同步流(遵循约定 [[feedback_vllm_pd_experiment_workflow]])**:本地 `/ceph` 写好 → `tar over ssh` 推到
`h200-6:/home/lychee/mycode/vllm/develop/`(留档 + 发压端参考)。**执行侧 h200-2 因挂载 /ceph,直接用
a100-2 的权威副本**,无需经 h200-6 中转。日志/结果落 `/ceph` 实验 `results/`,a100-2 直接 `tail`。

## 4. 目录结构与归档规范

```
develop/
├── EXPERIMENTS.md                  # 本文件(总纲)
├── common/                         # 跨实验共享 harness(单点维护,勿复制)
│   ├── common.env                  #   全局默认 + 统一 bench 口径
│   ├── serve_pd.sh                 #   起服务核心(参数化 connector×bypass×mode)
│   ├── bench_ramp.sh               #   统一 ramp 发压 + 结果回拉
│   ├── stop.sh / status.sh / lib.sh
└── experiments/<exp>/              # ★每实验独立自包含★
    ├── README.md                   #   目的/配置/命令/预期/执行记录
    ├── config.env                  #   该实验相对 common 的全部差异(唯一改点)
    ├── run.sh / bench.sh / stop.sh #   薄封装(内容统一,source common)
    ├── code/                       #   ★该实验的 vLLM 代码改动★(base/test1-3 为空)
    └── results/                    #   ★该实验的测试数据/报告★(bench 自动回拉)
        └── logs/                   #   容器日志(a100-2 可直接 tail)
```

**长期课题四类产物各自独立保存**:代码改动→`code/`;启动脚本→`config.env`+薄封装;
测试数据→`results/<exp>_<ts>/`;分析报告→各 `README.md` 执行记录 + 需要时的 `results/report.md`。
新增 case 时复制一个 `experiments/<exp>/` 骨架、改 `config.env` 即可,不动 `common/`。

## 5. 运行手册

```bash
# 0)【前置】h200-2 腾空 8 卡(见 §6)
# 1) 起服务(h200-2 本机;或 a100-2: ssh -l root h200-2 "bash <abs>/run.sh")
cd develop/experiments/test1-nixl-pd && bash run.sh
# 2) 发压(a100-2/监控端;内部 ssh h200-6 打 h200-2,结果回拉 results/)
bash bench.sh
# 3) 监控(a100-2 直接看 /ceph 日志)
tail -f results/logs/*.log
# 4) 停止(h200-2 本机)
bash stop.sh
```

建议顺序:base1 → base2 → test1 → test3 →(装好 mooncake)test2 →(开发完)test4。
每个实验 **先冒烟通过再 ramp**(serve_pd.sh 内置单请求冒烟)。

## 6. 前置阻塞与风险

- 🔴 **h200-2 8 卡当前被 SGLang 生产容器 `sglang-glm-sglang-1` 占满**(up 33h)。跑任何实验前需先停,
  属停生产服务,须用户确认后执行。
- 🟡 **test2/test4 依赖 `mooncake.engine`**,v0.25.0 镜像默认不含;建议构建派生镜像
  `vllm-openai:v0.25.0-mooncake`(h200-6 构建→传 h200-2)。
- 🟡 **kv_role 语义**:现用 `kv_producer/kv_consumer`;既有分析怀疑 v0.25.0 官方集成测试统一用
  `kv_both`。test1 首跑需验证 KV 是否真正传输(否则 decode 可能空等)。
- 🟡 **发压连通性**:需确认 h200-6 能访问 h200-2 的服务端口(`--network host` + 内网 IP)。
  bench_ramp.sh 已内置探活,失败会明确报错。
- 🟡 **10min/组**:长 context(codex ~30k prompt)+ TP4 时 prefill 慢,C=1 组可能偏长;用
  `--trials-per-user` / `--max-tokens` 收敛,首跑校准。

## 7. 分析与横向对比

各实验 `summary.json` 汇总后,统一出一张跨实验对比表(TTFT/TPOT/吞吐/错误率 × 并发 1/4/8/16),
按 §1 的对比逻辑归因。图表脚本产出后放 `develop/`(参照既有
`glm52-test` 的 `make_report_charts_*.py` 风格,直接读 summary.json,不手抄数字)。
