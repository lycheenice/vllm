# WeightTransferConfig（weight_transfer.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > WeightTransferConfig

源码：`vllm/config/weight_transfer.py`（约 15 行，最小配置之一）。`WeightTransferConfig` 描述 RL 训练场景下权重迁移的 backend 选择。它是 `VllmConfig.weight_transfer_config`（`None` 表示未启用），由 `WeightTransferEngineFactory` 注册表在引擎创建时验证。

## 是什么

`@config` 装饰（`weight_transfer.py:8`）。

| 字段 | 默认 | 含义 |
|---|---|---|
| `backend` | `"nccl"` | 权重迁移 backend：`"nccl"`/`"ipc"`/`"sparse_nccl"` 或自定义 str |

`backend` 是 `Literal["nccl","ipc","sparse_nccl"] | str`——开放 str 路径支持树外 backend 注册到 `WeightTransferEngineFactory`。

> 无 `compute_hash`（未在 `VllmConfig.compute_hash` 因子中），因权重迁移不影响前向图形状。

## 为什么

- **RL 训练权重热更新**：RL 训练中 policy 模型权重频繁更新，需把新权重从 trainer 迁移到 vLLM 推理实例。不同 backend 适配不同拓扑：
  - `"nccl"`：标准 NCCL 集合通信，同集群内。
  - `"ipc"`：进程间共享内存，本机低延迟。
  - `"sparse_nccl"`：稀疏更新（仅变动的 expert 权重），减通信量。
- **backend 字符串化**：`WeightTransferEngineFactory` 注册表按名 dispatch，支持树外扩展无需改核心。
- **与 EPLB 协同**：`sparse_nccl` 配合 EPLB（[parallel-config.md](parallel-config.md) 的 `enable_eplb`），专家重排后仅迁移变动 expert。

## 怎么做

- **NCCL 全量**：`--weight-transfer-config.backend nccl`（默认）。
- **IPC 本机**：`--weight-transfer-config.backend ipc`。
- **稀疏 expert**：`--weight-transfer-config.backend sparse_nccl`（配合 `--enable-eplb`）。

## 与其它模块/系统配合

- **EPLB（[parallel-config.md](parallel-config.md)）**：`enable_eplb` + `sparse_nccl` 协同专家重排与稀疏迁移。
- **Weight live patch（[`07-distributed/`](../07-distributed/README.md)）**：backend 驱动 `WeightTransferEngine` 选择，做权重在线热更新。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`weight_transfer_config` 非 None 时启用权重迁移路径；当前未在 `__post_init` 做强校验（backend 校验延迟到 `WeightTransferEngineFactory` 创建引擎时）。

## 历史版本演进

- **v0.9.x**：`WeightTransferConfig` 引入（weight live patch / RL 训练需求），初版 `nccl`/`ipc`。
- **v0.10/v0.11**：`sparse_nccl`（配合 EPLB 专家重排）；`WeightTransferEngineFactory` 注册表成形。
- **v0.12 / main**：与 EPLB async + nixl/gloo communicator 协同；树外 backend 扩展。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [parallel-config.md](parallel-config.md) — `enable_eplb`/`EPLBConfig` 与 `sparse_nccl` 协同。
- [ec-transfer-config.md](ec-transfer-config.md) — 编码器缓存迁移（类似 role 模式）。
- [../07-distributed/README.md](../07-distributed/README.md) — weight live patch 消费方。
