# ECTransferConfig（ec_transfer.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > ECTransferConfig

源码：`vllm/config/ec_transfer.py`（约 107 行）。`ECTransferConfig` 描述分布式 EC（Encoder Cache，编码器缓存）迁移：connector 名、引擎 ID、buffer 设备/大小、role(producer/consumer/both)、rank、并行度、IP/端口、额外配置。它是 `VllmConfig.ec_transfer_config`（`None` 表示未启用），被 `ECConnector` 体系（`vllm/distributed/ec_transfer/`）与 scheduler 的 `ec_connector` 钩子消费。用于多模态 disagg（Encoder 进程与 PD 实例分离）场景。

## 是什么

`@config` 装饰（`ec_transfer.py:15`）。`ECRole = Literal["ec_producer","ec_consumer","ec_both"]`。结构与 `KVTransferConfig` 高度对称。

| 字段 | 默认 | 含义 |
|---|---|---|
| `ec_connector` | `None` | EC connector 名 |
| `engine_id` | `None`(→`uuid4`) | EC 迁移引擎 ID |
| `ec_buffer_device` | `"cuda"` | connector 缓冲 EC 的设备（当前仅 cuda） |
| `ec_buffer_size` | `1e9`(≈1GB) | 缓冲字节数 |
| `ec_role` | `None` | `ec_producer`/`ec_consumer`/`ec_both` |
| `ec_rank` | `None` | EC 迁移 rank（0=encoder，1=pd；当前仅 1P1D） |
| `ec_parallel_size` | `1` | 并行实例数（PyNcclConnector 应为 2） |
| `ec_ip` | `"127.0.0.1"` | connector IP |
| `ec_port` | `14579` | connector 端口 |
| `ec_connector_extra_config` | `{}` | connector 专用额外配置 |
| `ec_connector_module_path` | `None` | 动态加载 connector 的模块路径（仅 V1） |

校验（`__post_init__`）：`engine_id` 默认 `uuid4`；`ec_role` 须在 `ECRole`；`ec_connector` 非 None 时 `ec_role` 必填。

属性：`is_ec_transfer_instance`/`is_ec_producer`/`is_ec_consumer`。方法：`get_from_extra_config(key, default)`。

`compute_hash`：空 factors——EC 迁移不影响编译图形状。（注意用 `hashlib.md5` 而 `KVTransferConfig` 用 `safe_hash`，二者均为空 factors 故无实际差异，属历史不一致 `(待核实)`。）

## 为什么

- **多模态 disagg**：大 ViT 编码器计算昂贵，可独立部署 Encoder 进程（`ec_producer`），其编码器输出（EC cache）经 connector 迁移到 PD 实例（`ec_consumer`），让 PD 实例跳过编码器前向。这是 `KVTransferConfig`（KV 迁移）的多模态对应物。
- **与 KV 迁移对称设计**：`ECTransferConfig` 字段命名/role 模式/校验逻辑与 `KVTransferConfig` 完全对称，降低学习成本；scheduler 侧 `ec_connector` 钩子与 `kv_connector` 钩子并行。
- **当前仅 cuda 缓冲**：EC cache（编码器输出张量）体积大，cuda 缓冲避免 H2D；CPU 缓冲暂不支持（`ec_buffer_device` 当前仅 cuda）。
- **V2 限制**：`VllmConfig._get_v2_model_runner_unsupported_features` 把 `ec_transfer_config is not None` 列为 V2 不支持特性（PR #38390 待加），故 V2 实例启用 EC 迁移会 raise。

## 怎么做

- **Encoder disagg**：Encoder 进程 `--ec-transfer-config '{"ec_connector":"PyNcclConnector","ec_role":"ec_producer","ec_rank":0,"ec_parallel_size":2}'`，PD 实例 `ec_role=ec_consumer`/`ec_rank=1`。
- **额外配置**：`--ec-transfer-config.ec-connector-extra-config '{}'`。
- **树外 connector**：`--ec-transfer-config.ec-connector-module-path mypkg:MyECConnector`。

## 与其它模块/系统配合

- **EC 迁移子系统（`vllm/distributed/ec_transfer/` 与 [`02-execution/worker/ec-connector-mixin.md`](../02-execution/worker/ec-connector-mixin.md)）**：`ec_connector` 驱动 `ECConnector` 选择；`ECConnectorMixin` 在 Worker 侧集成生命周期钩子。
- **Scheduler（[`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)）**：`ec_connector` 钩子（`build_connector_meta` 等），与 `kv_connector` 并行的 `ec_connector_metadata` 在 `SchedulerOutput`。
- **MultiModalConfig（[multimodal-config.md](multimodal-config.md)）**：`mm_encoder_only=True` 配合 `ec_role=ec_producer`（仅跑编码器的 disagg Encoder 进程）。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`ec_transfer_config` 非 None 时在 `_get_v2_model_runner_unsupported_features` 列为 V2 不支持；`_validate_v2_model_runner` raise。
- **kv-transfer（[kv-transfer-config.md](kv-transfer-config.md)）**：二者可共存（KV 迁移 + EC 迁移并行），分别走 `kv_connector`/`ec_connector` 钩子。

## 历史版本演进

- **v0.9.x**：`ECTransferConfig` 引入（EC transfer 成形）；`ECConnector`/`ECConnectorMixin` 落地；多模态 disagg Encoder 进程。
- **v0.10**：与 `mm_encoder_only` 协同；scheduler `ec_connector_metadata` 进 `SchedulerOutput`。
- **v0.11 / v0.12 / main**：V2 model runner 不支持 EC 迁移（PR #38390 待加）；connector 全生命周期与 scheduler 集成深化。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [kv-transfer-config.md](kv-transfer-config.md) — 对称设计的 KV 迁移配置。
- [multimodal-config.md](multimodal-config.md) — `mm_encoder_only` 配合 EC producer。
- [vllm-config.md](vllm-config.md) — V2 不支持 EC 迁移校验。
- [../02-execution/worker/ec-connector-mixin.md](../02-execution/worker/ec-connector-mixin.md) — Worker 侧 EC connector mixin。
