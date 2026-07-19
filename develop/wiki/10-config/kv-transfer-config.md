# KVTransferConfig（kv_transfer.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > KVTransferConfig

源码：`vllm/config/kv_transfer.py`（约 121 行）。`KVTransferConfig` 描述分布式 KV cache 迁移：connector 名、引擎 ID、buffer 设备/大小、role(producer/consumer/both)、并行度、IP/端口、连接器额外配置、加载失败策略等。它是 `VllmConfig.kv_transfer_config`（`None` 表示未启用），被 `vllm/distributed/kv_transfer/` 的 connector 体系与 scheduler 消费。PD disagg 与单实例 KV 卸载/共享都经此配置。

## 是什么

`@config` 装饰（`kv_transfer.py:22`）。`KVRole = Literal["kv_producer","kv_consumer","kv_both"]`。

| 字段 | 默认 | 含义 |
|---|---|---|
| `kv_connector` | `None` | connector 名（如 `NixlConnector`/`MooncakeConnector`/`LMCacheMPConnector`/`OffloadingConnector`/`SimpleCPUOffloadConnector`） |
| `engine_id` | `None`(→`uuid4`) | KV 迁移引擎 ID |
| `kv_buffer_device` | `current_platform.device_type` | connector 缓冲 KV 的设备（cuda/cpu/xpu） |
| `kv_buffer_size` | `1e9`(≈1GB) | `TorchDistributedConnector` 缓冲字节数 |
| `kv_role` | `None` | `kv_producer`/`kv_consumer`/`kv_both` |
| `kv_rank` | `None` | KV 迁移 rank（0=prefill，1=decode；当前仅 1P1D） |
| `kv_parallel_size` | `1` | KV 迁移并行实例数 |
| `kv_ip` | `"127.0.0.1"` | connector IP |
| `kv_port` | `14579` | connector 端口 |
| `kv_connector_extra_config` | `{}` | connector 专用额外配置 |
| `kv_connector_module_path` | `None` | 动态加载 connector 的 Python 模块路径（仅 V1） |
| `enable_permute_local_kv` | `False` | 实验：HND↔NHD KV 迁移 |
| `kv_load_failure_policy` | `"fail"` | 加载失败策略：`recompute`(重算)/`fail`(FINISHED_ERROR) |

校验（`__post_init__`）：`engine_id` 默认 `uuid4`；`kv_role` 须在 `KVRole`；`kv_connector` 非 None 时 `kv_role` 必填。

属性：`is_kv_transfer_instance`（`kv_connector`+`kv_role` 均设）、`is_kv_producer`/`is_kv_consumer`。方法：`get_from_extra_config(key, default)`。

`compute_hash`：返回空 factors——KV 迁移不影响编译图形状（connector 在图外做 KV 搬运）。

## 为什么

- **PD 解耦**：Prefill/Decode 分离部署时，P 实例 `kv_role=kv_producer`，D 实例 `kv_role=kv_consumer`，KV 经 connector 跨实例迁移，让 D 跳过 prefill 计算。`kv_both` 用于单实例 KV 卸载/共享（前缀缓存跨实例共享）。
- **connector 可插拔**：`kv_connector` 是字符串名，`KVConnectorFactory.get_connector_class` 按名 dispatch；`kv_connector_module_path` 支持树外 connector 动态加载。各 connector 用 `kv_connector_extra_config` 传专用参数（如 NIXL/Mooncake 的内存注册配置、LMCache 的 host/port）。
- **失败策略**：`kv_load_failure_policy` 决定 KV 迁移加载失败时 `recompute`（重算该 block）还是 `fail`（请求 FINISHED_ERROR）。`VllmConfig.__post_init` 校验与之相关的 `_handle_invalid_blocks` 恢复路径。
- **缓冲设备分离**：`kv_buffer_device` 让 connector 在 CPU 或 GPU 缓冲 KV，适配不同 connector（如 `SimpleCPUOffloadConnector` 用 CPU）。
- **自动派生**：`CacheConfig.kv_offloading_size` 经 `VllmConfig._post_init_kv_transfer_config` 自动生成 `KVTransferConfig`（设 connector 名、`kv_role=kv_both`、`cpu_bytes_to_use`），让用户用 `--kv-offloading-size` 简便开关。

## 怎么做

- **PD disagg**：P 实例 `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_rank":0}'`，D 实例 `kv_role=kv_consumer`/`kv_rank=1`。
- **单实例卸载**：`--kv-offloading-size 8`（GiB）+ 可选 `--kv-offloading-backend lmcache`，`VllmConfig` 自动生成 `kv_transfer_config`。
- **失败重算**：`--kv-transfer-config.kv-load-failure-policy recompute`。
- **树外 connector**：`--kv-transfer-config.kv-connector-module-path mypkg.connectors:MyConnector`。
- **额外配置**：`--kv-transfer-config.kv-connector-extra-config '{"cpu_bytes_to_use":8589934592}'`。

## 与其它模块/系统配合

- **KV 迁移子系统（[`07-distributed/`](../07-distributed/README.md) 与 [`15-kv-cache-offload/`](../15-kv-cache-offload/README.md)）**：`kv_connector` 驱动 `KVConnectorFactory` 选 connector 类；`KVConnectorBase_V1` 全生命周期钩子（`get_num_new_matched_tokens`/`update_state_after_alloc`/`request_finished`/`update_connector_output`）被 scheduler 调用。
- **Scheduler（[`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)）**：connector 钩子；`WAITING_FOR_REMOTE_KVS` 状态；`_handle_invalid_blocks` 恢复；`_update_waiting_for_remote_kv`；`defer_block_free`（KV consumer + 多 inflight batch 时防写后释放）。
- **CacheConfig（[cache-config.md](cache-config.md)）**：`kv_offloading_size`/`kv_offloading_backend` 经 `VllmConfig._post_init_kv_transfer_config` 翻译为本配置。
- **CompilationConfig（[compilation-config.md](compilation-config.md)）**：`is_kv_transfer_instance` + full cudagraph 时，`VllmConfig.__post_init` 调 `KVConnectorFactory.get_connector_class(...).requires_piecewise_for_cudagraph(...)`，需 PIECEWISE 的 connector 强制降 `cudagraph_mode=PIECEWISE`。
- **HMA（Hybrid KV cache manager）**：`is_kv_transfer_instance` + `disable_hybrid_kv_cache_manager=None` 时，`VllmConfig` 调 `KVConnectorFactory.supports_hma_config(self.kv_transfer_config)`，不支持的 connector 自动关 HMA 并 warning。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`_verify_kv_transfer_compat` 校验 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与 KV connector 冲突（除非 `enable_cumem_allocator`，因 PyTorch VMM 重映射会废 pinned KV 内存）；`enable_return_routed_experts` 与任何 KV connector 互斥。

## 历史版本演进

- **v0.7/v0.7.x**：`KVTransferConfig` 引入，初版 `PyNcmlConnector`/`TorchDistributedConnector`；1P1D PD disagg。
- **v0.8**：MooncakeConnector/NixlConnector；`kv_connector_module_path` 动态加载；`kv_load_failure_policy`；`is_kv_transfer_instance` 属性。
- **v0.9**：`_post_init_kv_transfer_config` 把 `cache_config.kv_offloading_*` 翻译为本配置；`supports_hma_config` 协议（connector 声明是否支持 hybrid KV cache manager）；`_verify_kv_transfer_compat`（expandable_segments 冲突）；multi-connector / FlexKV。
- **v0.10**：`enable_permute_local_kv`（HND↔NHD）；connector `requires_piecewise_for_cudagraph` 协议（layerwise async op 不能进 cudagraph）；`enable_return_routed_experts` 与 KV connector 互斥校验。
- **v0.11 / v0.12 / main**：`SimpleCPUOffloadConnector` vs `OffloadingConnector` 由 `VLLM_USE_SIMPLE_KV_OFFLOAD` 选；`LMCacheMPConnector` 默认 MP 模式；connector 全生命周期与 scheduler 深度集成。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [cache-config.md](cache-config.md) — `kv_offloading_*` 经 `VllmConfig` 翻译为本配置。
- [kv-events-config.md](kv-events-config.md) — KV 事件发布（与迁移正交的观测面）。
- [vllm-config.md](vllm-config.md) — `_post_init_kv_transfer_config`/`_verify_kv_transfer_compat`/HMA/`requires_piecewise_for_cudagraph`。
- [compilation-config.md](compilation-config.md) — connector 影响 `cudagraph_mode`。
- [../15-kv-cache-offload/README.md](../15-kv-cache-offload/README.md) — connector 体系消费方。
- [../01-engine-core/scheduler/scheduler.md](../01-engine-core/scheduler/scheduler.md) — connector 钩子消费方。
