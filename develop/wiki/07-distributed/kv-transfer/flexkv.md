# flexkv_connector.py — FlexKVConnectorV1

[← Wiki 首页](../../README.md) > [分布式](../../README.md) > [kv-transfer](README.md) > flexkv

源码：`vllm/distributed/kv_transfer/kv_connector/v1/flexkv_connector.py`（约 260 行）。这是 vLLM 对 [FlexKV](https://github.com/taco-project/FlexKV) 的门面 connector：FlexKV 是外部分布式 KV Store + 多级缓存（CPU/SSD/远端）系统，针对超大规模 LLM 推理。本 connector 实现极薄——几乎是 `KVConnectorBase_V1` 接口到 FlexKV 自有 adapter（`flexkv.integration.vllm.vllm_v1_adapter.FlexKVConnectorV1Impl`）的转发。

## 是什么

### FlexKVConnectorV1（`:35`）

`FlexKVConnectorV1(KVConnectorBase_V1)`。

`__init__(vllm_config, role, kv_cache_config)`（`:56`）：
- `super().__init__(...)`；
- try `from flexkv.integration.vllm.vllm_v1_adapter import FlexKVConnectorV1Impl`，失败 raise `ImportError`（提示安装 FlexKV 并指 GitHub 链接）；
- `self._flexkv_connector = FlexKVConnectorV1Impl(vllm_config, role)`（注意 FlexKV impl 只接 2 参，无 `kv_cache_config`，由其内部自取）。

之后所有 `KVConnectorBase_V1` 钩子转发到 `self._flexkv_connector`：

- 转发 `bind_connector_metadata`/`clear_connector_metadata`/`register_kv_caches`/`register_cross_layers_kv_cache`/`set_host_xfer_buffer_ops`/`handle_preemptions`/`start_load_kv`/`wait_for_layer_load`/`save_kv_layer`/`wait_for_save`/`get_finished`/`get_block_ids_with_load_errors`/`shutdown`/`get_kv_connector_stats` 等 worker 侧；
- 转发 `get_num_new_matched_tokens`/`update_state_after_alloc`/`build_connector_meta`/`on_new_request`/`update_connector_output`/`request_finished`/`take_events`/`has_pending_push_work`/`reset_cache` 等 scheduler 侧；
- 类方法 `get_required_kvcache_layout`/`requires_piecewise_for_cudagraph`/`get_finished_count`/`build_kv_connector_stats`/`build_prom_metrics` 等转发到 impl 类方法。

`__getattr__` 风格通用转发：调用 connector 上任意方法 → 转给 `self._flexkv_connector`。

### 注册

`KVConnectorFactory.register_connector("FlexKVConnectorV1", "vllm.distributed.kv_transfer.kv_connector.v1.flexkv_connector", "FlexKVConnectorV1")`（factory.py:229）。

## 为什么

- **门面模式**：FlexKV 是独立项目，演进节奏与 vLLM 不同；保持 vLLM 侧极薄转发让 FlexKV 团队在自己的 `vllm_v1_adapter.py` 内跟 vLLM 协议迭代，避免 vLLM 主仓反复改。
- **懒 import 隔离依赖**：FlexKV 未装时直接 `ImportError`，不拖累其它 connector 启动；与 factory lazy loader 配合。
- **不强制 3 参构造**：FlexKV impl 用 `(vllm_config, role)`，`kv_cache_config` 由其内部 `get_current_vllm_config` 自取；本 connector 兼容该旧签名（不调 `supports_kw(kv_cache_config)` 校验路径——是其选择而非外部模块路径）。
- **完整钩子覆盖**：转发所有 v1 钩子让 FlexKV 与 vLLM 调度/前向/事件/统计全链路对齐，行为等价于"vLLM 内置 connector"。
- **超大规模定位**：FlexKV 设计目标"ultra-large-scale"（多机 SSD + 远端 + 分布式索引），与 [LMCache](lmcache.md)、[Mooncake](transports/mooncake.md) 互补——用户按规模/部署偏好选。
- **多级缓存**：注释 `:39` 指出 FlexKV 支持 offload 到 CPU/SSD/remote，定位与 [OffloadingConnector](offloading.md) 重叠但更侧重"分布式 KV store"语义。

## 怎么做

### 配置示例

`--kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'` + 安装 FlexKV（`git clone ... FlexKV && cd FlexKV && bash build.sh`，注释 `:33`/`:46`）。

### 调用链

```mermaid
flowchart LR
    SCH[Scheduler/ModelRunner] -->|"v1 hook"| FVK[FlexKVConnectorV1]
    FVK -->|"getattr/显式转发"| IMPL[FlexKVConnectorV1Impl<br/>flexkv.integration.vllm.vllm_v1_adapter]
    IMPL --> FK[FlexKV engine<br/>CPU/SSD/remote store]
```

### 与 vLLM 协议对齐

FlexKV impl 必须实现 `KVConnectorBase_V1` 的全部抽象方法（含 `start_load_kv`/`wait_for_layer_load`/`save_kv_layer`/`wait_for_save`/`get_num_new_matched_tokens`/`update_state_after_alloc`/`build_connector_meta`/`request_finished` 等）；vLLM 侧不补充任何实现。`requires_piecewise_for_cudagraph`/`get_required_kvcache_layout` 类方法由 FlexKV impl 自行决定。

## 与其它模块/系统配合

- **[base](base.md)**：协议——本 connector 全转发。
- **[utils](utils.md)**：FlexKV impl 内部可能用 `TransferTopology`/`KVOutputAggregator`（待核实）。
- **[01-engine-core](../../01-engine-core/README.md)**：scheduler 钩子由 FlexKV impl 驱动。
- **[02-execution](../../02-execution/README.md)**：worker 钩子由 FlexKV impl 驱动。
- **[09-compilation-ir](../../09-compilation-ir/README.md)**：`requires_piecewise_for_cudagraph` 由 FlexKV impl 决定。
- **[kv-events](../kv-events.md)**：`take_events`/`get_kv_connector_kv_cache_events` 由 FlexKV impl 提供。

## 历史版本演进

- **v0.10**：`FlexKVConnectorV1` 引入（晚于 LMCache/Mooncake），初版全转发。
- **v0.11/v0.12/main**：与 FlexKV 上游协议对齐持续；接入 `get_required_kvcache_layout`/`build_prom_metrics` 等新钩子（待核实）。

[← 返回 kv-transfer 首页](README.md)

## 参见

- [base.md](base.md) — 协议来源。
- [lmcache.md](lmcache.md) — 同类外部 KV store。
- [transports/mooncake.md](transports/mooncake.md) / [transports/hf3fs.md](transports/hf3fs.md) — 其它分布式存储路径。
