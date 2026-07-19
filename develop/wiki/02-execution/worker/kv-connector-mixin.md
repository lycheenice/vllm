[← Wiki 首页](../../README.md) > [执行层](../README.md) > [Worker](./README.md) > KV Connector Mixin

# KV Connector Model Runner Mixin

源码：`vllm/v1/worker/kv_connector_model_runner_mixin.py`（251 行）+ V2 对应 `vllm/v1/worker/gpu/kv_connector.py`（123 行）

## 是什么

`KVConnectorModelRunnerMixin` (`:34`) 把 **KV connector**（跨实例/跨节点 KV cache 迁移，见 [15-kv-cache-offload](../../15-kv-cache-offload/README.md)）的钩子注入 `GPUModelRunner`。它提供：

- `kv_connector_no_forward(scheduler_output, vllm_config)`：当某步无 token 需要前向（`num_scheduled_tokens==0`）但仍有 KV send/recv 任务时，仍然推进 connector 的 load/save，返回 `ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)`。
- `maybe_get_kv_connector_output(scheduler_output, defer_finalize=False)` → context manager：包裹整个 forward，在 `yield` 前调 `start_load_kv`，在 `finally` 调 `get_finished` / `get_kv_connector_stats` / `clear_connector_metadata`。
- `finalize_kv_connector()`：当 `defer_finalize=True`（spec decode 场景）时，draft model 跑完后再 wait_for_save + clear。
- `use_uniform_kv_cache(attn_groups)` / `allocate_uniform_kv_cache(...)`：判断并分配 cross-layer 连续 KV buffer，让 connector 一次拷贝所有层。

V2 路径：`gpu/kv_connector.py` 把同样逻辑收纳成 `KVConnector` 对象（`NO_OP_KV_CONNECTOR` 默认），`gpu/model_runner.py:execute_model` 直接调 `kv_connector.pre_forward` / `post_forward` / `no_forward`。

## 为什么

- **解耦 P/D disaggregation**：prefill 实例（producer）和 decode 实例（consumer）之间需要异步迁移 KV，mixin 让 connector 在 forward 前后自动驱动，不污染主 forward 路径。
- **defer finalize for spec decode**：draft model 也用目标模型的 KV cache，`defer_finalize=True` 让 `wait_for_save` 延迟到 draft 跑完，避免假投被错误保存。
- **cross-layer uniform KV**：connector 一次拷贝一个 block 内所有层比逐层拷贝快几倍；mixin 在 `initialize_kv_cache` 阶段判定是否启用并分配连续 buffer。
- **空步兼容**：调度器可能产出 `num_scheduled_tokens==0` 的 step（仅有 finished_req_ids），connector 仍需推进收尾，`kv_connector_no_forward` 处理。
- **多 backend 一致**：`has_kv_transfer_group()` 让 mixin 在无 connector 时退化为 `nullcontext`，零开销。

## 怎么做

### 静态方法 maybe_get_kv_connector_output（`:50`）

```python
@staticmethod
def maybe_get_kv_connector_output(scheduler_output, defer_finalize=False):
    return (
        KVConnectorModelRunnerMixin._get_kv_connector_output(
            scheduler_output, defer_finalize=defer_finalize)
        if has_kv_transfer_group() else nullcontext())
```

V1 `execute_model` 在进入 `set_forward_context` 后：

```python
with set_forward_context(...), maybe_get_kv_connector_output(
        scheduler_output, defer_finalize=defer_kv_connector_finalize) as kv_connector_output:
    model_output = self._model_forward(...)
```

### _get_kv_connector_output（`:78`）

context manager，必须在已激活 `set_forward_context` 下使用：

1. `kv_connector = get_kv_transfer_group()`。
2. `kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)`：把调度器产出的 metadata 绑定到 connector。
3. `kv_connector.start_load_kv(get_forward_context())`：启动后台 KV 加载（异步传输）。
4. `yield output`（forward 在此执行）。
5. `finally`：
   - `if wait_for_save and not defer_finalize: kv_connector.wait_for_save()`。
   - `output.finished_sending, output.finished_recving = kv_connector.get_finished(scheduler_output.finished_req_ids)`。
   - `output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()`。
   - `output.kv_connector_stats = kv_connector.get_kv_connector_stats()`。
   - `output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()`。
   - `output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()`。
   - `if not defer_finalize: kv_connector.clear_connector_metadata()`。

### kv_connector_no_forward（`:36`）

空步路径：

```python
with set_forward_context(None, vllm_config), \
     KVConnectorModelRunnerMixin._get_kv_connector_output(
        scheduler_output, wait_for_save=False) as kv_connector_output:
    pass
return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

`wait_for_save=False`——空步不等 save（避免阻塞下一步）。

### finalize_kv_connector（`:64`）

```python
if has_kv_transfer_group():
    kv_connector = get_kv_transfer_group()
    kv_connector.wait_for_save()
    kv_connector.clear_connector_metadata()
```

V1 spec decode 路径在 draft model 跑完后显式调用。

### use_uniform_kv_cache（`:114`）

判定是否启用 cross-layer 连续 KV buffer（3 条件全真）：

1. `has_kv_transfer_group()` 且 `get_kv_transfer_group().prefer_cross_layer_blocks` 返回 True。
2. `len(attn_groups) == 1 and len(attn_groups[0]) == 1`（只有一组同构 attention）。
3. `kv_cache_spec.indexes_kv_by_block_stride` 为 True（attention backend 按 block stride 索引）。

### allocate_uniform_kv_caches（`:161`）

分配单块连续 buffer 覆盖所有层：

- 取 `kv_cache_tensor.size` / `page_size`、`num_blocks`、`num_layers`。
- `kv_cache_shape = (num_layers,) + attn_backend.get_kv_cache_shape(...)`。
- 按 `attn_backend.get_kv_cache_stride_order(include_num_layers_dimension=True)` 重排 stride（让 connector 拷贝时 per-block all-layers 连续）。
- `cross_layers_kv_cache = torch.zeros(total_size, int8).view(dtype).view(shape)`。
- `permuted_kv_cache = cross_layers_kv_cache.permute(*inv_order)` 恢复每层视角。
- 给 `kv_cache_tensor.shared_by` 列出的层名赋同一 tensor view，写入 `kv_caches` dict。
- 返回 `(kv_caches, cross_layers_kv_cache, attn_backend)`。

### V2：gpu/kv_connector.py

| 类 | 行为 |
|---|---|
| `KVConnector` (base, NO_OP) | 所有方法 no-op；`no_forward` 返回 `EMPTY_MODEL_RUNNER_OUTPUT`；`set_disabled` no-op |
| `ActiveKVConnector` | 持有 `KVConnectorBase`，`pre_forward`/`post_forward`/`no_forward`/`set_disabled` 真实实现 |
| `get_kv_connector(vllm_config, kv_caches_dict)` | `has_kv_transfer_group()` ? `ActiveKVConnector` : `NO_OP_KV_CONNECTOR` |

V2 调用（`gpu/model_runner.py`）：

- `execute_model`：`kv_connector.pre_forward(scheduler_output)` → `_model_forward` → `_handle_kv_connector_output = kv_connector.post_forward(finished_req_ids, wait_for_save=not defer)`。
- 空步 / `num_tokens==0`：`return kv_connector.no_forward(scheduler_output)`（内部 = `pre_forward` + 立即 `post_forward(wait_for_save=False)` + 包装成 ModelRunnerOutput）。
- `_dummy_run`：`self.kv_connector.set_disabled(True)` 禁用 connector 钩子，结束后恢复。

`ActiveKVConnector.pre_forward` (`kv_connector.py:61`)：`handle_preemptions` → `bind_connector_metadata` → `start_load_kv(forward_context or new)`。
`ActiveKVConnector.post_forward` (`:77`)：`wait_for_save` → `get_finished` → `get_block_ids_with_load_errors` → `get_kv_connector_stats` → `get_kv_cache_events` → `build_connector_worker_meta` → `clear_connector_metadata`。
`ActiveKVConnector.set_disabled` (`:107`)：把 `kv_transfer_state._KV_CONNECTOR_AGENT` 设为 None/connector，禁用层级钩子（防 dummy run 误触）。

## 与其它模块/系统配合

- [GPU Model Runner V1](gpu-model-runner.md) / [V2](model-runner-v2.md)：多继承注入；V1 用 ctx manager，V2 用对象方法。
- [GPU Worker](gpu-worker.md)：`get_kv_connector_handshake_metadata` 在 Worker 层返回 `{(pp_rank, tp_rank): metadata}`；`ensure_kv_transfer_initialized` 在 `initialize_from_config` 头调。
- [Executor ABC](../executor/abstract.md)：`init_kv_output_aggregator(connector)` + `execute_model` 用 `kv_output_aggregator.aggregate` 聚合多 Worker 输出。
- [引擎核心](../../01-engine-core/README.md)：`SchedulerOutput.kv_connector_metadata` 由调度器填充；`ModelRunnerOutput.kv_connector_output` 回流给引擎核心处理 finished_sending/recving。
- [KV 缓存卸载](../../15-kv-cache-offload/README.md)：`KVConnectorBase` 实现（PyNvNIXL/SysBypass/SimpleKV 等）。
- [分布式](../../07-distributed/README.md)：`get_kv_transfer_group`、`copy_kv_blocks` utility。
- [采样与解码](../../06-sampling-decoding/README.md)：spec decode 路径用 `defer_finalize=True`，draft 跑完调 `finalize_kv_connector()`。
- [注意力后端](../../05-attention/README.md)：`indexes_kv_by_block_stride` 决定能否用 uniform kv cache；`get_kv_cache_shape`/`get_kv_cache_stride_order` 决定 layout。

## 历史版本演进

- **v0.7.0**：V1 引入 `KVConnectorModelRunnerMixin`，`has_kv_transfer_group()` 守卫 + `nullcontext` 退化。
- **v0.8.0**：`_get_kv_connector_output` context manager 落地，封装 bind/load/forward/wait/clear 生命周期。
- **v0.9.0**：`defer_finalize` 引入（spec decode）；`kv_connector_no_forward` 处理空步。
- **v0.10.0**：`use_uniform_kv_cache` / `allocate_uniform_kv_caches` 引入 cross-layer 连续 KV，配合 NIXL 高效迁移。
- **v0.11.0**：handshake metadata（`KVConnectorHandshakeMetadata`）+ `get_kv_connector_handshake_metadata` 在 Worker 层返回；`invalid_block_ids` / `get_block_ids_with_load_errors` 处理加载失败的 block。
- **v0.12 / main**：V2 `KVConnector`/`ActiveKVConnector`/`NO_OP_KV_CONNECTOR` 对象化；`kv_cache_events` + `build_connector_worker_meta`；`set_disabled` 通过 `kv_transfer_state._KV_CONNECTOR_AGENT` 控制层级钩子，让 dummy run 安全。

[← 返回执行层首页](../README.md)

## 参见

- [GPU Model Runner V1](gpu-model-runner.md)
- [Model Runner V2](model-runner-v2.md)
- [EC Connector Mixin](ec-connector-mixin.md)
- [GPU Worker](gpu-worker.md)
- [Executor ABC](../executor/abstract.md)
- [KV 缓存卸载子系统](../../15-kv-cache-offload/README.md)
