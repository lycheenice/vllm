[← Wiki 首页](../../README.md) > [执行层](../README.md) > [Worker](./README.md) > EC Connector Mixin

# EC Connector Model Runner Mixin

源码：`vllm/v1/worker/ec_connector_model_runner_mixin.py`（78 行）

## 是什么

`ECConnectorModelRunnerMixin` (`:25`) 把 **EC connector**（Encoder Cache connector，跨实例迁移多模态 encoder 的中间输出）的钩子注入 `GPUModelRunner`。它是 KV connector 的"编码器版"：在 EPD（Encoder-Prefill-Decode）disaggregation 模式下，producer 实例跑图像/音频 encoder 把输出缓存（`encoder_cache`）发送给 consumer 实例，consumer 跳过 encoder 直接用收到的 cache 跑 language model。

提供的静态方法（全是 mixin 静态方法，不持有 self 状态）：

- `maybe_save_ec_to_connector(encoder_cache, mm_hash)`：把本步产生的 encoder cache 按 mm_hash 保存到 connector。
- `maybe_get_ec_connector_output(scheduler_output, encoder_cache, **kwargs)` → context manager：包裹 forward，consumer 在 `yield` 前 `start_load_caches`，`finally` 调 `get_finished` + `clear_connector_metadata`。
- `_get_ec_connector_output(scheduler_output, encoder_cache, **kwargs)`：实际 ctx manager 实现。

EC connector 实例 `get_ec_transfer()` 来自 [分布式子系统](../../07-distributed/README.md) 的 `distributed.ec_transfer`，实现见 `distributed/ec_transfer/ec_connector/base.py:ECConnectorBase`。

## 为什么

- **解耦编码器与语言模型**：多模态模型的 encoder 在首节点占显存/算力可观，EPD 模式让 encoder 独立部署在 producer，consumer 只持有 LM。
- **复用 KV connector 模式**：与 KV connector 同样的"ctx manager 包 forward + defer finalize"骨架，便于维护。
- **mm_hash 索引**：encoder 输出按 hash 索引而非按请求，同一张图在多请求间可复用，节省带宽。
- **空 forward 兼容**：consumer 角色下若已收齐 cache、本步无新 encoder 输入，`start_load_caches` 仍可推进；producer 角色下若 `is_consumer=False`，只 save 不 load。

## 怎么做

### maybe_save_ec_to_connector（`:27`）

```python
@staticmethod
def maybe_save_ec_to_connector(encoder_cache, mm_hash):
    if not has_ec_transfer():
        logger.debug("Not have ec transfer please check")
        return
    connector = get_ec_transfer()
    connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)
```

调用时机由具体 ModelRunner 决定（V1 `gpu_model_runner._execute_mm_encoder` 完成后）。

### maybe_get_ec_connector_output（`:38`）

```python
@staticmethod
def maybe_get_ec_connector_output(scheduler_output, encoder_cache, **kwargs):
    return (
        ECConnectorModelRunnerMixin._get_ec_connector_output(
            scheduler_output, encoder_cache, **kwargs)
        if has_ec_transfer() else nullcontext())
```

V1 `execute_model` 在 EC producer 分支会用到（见 [gpu-model-runner.md](gpu-model-runner.md)）：

```python
if has_ec_transfer() and not get_ec_transfer().is_consumer:
    with self.maybe_get_ec_connector_output(
            scheduler_output,
            encoder_cache=self.encoder_cache) as ec_connector_output:
        self._execute_mm_encoder(scheduler_output)
        return make_empty_encoder_model_runner_output(scheduler_output)
```

producer 只跑 MM encoder，把结果送出去，本步 language model 不跑。

### _get_ec_connector_output（`:55`）

必须在 active `forward_context` 下使用（同 KV connector 约束）：

1. `ec_connector = get_ec_transfer()`，断言是 `ECConnectorBase`。
2. `assert scheduler_output.ec_connector_metadata is not None`。
3. `ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)`。
4. `if ec_connector.is_consumer: ec_connector.start_load_caches(encoder_cache, **kwargs)`：consumer 启动加载（producer 跳过）。
5. `yield output`（forward 在此执行）。
6. `finally`：
   - `output.finished_sending, output.finished_recving = ec_connector.get_finished(scheduler_output.finished_req_ids)`。
   - `ec_connector.clear_connector_metadata()`。

注意：没有 `wait_for_save` 选项（与 KV connector 不同），EC 的 save 在 `maybe_save_ec_to_connector` 里即刻触发；ctx 只负责 load + finished 标志 + 清理。

### 角色判定

- `has_ec_transfer()`：是否配置了 `ec_transfer_config`。
- `get_ec_transfer().is_consumer`：当前实例是 consumer（decode 端）还是 producer（encoder 端）。
- producer：每步 `_execute_mm_encoder` 跑编码器，`maybe_save_ec_to_connector` 存 cache + 通过 connector 发送，返回 `make_empty_encoder_model_runner_output`；本步 LM 不跑。
- consumer：`maybe_get_ec_connector_output` 上下文 `start_load_caches(encoder_cache)` 异步接收，正常跑 LM 前向，使用 `encoder_cache` 中已收到的 embeddings。

## 与其它模块/系统配合

- [GPU Model Runner V1](gpu-model-runner.md)：多继承注入；`_execute_mm_encoder` 内调 `maybe_save_ec_to_connector`，`execute_model` 头判定 `is_consumer` 分支。
- [GPU Worker](gpu-worker.md)：`ensure_ec_transfer_initialized(vllm_config)` 在 `init_worker_distributed_environment` 末尾调；`shutdown` 调 `ensure_ec_transfer_shutdown`。
- [Executor ABC](../executor/abstract.md)：`uses_sampler` 在 `RayDistributedExecutor._init_executor` 里根据 `ec_transfer_config.is_ec_consumer` 决定。
- [多模态](../../11-multimodal/README.md)：`encoder_cache` 是 `EncoderCache` 实例（V1 在 `gpu_model_runner` 内、V2 在 `mm/encoder_cache.py`）；`mm_hash` 由多模态 registry 生成。
- [分布式](../../07-distributed/README.md)：`distributed.ec_transfer.get_ec_transfer` / `has_ec_transfer` / `ensure_ec_transfer_initialized` / `ensure_ec_transfer_shutdown`；`ECConnectorBase` 由各 backend（NIXL 等）实现。
- [KV Connector Mixin](kv-connector-mixin.md)：模式相同、可同时启用（KV + EC 双 disagg）。
- [引擎核心](../../01-engine-core/README.md)：`SchedulerOutput.ec_connector_metadata` 由调度器填充；`ECConnectorOutput.finished_sending/recving` 回流。

## 历史版本演进

- **v0.9.0**：`ECConnectorModelRunnerMixin` 首次合入，服务 EPD（Encoder-Prefill-Decode）disaggregation 实验特性。
- **v0.10.0**：`maybe_get_ec_connector_output` / `maybe_save_ec_to_connector` API 稳定；`is_consumer` 角色判定接入 `execute_model` producer 分支跳过 LM。
- **v0.11.0**：`ensure_ec_transfer_initialized/shutdown` 在 Worker 生命周期接入；`uses_sampler` 在 Executor 层考虑 EC consumer 角色。
- **v0.12 / main**：与多模态 `EncoderCache` 持续协同；V2 `gpu/mm/encoder_cache.py` + `encoder_runner.py` 与 EC connector 兼容；具体 backend 实现（NIXL 等）仍在 [分布式子系统](../../07-distributed/README.md) 演进 `(待补充)`。

[← 返回执行层首页](../README.md)

## 参见

- [KV Connector Mixin](kv-connector-mixin.md)
- [GPU Model Runner V1](gpu-model-runner.md)
- [GPU Worker](gpu-worker.md)
- [Executor ABC](../executor/abstract.md)
- [多模态子系统](../../11-multimodal/README.md)
- [分布式子系统](../../07-distributed/README.md)
