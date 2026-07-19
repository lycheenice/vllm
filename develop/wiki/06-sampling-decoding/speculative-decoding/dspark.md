[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > DSpark

# DSpark（DSV4 DSpark）

> 源码：drafter 类位于 `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`；模型实现位于 `vllm/models/deepseek_v4/nvidia/dspark.py` 与 `vllm/model_executor/models/qwen3_dspark.py`；配置位于 `vllm/config/speculative.py`。

---

## 是什么

DSpark 是 DeepSeek V4（DSV4）模型的并行 drafting 特化方法，方法字符串为 `"dspark"`。它继承 DFlash 的并行 drafting 理念——一次 drafter forward 同时算 K 个 draft token——但针对 DSV4 的模型架构（sparse attention、Markov rank、bonus anchor 等）做了专门优化。

**注意 DSpark 没有对应的 `vllm/v1/spec_decode/dspark.py` 文件**：

- drafter 类（proposer）：`DSparkProposer` 位于 `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`。
- speculator（verifier）：`DSparkSpeculator` 在 `vllm/v1/worker/gpu/spec_decode/__init__.py:18` 处 lazy import。
- 模型定义：`vllm/models/deepseek_v4/nvidia/dspark.py` 提供 NVIDIA 平台特化，从 `vllm/model_executor/models/qwen3_dspark.py` 复用基础 Qwen3 DSpark 架构。
- 模型注册：`Qwen3DSparkModel` 在 `vllm/model_executor/models/registry.py:597` 注册。

## 为什么

- **DSV4 架构专门化**：DSV4 用 sparse attention（lightning-indexer）与 MLA，drafter 必须能复用 sparse index 计算。DFlash 的 generic `precompute_and_store_context_kv` 不足以表达 DSpark 的 multi-layer KV 投影与 sparse mask。
- **独立 speculator**：DSpark 的 verifier 端（target verify + reject）需要更多 DSV4 专门逻辑（如 sparse SWA index kernel），因此在 `vllm/v1/worker/gpu/spec_decode/dspark/` 下有独立 speculator 类，而非复用 `RejectionSampler`。
- **Markov 预测**：DSV4 的 dspark hf_config 含 `dspark_markov_rank` 字段——drafter 用 Markov chain 增强预测，需要专门接口。
- **bonus anchor**：`dspark_bonus_anchor` 字段（在 `transformers_utils/configs/speculators/algos.py` 中 `update_dspark` 设置为 True）决定 bonus token 的 anchor 位置（待核实：实际语义）。
- **noise token**：`dspark_noise_token_id` 是 DSpark 专用 mask token，类似 DFlash 的 `mask_token_id`。

## 怎么做

由于 DSpark 实现细节分散在多个非 `vllm/v1/spec_decode/` 目录中，本节只列关键文件与入口，深入细节请直接读源码。

###drafter 创建入口

`vllm/v1/worker/gpu/spec_decode/__init__.py:17`（speculator factory）：

```python
elif speculative_config.method == "dspark":
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    return DSparkSpeculator(vllm_config, device)
```

这是 speculator（verifier），与 proposer 区分。drafter 实例化在 `GPUModelRunner` 中（待核实：DSpark 是否走 `GPUModelRunner.execute` 中的通用 drafter API，还是独立路径；从 `gpu_model_runner.py:196` 仅看到 `eagle3 / dflash / dspark` 共同触发"drafter"逻辑）。

### drafter load

`vllm/v1/worker/gpu/spec_decode/dspark/utils.py:12`：

```python
def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    with set_model_tag("dspark_head"):
        ...
```

`set_model_tag("dspark_head")` 隔离 compile cache。

### 配置识别

`SpeculativeConfig.use_dspark()`（`vllm/config/speculative.py:1247`）：

```python
return self.method == "dspark"
```

`SpeculativeConfig.__post_init__` 中自动识别：当 `draft_model_config.model` 含 `"dspark"` 字串时强制 `method = "dspark"`（行 849-852）。

### 模型架构（NVIDIA 特化）

`vllm/models/deepseek_v4/nvidia/dspark.py` 主要类：

- `DSparkForCausalLM`：drafter top-level 模型，继承 `Qwen3DSparkForCausalLM`（基础架构）。
- `target_layer_ids = config.dspark_target_layer_ids`：指定哪些 target layer 的 hidden state 用于 drafter 输入。
- `num_dspark_layers = config.n_mtp_layers or 3`：drafter 层数。
- `dspark_markov_rank`：Markov chain rank 字段。
- `combine_hidden_states`：与 EAGLE3 类似，合并 target 多层 hidden。
- `_remap_dspark_name`：权重名映射，从 DSV4 checkpoint 加载 drafter 参数。

### sparse attention 集成

`vllm/v1/attention/backends/mla/sparse_swa.py` 中的 `DeepseekV4ROCMAiterSparseSWAMetadata` 与 DSpark 紧密配合：

```python
self.is_dspark = spec_config is not None and spec_config.use_dspark()
...
if self.is_dspark:
    ...
    _compute_dspark_noncausal_swa_indices_kernel(num_decode_tokens,)
```

DSpark 在 ROCm 上使用 sparse SWA（sliding window attention）+ noncausal index kernel。

### speculator initializer 与 dispatch

`init_speculator(vllm_config, device)`（`vllm/v1/worker/gpu/spec_decode/__init__.py`）在 `method == "dspark"` 时返回 `DSparkSpeculator`。注意此函数命名是 `init_speculator` 而非 `init_proposer`，且仅 eagle/gemma4/mtp/dflash/dspark 走这条路径——ngram/suffix/medusa 等不进入。

## 与其它模块/系统配合

- [dflash.md](dflash.md)：概念上的"父"算法，DSpark 借鉴其并行 drafting 思想。
- [eagle.md](eagle.md)：`combine_hidden_states` 接口同 EAGLE3。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：`gpu_model_runner.py:196` 触发 eagle3/dflash/dspark 共同样的 `use_aux_hidden_state_outputs` 逻辑。
- [模型库-DeepSeek V4](../../04-model-zoo/README.md)（待补充）：DSV4 主模型与 DSpark drafter 的关系。
- [注意力后端-MLA sparse SWA](../../05-attention/README.md)：DSpark 直接调用 sparse SWA 代码。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：`vllm/v1/core/sched/scheduler.py:253` 通过 `speculative_config.use_dspark()` 触发 DSpark 专用调度路径。

## 历史版本演进

- **v0.11.0**：DSpark 与 DFlash、DSV4 同期 landfall；`dspark_target_layer_ids` / `dspark_markov_rank` / `dspark_bonus_anchor` / `dspark_noise_token_id` 字段加入。
- **v0.11.5**：`update_dspark` speculator 算法在 `transformers_utils/configs/speculators/algos.py:125` 注册，自动设置 `dspark_bonus_anchor=True`。
- **v0.12 / main**：sparse SWA kernel 集成；dspark + DSV4 在 NVIDIA + ROCm 两平台均支持。

[← 返回投机解码](../README.md)

## 参见

- [dflash.md](dflash.md)
- [eagle.md](eagle.md)
- [../README.md](../README.md)：drafter 与 speculator 的概览区分
