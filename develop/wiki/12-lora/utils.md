[← Wiki 首页](../README.md) > [LoRA](README.md) > Utils

# LoRA Utils

> `vllm/lora/utils.py`：层选型工厂、权重名解析、模块匹配、路径解析、packed mapping 处理、全局 id 分配——LoRA 子系统的"工具腰带"。

## 是什么

`vllm/lora/utils.py` 是一组无状态函数 + 两个模块级表，被 model_manager / worker_manager / lora_model 反复调用。

| 符号 | 行 | 作用 |
|---|---|---|
| `get_captured_lora_counts` | `:49` | cudagraph 捕获的 active LoRA 计数列表（ specialize 为 2 的幂+max+1，否则仅 max+1） |
| `_GLOBAL_LORA_ID`/`get_lora_id` | `:67` | 全局自增 id，`from_local_checkpoint` 未给 id 时用 |
| `_all_lora_classes` | `:78` | 有序 LoRA 层类元组，`from_layer` 按序匹配（先具体后通用） |
| `is_moe_model` | `:98` | 模型含 `MoERunner` 则 True |
| `from_layer` | `:106` | 给原层选第一个 `can_replace_layer=True` 的 LoRA 类并建权重 |
| `from_layer_logits_processor` | `:127` | 专门建 `LogitsProcessorWithLoRA` |
| `replace_submodule` | `:145` | 用 `setattr` 把新模块挂回父模块 |
| `parse_fine_tuned_lora_name` | `:155` | 解析 `base_model.model.X.lora_A.weight` → `(module_name, is_lora_a)` |
| `is_base_embedding_weights` | `:210` | 判 base embedding 权重（加载时跳过） |
| `get_supported_lora_modules` | `:219` | 扫模型 `LinearBase`/`MoERunner`/`embedding_modules` 得支持后缀 |
| `is_supported_lora_module` | `:243` | regex 后缀匹配 |
| `is_in_target_modules` | `:271` | 对照 `LoRAConfig.target_modules`，含 packed 父子互匹配 |
| `get_adapter_absolute_path` | `:314` | 本地绝对化或 HF/ModelScope 下载快照 |
| `process_packed_modules_mapping` | `:371` | 模型 packed mapping + MoE experts 展开 + 2D/3D 选择 |

## 为什么

- **选型有序**：`_all_lora_classes` 把 `VocabParallelEmbeddingWithLoRA`、`QKVParallelLinearWithLoRA` 等具体类排在通用 `MergedColumnParallelLinearWithLoRA` 前，避免通用类抢匹配；MoE 类在最后。注释明示"Order matters"，`vllm/lora/utils.py:76`。
- **部署 vs 模型双层过滤**：`is_supported_lora_module` 看模型声明能力，`is_in_target_modules` 看运维 `target_modules` 限制，二者相与决定实际替换。packed 父子互匹配让 `target_modules=["q_proj"]` 仍能命中运行时 `qkv_proj`（`vllm/lora/utils.py:303-311`）。
- **权重名健壮**：`parse_fine_tuned_lora_name` 处理 `base_model.model.` 前缀、`weights_mapper` 映射、`lora_embedding_A/B`、非标准前缀（granite-speech，`vllm/lora/utils.py:191`）。
- **下载容错**：`get_adapter_absolute_path` 绝对路径原样返回；HF 下载失败返回原 path 让上层抛 `LoRAAdapterNotFoundError` 而非崩溃；ModelScope 分支按 `VLLM_USE_MODELSCOPE`。
- **cudagraph 单一真源**：`get_captured_lora_counts` 同时被 `CudagraphDispatcher`（经 `vllm/v1/worker/gpu/lora_utils.py:34`）与 `PunicaWrapperGPU`（`vllm/lora/punica_wrapper/punica_gpu.py:53`）消费，确保两端 active 计数一致。

## 怎么做

### from_layer（层选型，`vllm/lora/utils.py:106`）

```python
for lora_cls in _all_lora_classes:
    if lora_cls.can_replace_layer(source_layer, lora_config, packed_modules_list, model_config):
        instance = lora_cls(layer)
        instance.create_lora_weights(max_loras, lora_config, model_config)
        return instance
return layer  # 无匹配，原样返回
```

`from_layer_logits_processor`（`:127`）专建 `LogitsProcessorWithLoRA`，用 `lm_head.embedding_dim/weight.dtype/device/get_sharded_to_full_mapping()`。

### parse_fine_tuned_lora_name（`vllm/lora/utils.py:155`）

- 去掉 `base_model.model.` 前缀；若 `weights_mapper` 给定，映射后补回前缀。
- 非 `base_model.model.` 开头（granite-speech）保持前缀，`start_index=0`。
- `*.lora_A.weight`/`*.lora_B.weight` → `(前缀去掉末两段, 是否 A)`。
- `*.lora_embedding_A/B` → `(去掉末一段, 是否 A)`。

### is_in_target_modules（`vllm/lora/utils.py:271`）

- `target_modules is None` → 全通过。
- 后缀或全名命中 → True。
- packed 父（如 `gate_up_proj`）的子（`gate_proj`/`up_proj`）任一在 target → True。
- packed 子的后缀在某个 packed 父的 children 且父在 target → True。

### process_packed_modules_mapping（`vllm/lora/utils.py:371`）

- MoE 模型：`get_packed_modules_mapping(model)` + 按 `is_3d_moe_weight`/`force_2d_moe` 决定是否展开 `experts` 为各 w1/w2/w3 名（过滤 `..` 畸形项，对应非 gated MoE 空 `ckpt_up_proj_name`，`vllm/lora/utils.py:388`）。
- 非 MoE：直接 `get_packed_modules_mapping(model)`。

## 与其它模块/系统配合

- **LoRAModelManager**：`_create_lora_modules`/`_match_target_modules`/`process_packed_modules_mapping`；见 [model-manager.md](model-manager.md)。
- **WorkerLoRAManager**：`get_adapter_absolute_path`/`hf_to_vllm_mapper`；见 [worker-manager.md](worker-manager.md)。
- **LoRAModel**：`parse_fine_tuned_lora_name`/`is_base_embedding_weights`/`get_lora_id`；见 [lora-model.md](lora-model.md)。
- **layers/**：`_all_lora_classes` 引用所有 LoRA 层类；见 [layers.md](layers.md)。
- **v1 cudagraph**：`get_captured_lora_counts` → `get_lora_capture_cases`；见 [v1-integration.md](v1-integration.md)。
- [模型执行-Linear](../03-model-execution/layers/linear.md)：`LinearBase`/`MoERunner` 是支持模块探测源。
- [配置-LoRA](../10-config/lora-config.md)：`LoRAConfig.target_modules`/`fully_sharded_loras`。

## 历史版本演进

- **v0.5（首版）**：`from_layer`/`_all_lora_classes`/`parse_fine_tuned_lora_name`/`get_supported_lora_modules`/`replace_submodule`；`get_adapter_absolute_path` 走 HF。
- **v0.6+（ModelScope）**：`get_adapter_absolute_path` 按 `VLLM_USE_MODELSCOPE` 走 `snapshot_download`（`vllm/lora/utils.py:344`）。
- **v0.9（cudagraph specialize）**：`get_captured_lora_counts` 作为单一真源引入；`is_in_target_modules` 加入 packed 父子互匹配。
- **v0.11（MoE/3D）**：`process_packed_modules_mapping` 加 `force_2d_moe` 与 `..` 过滤；`is_moe_model` 用 `MoERunner` 探测。
- **main**：`get_captured_lora_counts.specialize` 分支；weights_mapper 在 `parse_fine_tuned_lora_name` 里补回前缀以支持 granite-speech 等非标准 ckpt。

## 参见

- [← 返回 LoRA 首页](README.md)
- [model-manager.md](model-manager.md)
- [worker-manager.md](worker-manager.md)
- [lora-model.md](lora-model.md)
- [layers.md](layers.md)
- [v1-integration.md](v1-integration.md)
