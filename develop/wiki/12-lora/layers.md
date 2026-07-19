[← Wiki 首页](../README.md) > [LoRA](README.md) > Layers

# LoRA Layers（`vllm/lora/layers/`）

> 把 vLLM 原模型层替换成"基座层 + LoRA 增量"的包装层。每个 LoRA 层持有 GPU 预分配的 `lora_a/b_stacked`，前向时调 Punica wrapper 叠加增量。

## 是什么

`vllm/lora/layers/__init__.py:1` 导出基于 `BaseLayerWithLoRA` 的层族，按原层类型选型替换：

| LoRA 层类 | 原层 | 文件 | 场景 |
|---|---|---|---|
| `BaseLayerWithLoRA` | — | `base.py:16` | 抽象基类 |
| `BaseLinearLayerWithLoRA` | `LinearBase` | `base_linear.py:70` | 线性层通用前向/双流 |
| `ColumnParallelLinearWithLoRA` | `ColumnParallelLinear`/`MergedColumnParallelLinear` | `column_parallel_linear.py:85` | 列并行 |
| `MergedColumnParallelLinearWithLoRA` | `MergedColumnParallelLinear`（2 slice） | `column_parallel_linear.py:184` | gate_up |
| `QKVParallelLinearWithLoRA` | `QKVParallelLinear`（单 LoRA） | `column_parallel_linear.py:371` | chatglm/baichuan qkv |
| `MergedQKVParallelLinearWithLoRA` | `QKVParallelLinear`（3 slice） | `column_parallel_linear.py:437` | 标准 3-LoRA qkv |
| `MergedColumnParallelLinearVariableSliceWithLoRA` | `MergedColumnParallelLinear`（3+ slice） | `column_parallel_linear.py:674` | in_proj_qkvz 等 |
| `*WithShardedLoRA`（4 个） | 同上 | `column_parallel_linear.py:504`+ / `row_parallel_linear.py:101` | S-LoRA 全分片 |
| `RowParallelLinearWithLoRA`/`...ShardedLoRA` | `RowParallelLinear` | `row_parallel_linear.py:22` / `:101` | 行并行 |
| `ReplicatedLinearWithLoRA` | `ReplicatedLinear` | `replicated_linear.py:16` | 复制层（如 GateLinear） |
| `VocabParallelEmbeddingWithLoRA` | `VocabParallelEmbedding` | `vocal_parallel_embedding.py:17` | embedding LoRA |
| `LogitsProcessorWithLoRA` | `LogitsProcessor` | `logits_processor.py:20` | lm_head logits |
| `FusedMoEWithLoRA` | `MoERunner`（2D） | `fused_moe.py:33` | 通用 MoE |
| `FusedMoE3DWithLoRA` | `MoERunner`（3D 融合） | `fused_moe.py:461` | GPT-OSS 等 3D |
| `LoRAMapping`/`LoRAMappingType` | — | `utils.py:27`/`:33` | 映射数据结构 |

## 为什么

- **不侵入基座**：以装饰器模式包装原 `base_layer`，前向先跑基座量化方法再叠加 LoRA，量化（GPTQ/AWQ/CompressedTensor/Marlin）零改动。
- **预分配 stacked buffer**：每层建 `(max_loras, 1, rank, dim)` 形状 buffer，激活时 `set_lora(index)` 拷贝，前向按 index 取，图静态、cudagraph 可捕获。
- **TP 切片语义**：`slice_lora_a/b` 按 base_layer 类型（列/行/合并/qkv）正确切分，sharded 变体按 S-LoRA 切 rank 维并配合 all-gather/all-reduce。
- **MoE 融合**：`FusedMoEWithLoRA` 注入 `MoELoRAContext` 到专家 kernel，w13/w2 分阶段在路由前后叠加 LoRA，避免专家循环内核 launch。
- **双流重叠**：`VLLM_LORA_ENABLE_DUAL_STREAM` 让基座 GEMM 与 LoRA 在不同 CUDA stream 并行（`base_linear.py:33`+`_apply_async_impl`）。
- **选型有序**：`utils._all_lora_classes` 先具体后通用，`can_replace_layer` 各自声明适用条件（含 `fully_sharded_loras` 装饰器）。

## 怎么做

### BaseLayerWithLoRA（`base.py:16`）

抽象接口：`slice_lora_a/b`、`create_lora_weights`、`reset_lora`、`set_lora`、`set_mapping`（存 `punica_wrapper`）、`can_replace_layer`。

### BaseLinearLayerWithLoRA（`base_linear.py:70`）

线性层通用骨架：
- `create_lora_weights`（`:100`）：按 `ReplicatedLinear`/`ColumnParallelLinear`/`RowParallelLinear` 推断 `lora_a/b_stacked` 形状，fully_sharded 时按 `divide(rank, tp_size)` 切。
- `apply`（`:194`）：双流开则 `torch.ops.vllm.lora_linear_async`（注册于 `:63`），否则 `_apply_sync`：基座 `quant_method.apply` → `_apply_lora_to_output`。
- `_apply_lora_to_output`（`:215`）：拍平 batch 维 → `punica_wrapper.add_lora_linear(output, x, a_stacked, b_stacked, 1.0, output_slices)` → 还原形状。
- `_apply_async_impl`（`:240`）：`maybe_execute_in_parallel(base_fn, lora_fn, events, stream)`，LoRA 输出初始化为 zeros（防 `lora_id==-1` 早退污染），最后 `output.add_(lora_result)`。
- `weight`/`bias` 属性代理基座各种量化字段（`weight`/`weight_packed`/`qweight`/`B`）。

### 列并行族（`column_parallel_linear.py`）

- `ColumnParallelLinearWithLoRA.slice_lora_b`（`:107`）：`MergedColumnParallelLinear` 切两半，普通列并行切一段。
- `MergedColumnParallelLinearWithLoRA`（`:184`）：`n_slices=2`，`output_slices` 按 `divide(output_sizes, tp_size)`；`set_lora` 支持非等长 packed group 的 `expand_packed_lora`（如 in_proj_qkv + in_proj_z → 4 slice）。
- `QKVParallelLinearWithLoRA`（`:371`）：单一 LoRA，按 q/kv shard 切 B。
- `MergedQKVParallelLinearWithLoRA`（`:437`）：3 LoRA，q 与 kv shard 不同大小。
- sharded 变体用 `_mcp_apply`（`:24`）：shrink 后 `tensor_model_parallel_all_gather` buffer，再 expand；装饰器 `_fully_sharded_can_replace` 限定 `fully_sharded_loras=True`。
- `MergedColumnParallelLinearVariableSliceWithLoRA`（`:674`）：3+ slice，`set_lora` 支持单 tensor 切多 slice。

### 行并行族（`row_parallel_linear.py`）

- `RowParallelLinearWithLoRA.slice_lora_a`（`:32`）：按 `input_size_per_partition` 列切 A。
- `RowParallelLinearWithShardedLoRA.apply`（`:118`）：shrink → `all_reduce(buffer)` → `add_expand(offset_start=tp_rank*shard, add_input=True)`，融合 all-gather/all-reduce（S-LoRA）。

### ReplicatedLinearWithLoRA（`replicated_linear.py:16`）

`apply` 走 `_apply_base_forward`，保留子类（如 `GateLinear`）自定义 forward；`can_replace_layer` 无条件 True（复制层本就每 GPU 一份）。

### VocabParallelEmbeddingWithLoRA（`vocal_parallel_embedding.py:17`）

- `create_lora_weights`：`lora_a_stacked (max_loras, org_vocab_size, rank)` + `lora_b_stacked (max_loras, 1, embed_dim, rank)`；清零 added embedding 区。
- `forward`（`:96`）：`F.embedding(x + indices_1, lora_a_stacked_2d)` 取 A 嵌入 → 基座 embedding → `add_lora_embedding` 叠加 B。
- `set_lora`：A 需转置（stacked 行主，lora_a 列主）。

### LogitsProcessorWithLoRA（`logits_processor.py:20`）

- 词汇上限 258048（`:91`）。
- `_get_logits`：基座 lm_head → `_gather_logits` → 可选 `sharded_to_full_mapping_gpu` 重排 → `add_lora_logits` → 切到 `vocab_size`。
- `can_replace_layer` 返回 False（由 `from_layer_logits_processor` 专门构造）。

### FusedMoEWithLoRA / FusedMoE3DWithLoRA（`fused_moe.py`）

- 构造（`:34`）：取 `MoERunner`，校验非 monolithic kernel；构建 `FusedMoEKernel`（`LoRAExpertsMixin`），`_replace_quant_method` 成 `FusedMoEModularMethod`。`_w13_slices = 2`（gated）或 `1`（non-gated）。
- `create_lora_weights`（`:239`）：`w13_lora_a_stacked`/`w13_lora_b_stacked`（`_w13_slices` 个）、`w2_lora_a_stacked`/`w2_lora_b_stacked`；`adapter_enabled` tensor。
- `set_lora`（`:355`）：解 `w1/w2/w3`，`_slice_w13_*`/`_slice_w2_*` 按 fully_sharded/TP 切，copy 进 stacked。
- `_build_lora_context`（`:132`）：建 `MoELoRAContext` 传给 `fused_experts.set_lora_context` 与 `prepare_finalize.set_lora_context`，含 aux stream/events。
- 3D 版（`:461`）：`_w13_slices=1`，`w13_lora_b_stacked` 形状用 `intermediate*2`；`_slice_w13_b` 处理 GPT-OSS 交错布局（`::2`/`1::2`，`:529`）。
- `can_replace_layer`：MoE + `packed_modules_list` 长度 2（2D）或 1（3D）。

### LoRAMapping / LoRAMappingType（`utils.py`）

`LoRAMapping(index_mapping, prompt_mapping, is_prefill, type)`；`LoRAMappingType.LANGUAGE/TOWER/CONNECTOR`。`_get_lora_device` 兼容各种量化层找 device；`try_get_optimal_moe_lora_config` 按 rank 调 BLOCK_SIZE_N/K。`_get_lora_aux_cuda_stream` 单例 aux stream（受 `VLLM_LORA_ENABLE_DUAL_STREAM` 控制）。

## 与其它模块/系统配合

- **LoRAModelManager**：构造期 `from_layer` 选型、`activate_adapter` 调 `set_lora`/`reset_lora`、`set_mapping` 注入 wrapper；见 [model-manager.md](model-manager.md)。
- **Punica wrapper**：层前向调 `add_lora_linear`/`add_lora_logits`/`add_lora_embedding`/`add_lora_fused_moe`；见 [punica.md](punica.md)。
- **utils._all_lora_classes**：选型顺序表；见 [utils.md](utils.md)。
- [模型执行-Linear](../03-model-execution/layers/linear.md)：被包装的原层。
- [模型执行-FusedMoE](../03-model-execution/layers/fused-moe.md)：`MoERunner`/`MoELoRAContext`/`LoRAExpertsMixin`。
- [09-编译与 IR](../09-compilation-ir/README.md)：`lora_linear_async` 自定义 op 与 `static_forward_context` 注册。

## 历史版本演进

- **v0.5（首版）**：`BaseLayerWithLoRA` + 列/行/merged/qkv/embedding/logits 层；Punica shrink/expand；`_not_fully_sharded_can_replace`/`_fully_sharded_can_replace` 装饰器。
- **v0.7（v1）**：层接口稳定；`BaseLinearLayerWithLoRA` 抽出通用前向骨架，减少重复。
- **v0.8/v0.9（S-LoRA）**：`*WithShardedLoRA` 全分片族 + `_mcp_apply`/`RowParallelLinearWithShardedLoRA.apply` all-gather/all-reduce 融合。
- **v0.10（MoE LoRA 2D）**：`FusedMoEWithLoRA` + `MoELoRAContext` + `LoRAExpertsMixin`，w13/w2 分阶段。
- **v0.11（3D + 双流）**：`FusedMoE3DWithLoRA` 与 GPT-OSS 交错切片；`VLLM_LORA_ENABLE_DUAL_STREAM` 注册 `lora_linear_async` op + `maybe_execute_in_parallel`；`MergedColumnParallelLinearVariableSliceWithLoRA` 支持 3+ slice。
- **main**：非 gated MoE `_w13_slices=1`；tower/connector LoRA 经 `LoRAMappingType` 路由；MoE aux stream 4 events。

## 参见

- [← 返回 LoRA 首页](README.md)
- [model-manager.md](model-manager.md)
- [punica.md](punica.md)
- [ops.md](ops.md)
- [utils.md](utils.md)
- [模型执行-Linear](../03-model-execution/layers/linear.md)
- [模型执行-FusedMoE](../03-model-execution/layers/fused-moe.md)
