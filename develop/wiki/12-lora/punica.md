[← Wiki 首页](../README.md) > [LoRA](README.md) > Punica Wrapper

# Punica Wrapper（`vllm/lora/punica_wrapper/`）

> LoRA 多租户内核的"状态机 + 派发器"：维护 token→adapter 索引、prefill/decode 元数据，把不同前向场景（linear/embedding/logits/MoE）派发给对应底层算子。按平台分 GPU/CPU/XPU 三实现。

## 是什么

| 文件 | 类 | 平台 |
|---|---|---|
| `punica_base.py:124` | `PunicaWrapperBase` | 抽象，持有共享 metadata |
| `punica_base.py:22` | `PunicaWrapperABC` | 纯 ABC 接口 |
| `punica_gpu.py:33` | `PunicaWrapperGPU`（`@final`） | CUDA（Triton） |
| `punica_cpu.py:22` | `PunicaWrapperCPU` | CPU（torch_ops）；兼容平台可继承 |
| `punica_xpu.py:31` | `PunicaWrapperXPU`（`@final`） | Intel XPU（C ext + 部分 Triton） |
| `punica_selector.py:13` | `get_punica_wrapper` | 工厂，按 `current_platform.get_punica_wrapper()` 选型 |
| `utils.py:15` | `compute_meta`/`convert_mapping` | metadata 计算工具 |

`__init__.py` 重导出 `PunicaWrapperBase` 与 `get_punica_wrapper`。

### 共享状态（`PunicaWrapperBase.__init__`，`punica_base.py:131`）

| 字段 | 说明 |
|---|---|
| `_token_lora_indices` | token→slot 索引（-1=无 LoRA） |
| `_sampler_indices`/`_sampler_indices_padded` | 请求级采样索引（logits 用） |
| `_embeddings_indices` | embedding LoRA 索引（2×N） |
| `_seq_start_locs`/`_seq_lengths`/`_lora_indices_per_batch` | SGMV 段化元数据 |
| `indices_len` | 上述四张量有效长度 |
| `is_prefill`/`no_lora`/`batch_size`/`max_length`/`token_nums` | 步级状态 |

### 抽象方法

`update_metadata`、`add_shrink`、`add_expand`、`add_lora_embedding`、`add_lora_linear`、`add_lora_logits`、`moe_lora_align_block_size`、`add_lora_fused_moe`、`add_lora_w13`、`add_lora_w2`（`punica_base.py:284-564`）。

## 为什么

- **状态与算子解耦**：索引/段化等"状态"在 base 一份，"算子调用"随平台实现，避免三后端各写一遍 metadata 逻辑。
- **平台即插即用**：`current_platform.get_punica_wrapper()` 返回 qualname，`resolve_obj_by_qualname` 动态导入（`punica_selector.py:13`），新平台只需在 platform 注册 qualname。
- **prefill/decode 双路径**：`is_prefill` 决定 SGMV（段化 GEMM）vs BGMV（逐 token），`_update_prefill_metadata` 用 `torch.unique_consecutive` 合并连续同 id 段，`no_lora` 短路跳过 kernel。
- **MoE 一体化**：`add_lora_w13`/`add_lora_w2` 分阶段供 `FusedMoEWithLoRA`，`moe_lora_align_block_size` 对齐 token×expert block，`add_lora_fused_moe` 单入口。
- **cudagraph 对齐**：GPU 版用 `LoRAKernelMeta` 按 `captured_lora_counts` 分桶，与 `v1/worker/gpu/lora_utils.get_lora_capture_cases` 一致。

## 怎么做

### update_metadata（`punica_base.py:284`）

1. `_update_base_metadata`：`convert_mapping(mapping, lora_index_to_id, max_loras, vocab_size, 0, device)` 把 `LoRAMapping` 转成 4 张索引张量，拷进预分配 buffer。`convert_mapping`（`utils.py:54`）建 `lora_id_to_index` 反查表（O(1) 而非 `list.index`），生成 `base_indices`/`sampler_indices`/`sampler_indices_padded`/`embeddings_indices`。
2. `is_prefill` 时 `_update_prefill_metadata`：`compute_meta(token_lora_indices)`（`utils.py:15`）做 `unique_consecutive` 得段起止/长度/每段 id，填 `_seq_start_locs`/`_seq_lengths`/`_lora_indices_per_batch`，算 `batch_size`/`max_length`/`no_lora`。
3. GPU 版额外 `token_mapping_meta.prepare_tensors`/`prompt_mapping_meta.prepare_tensors`（`punica_gpu.py:75`）。

### add_shrink / add_expand

- GPU（`punica_gpu.py:90`/`:123`）：`x.view(-1, dim)` → `lora_shrink(x, a_stacked, y, *meta_args, scale)` / `lora_expand(y, x, b_stacked, output_slices, offset_start, add_inputs, *meta_args)`。
- CPU（`punica_cpu.py:38`+）：`is_prefill` 走 `sgmv_*`（段化），否则 `bgmv_*`；`no_lora` 直接 return。

### add_lora_linear（linear 层主入口）

组装 shrink→expand 两步；`buffer` 可外部传入复用。`output_slices` 决定 expand 按 slice 写入区间。GPU 版直接两调 Triton。

### add_lora_logits（`LogitsProcessorWithLoRA` 用）

`buffer = (x @ lora_a) * scale; y += buffer @ lora_b`，输出维度为 vocab。

### add_lora_embedding（`VocabParallelEmbeddingWithLoRA` 用）

只 expand：`y += x @ lora_b`（A 嵌入已在层里用 `F.embedding` 取）。

### MoE 系列

- `moe_lora_align_block_size`：把 token×expert 对齐到 block，输出 `sorted_token_ids_lora`/`expert_ids_lora`/`num_tokens_post_padded_lora`/`token_lora_mapping`。
- `add_lora_w13`：w13 shrink 前，返回路由张量供 w2 复用。
- `add_lora_w2`：w2 expand 前，复用 w13 路由。
- `add_lora_fused_moe`：单入口融合 shrink+expand，供量化/非量化 MoE。

### 工厂选型（`punica_selector.py:13`）

```python
qualname = current_platform.get_punica_wrapper()  # e.g. "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"
cls = resolve_obj_by_qualname(qualname)
return cls(*args, **kwargs)
```

`LoRAModelManager._init_punica_wrapper`（`vllm/lora/model_manager.py:148`）按语言/tower/connector 各调一次建独立 wrapper。

## 与其它模块/系统配合

- **LoRAModelManager**：建/选 wrapper，`_set_adapter_mapping` 调 `update_metadata`；见 [model-manager.md](model-manager.md)。
- **BaseLayerWithLoRA 系**：前向调 `add_lora_*`；见 [layers.md](layers.md)。
- **ops/**：底层算子；见 [ops.md](ops.md)。
- **v1 ModelRunner**：每步 `set_active_loras`→`set_active_adapters`→`set_adapter_mapping`→`update_metadata`；见 [v1-integration.md](v1-integration.md)。
- [08-硬件平台](../08-platforms/README.md)：`Platform.get_punica_wrapper()` 提供 qualname。
- [多模态](../11-multimodal/README.md)：tower/connector 各自 wrapper。

## 历史版本演进

- **v0.5（首版）**：单 `PunicaWrapper`（CUDA Triton），shrink/expand/embedding/logits；`convert_mapping`/`compute_meta`。
- **v0.7（v1）**：metadata 流稳定；`PunicaWrapperBase` 与子类分离。
- **v0.9（多后端）**：拆 `PunicaWrapperGPU`/`CPU`/`XPU` + `punica_selector.get_punica_wrapper` + `resolve_obj_by_qualname`；ops 同步拆三子包。`@final` 防误继承。
- **v0.10/v0.11（MoE + cudagraph）**：`add_lora_w13`/`add_lora_w2`/`add_lora_fused_moe`/`moe_lora_align_block_size`；GPU `LoRAKernelMeta` + `captured_lora_counts` 对齐 cudagraph specialize；`token_lora_mapping` 覆盖支持 EP+LoRA 本地映射。
- **main**：`convert_mapping` 用反查表消除 O(n²) `list.index`；`no_lora` 短路；PDL/TMA 在 op 层。

## 参见

- [← 返回 LoRA 首页](README.md)
- [ops.md](ops.md)
- [layers.md](layers.md)
- [model-manager.md](model-manager.md)
- [v1-integration.md](v1-integration.md)
- [08-硬件平台](../08-platforms/README.md)
