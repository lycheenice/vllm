# 词表并行嵌入（vocab_parallel_embedding.py）

[← Wiki 首页](../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > 词表并行嵌入

`vllm/model_executor/layers/vocab_parallel_embedding.py`（~561 行）实现 vLLM 全部词表（vocabulary）维度的并行嵌入层，包括输入端的 `VocabParallelEmbedding` 与输出端的 `ParallelLMHead`，并通过 `UnquantizedEmbeddingMethod` 与量化子树共享 `QuantizeMethodBase` 抽象。

## 是什么

| 类/函数 | 行 | 角色 |
|---|---|---|
| `UnquantizedEmbeddingMethod` | `vocab_parallel_embedding.py:35` | 默认嵌入方法：`create_weights` 建 `Parameter`；`apply` 走 GEMM；`embedding` 走 `F.embedding`；`tie_weights` 把 LMHead weight 与 embed_tokens 共享 |
| `pad_vocab_size` | `:87` | 把 vocab 维按 `DEFAULT_VOCAB_PADDING_SIZE=64` 向上取整，保证 `tp_size` 整除 |
| `vocab_range_from_per_partition_vocab_size` / `vocab_range_from_global_vocab_size` | `:92` / `:100` | 给定 rank/world_size 计算本 rank 持有的词表区间 |
| `VocabParallelEmbeddingShardIndices` | `:109` | dataclass：固定 8 个 index，描述"base / base_padding / LoRA-added / LoRA-padding" 在本 rank 的起止 |
| `get_masked_input_and_mask` | `:168` | torch.compile 的纯函数：把超出本 rank 词表区间的 token id 置 0 并返回 mask |
| `VocabParallelEmbedding` | `:198` | 词表并行嵌入层，注册为 `PluggableLayer` 命名 `vocab_parallel_embedding` |
| `ParallelLMHead` | `:505` | 输出 logits 的并行 LM head，注册为 `PluggableLayer` 命名 `parallel_lm_head`；继承自 `VocabParallelEmbedding` 但 `forward` 抛错（设计上只能被 Sampler 直接当权重用） |

## 为什么

把 Embedding 单独工程化的动机：

1. **词表巨大**：32k–256k 词表，按 hidden_dim 投影下来动辄数百 MB，必须按词表维度切到多卡。
2. **支持 LoRA-added 词表**：训练时词表固定，但在 SFT/LoRA 场景常常扩展几十到几千个新 token。vLLM 顶层布局是 `[BASE | BASE_PADDING | LORA | LORA_PADDING]` 四段（`vocab_parallel_embedding.py:212-225`），base 和 LoRA 各自独立 TP 切分并独立 padding，使得 LoRA 替换不需要重排 base。这是 vLLM LoRA 子系统的底层前提（见 [LoRA #12](../../12-lora/README.md)）。
3. **支持 tied weights**：很多模型 `lm_head.weight = embed_tokens.weight`，通过 `ParallelLMHead.tie_weights(embed_tokens)` 把 `layer.weight` 直接指向 `embed_tokens.weight`，省一份。
4. **量化可注入但又与 Linear 共享抽象**：`Fp8EmbeddingMethod` 等量化方法通过 `quant_config.get_quant_method(self, prefix)` 挂入，与 `LinearBase.quant_method` 用同一套 `QuantizeMethodBase` 接口（见 [quantization/](quantization/README.md)）。
5. **TP gather 顺序与 Sampler 对齐**：`VocabParallelEmbedding.forward` 做 all-reduce（`vocab_parallel_embedding.py:491`），等价于"每 rank 各算自己的词表段，最后求和"；`ParallelLMHead` 则不前向，只把权重交给 `LogitsProcessor`，后者调 `lm_head.quant_method.apply`（实际是 GEMM）并按 `get_sharded_to_full_mapping` 重新排列 gather 后的 logits，使"采样视角下的 index ↔ token_id 1:1"。

## 怎么做

### 四段布局与 `_get_indices`

`VocabParallelEmbedding.__init__`（`vocab_parallel_embedding.py:239-325`）：

1. 计算 `org_vocab_size`（来自 `org_num_embeddings` 或全词表）、`num_added_embeddings = num_embeddings - org_vocab_size`。
2. `org_vocab_size_padded = pad_vocab_size(org_vocab_size, 64)`；`num_embeddings_padded = pad_vocab_size(org_vocab_size_padded + num_added_embeddings, 64)`。
3. `_get_indices(...)` 给定 `tp_rank/tp_size` 计算本 rank 的 8 个 index（padded 和 unpadded 各四段）。`assert num_elements_padded == num_embeddings_per_partition` 保证张量长度对齐。
4. 通过 `quant_config.get_quant_method(self, prefix)` 取得 `quant_method`；若 `type(self) is VocabParallelEmbedding` 则要求 quant method 实现了 `embedding` 方法（`vocab_parallel_embedding.py:285-293`），保证 `forward` 可以调到 `quant_method.embedding(self, masked_input.long())`。
5. `quant_method.create_weights(...)` 创建权重（`UnquantizedEmbeddingMethod` 直接 `Parameter(torch.empty(num_emb_per_part, embedding_dim))` + `set_weight_attrs(weight, {"input_dim":1, "output_dim":0})`）。

### `weight_loader`

`vocab_parallel_embedding.py:430-470`：

- 若 `param.output_dim is None`（如 `g_idx`），直接复制到所有 rank。
- 否则按 `shard_indices.org_vocab_start_index` 切 base 段，若 `packed_dim == output_dim` 则按 `packed_factor` 调整 offset/size。
- `param[:shard_size].copy_(); param[shard_size:].fill_(0)`——已切的部分写入，剩余填 0（padding 段即为 0）。

### `forward`

```
if tp_size > 1:
    masked_input, input_mask = get_masked_input_and_mask(...)  # torch.compile fused
output_parallel = quant_method.embedding(self, masked_input.long())
if tp_size > 1:
    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
output = tensor_model_parallel_all_reduce(output_parallel)
```

`get_masked_input_and_mask`（`vocab_parallel_embedding.py:168-193`）把"在 base 区间内"、"在 LoRA-add 区间内"两种 mask 合并；超出区间的 token id 被置为 0，后续 `masked_fill_` 把对应输出清零以避免读出垃圾行再 all-reduce 求和时污染。

### `ParallelLMHead` 与 Sampler

`ParallelLMHead.__init__` 调父类构造（`vocab_parallel_embedding.py:523-547`），但 `forward(input_)` 直接 `raise RuntimeError("LMHead's weights should be used in the sampler.")`（`:559-561`）。这是有意的——LM head 在 vLLM 里不被"前向"调用，而是把 `weight`（和 `bias`）传给 [LogitsProcessor](embedding.md) 与 [Sampler](sampler-layer.md)，由它们用 GEMM 计算 logits。

`tie_weights(embed_tokens)` 把 `layer.weight` 替换为 `embed_tokens.weight`（`UnquantizedEmbeddingMethod.tie_weights`，`:80-84`）。

### `get_sharded_to_full_mapping`

`vocab_parallel_embedding.py:365-428`：当 `tp_size >= 2` 时返回一个长度为 `num_embeddings_padded` 的重排表 `[base..., lora..., padding...]`，让 Sampler 在 all-gather logits 后按这个表 reindex，使得最终 logits 张量的第 `i` 行严格对应 token_id `i`。`tp_size<2` 时返回 `None` 表示无需重排。

## 与其它模块/系统配合

- [模型库 #04](../../04-model-zoo/README.md)：所有 LLM 的 `embed_tokens` 与 `lm_head` 都用这两个类；tied weight 由模型 `__init__` 调 `lm_head.tie_weights(embed_tokens)`。
- [linear.md](linear.md)：`UnquantizedEmbeddingMethod.apply` 与 `UnquantizedLinearMethod.apply` 都走 `dispatch_unquantized_gemm()`，让 LM head 和 Linear 享用同一 GEMM 分派；`VLLM_BATCH_INVARIANT` 时都改走 `linear_batch_invariant`（`:73-75`）。
- [LoRA #12](../../12-lora/README.md)：LoRA-added 词表的四段布局直接服务 LoRA；LoRA 子系统在加载 lora 时把 added embedding 写到对应 partition。
- [sampler-layer.md](sampler-layer.md)：`Sampler` 直接消费 `LogitsProcessor.forward(lm_head, hidden_states)` 产出的 logits；`get_sharded_to_full_mapping` 由 Sampler/LogitsProcessor 使用。
- [distributed #07](../../07-distributed/README.md)：`tensor_model_parallel_all_reduce`（embedding forward）、`tensor_model_parallel_all_gather` / `tensor_model_parallel_gather`（logits gather，由 `LogitsProcessor` 调，参见 [embedding.md](embedding.md) 同名小节）。
- [quantization/](quantization/README.md)：`Fp8EmbeddingMethod`、`MarlinEmbeddingMethod` 等量化嵌入方法都通过 `quant_config.get_quant_method(self, prefix)` 注入。
- [custom-op.md](custom-op.md)：`VocabParallelEmbedding` 与 `ParallelLMHead` 都是 `PluggableLayer.register(...)`，可由厂商通过 `register_oot` 整层替换。
- [compilation-ir #09](../../09-compilation-ir/README.md)：`get_masked_input_and_mask` 用 `@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)` 装饰（`:168`），把多个 pointwise mask 合成单 kernel；`PluggableLayer` 留给 OOT 替换。

## 历史版本演进

- **早期**：`VocabParallelEmbedding` 实现朴素，base + padding 两段，不支持 LoRA-added。
- **v0.5–v0.6**：引入"base + base padding + LoRA + LoRA padding"四段布局，支持 vLLM LoRA 子系统新增词表。
- **v0.6**：`pad_vocab_size` 默认 `DEFAULT_VOCAB_PADDING_SIZE=64`；`get_sharded_to_full_mapping` 引入服务 TP>1 时 logits reindex。
- **v0.7–v0.8**：`UnquantizedEmbeddingMethod` 抽象引入，与 `LinearMethodBase` 共享 `QuantizeMethodBase` 基类；`embedding` 方法独立于 `apply`。
- **v0.8–v0.9**：`ParallelLMHead` 改为继承 `VocabParallelEmbedding` 但禁用 `forward`，让 LM head 专门为 Sampler 服务；`tie_weights` 通过 quant method 间接调用支持量化嵌入的 tied 权重。
- **v0.9–v0.10**：`get_masked_input_and_mask` 改用 `@torch.compile` 装饰（`vocab_parallel_embedding.py:168`），把 mask 逻辑融合成单 kernel；`VocabParallelEmbeddingShardIndices` dataclass 引入，把 index 计算与 sanity 检查集中化。
- **v0.10（PR #32744）**：`PluggableLayer` 抽象落地，`VocabParallelEmbedding` 与 `ParallelLMHead` 改用 `PluggableLayer.register("vocab_parallel_embedding")` / `register("parallel_lm_head")`，可整层 OOT 替换。
- **v0.10末–v0.11**：`weight_loader_v2` 在嵌入层还未完整迁移（`(待核实)`），仍主要走 `weight_loader`；量化嵌入 method（如 `Fp8EmbeddingMethod`）逐步补全。
- **v0.12 / main**：`method_has_implemented_embedding` 校验引入（`:286`），强制 `VocabParallelEmbedding` 直接持有的 quant method 必须实现 `embedding` 方法；`use_all_gather` 由 `current_platform.use_all_gather()` 决定，进入 `LogitsProcessor` 而非嵌入层。

[← 返回层库首页](../README.md)

## 参见

- [linear.md](linear.md)：`UnquantizedEmbeddingMethod.apply` 与 `UnquantizedLinearMethod.apply` 共享 GEMM 分派。
- [sampler-layer.md](sampler-layer.md)：`Sampler` 与 `LogitsProcessor` 消费 LM head 权重。
- [LoRA #12](../../12-lora/README.md)：四段布局与 LoRA-added 词表。
- [`./quantization/README.md`](quantization/README.md)：各 `EmbeddingMethod` 子类。
