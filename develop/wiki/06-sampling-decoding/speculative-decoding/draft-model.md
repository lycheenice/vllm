[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > DraftModelProposer

# DraftModelProposer（独立小模型 drafter）

> 源码：`vllm/v1/spec_decode/draft_model.py`

---

## 是什么

`DraftModelProposer` 是用"一个完全独立的、参数量比 target 小很多的 LM"作 drafter 的 spec decode 实现——它**不**接收 target 的 hidden states（`pass_hidden_states_to_model=False`），而以自己采样的 token 作为 input_ids 走自回归 K 步。drafter 与 target 仅共享 vocab（同 tokenizer，或通过 `VocabMapping` 处理异构 vocab 场景）。

类签名（`vllm/v1/spec_decode/draft_model.py:19`）：

```python
class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config=vllm_config, device=device,
                         pass_hidden_states_to_model=False, runner=runner)
        self._raise_if_draft_tp_mismatch()
        self.use_heterogeneous_vocab = self.speculative_config.use_heterogeneous_vocab
        if self.use_heterogeneous_vocab:
            self.vocab_mapping = VocabMapping(...)
        else:
            self._raise_if_vocab_size_mismatch()
            self.vocab_mapping = None
```

## 为什么

- **经典 SD 设定**：Leviathan 2022 原论文的设定就是"小 model + 大 model"。draft_model 是最直接的实现，无需 target 模型架构支持 hidden state 接口。
- **TP size 必须一致**：注释行 65–78 说明——target TP>1 + draft TP=1 时所有 rank 都编译 drafter，破坏 torch compile cache。当前强制 `draft_tensor_parallel_size == tensor_parallel_size`。
- **vocab 等长 OR 异构**：默认要求 `target_vocab_size == draft_vocab_size`；开启 `use_heterogeneous_vocab=True` 时跳过断言并建立 `VocabMapping`——drafter logits 被约束到交集、token id 双向映射。
- **独立 EPLB / quant_config**：`_create_draft_vllm_config` 覆盖让 drafter 用自己的 `parallel_config`（draft_parallel_config）与 `quant_config`（清空，让 draft 走默认或 get_draft_quant_config）。
- **不共享 embed/lm_head**：`_maybe_share_embeddings` 与 `_maybe_share_lm_head` 覆盖为 no-op——与 EAGLE 不同，draft_model 完全独立权重。

## 怎么做

### 关键覆盖

- `_create_draft_vllm_config`（行 81）：清空 `quant_config`、用 `draft_parallel_config` 替换 `parallel_config`、保留 `draft_model_config`。`rank` 字段从主 vllm_config 继承以避免多 rank 间冲突。
- `_get_model`（行 95）：与基类相同，但用 `set_model_tag("draft_model")` 隔离编译 cache。
- `_maybe_share_embeddings` / `_maybe_share_lm_head`：no-op。
- `_raise_if_draft_tp_mismatch`（行 63）：构造时 assert。
- `_raise_if_vocab_size_mismatch`（行 60）：调 `speculative_config.verify_equal_vocab_size_if_draft_model`。

### propose 流程

走基类 `SpecDecodeBaseProposer.propose`，但因为 `pass_hidden_states_to_model=False`：

1. `set_inputs_first_pass` 中走 `needs_extra_input_slots=True` 分支——drafter 没有 hidden state 输入，但 position/slot/input_ids 仍需正确填充。
2. `build_model_inputs_first_pass` 不传 `hidden_states` kwarg——drafter 用自己的 embedding 层。
3. multi-pass loop 中 `self.hidden_states[:batch_size] = hidden_states` 这行实际不影响 drafter forward（drafter 不读 hidden_states 输入）；只是 buffer 维护。
4. `_sample_draft_tokens` 走默认 `compute_logits + argmax` 流程；`use_heterogeneous_vocab` 时在算 logits 后 constrain + 映射。

###drafter 模型加载（继承基类）

`SpecDecodeBaseProposer.load_model`（行 1321）被调用——drafter attn 层在 target 之外的部分被识别为 `_draft_attn_layer_names`，每个层都被分配 `AttentionGroup` 与 metadata builder。但与 EAGLE 不同：drafter 不与 target 共享 KV cache group，独立 KV cache。

###drafter KV cache 管理

drafter有自己的 KV cache 空间，与 target 分开。`initialize_attn_backend` 把 drafter attn 层归入独立 `AttentionGroup`；每步 drafter forward 维护自己的 KV cache（其 BlockManager 在 ModelRunner 的 `kv_cache_config` 中按 drafter layer 单独配置）。

### VocabMapping 集成（use_heterogeneous_vocab=True）

```python
target_tokenizer = get_tokenizer(spec.target_model_config.tokenizer, ...)
draft_tokenizer = get_tokenizer(spec.draft_model_config.model, ...)
self.vocab_mapping = VocabMapping(
    target_tokenizer=target_tokenizer, draft_tokenizer=draft_tokenizer,
    target_vocab_size=spec.target_model_config.get_vocab_size(),
    draft_vocab_size=spec.draft_model_config.get_vocab_size(),
    device=device)
```

详见 [vocab-mapping.md](vocab-mapping.md)。

## 与其它模块/系统配合

- [llm-base-proposer.md](llm-base-proposer.md)：复用基类 propose 流程；不需要 hidden state 路径。
- [vocab-mapping.md](vocab-mapping.md)：异构 vocab 时必备。
- [../rejection-sampler.md](../rejection-sampler.md)：drafter forward 完成后产出 draft_token_ids（target vocab 空间）+ draft_probs（可选）；走标准 rejection sampling。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：`speculative_config.uses_draft_model()` 触发实例化（`gpu_model_runner.py:591`）。
- [分布式-TP](../../07-distributed/README.md)：强制 draft_tp == target_tp；EPLB 在 drafter 不与 MoE 共启。
- [配置体系-SpeculativeConfig](../../10-config/README.md)（待补充）：`draft_model_config`、`draft_parallel_config`、`use_heterogeneous_vocab`。

## 历史版本演进

- **v0.5–v0.6（V0）**：V0 时代已有 `DraftModel` 概念，走 `MultiStepWorker`。
- **v0.7.0**：V1 `DraftModelProposer` landfall；继承 `SpecDecodeBaseProposer`；强制 TP 一致。
- **v0.10.0**：`use_heterogeneous_vocab` + VocabMapping 接入；支持 target/draft vocab 不同。
- **v0.10.5**：`disable_padded_drafter_batch` 选项加入；draft_model 不再支持 `disable_padded_drafter_batch=True`（基类 `_raise_if_padded_drafter_batch_disabled`）。
- **v0.11.0**：与 EPLB / MOE drafter 兼容性完善；`get_draft_quant_config` 让 drafter 走独立量化。
- **v0.12 / main**：与 multimodal 路径不完全兼容——`_warn_if_multimodal` warning；待后续 multimodal drafter 完善。

[← 返回投机解码](../README.md)

## 参见

- [llm-base-proposer.md](llm-base-proposer.md)
- [vocab-mapping.md](vocab-mapping.md)
- [eagle.md](eagle.md)：vLLM 推荐的更高效替代
