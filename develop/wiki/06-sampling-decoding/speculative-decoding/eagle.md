[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > EagleProposer

# EagleProposer（EAGLE / EAGLE3）

> 源码：`vllm/v1/spec_decode/eagle.py`

---

## 是什么

`EagleProposer` 是 vLLM V1 中 EAGLE 与 EAGLE3 算法的 drafter 实现。它继承 `SpecDecodeBaseProposer`，构造时固定 `pass_hidden_states_to_model=True`——drafter 模型不接收 raw `input_ids`，而是吃 target 末层（或 aux 层）hidden states 作为 hidden 输入，再用自己的 lm_head 算 logits。

类签名极其简短（`vllm/v1/spec_decode/eagle.py:10`）：

```python
class EagleProposer(SpecDecodeBaseProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device,
                         pass_hidden_states_to_model=True, runner=runner)
```

因为所有重活都被基类 `SpecDecodeBaseProposer.propose` 承担，子类只需声明"我要吃 hidden states"。EAGLE3 的差异在 `model.combine_hidden_states`（target 多个 aux hidden states 合并）与 `eagle3_use_aux_hidden_state` 标志，由 `_get_eagle3_use_aux_hidden_state_from_config` 在基类构造中读取。

## 为什么

- **理论依据**：EAGLE（[Li et al. 2024](https://arxiv.org/abs/2401.15077），待核实链接）骨干是"用 target 末层 hidden states 作为 draft decoder 输入"——draft 模型只学 1 层 transformer + 一个lm_head，参数极少，但能复用 target 的语义表征。
- **EAGLE3 增量**：使用 target 的多个中间层 (aux hidden states) 而非只末层；`eagle_aux_hidden_state_layer_ids` 在 hf_config 中指定。当 `eagle3_use_aux_hidden_state=True` 时 ModelRunner 会在 target forward 中用 `use_aux_hidden_state_outputs` 标志收集这些中间 hidden。
- **KV 共享**：EAGLE draft decoder 通常与 target 共享 KV cache（同 vocab、同 layers config 时）。基类 `initialize_attn_backend` 会把所有 draft 层归到同一 KV cache group；当 target 与 draft layer 名重合时通过 `kv_sharing_target_layer_name` 直接复用（见 `vllm/model_executor/models/utils.py`，待核实文件位置）。
- **本地 argmax 缩减**：`use_local_argmax_reduction` 字段开启时调 `model.get_top_tokens(hidden_states)`——EAGLE3 多 LM head 分支情况下，这条路径避免把多分支 logits 拼成全 vocab 再 argmax，直接在分支内取 top-1。
- **支持 multimodal**：基类 `_warn_if_multimodal` 在 EAGLE 路径下警告"EAGLE + multimodal 不完全支持"，但允许 text-only 继续。

## 怎么做

EAGLE 的实际执行完全走基类 `propose` 流程，参见 [llm-base-proposer.md](llm-base-proposer.md)。具体细节：

###drafter 模型加载

`SpecDecodeBaseProposer.load_model`（行 1321）：

1. 调 `_get_model()` 加载 drafter 权重。
2. 找出 `target_attn_layer_names` 与 drafter 全 attn 层的差集，作为 `_draft_attn_layer_names`。
3. 若 target 是多模态模型，根据模型名映射 `image_token_index`（行 1359–1387）。
4. `_maybe_share_embeddings` 与 `_maybe_share_lm_head`：EAGLE 通常会共享（当 draft 模型无自身 embed/lm_head 时，直接 alias target 的）；子类可覆盖。

###EAGLE3 combine_hidden_states

`propose` 行 526：

```python
if self.method in ("eagle3", "dflash"):
    target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
    assert target_hidden_states.shape[-1] == self.hidden_size
```

`combine_hidden_states` 是 drafter 模型的方法——把 target 的多个 aux hidden states（如 layer 4、16、28 的输出）按模型定义的权重/MLP 合并成单条 (T, hidden)。

###aux hidden state 路径

`GPUModelRunner` 在 spec decode + EAGLE3 场景下：

- `use_aux_hidden_state_outputs = self.drafter.eagle3_use_aux_hidden_state`（`gpu_model_runner.py:626`）。
- target forward 时收集 `aux_hidden_state_layer_ids` 指定层的 hidden states，作为 `target_hidden_states` 列表传入 proposer。
- DFlash 也复用此路径（`EagleProposer` 的 `_get_eagle3_use_aux_hidden_state_from_config` 在 DFlash 中读 `dflash_config.use_aux_hidden_state`，默认 True）。

### use_local_argmax_reduction

`_greedy_sample`（基类行 428）：

```python
if self.use_local_argmax_reduction:
    return self.model.get_top_tokens(hidden_states)
```

`get_top_tokens` 由 drafter 模型实现，常见实现是逐 head 求 argmax 再合并（EAGLE3 多分支 lm_head 场景比"merge 后全 vocab argmax"省显存）。

### probabilistic draft probs

EAGLE 默认走 `_greedy_sample`（仅 argmax，draft_probs=None）。当 `rejection_sample_method="standard"` 且 `draft_sample_method="probabilistic"` 时走 `_sample_from_logits` 算出 (token_ids, probs) tuple，probs 被 rejection kernel 用于概率比较（更精确但更耗）。当前默认仍为 greedy（注释 `llm_base_proposer.py:1813` 说明 draft_probs management pending）。

## 与其它模块/系统配合

- [llm-base-proposer.md](llm-base-proposer.md)：EAGLE 完全复用基类流程。
- [../rejection-sampler.md](../rejection-sampler.md)：EAGLE 默认 draft_probs=None，rejection kernel 走 `NO_DRAFT_PROBS` 分支（仅 ngram-style 比较）；probabilistic 模式才走标准拒绝采样。
- [mtp.md](mtp.md)：DeepSeek MTP 与 EAGLE 在结构上类似，但 MTP 有 `model_returns_tuple=True` 的特殊分支。
- [模型库-Eagle3 架构](../../04-model-zoo/architecture-families/README.md)：`Eagle3LlamaForCausalLM` / `Eagle3DeepseekV2ForCausalLM` / `Eagle3Qwen3ForCausalLM` 等 drafter 模型实现 `combine_hidden_states` 与 `get_top_tokens`。
- [编译与 IR](../../09-compilation-ir/README.md)：EAGLE drafter 默认走 PIECEWISE cudagraph；`CudagraphDispatcher.initialize_cudagraph_keys` 在 `initialize_cudagraph_keys` 中调用。
- [LoRA](../../12-lora/README.md)：EAGLE + LoRA 的支持矩阵——通常 target lm_head 被 LoRA 替换时，drafter `compute_logits` 也需要兼容（待核实）。

## 历史版本演进

- **v0.7.0**：V1 首批 EAGLE 落地，仅支持单 hidden state 输入；`EagleProposer` 类代码 < 20 行（与今天一致）。
- **v0.8.0**：padded drafter batch 加入；EAGLE 与 cuda graph 完全兼容。
- **v0.8.5**：EAGLE3 引入——`use_aux_hidden_state_outputs` 字段、`combine_hidden_states`、`get_top_tokens` 接口落地；`eagle3_use_aux_hidden_state` 从 hf_config `eagle_aux_hidden_state_layer_ids` 推断。
- **v0.9.0**：`use_local_argmax_reduction` 字段加入（默认从 speculative_config 读取）；probabilistic draft sampling 路径加入但默认关闭。
- **v0.10.0**：`use_heterogeneous_vocab` + VocabMapping 接入；EAGLE + 异构 vocab 走 TLI 算法（target→draft / draft→target 双向映射）。
- **v0.10.5**：MTP（DeepSeek3 MTP）的非 EAGLE 风格 draft model 路径与之并行；`model_returns_tuple` 区分。
- **v0.11.0**：EPLB 支持；EAGLE 与 Drafter MOE 模型的 EPLB limit（禁止组合）写入。
- **v0.12 / main**：`allowed_attn_types` 在 ROCm 上扩展（含 MLA / sparse / MiniMax M3 等稀疏注意力）以支持 EAGLE 在更多模型族上的多步骤 drafting；`BreakableCUDAGraphWrapper.unwrap` 兼容 breakable graph。

[← 返回投机解码](../README.md)

## 参见

- [llm-base-proposer.md](llm-base-proposer.md)
- [step3p5.md](step3p5.md)
- [gemma4.md](gemma4.md)
- [mtp.md](mtp.md)
- [vocab-mapping.md](vocab-mapping.md)
