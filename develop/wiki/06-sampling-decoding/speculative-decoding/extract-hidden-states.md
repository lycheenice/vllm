[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > ExtractHiddenStatesProposer

# ExtractHiddenStatesProposer（Hidden States 缓存器）

> 源码：`vllm/v1/spec_decode/extract_hidden_states.py`

---

## 是什么

`ExtractHiddenStatesProposer` 是一个**不真正进行 spec decoding**的特殊 drafter：它把 target 模型的若干 aux hidden states 通过"cache-only attention layer"写入 KV cache，但返回的"draft token"就是 target 真正采样的 token 本身（`sampled_token_ids[:, :1]`）。这样 rejection sampler 总是"接受"它，没有真正的预测行为——其唯一用途是为 **KV transfer**（如 KV connector / KV offload）让 target hidden states 被预先缓存到特定 layer 的 KV cache 中。

类签名（`vllm/v1/spec_decode/extract_hidden_states.py:29`）：

```python
class ExtractHiddenStatesProposer:
    def __init__(self, vllm_config, device): ...
    def propose(self, num_speculative_tokens, sampled_token_ids,
                target_hidden_states, common_attn_metadata, slot_mappings=None) -> torch.Tensor:
        ...
        return sampled_token_ids[:, :1]
```

## 为什么

- **KV transfer 场景**：某些 KV connector（如远程 KV 复制、跨进程 KV 共享）需要把 target 模型多层 hidden states 缓存进特定 layer 的 KV cache，便于其他 rank 进程读取。本 proposer 让此过程复用 spec decode infra（已有的 spec metadata、sampler、rejection sampler 接口、spec_token_ids 拼接），无需另起独立路径。
- **`num_speculative_tokens == 1`**：本方法强制 K=1（行 36 assert），因为不是真正"预测 1+ 个 token"，而是借 spec decode 接口走一次"伪 draft + 验证"流程。
- **不消耗 hidden state 作 logits**：与其他 drafter 不同，本 proposer 不调用 `compute_logits` 也不采样——直接把 target 真正采样的 token 当 draft。`target_hidden_states` 只是写入 KV cache 的载体。
- **cache-only attention layer**：drafter "model" 是一个特殊的 `ExtractHiddenStatesModel`，其 attention 层不做任何实际计算，只把 `hidden_states` 写到对应 slot 的 KV cache。`attn_layer_names` 列表仅含一个这种特殊层。
- **`eagle_aux_hidden_state_layer_ids`**：hf_config 字段指定 target 哪些层的 hidden states 被收集（行 70）；`num_hidden_states = len(layer_ids)` 决定 buffer 第三维大小。

## 怎么做

### __init__ 准备

- 读 `eagle_aux_hidden_state_layer_ids`；`num_hidden_states = len(layer_ids)`；`hidden_size = vllm_config.model_config.get_hidden_size()`。
- 分配 `hidden_states: [max_num_tokens, num_hidden_states, hidden_size]` 三维 buffer（与 EAGLE 3 一致）。
- `disable_padded_drafter_batch` 显式禁止（行 37）——padded 模式下本方法不支持。
- 持有 `cudagraph_dispatcher` + `_slot_mapping_buffer` 与其他 drafter 同。

### propose（行 93）

```python
def propose(self, num_speculative_tokens, sampled_token_ids,
            target_hidden_states, common_attn_metadata, slot_mappings=None):
    # target_hidden_states: list of tensors (one per layer), shape [num_tokens, hidden]
    stacked_hidden_states = torch.stack(target_hidden_states, dim=1)  # [T, L, H]
    num_tokens = stacked_hidden_states.shape[0]
    self.hidden_states[:num_tokens] = stacked_hidden_states  # copy
    
    # build attn metadata
    attn_metadata = self.attn_metadata_builder.build_for_drafting(
        common_attn_metadata=common_attn_metadata, draft_index=0)
    per_layer_attn_metadata = {name: attn_metadata for name in self.attn_layer_names}
    
    # determine padding
    cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = \
        self._determine_batch_execution_and_padding(num_tokens)
    
    if self.eplb_state is not None:
        self.eplb_state.prepare_forward(self.draft_model_config, num_tokens)
    
    with set_forward_context(per_layer_attn_metadata, ...):
        self.model(hidden_states=self.hidden_states[:num_input_tokens])
    
    # 返回 sampled 作为 draft，shape [B, 1]
    return sampled_token_ids[:, :1]
```

drafter "model" forward 不产生 logits，只把 `hidden_states` 通过 cache-only attention layer 写入 KV cache 中。

### load_model（行 357）

- 用 `set_model_tag("extract_hidden_states")` 隔离 compile。
- 加载 draft model config 后，找出 drafter 独有的 attn 层（与 target 的差集），assert 长度 == 1。
- 通过 `_build_attn_metadata_builder` 用该层 backend 的 builder 类创建 metadata builder。

### validate_same_kv_cache_group（行 401）

只支持单 KV cache group：找出唯一 cache-only layer 所属 group，记录 `self.kv_cache_gid`。

### prepare_next_token_ids_padded（行 319）

由于 K=1，本方法与 EAGLE 的版本不同：

- `sampled_token_ids` shape `(batch_size, 1)`。
- 简单地用 `torch.where(use_sampled, sampled, backup_tokens_gpu)` 选择每请求的 next token。
- `valid_sampled_tokens_count = is_valid.to(int32)`。

### dummy_run（行 264）

`@torch.inference_mode()` 装饰；构造 zero hidden_states 跑一次 `self.model(hidden_states=...)`，用于 memory profiling / cudagraph 捕获。

## 与其它模块/系统配合

- [llm-base-proposer.md](llm-base-proposer.md)：接口形似但**不继承** `SpecDecodeBaseProposer`——独立实现，因为它不真正 propose spec token。
- [../rejection-sampler.md](../rejection-sampler.md)：draft = real sampled，rejection 100% 接受；output 长度 K+1 中只 sampled 那 1 个 token 有效，附 0 个 bonus。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：`speculative_config.method == "extract_hidden_states"` 触发实例化；`use_aux_hidden_state_outputs = True`（行 637）。
- [KV cache offload 子系统](../../15-kv-cache-offload/README.md)（待补充）：本 drafter 是为 KV transfer 设计，与 offload / connector 配合。
- [编译与 IR](../../09-compilation-ir/README.md)：`set_model_tag("extract_hidden_states")` 让 compile backend 隔离；`initialize_cudagraph_keys` 仅 PIECEWISE 模式。

## 历史版本演进

- **v0.12 / main**：`ExtractHiddenStatesProposer` 引入；与 KV connector / KV transfer 体系同期 landfall。
- 待核实：是否在 v0.11 已有早期版本；具体 KV connector 触发 `extract_hidden_states` method 的实例化路径。

[← 返回投机解码](../README.md)

## 参见

- [eagle.md](eagle.md)：aux hidden state 收集流程源头
- [llm-base-proposer.md](llm-base-proposer.md)：接口参考
- [KV cache offload](../../15-kv-cache-offload/README.md)
