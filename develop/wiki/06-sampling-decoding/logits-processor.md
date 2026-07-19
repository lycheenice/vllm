[← Wiki 首页](../README.md) > [采样与解码](../README.md) > LogitsProcessor

# LogitsProcessor（Logits 处理器框架）

> 源码：`vllm/v1/sample/logits_processor/`

---

## 是什么

`LogitsProcessor` 框架是 V1 采样层专用的"批级 logits 修改器"——与 V0 中 per-request 的 `vllm.logits_process.LogitsProcessor` 不同，V1 processor 在每次 `Sampler.forward` 时被一次性 apply 到整批 logits 上，且必须实现：

- `apply(logits: torch.Tensor) -> torch.Tensor`：原地或非原地修改 `[num_tokens, vocab]`。
- `is_argmax_invariant() -> bool`：是否不影响 greedy 采样的 argmax 结果。
- `update_state(batch_update: BatchUpdate | None) -> None`：当持久 batch 改变时同步状态。
- `validate_params(sampling_params)`（classmethod）：请求校验阶段调用。

容器 `LogitsProcessors`（`vllm/v1/sample/logits_processor/state.py:148`）把所有 processor 分到 `argmax_invariant` / `non_argmax_invariant` 两个列表，`Sampler` 按序调用。

### 子目录文件

| 文件 | 内容 |
|---|---|
| `interface.py` | `LogitsProcessor` ABC、`BatchUpdate` dataclass、`MoveDirectionality` 枚举、`AddedRequest`/`RemovedRequest`/`MovedRequest` 类型别名 |
| `state.py` | `BatchUpdateBuilder`（scheduler 端用）+ `LogitsProcessors` 容器 |
| `builtin.py` | `MinPLogitsProcessor`、`LogitBiasLogitsProcessor`、`MinTokensLogitsProcessor` 三个内置 processor + `process_dict_updates` 工具 |
| `__init__.py` | `AdapterLogitsProcessor`（包装 V0 per-request processor）、`build_logitsprocs` 工厂、FQCN 插件加载 |

## 为什么

- **从 per-request 切到 per-batch**：V0 时代每请求一个 `LogitsProcessor` 实例，每个 step 都需要 Python 循环。"V1 模式"把所有请求的处理器状态合并到批级张量（如 `LogitBiasLogitsProcessor.biases` 是 `dict[req_index, dict[token_id, bias]]`），在 `apply` 里用一次 `logits[reqs, toks] += bias_tensor` 完成。
- **argmax 不变量分类**：这是 V1 的关键设计——
  - **非 argmax 不变量**（min_tokens 把 stop token 设 -inf、logit_bias 改写偏置）必须在 greedy 决定**之前**跑。
  - **argmax 不变量**（min_p：把概率低于 `max_prob * min_p` 的 token 设 -inf，不影响 argmax）放在 temperature 之后、top-k/top-p 之前。
  
  这条二分让 greedy 请求可以完全跳过 min_p（节省一次 softmax）而仍受 min_tokens/logit_bias 约束。
- **batch 增量更新**：scheduler 每步通过 `BatchUpdateBuilder` 收集 added/removed/moved 集合，构造 `BatchUpdate`。processor 的 `update_state` 据此更新自身的 per-request dict；移除时必须 `pop`，移动时按 `MoveDirectionality.SWAP` 或 `UNIDIRECTIONAL` 分别处理。
- **可插拔**：通过 entry point `vllm.logits_processors` 或 `logits_processors=[...]` 配置项可加载自定义 processor；FQCN 语法 `<module>:<ClassName>` 走 `importlib` 反射。

## 怎么做

### 内置 processor 详解

#### MinPLogitsProcessor（`builtin.py:23`，argmax-invariant）

- `is_argmax_invariant() -> True`。
- `update_state`：维护 `min_p_cpu`（pinned numpy）、`min_p_device`（GPU 张量）、`min_p_count`（非零请求数）；batch 改变时只重新 `copy_` 与 `unsqueeze`，避免每步都同步。
- `apply`：`probability_values = softmax(logits)`；`max_probabilities = amax(...)`；`invalid = probabilities < max_probabilities * self.min_p`；`masked_fill_(-inf)`。当 `min_p_count == 0` 时直接 return 原 logits。

#### LogitBiasLogitsProcessor（`builtin.py:119`，non-argmax-invariant）

- `is_argmax_invariant() -> False`（logit bias 改变 argmax 结果）。
- `update_state`：把 `params.logit_bias` dict 取出转成扁平的 `(reqs, tok_ids, biases)` 三个等长列表，存成两个 int32 索引 tensor + 一个 float32 偏置 tensor。
- `apply`：`logits[self.logits_slice] += self.bias_tensor`，单次 advanced indexing 加法完成全部请求的 bias 应用。

#### MinTokensLogitsProcessor（`builtin.py:165`，non-argmax-invariant）

- 维护 `min_toks: dict[req_idx, (min_tokens, output_token_ids, stop_token_ids_set)]`；当请求达到 `min_tokens` 长度自动移除。
- 默认 `apply`（非 spec decode）：用 `index_put_(self.logits_slice, neg_inf)` 把所有 stop token 在该 request 行设 -inf。`logits_slice` 是 `(reqs, toks)` 二维索引。
- `apply_with_spec_decode(logits, num_draft_tokens)`（行 235）：spec decode 专用。按 `cumsum(num_draft_tokens)` 找到每请求的 draft 位置区间，对前 `min(remaining, num_draft_tokens[req_idx])` 个位置（即"还没达到 min_tokens 的位置"）屏蔽所有 stop tokens；用 numpy 拼出 `(rows, toks)` 索引再 `index_put_`。这是 spec decode + min_tokens 能正确触发的关键。

### build_logitsprocs 工厂（`__init__.py:184`）

```python
def build_logitsprocs(vllm_config, device, is_pin_memory, is_pooling_model,
                      custom_logitsprocs=()) -> LogitsProcessors:
    ...
```

- Pooling model 不支持任何 processor（直接返回空容器）。
- spec decode 开启时仅注册 `MinTokensLogitsProcessor`，并 warning `min_p/logit_bias 在 spec decode 中不生效`（`STR_SPEC_DEC_REJECTS_LOGITSPROCS`）。
- 普通 generation：`BUILTIN_LOGITS_PROCESSORS`（三个内置）+ `_load_custom_logitsprocs`（entry point + FQCN）。TPU 不支持 custom processor。

### AdapterLogitsProcessor（`__init__.py:234`）

为了让 V0 风格的 per-request `RequestLogitsProcessor`（接口 `__call__(row_logits, output_token_ids[, prompt_token_ids])`）能在 V1 中复用，`AdapterLogitsProcessor` 提供基底类：子类实现 `new_req_logits_processor(params) -> RequestLogitsProcessor | None`，框架在 `apply` 时按 `req_idx` for-loop 调用每个请求的 processor（行 331）。这条路径在性能上不如原生 V1 processor，但兼容性最好（如 `LMGuidanceLogitsProcessor`、`ReasoningEndLogitsProcessor` 等旧实现）。

### BatchUpdateBuilder（`state.py:18`）

scheduler 端构建 `BatchUpdate`：

- `removed_append(index)`：在 step 开始时累积被移除请求索引；调用 `removed`/`pop_removed`/`peek_removed` 后不可再 append（会 raise）。
- `added` / `moved`：分别是 `[(index, params, prompt_ids, output_ids)]` 与 `[(i1, i2, direction)]` 列表。
- `get_and_reset(batch_size)`：返回一个 frozen `BatchUpdate` 并清空内部状态；无变化时返回 None 短路。

## 与其它模块/系统配合

- [sampler.md](sampler.md)：`SamplingMetadata.logitsprocs` 字段类型是 `LogitsProcessors`，sampler 在两个位置分别 apply non-argmax-invariant（`forward` 中）与 argmax-invariant（`sample` 中）。
- [rejection-sampler.md](rejection-sampler.md)：spec decode 路径下 `MinTokensLogitsProcessor.apply_with_spec_decode` 被显式调用；其他 processor 在 spec decode 下被禁用。
- [thinking-budget.md](thinking-budget.md)：`ThinkingBudgetStateHolder` 不实现 `LogitsProcessor` 接口，但语义类似——同样在 `apply_logits_processors` 内被调用，受 `predict_bonus_token` 标志影响。
- [引擎核心-调度](../01-engine-core/scheduler/README.md)：scheduler 端 `BatchUpdateBuilder` 实例存在于多个 manager（gpu_input_batch、logitsprocs、structure output）中；scheduler 每步 `finish_request`/`add_request` 时调用对应 builder 方法。
- [结构化输出](structured-output/README.md)：结构化输出走 bitmask 而非 `LogitsProcessor`（避免 FSM 状态在 batch 维度的复杂更新），但 `AdapterLogitsProcessor` 仍可用于把 V0 的 per-request grammar processor 包装过来（待核实：vLLM 当前是否仍走这条路径，还是已完全切换到 bitmask）。
- [tokenizers](../14-tokenizers-transformers/README.md)：`LogitBiasLogitsProcessor` 的 token_id 来自 `params.logit_bias`，由前端 from tokenizer vocab 索引而来。

## 历史版本演进

- **v0.5–v0.6（V0）**：`vllm/logits_process.py` 提供 `LogitsProcessor` ABC + 一堆具体实现（`RepetitionPenaltyLogitsProcessor`、`FrequencyPenaltyLogitsProcessor`、`TemperatureLogitsProcessor` 等），per-request 调用、每次 forward 都重新创建。
- **v0.7.0**：V1 引入 `LogitsProcessor` ABC（新接口）、`BatchUpdate`、`BatchUpdateBuilder`、`LogitsProcessors` 容器；内置只剩 `MinTokens`、`LogitBias`、`MinP` 三个——penalties/temperature/top-k/top-p 不再是 processor，而是直接由 sampler / ops 内联调用。
- **v0.7.5**：argmax-invariant 二分落地，`Sampler.apply_logits_processors` 与 `Sampler.sample` 分担 processor 调用。
- **v0.8.0**：`AdapterLogitsProcessor` 引入，让 V0 per-request processor 在 V1 中存活（迁移期兼容）。
- **v0.8.5**：FQCN 加载日志完善；TPU 不支持 custom processor 的限制写入。
- **v0.9.0**：spec decode + LogitsProcessor 禁用规则确立（除 MinTokens 外全部禁用），`STR_SPEC_DEC_REJECTS_LOGITSPROCS` 引入。
- **v0.10.5**：`MinTokensLogitsProcessor.apply_with_spec_decode` 加入，spec decode + min_tokens 的接受率退化 bug 修复。
- **v0.11.0**：`process_dict_updates` 工具 helper 统一三个内置 processor 的 batch update 逻辑，减少代码重复。
- **v0.12 / main**：插件 entry point `vllm.logits_processors` 文档化；`_load_logitsprocs_by_fqcns` 支持 `module:type.attr.attr` 多级 qualname。

[← 返回采样与解码](../README.md)

## 参见

- [sampler.md](sampler.md)
- [sampling-ops.md](sampling-ops.md)
- [rejection-sampler.md](rejection-sampler.md)
- [thinking-budget.md](thinking-budget.md)
