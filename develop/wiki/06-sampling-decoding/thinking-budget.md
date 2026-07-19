[← Wiki 首页](../README.md) > [采样与解码](../README.md) > ThinkingBudgetStateHolder

# ThinkingBudgetStateHolder（思考预算状态机）

> 源码：`vllm/v1/sample/thinking_budget_state.py`

---

## 是什么

`ThinkingBudgetStateHolder` 是一个 per-batch 状态机，用于在采样时强制让"思考段（thinking section）"在用户指定的 token 预算耗尽时立即结束——即使模型自己想继续输出思考内容也会被强行截断为 `reasoning_end_token_ids`。

它由 `ReasoningConfig` 开启：当 `reasoning_config is not None` 时 `maybe_create_thinking_budget_state_holder` 返回 holder 实例（`vllm/v1/sample/thinking_budget_state.py:20`），否则返回 None。Holder 持有：

- `think_start_token_ids` / `think_end_token_ids`：从 `reasoning_config.reasoning_start_token_ids` / `reasoning_end_token_ids` 取。
- `_state: dict[req_index, dict[str, Any]]`：每请求一个 state 字典，含 `in_think` / `in_end` / `check_count_down` / `think_count` / `force_index` / `spec_token_ids` / `prev_output_length` 等字段。
- `cu_num_tokens`：每请求的 cumulative token offset，用于在 spec decode 展开 logits 时定位 `force_index`。
- `_mask_capacity`：`max_num_reqs * (num_spec_tokens+1)` 或 `max_num_reqs`，限制 advance indexing 不越界。

Holder 不实现 `LogitsProcessor` 接口——它由 `Sampler.apply_logits_processors` 与 `RejectionSampler.apply_logits_processors` 显式回调。

## 为什么

- **思考模型的可预测性**：Qwen3、DeepSeek-R1 等思考模型在某些 prompt 上会无限输出思考内容，超出部署预算。`thinking_token_budget` 参数让用户显式设定"最多允许 K 个思考 token"，到期强制结束。
- **state 跟踪复杂**：思考段可能跨越多个调度步、被 spec draft token 提前预测、被 rejection sampling 部分拒绝；holder 需要在每个采样步后根据实际接受/拒绝的 token 更新 `think_count` 与 `check_count_down`。
- **位置精度**：spec decode 下，bonus token 与 spec token 在 logits 张量里是相邻位置；holder 必须知道在哪个 spec 位置触发 end-of-thinking（`force_index = [remaining_budget]` / `[0]` / `[spec_len]`），并把对应 logits 行的 `think_end_token_ids[end_count]` 列拉到 1e9。
- **多次思考循环**：模型可能 `start→…→end→…→start→…→end` 多次进入思考段；`_update_think_state` 用 `start_thinking` / `end_thinking` 双指针扫描，并在 `in_end` 完成后清零、记录 `scan_offset` 防止误触发 old start（`vllm/v1/sample/thinking_budget_state.py:268`）。
- **补丁兼容性**：holder 与 spec_token_ids 拼接逻辑深度耦合——`Sampler._combine_outputs_with_spec_tokens` 在 `predict_bonus_token` 时把 spec 历史拼到 output 上，holder 据此推进 `think_count`。

## 怎么做

### sync_batch（行 82）

scheduler 端调用，传入 `BatchUpdate`：

- `removed`：`_state.pop(index)`。
- `added`：仅当 `params.thinking_token_budget is not None` 才 `_init_state_entry`；否则 pop。`output_tok_ids` 与 `spec_token_ids=[]` 都按引用存（后续会随请求推进自动更新）。
- `moved`：按 SWAP / UNIDIRECTIONAL 分别 swap 或 pop+set。

`has_tracked_requests()` 仅当 `_state` 非空时返回 True，用于 sampler 决定是否需要触发"combine spec + output"路径。

### update_state（行 113）

`sampler.apply_logits_processors` 在 penalties 之后调用，传入最新 `output_token_ids` 与 `spec_token_ids`：

1. 当 `repeat_indices is not None`（spec decode 路径）：按 `repeat_indices.cpu().tolist()` 找每请求在展开张量中的最后一行；否则直接按 `seq_idx` 索引。
2. 把 `output_tok_ids` / `spec_token_ids` 写回 state。
3. spec_token_ids 从 output 末尾剥离（行 150–152，注意 `[:-0]` 陷阱）。
4. 调 `_update_think_state(state)` 推进 state（详见下文）。

### _update_think_state（行 250）

state 更新的核心，按"是否在 end 模式"分两大路径：

#### 非 in_end 模式（行 283–453）

1. 用 `_find_last_sequence_index_from` 在 `output_tok_ids[search_start:]` 中找最近的 `think_start_token_ids` 与 `think_end_token_ids` 出现位置。
   - `start_thinking` 找到时记录；找不到则更新 `start_search_pos`。
   - `end_thinking` 同理。
2. 根据 `start_thinking` / `end_thinking` 的相对位置决定进入/退出思考：
   - `start > end`：进入 think，`think_count = current_length - (start + start_len)`。
   - `end > start`：退出 think，`think_count = 0`。
3. 维护 `check_count_down = thinking_token_budget - think_count`（行 417）。
4. 当 `think_count + spec_len + 1 > budget` 时转型为 in_end 模式，计算 `force_index`：
   - `remaining_budget > 0 && < spec_len`：在 spec 第 `remaining_budget` 位强制 end。
   - `remaining_budget <= 0`：在 spec 第 0 位强制 end。
   - `remaining_budget >= spec_len`：在 bonus 位（即 spec_len）强制 end。

#### in_end 模式（行 452–483）

逐 spec token 检查是否匹配 `think_end_token_ids[end_count]`：

- 匹配：`end_count += 1`；若 `end_count == len(think_end_token_ids)` 完成退出，重置 state、记录 `scan_offset = len(output_tok_ids)`（防止下次 _find 误触发已经处理过的 end token）。
- 不匹配：`end_count += 1`、`force_index = [i]` 当前 spec 位——强制把下一个 `think_end_token_ids[end_count]` 推到 1e9。

### apply_to_logits（行 155）

入口 `_apply_forcing_to_logits`（行 485）：

1. 重建 `cu_num_tokens`：spec mode 下按 `len(spec_tokens)`（非 bonus）累加；非 spec mode 下每请求 +1。
2. 遍历所有 tracked requests：
   - 仅当 `in_end and not bonus_token_forced` 时处理。
   - `predict_bonus_token=True`：若 `force_index[0] < len(spec_token_ids)` 跳过（该位置属于 spec token，bonus 阶段不应处理）；否则重置 `force_index=[0]`。
   - 对每个 `force_idx`，计算 `mask_idx = cu_num_tokens[seq_idx] + force_idx`，在 `logits[mask_idx, think_end_token_ids[end_count]]` 写 1e9。
3. ROCm 路径用 flattened `index_fill_` 避免 2-D advanced-indexing 写 fault（行 560–571）；NVIDIA 路径用 `index_put_`。
4. `bonus_token_forced` 标志防止同一 step 内 bonus 与 spec 路径重复触发。

### 实例化与生命周期

`maybe_create_thinking_budget_state_holder(reasoning_config, max_num_seqs, num_spec_tokens, device)` 在 `GPUModelRunner` 启动时被调用，存入 `sampling_metadata.thinking_budget_state_holder`（每步 metadata 构造时填入）。scheduler 端通过 `BatchUpdateBuilder` 同步 add/remove/move；sampler 端通过 `update_state` + `apply_to_logits` 触发实际行为。

## 与其它模块/系统配合

- [sampler.md](sampler.md)：`Sampler.apply_logits_processors` 行 409–419：先 `update_state`（用 `output_token_ids` + `spec_token_ids`），再 `apply_to_logits`。`predict_bonus_token=True` 路径下 `_combine_outputs_with_spec_tokens` 把 spec 历史拼到 output。
- [rejection-sampler.md](rejection-sampler.md)：`RejectionSampler.apply_logits_processors` 行 340–345：spec decode 下第二次 `apply_to_logits`，针对 target_logits 而非 bonus_logits；`force_index` 调整遵循 spec_token_ids 长度。
- [引擎核心-调度](../01-engine-core/scheduler/README.md)：scheduler 在调度请求时维护 `params.thinking_token_budget` 字段；当请求被 preempt/resume 时 metadata 通过 `BatchUpdate.moved` 传递。
- [结构化输出](structured-output/README.md)：thinking budget 与 reasoning parser 联动——结构化输出 manager 在 `should_fill_bitmask` 中调用 `reasoner.is_reasoning_end` 决定是否停止 mask；thinking budget 在 sampling 端强制结束思考，二者协作避免"思考超预算且 grammar 卡死"的尴尬状态（待核实：thinking_budget + structured_output 的精确交互时序）。
- [配置体系-ReasoningConfig](../10-config/README.md)：`reasoning_config.reasoning_start_token_ids` / `reasoning_end_token_ids` / `enable_thinking` 等字段决定 holder 行为；`thinking_token_budget` 是 per-request 参数（在 `SamplingParams` 中）。

## 历史版本演进

- **早期（V0）**：V0 无思考预算；思考模型在 V0 中依赖 stop strings 或 max_tokens 截断，无法精确控制"思考 token 数"。
- **v0.10.0**：思考模型（DeepSeek-R1、Qwen3）批量进入主线，但 hold 与采样解耦仍不完整；早期实现以 V0 风格的 `LogitsProcessor` 包装为主。
- **v0.10.5**：`ThinkingBudgetStateHolder` 在 V1 中 landfall，按"非 LogitsProcessor 但语义近似"插入到 `Sampler.apply_logits_processors` 末尾；`predict_bonus_token` 路径同步加。
- **v0.11.0**：spec decode 路径接入——`_combine_outputs_with_spec_tokens`、`force_index` 按 spec_token_ids 长度精确计算；`scan_offset` 字段加入，修复"end token 被 rejection 拒绝后误重新触发 start"的 bug。
- **v0.12 / main**：ROCm 1-D `index_fill_` 路径加入（规避 2-D advanced-indexing fault）；`_mask_capacity` 显式化为 `max_num_reqs * (num_spec_tokens+1)`；并行 drafting（DFlash/DSpark）下 `cu_num_tokens` 计算需考虑 `parallel_drafting_token_id`（待核实：并行 drafting + thinking_budget 是否已完全支持）。

[← 返回采样与解码](../README.md)

## 参见

- [sampler.md](sampler.md)
- [rejection-sampler.md](rejection-sampler.md)
- [logits-processor.md](logits-processor.md)
- [结构化输出-manager](structured-output/manager.md)：reasoning parser 与 thinking budget 的协同
- [配置体系-ReasoningConfig](../10-config/README.md)（待补充）
