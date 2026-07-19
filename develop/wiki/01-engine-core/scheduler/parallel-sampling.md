# 并行采样（Parallel Sampling, n>1）

[← Wiki 首页](../../README.md) > [引擎核心](../README.md) > [Scheduler](README.md) > 并行采样

源码：`vllm/v1/engine/parallel_sampling.py`（约 150 行）。`ParentRequest` 是 v1 处理 `SamplingParams.n > 1` 的核心类，逻辑上"父请求 fan-out 出 n 个子请求"，子请求各自独立调度，输出在前端按输出模式聚合。

## 是什么

`ParentRequest`（`parallel_sampling.py:13`）持有：
- `request_id` / `external_req_id` / `sampling_params`：父请求的标识与原始参数（`n` 来自 `sampling_params.n`）。
- `child_requests: set[str]`：未完成子请求的 internal id 集合。
- `output_aggregator: list[CompletionOutput | None]`：仅 `FINAL_ONLY` 模式使用，预分配 `n` 个槽位（按 `completion_output.index` 写入）。
- `max_num_generation_tokens: int`：所有子请求中最大的生成 token 数，用于 iteration stats 的 `n` 归一化。
- `cached_child_sampling_params: SamplingParams | None`：seed=None 时复用同一 `n=1` 的克隆对象，避免重复 clone。

核心方法：
- `get_child_info(index) -> (child_req_id, child_sampling_params)`（`parallel_sampling.py:83`）：child_req_id 形式 `f"{index}_{request_id}"`；`_get_child_sampling_params` 在 seed=None 时 cache，seed 给定时每个子请求 `seed + index` 唯一。
- `get_outputs(child_request_id, completion_output) -> (list[CompletionOutput], finished)`（`parallel_sampling.py:100`）：聚合核心。
- `observe_num_generation_tokens(num)` / `observe_finished_request(parent_req, iteration_stats, num)`（`parallel_sampling.py:128`）：统计聚合。

## 为什么

- **子请求独立调度**：v1 与 v0 不同，`n>1` 不是"一个请求采样 n 次"，而是 fan-out 成 n 个内部独立请求，每个 `n=1`，让调度器把它们当成 n 条普通 sequence 处理。这样 n 条序列可以抢占、迁移、prefix-cache 命中各自独立，无特殊调度路径。
- **seed 隔离**：seed 给定时若所有子请求共用同一 `SamplingParams`，rng 状态会被共享导致 n 条输出相同；故 seed 给定时每个子请求 `seed + index` 唯一克隆，确保 n 条输出独立同分布。
- **缓存复用**：seed=None 时所有子请求共用同一 `n=1` 的克隆对象（`cached_child_sampling_params`），大幅减少内存分配与参数 clone 开销；只有最后一个子请求复用 `request` 对象本身（`AsyncLLM.add_request` 中 `child_request = request if idx == n-1 else copy(request)`，`async_llm.py:392`）。
- **输出模式区分**：
  - `DELTA` / `UNFINAL`：每步把子请求的 `CompletionOutput` 直接转发给客户端（带 `index` 字段），客户端看到 n 条独立流。
  - `FINAL_ONLY`：用 `output_aggregator[index]` 累积，等所有子请求完成时一次性返回 n 条 `CompletionOutput`。
- **去重 child_requests**：子请求可能在多个 step 中触发 `finished`，`already_finished_and_returned` 标志防止重复发出（`parallel_sampling.py:108-113`）。
- **iteration stats 归一化**：单条 generate 调用即使 n=8 也应记一次 `n_params_iter`，但 `max_num_generation_tokens` 取所有子请求的最大值；`observe_finished_request` 在最后一个子请求完成时才写入 stats。

## 怎么做

### fan-out 入口（`AsyncLLM.add_request`，`async_llm.py:381`）

```python
if is_pooling or params.n == 1:
    await self._add_request(request, prompt_text, None, 0, queue)
    return queue

parent_params = params
parent_request = ParentRequest(request)
for idx in range(parent_params.n):
    request_id, child_params = parent_request.get_child_info(idx)
    child_request = request if idx == parent_params.n - 1 else copy(request)
    child_request.request_id = request_id
    child_request.sampling_params = child_params
    await self._add_request(child_request, prompt_text, parent_request, idx, queue)
return queue
```

- 最后一个子请求复用原 `EngineCoreRequest` 对象（避免 clone n-1 次）；其它子请求 `copy(request)` 浅拷贝。
- 所有子请求共享同一个 `RequestOutputCollector`，由 `RequestState.parent_req` 关联到 `ParentRequest`。
- child request_id 形式 `f"{idx}_{parent_request_id}"`——前缀索引便于 abort 时反查 parent（`OutputProcessor.abort_requests` 检查 `parent_requests` dict）。

### 聚合路径（`RequestState.make_request_output`，`output_processor.py:321`）

```python
output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)

if self.parent_req is None:
    outputs = [output]
else:
    outputs, finished = self.parent_req.get_outputs(self.request_id, output)
    if not outputs:           # FINAL_ONLY 且未完成全部 → 不发
        return None
    external_req_id = self.parent_req.external_req_id

return self._new_request_output(external_req_id, outputs, finished, kv_transfer_params)
```

### ParentRequest.get_outputs 内部

```python
def get_outputs(self, child_request_id, completion_output):
    already_finished_and_returned = False
    if completion_output.finished():
        if child_request_id in self.child_requests:
            self.child_requests.remove(child_request_id)
        else:
            already_finished_and_returned = True  # 此前已完成并发出过

    if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
        # DELTA / UNFINAL：直接转发（除非重复完成）
        outputs = [] if already_finished_and_returned else [completion_output]
    else:
        # FINAL_ONLY：按 index 入槽，全部完成时一次性发出
        self.output_aggregator[completion_output.index] = completion_output
        outputs = [] if self.child_requests else self.output_aggregator

    finished = not self.child_requests
    return outputs, finished
```

### abort 级联

`OutputProcessor.abort_requests`（`output_processor.py:450`）检查 `parent_requests`：
```python
elif parent := self.parent_requests.get(request_id):
    if parent.child_requests:
        child_reqs = list(parent.child_requests)
        child_reqs = self.abort_requests(child_reqs, internal=True)
        request_ids_to_abort.extend(child_reqs)
    self.parent_requests.pop(request_id, None)
```

abort parent 时递归 abort 所有未完成子请求，避免泄漏。

### 统计聚合（`observe_finished_request`）

```python
@staticmethod
def observe_finished_request(parent_req, iteration_stats, num_generation_tokens):
    n_param = parent_req.n if parent_req is not None else 1
    if parent_req is not None:
        num_generation_tokens = parent_req.observe_num_generation_tokens(
            num_generation_tokens
        )
    if parent_req is None or not parent_req.child_requests:
        # 单请求 或 最后一个子请求完成 → 写入 stats
        iteration_stats.max_num_generation_tokens_iter.append(num_generation_tokens)
        iteration_stats.n_params_iter.append(n_param)
```

只在所有子请求完成时记一次，n_param 取 parent.n，token 数取所有子请求的最大值。

## 与其它模块/系统配合

- **[AsyncLLM](../async-llm-frontend.md)**：`add_request` 是 fan-out 入口；`_add_request` 把 `parent_req` 透传给 OutputProcessor。
- **[OutputProcessor](../output-processor.md)**：`RequestState.parent_req` 在 `make_request_output`/`abort_requests`/`_finish_request` 中消费；`_finish_request` 在 parent 无 child 时清理 `parent_requests`。
- **[Request 状态机](../data-model.md)**：child request 的 `request_index` 字段填入 `CompletionOutput.index`，是 `output_aggregator` 的索引。
- **[IterationStats / metrics](../../16-observability/README.md)**：`n_params_iter` 与 `max_num_generation_tokens_iter` 用于吞吐量统计。
- **[scheduler.md](./scheduler.md)**：调度器对子请求完全透明，按普通 request 处理；它们可独立 prefix-cache 命中、抢占、KV connector 转移。
- **[SamplingParams](../../10-config/README.md)**：`n`、`seed`、`output_kind` 是关键参数；`RequestOutputKind.DELTA/UNFINAL/FINAL_ONLY` 决定聚合策略。

## 历史版本演进

- **v0.5/v0.6（v0）**：v0 在 `SequenceGroup` 内部维护多个 `Sequence`，n=8 共用 group 级 sampling params；调度器需感知 group-sqe 关系。
- **v0.7（v1 落地）**：`ParentRequest` 抽出，fan-out 在前端完成；调度器完全去耦合；`cached_child_sampling_params` 优化复用。
- **v0.8（v1 默认）**：`output_kind` 三态引入，`output_aggregator` 仅 FINAL_ONLY 预分配；`already_finished_and_returned` 去重保护。
- **v0.9**：abort parent 递归 abort child 修复；`observe_finished_request` 调整为只在最后一个子完成时记 stats，避免重复计数。
- **v0.10 / main**：与 streaming-input 不兼容（`_validate_streaming_input_sampling_params` 禁止 n>1 + streaming）；并行采样与 structured output 协同（每个子请求独立 grammar）。具体版本归属（待核实）。

[← 返回引擎核心首页](../README.md)

## 参见

- [output-processor.md](../output-processor.md) — `ParentRequest.get_outputs` 的消费方。
- [async-llm-frontend.md](../async-llm-frontend.md) — fan-out 入口与 child request_id 格式。
- [data-model.md](../data-model.md) — `CompletionOutput.index` 与 `external_req_id` 的关系。
