[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [结构化输出](../README.md) > StructuredOutputManager

# StructuredOutputManager（结构化输出管理器）

> 源码：`vllm/v1/structured_output/__init__.py`

---

## 是什么

`StructuredOutputManager` 是 engine-level 的结构化输出协调者——它不属于任何 worker，而是挂在 `EngineCore` / `Scheduler` 进程上，负责：

1. 在请求入队时编译 grammar（async via ThreadPoolExecutor）。
2. 每步调度时为所有 structured-output 请求构造 grammar bitmask。
3. 协调思考模型与结构化输出的联动（reasoning parser、`should_fill_bitmask`、`should_advance`、`trim_reasoning_for_advance`）。
4. 在请求推进时调用 `grammar.accept_tokens` 推进 FSM，并在 spec draft 被拒绝后 `grammar.rollback` 回滚。
5. 维护 bitmask 张量复用（`_grammar_bitmask` 缓存）、并行 fill 优化。

类签名（`vllm/v1/structured_output/__init__.py:36`）：

```python
class StructuredOutputManager:
    def __init__(self, vllm_config: VllmConfig): ...
    def grammar_init(self, request: Request) -> None: ...
    def grammar_bitmask(self, requests, structured_output_request_ids,
                        scheduled_spec_decode_tokens) -> npt.NDArray[np.int32] | None: ...
    def should_fill_bitmask(self, request) -> bool: ...
    def should_advance(self, request) -> bool: ...
    def trim_reasoning_for_advance(self, request, new_token_ids) -> list[int]: ...
    def clear_backend(self) -> None: ...
```

## 为什么

- **跨进程序列化**：grammar bitmask 在 engine core 进程生成，但要在 worker 进程应用。numpy ndarray 比 torch tensor 更高效序列化（行 99 注释），所以 `grammar_bitmask` 返回 ndarray 给 scheduler；scheduler 把它包进 `GrammarOutput` 走 MessageQueue。
- **异步编译避免调度阻塞**：JSON Schema / regex 编译耗时与 vocab 大小相关（数毫秒到秒）；用 `ThreadPoolExecutor(max_workers=cpu_count//2)` 在后台编译，请求状态提前转 `WAITING`。
- **external_launcher determinism**：多 TP rank 走 external_launcher 时各 rank 各自调度，async grammar 会导致 rank 间 WAITING→WAITING 转换时机不同，破坏 SPMD。`_use_async_grammar_compilation = distributed_executor_backend != "external_launcher"` 在此场景下 disable async（行 53）。
- **per-request reasoning state**：思考模型与结构化输出联动需要 per-request 跟踪 "reasoning 是否已结束"——结构化输出 manager 持有 `reasoner_cls`，按需 lazy 实例化 `ReasoningParser`，并把 `reasoning_ended` / `reasoning_end_token_index` 存入 `StructuredOutputRequest`。
- **大 batch parallel fill**：当 `len(structured_output_request_ids) > 128` 且 non-spec 时（行 62–69），用 `ThreadPoolExecutor` 并行 fill bitmask chunks；小 batch 走串行路径（与 reasoning / spec 处理逻辑共享状态更复杂）。

## 怎么做

### __init__（行 39）

```python
def __init__(self, vllm_config):
    self.backend: StructuredOutputBackend | None = None
    self.reasoner_cls: type[ReasoningParser] | None = None
    self.vllm_config = vllm_config
    self._use_async_grammar_compilation = (
        vllm_config.parallel_config.distributed_executor_backend != "external_launcher")
    self._grammar_bitmask: torch.Tensor | None = None
    self._full_mask = torch.tensor(-1, dtype=torch.int32)
    max_batch_size = vllm_config.scheduler_config.max_num_seqs
    self.fill_bitmask_parallel_threshold = 128
    if max_batch_size > self.fill_bitmask_parallel_threshold:
        self.fill_bitmask_parallel_batch_size = 16
        max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
        self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)
    if not skip_tokenizer_init:
        max_workers = max(1, (cpu_count + 1) // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tokenizer = cached_tokenizer_from_config(model_config)
        if reasoning_parser_plugin and len(reasoning_parser_plugin) > 3:
            ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)
        if reasoning_parser:
            self.reasoner_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
    self.enable_in_reasoning = vllm_config.structured_outputs_config.enable_in_reasoning
```

关键参数：

- `_use_async_grammar_compilation`：上文。
- `_grammar_bitmask`：lazy 分配的 bitmask 张量，按 `max_batch_size * (1 + num_speculative_tokens)` 行预分配。
- `_full_mask`：所有位全 1（-1 as int32）的标量，用于"不约束的请求"行。
- `fill_bitmask_parallel_threshold = 128`：batch > 128 时启用并行 fill。
- `executor_for_fillmask`：并行 fill bitmask 用的 thread pool。
- `executor`：grammar 编译用的 thread pool。
- `reasoner_cls`：per request 实例化 parser（不缓存实例，因为依赖 per-request template kwargs）。
- `enable_in_reasoning`：是否在 reasoning 段也强制 grammar 约束。

### grammar_init（行 115）

第一次调用时锁定 backend，之后 `_create_grammar` 提交到 `executor`（async）或同步执行。`_create_grammar`：

```python
def _create_grammar(self, request):
    key = request.structured_output_request.structured_output_key
    request_type, grammar_spec = key
    assert self.backend is not None
    return self.backend.compile_grammar(request_type, grammar_spec)
```

backend 选择（行 130）按字符串：`"xgrammar"` / `"guidance"` / `"outlines"` / `"lm-format-enforcer"`。

### grammar_bitmask（行 204）

主流程：

1. `if not structured_output_request_ids: return None` 短路。
2. lazy 分配 `_grammar_bitmask`：`backend.allocate_token_bitmask(max_batch_size * (1 + num_speculative_tokens))`。
3. 计算 `cumulative_index = 0`；按是否 spec + 是否大 batch 选路径：

#### 大 batch + 非 spec path（行 236）

按 `fill_bitmask_parallel_batch_size = 16` 切片，每片 submit 一个 `_async_submit_fill_bitmask` 任务，最后 `promise.result()` 等全部完成。

#### 串行 path + spec/reasoning（行 263）

对每请求：

- 取 `grammar`、`apply_bitmask` (`should_fill_bitmask(request)`)。
- lazily 取 `reasoner`，根据 `detect_reasoning_end` 决定是否在 window 内动态检测 reasoning 是否结束。
- 对每 spec draft token `token`（来自 `scheduled_spec_decode_tokens[req_id]`）：
  - `_fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))`。
  - 若 `token == -1`：跳过 mask + 不 advance（spec draft 的 padding 位）。
  - 若 `detect_reasoning_end and not apply_bitmask`：用 `simulated_buf` 模拟推进 token 序列，调 `reasoner.is_reasoning_end_streaming` 检测；若检测到结束，翻转 `apply_bitmask`、`advance_grammar=False`（不通过结束 marker token）、设置 `post_reasoning_end_in_window=True`。
  - 若 `advance_grammar and not grammar.is_terminated()`：`accepted = grammar.accept_tokens(req_id, [token])`；若未接受且不在 reasoning-end-window，raise AssertionError（grammar 状态与 sampler 输出不一致的 sanity check）。
- bonus token bitmask：`_fill_bitmasks(((grammar, cumulative_index, bonus_apply),))`。`bonus_apply = should_fill_bitmask(request) or apply_bitmask`（mid-window 翻转的情况）。
- 若 `state_advancements > 0`：`grammar.rollback(state_advancements)`——把 spec draft 期间的 FSM 推进全部回滚，等 scheduler 真正接受后再推进。
4. 转 numpy 返回。

### should_fill_bitmask（行 351）

```python
def should_fill_bitmask(self, request):
    reasoner = self._get_reasoner(request)
    if reasoner is not None:
        if self.enable_in_reasoning: return True
        if request.structured_output_request.reasoning_ended is None:
            request.structured_output_request.reasoning_ended = \
                reasoner.is_reasoning_end(request.prompt_token_ids or [])
        return request.structured_output_request.reasoning_ended
    return True
```

### should_advance（行 371）

```python
def should_advance(self, request):
    if not request.use_structured_output: return False
    reasoner = self._get_reasoner(request)
    if reasoner is None: return True
    if self.enable_in_reasoning: return True
    structured_req = request.structured_output_request
    if structured_req.reasoning_ended: return True
    delta_from = request.num_computed_tokens - request.num_output_placeholders
    all_token_ids = request.all_token_ids
    start = delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
    if reasoner.is_reasoning_end_streaming(all_token_ids, itertools.islice(all_token_ids, start, None)):
        structured_req.reasoning_ended = True
        # structural tag + spec decode：当 reasoning 结束在同一 step 内
        if spec_config and structured_req.structured_output_key[0] == STRUCTURAL_TAG:
            structured_req.reasoning_end_token_index = \
                self._find_reasoning_end_index(reasoner, all_token_ids, start)
            return True
    return False
```

对 structural_tag + spec decode，reasoning 结束 marker 与 grammar advance 同 step；`trim_reasoning_for_advance` 砍掉 reasoning 段后再 accept_tokens。

### trim_reasoning_for_advance（行 449）

```python
def trim_reasoning_for_advance(self, request, new_token_ids):
    structured_req = request.structured_output_request
    if structured_req is None: return new_token_ids
    end_idx = structured_req.reasoning_end_token_index
    if end_idx is None: return new_token_ids
    first_idx = len(request.all_token_ids) - len(new_token_ids)
    num_reasoning = end_idx + 1 - first_idx
    if num_reasoning <= 0: return new_token_ids
    return new_token_ids[num_reasoning:]
```

### clear_backend

```python
def clear_backend(self):
    if self.backend is not None:
        self.backend.destroy()
```

engine 关闭时调用，释放 backend 资源（如 xgrammar compiler cache）。

## 与其它模块/系统配合

- [backend-types.md](backend-types.md) / [backend-xgrammar.md](backend-xgrammar.md) 等：backend 实例化为 SOM 持有。
- [request.md](request.md)：`StructuredOutputRequest` 是 per-request 状态容器，SOM 在其上挂 grammar、reasoning_ended 等字段。
- [utils.md](utils.md)：`apply_grammar_bitmask` 是 SOM 输出 bitmask 的消费者（worker 端）。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：scheduler 在调度阶段调 `grammar_init`、`grammar_bitmask`、`should_advance`、`accept_tokens`、`rollback`。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：worker 端通过 `GrammarOutput` 接收 bitmask 并调 `apply_grammar_bitmask`。
- [tokenizers-ReasoningParser](../../14-tokenizers-transformers/README.md)（待补充）：`ReasoningParserManager` 与 `ReasoningParser` 实例化路径；`reasoning_parser_plugin` 字段。
- [../thinking-budget.md](../thinking-budget.md)：thinking budget 强制 `reasoning_ended`，SOM 据此调节 bitmask。

## 历史版本演进

- **v0.7.0**：V1 `StructuredOutputManager` landfall，仅 xgrammar backend；async grammar init via ThreadPoolExecutor。
- **v0.8.0**：spec decode + structured output 联动；`rollback(state_advancements)` 机制引入；per-position bitmask 行展开。
- **v0.9.0**：guidance backend 加入；`disable_any_whitespace` / `disable_additional_properties` 字段。
- **v0.10.0**：outlines + lm-format-enforcer 加入；ABC `StructuredOutputBackend` / `StructuredOutputGrammar` 稳定；`should_advance` / `should_fill_bitmask` / `trim_reasoning_for_advance` reasoning trio 引入。
- **v0.10.5**：structural_tag + spec decode 同 step advance 的特殊路径；`_find_reasoning_end_index` helper。
- **v0.11.0**：`fill_bitmask_parallel_threshold = 128` + `executor_for_fillmask` 并行 fill；`reasoning_parser_plugin` 字段支持 plugin 路径加载。
- **v0.12 / main**：diffusion LLM 兼容（`is_diffusion and req_tokens` 时跳过 bonus bitmask）；`reasoning_parser_kwargs` per-request 传递。

[← 返回结构化输出](../README.md)

## 参见

- [backend-types.md](backend-types.md)
- [backend-xgrammar.md](backend-xgrammar.md)
- [request.md](request.md)
- [utils.md](utils.md)
- [../thinking-budget.md](../thinking-budget.md)
