[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [结构化输出](../README.md) > StructuredOutputRequest

# StructuredOutputRequest（请求级状态）

> 源码：`vllm/v1/structured_output/request.py`

---

## 是什么

`StructuredOutputRequest` 是 request-level 状态容器，挂在 `Request.structured_output_request` 字段上。它持有：

- `params: StructuredOutputsParams`：用户从前端传入的 JSON/regex/grammar 等参数。
- `_grammar`：编译后的 `StructuredOutputGrammar` 或其 `Future`（异步编译期）。
- `reasoning_ended: bool | None`：思考模型 reasoning 是否结束的 per-request 缓存。
- `reasoning_end_token_index: int | None`：reasoning 结束 marker 的绝对 token index；用于 structural_tag + spec decode 同 step advance 场景的 trim。
- `reasoning_parser_kwargs: dict[str, Any] | None`：per-request parser 参数（依赖 chat template kwargs）。
- `reasoner: ReasoningParser | None`：lazy 实例化的 per-request parser，**不跨请求共享**因为依赖 kwargs。
- `structured_output_key: StructuredOutputKey`（cached property）：`(option, spec_string)` 元组，作为 grammar cache key。

类签名（`vllm/v1/structured_output/request.py:21`）：

```python
@dataclasses.dataclass
class StructuredOutputRequest:
    params: StructuredOutputsParams
    _grammar: Future[StructuredOutputGrammar] | StructuredOutputGrammar | None = None
    reasoning_ended: bool | None = None
    reasoning_end_token_index: int | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None
    reasoner: "ReasoningParser | None" = None

    @staticmethod
    def from_sampling_params(sampling_params: SamplingParams | None) -> "StructuredOutputRequest | None":
        ...
    def _check_grammar_completion(self) -> bool: ...
    @property
    def is_grammar_ready(self) -> bool: ...
    @property
    def grammar(self) -> StructuredOutputGrammar | None: ...
    @grammar.setter
    def grammar(self, grammar_or_future): ...
    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey: ...
```

## 为什么

- **请求级而非全局级**：每个请求的 grammar 是独立 FSM 实例，状态不可共享；`StructuredOutputRequest` 把"参数 + 编译结果 + reasoning 状态 + parser 实例"封装到一起，让 `Request` 类本身不感知 grammar 细节。
- **Future 兼容同步/异步编译**：`_grammar` 字段有三个态（None / Future / Grammar），`_check_grammar_completion` 用 100us timeout 轮询 Future——ready 时转 `WAITING`，未 ready 时 scheduler 维持 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 状态。
- **history_len reference semantics**：`reasoning_parser_kwargs` 等字典/列表保留引用语义，让 SOM 在 lazy 实例化 reasoner 时能看到当前请求的最新 template kwargs（chat template 可能因 request 不同而异）。
- **per-request parser 实例**：`reasoner` 字段不是 class-level 共享——`StructuredOutputManager._get_reasoner` 在 reasons cls 已配置但 per-request reasoner 未建时 lazy 实例化（`__init__.py:100`），传入当前请求的 parser_kwargs。这让 GPT-OSS 风格 reasoning parser 等依赖 template 的实现能 per-request 工作。
- **`structured_output_key` cached**：option + spec string 一旦计算就缓存——同请求多次 `compile_grammar` 或 `validate` 调用复用。

## 怎么做

### from_sampling_params（行 37）

```python
@staticmethod
def from_sampling_params(sampling_params):
    if sampling_params is None: return None
    params = sampling_params.structured_outputs
    if not params or params.all_constraints_none(): return None
    return StructuredOutputRequest(params=params)
```

入口：在 `Request.__init__` / `Request.from_params` 时调用，根据 `sampling_params.structured_outputs` 决定是否创建。`all_constraints_none()` 检查所有字段（json/regex/grammar/choice/structural_tag）都为 None / False。

### _check_grammar_completion（行 48）

```python
def _check_grammar_completion(self) -> bool:
    from vllm.v1.request import RequestStatus
    if isinstance(self._grammar, Future):
        try:
            self._grammar = self._grammar.result(timeout=0.0001)  # 100us
            self.status = RequestStatus.WAITING
        except TimeoutError:
            return False
    return True
```

每次 `is_grammar_ready` / `grammar` property 访问都触发——100us 超时让 scheduler 快速跳过未完成的请求。完成时 switch status 到 `WAITING`。

### grammar property

```python
@property
def grammar(self) -> StructuredOutputGrammar | None:
    completed = self._check_grammar_completion()
    return cast(...) if completed else None

@grammar.setter
def grammar(self, grammar_or_future):
    self._grammar = grammar_or_future
```

读写都通过 property——读时自动 Future-resolve，写时直接存 Future/Grammar。

### get_structured_output_key（行 83）

模块级函数，由 `structured_output_key` cached property 调用：

```python
def get_structured_output_key(params: StructuredOutputsParams) -> StructuredOutputKey:
    if params.json is not None:
        json_str = params.json if isinstance(params.json, str) else json.dumps(params.json)
        return StructuredOutputOptions.JSON, json_str
    if params.json_object:
        return StructuredOutputOptions.JSON_OBJECT, ""
    if params.regex is not None:
        return StructuredOutputOptions.REGEX, params.regex
    if params.choice is not None:
        json_str = params.choice if isinstance(params.choice, str) else json.dumps(params.choice)
        return StructuredOutputOptions.CHOICE, json_str
    if params.grammar is not None:
        return StructuredOutputOptions.GRAMMAR, params.grammar
    if params.structural_tag is not None:
        return StructuredOutputOptions.STRUCTURAL_TAG, params.structural_tag
    raise ValueError("No valid structured output parameter found")
```

优先级顺序：json > json_object > regex > choice > grammar > structural_tag。每个 option 对应不同的 spec string 形态（JSON_OBJECT 与 None spec 例外）。

### reasoning_end_token_index

只有 `should_advance` 在 spec decode + structural_tag + reasoning 结束同 step 的特殊场景下被设置（`manager.md` `should_advance` 行 421）：

```python
structured_req.reasoning_end_token_index = self._find_reasoning_end_index(reasoner, all_token_ids, start)
```

`trim_reasoning_for_advance` 用它砍掉 reasoning 段后再 `accept_tokens`：

```python
first_idx = len(request.all_token_ids) - len(new_token_ids)
num_reasoning = end_idx + 1 - first_idx
if num_reasoning <= 0: return new_token_ids
return new_token_ids[num_reasoning:]
```

## 与其它模块/系统配合

- [manager.md](manager.md)：SOM 在 `grammar_init` 写 `structured_output_request.grammar`，在 `should_fill_bitmask` / `should_advance` / `trim_reasoning_for_advance` 读 `reasoning_ended` / `reasoning_end_token_index` / `reasoner`。
- [backend-types.md](backend-types.md)：`_grammar` 字段类型是 `StructuredOutputGrammar`（或其 Future）。
- [utils.md](utils.md)：`apply_grammar_bitmask` 在 worker 端不直接读 StructuredOutputRequest——它通过 `SchedulerOutput.GrammarOutput.structured_output_request_ids` 间接索引。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：scheduler 在调度阶段检查 `is_grammar_ready` 决定请求状态转换；`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING` 转换。
- [引擎核心-data model](../../01-engine-core/data-model.md)（待补充）：`Request` 类如何持有 `StructuredOutputRequest`。
- [tokenizers-reasoning](../../14-tokenizers-transformers/README.md)（待补充）：`ReasoningParser` 实例化与 `reasoning_parser_kwargs` 来源（chat template kwargs）。
- [../thinking-budget.md](../thinking-budget.md)：thinking budget 与 `reasoning_ended` 联动——前者在 sampling 层强制结束思考，后者在 structured output 层触发 grammar 约束开启。

## 历史版本演进

- **v0.7.0**：V1 `StructuredOutputRequest` landfall，仅含 `params` 与 `_grammar` 字段；不支持 reasoning parser。
- **v0.9.0**：`reasoner` 字段加入；与 guidance backend 同期。
- **v0.10.0**：`reasoning_ended` 字段加入；`should_fill_bitmask` / `should_advance` 的 per-request state 开始。
- **v0.10.5**：`reasoning_end_token_index` 字段加入，支持 structural_tag + spec decode 同 step advance 场景；`trim_reasoning_for_advance` 修复 #44006（accept marker token 后 grammar 拒绝整个请求）。
- **v0.11.0**：`reasoning_parser_kwargs` 字段加入，支持 per-request chat template kwargs 传递给 reasoning parser。
- **v0.12 / main**：稳定；`from_sampling_params` 利用 `all_constraints_none()` 提前 short-circuit 全空请求。

[← 返回结构化输出](../README.md)

## 参见

- [manager.md](manager.md)
- [backend-types.md](backend-types.md)
- [request.md 不直接相关]
- [../thinking-budget.md](../thinking-budget.md)
