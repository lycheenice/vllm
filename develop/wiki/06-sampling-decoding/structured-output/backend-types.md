[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [结构化输出](../README.md) > Backend Types

# Backend Types（后端抽象）

> 源码：`vllm/v1/structured_output/backend_types.py`

---

## 是什么

`backend_types.py` 定义结构化输出子系统的两个核心 ABC 与一个选项枚举：

- `StructuredOutputOptions`（enum，行 19）：支持的请求类型——`JSON` / `JSON_OBJECT` / `REGEX` / `GRAMMAR` / `CHOICE` / `STRUCTURAL_TAG`。
- `StructuredOutputGrammar`（ABC，行 31）：request-level FSM 接口，定义 `accept_tokens` / `validate_tokens` / `rollback` / `fill_bitmask` / `is_terminated` / `reset` 六个抽象方法。
- `StructuredOutputBackend`（ABC，行 98）：engine-level 编译器接口，定义 `compile_grammar` / `allocate_token_bitmask` / `destroy` 三个抽象方法。
- `StructuredOutputKey = tuple[StructuredOutputOptions, str]`（行 28）：request 的 (option, spec_string) 元组，作为 grammar cache key。

## 为什么

- **后端可插拔**：vLLM 同时支持 xgrammar / guidance / outlines / lm-format-enforcer 四个 backend，每个的能力矩阵不同（见下文）。统一 ABC 让 `StructuredOutputManager` 只与 `StructuredOutputBackend` / `StructuredOutputGrammar` 打交道，不感知具体实现。
- **职责分离**：
  - `StructuredOutputBackend` 是 engine-level 单例，编译多个 grammar，自维护 cache（如 xgrammar 的 `GrammarCompiler`）。
  - `StructuredOutputGrammar` 是 request-level 实例，维护一个 FSM 的状态、可被多次 accept/rollback。Request 完成时丢弃。
- **bitmask 是通用人机接口**：所有 backend 最终通过 `fill_bitmask(bitmask, batch_index)` 写入同一个 `[vocab // 32]` int32 张量；上层 `apply_grammar_bitmask` 调 `xgrammar.apply_token_bitmask_inplace` 应用。这是 vLLM 与具体后端 FSM 库解耦的关键。
- **`validate_tokens` 与 `accept_tokens` 区分**：spec decode 路径需要"测试函数"——在不推进 FSM 的前提下检查 draft token 是否合法；`validate_tokens` 返回 prefix 长度，draft 被拒后不需要 rollback。`accept_tokens` 真正推进 FSM。
- **`is_terminated` 终止条件**：每个 backend 对"终止"的定义可能不同——xgrammar 是 FSM 接受态，outlines 是 DFA 接受态延迟一步，lm-format-enforcer 是 EOS token。ABC 强制接口一致但语义各 backend 自决。

## 怎么做

### StructuredOutputOptions

```python
class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()
```

由 `get_structured_output_key(StructuredOutputsParams)`（`request.py:83`）根据 `params.json` / `params.json_object` / `params.regex` / `params.choice` / `params.grammar` / `params.structural_tag` 字段决定。

### StructuredOutputGrammar ABC

```python
class StructuredOutputGrammar(ABC):
    @abstractmethod
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool: ...
    @abstractmethod
    def validate_tokens(self, tokens: list[int]) -> list[int]: ...
    @abstractmethod
    def rollback(self, num_tokens: int) -> None: ...
    @abstractmethod
    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None: ...
    @abstractmethod
    def is_terminated(self) -> bool: ...
    @abstractmethod
    def reset(self): ...
```

语义说明（docstring 摘录）：

- `accept_tokens`：返回 True 表示 FSM 完全接受；False 表示至少一个 token 不合法。request_id 用于 log。
- `validate_tokens`：返回 list[int]，是输入 tokens 的最长合法前缀。FSM 状态不前进。
- `rollback(num_tokens)`：回滚 n 次 accept 操作；spec decode 拒绝时调。
- `fill_bitmask(bitmask, idx)`：把"下一个合法 token"的 bitmask 写入 `bitmask[idx]`。`bitmask` 形状 `[batch, vocab//32]`，int32，bit i = 1 表示 token i 合法。
- `is_terminated`：FSM 是否到达接受态（再无合法 token）。
- `reset`：回到初始状态（极少用，可能 deprecated）。

### StructuredOutputBackend ABC

```python
@dataclass
class StructuredOutputBackend(ABC):
    vllm_config: VllmConfig
    tokenizer: TokenizerLike
    vocab_size: int

    @abstractmethod
    def compile_grammar(self, request_type: StructuredOutputOptions,
                         grammar_spec: str) -> StructuredOutputGrammar: ...
    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor: ...
    @abstractmethod
    def destroy(self): ...
```

- `compile_grammar`：把 `(JSON, schema_str)` / `(REGEX, pattern_str)` / etc. 编译成 `StructuredOutputGrammar` 实例。耗时操作，由 `StructuredOutputManager._create_grammar` 在 ThreadPool 中调用。
- `allocate_token_bitmask`：分配 bitmask 张量，形状 `[max_num_seqs, vocab // 32]` int32。各 backend 可能用不同分配路径（xgrammar 调 `xgr.allocate_token_bitmask`，others 手动 `torch.full`）。
- `destroy`：engine 关闭时清理资源。

### 各 Backend 能力矩阵

| Backend | JSON | JSON_OBJECT | REGEX | GRAMMAR | CHOICE | STRUCTURAL_TAG | Spec decode | 备注 |
|---|---|---|---|---|---|---|---|---|
| xgrammar | ✅ | ✅ | ✅ | ✅ (EBNF；自动 Lark→EBNF) | ✅ (转 grammar) | ✅ | ✅ (max_rollback_tokens) | 默认；Mistral tekken tokenizer 特殊处理 |
| guidance (llguidance) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 不支持 `patternProperties`；提供 `disable_additional_properties` |
| outlines (outlines_core) | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | regex-automata DFA；不支持 lookarounds / backreferences |
| lm-format-enforcer | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ (显式 raise) | 不支持 spec tokens |

由 ABC 强制 + 各 backend 自愿实现，不在运行时检查——前端 processor 校验阶段 backend-specific validate 函数（如 `validate_xgrammar_grammar`、`validate_structured_output_request_outlines`、`validate_guidance_grammar`、`validate_structured_output_request_lm_format_enforcer`）会 raise。

## 与其它模块/系统配合

- [manager.md](manager.md)：`StructuredOutputManager` 持有 backend 实例，调 `compile_grammar` / `allocate_token_bitmask` / `destroy`。
- [backend-xgrammar.md](backend-xgrammar.md) / [backend-guidance.md](backend-guidance.md) / [backend-outlines.md](backend-outlines.md) / [backend-lm-format-enforcer.md](backend-lm-format-enforcer.md)：四个具体实现。
- [request.md](request.md)：`StructuredOutputOptions` 由 `get_structured_output_key` 从参数推导；决定 `compile_grammar` 入参。
- [utils.md](utils.md)：`apply_token_bitmask_inplace` 是 bitmask 的统一消费者，与 backend 无关。
- [../sampler.md](../sampler.md)：sampler 不感知 grammar；bitmask apply 在 sampler 前。
- [../rejection-sampler.md](../rejection-sampler.md)：spec decode 路径下 `validate_tokens` 与 `rollback` 在 `grammar_bitmask` 中被调，对应 spec draft 的 advance + rollback 逻辑。

## 历史版本演进

- **v0.6.x（V0）**：V0 只有 xgrammar；`GrammarMatcher` 直接暴露给 sampler。
- **v0.7.0**：V1 引入 `StructuredOutputBackend` / `StructuredOutputGrammar` ABC；grammar init 异步化。
- **v0.9.0**：guidance backend 引入，促使 ABC 稳定；`StructuredOutputOptions.GRAMMAR` 加入。
- **v0.10.0**：outlines + lm-format-enforcer 加入；`STRUCTURAL_TAG` option 加入（用于思考段切换格式）；`validate_tokens` 接口明确（之前部分 backend 不实现）。
- **v0.10.5+**：各 backend 完善 spec decode 支持——`max_rollback_tokens` 字段接入；lm-format-enforcer 显式 raise 不支持 spec tokens。
- **v0.12 / main**：ABC 稳定；新 backend 接入只需实现 6+3 个抽象方法。

[← 返回结构化输出](../README.md)

## 参见

- [manager.md](manager.md)
- [backend-xgrammar.md](backend-xgrammar.md)
- [backend-guidance.md](backend-guidance.md)
- [backend-outlines.md](backend-outlines.md)
- [backend-lm-format-enforcer.md](backend-lm-format-enforcer.md)
- [request.md](request.md)
