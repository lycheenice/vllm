[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [结构化输出](../README.md) > GuidanceBackend

# GuidanceBackend（llguidance 后端）

> 源码：`vllm/v1/structured_output/backend_guidance.py`
> 依赖：`llguidance` 库（lazy import）

---

## 是什么

`GuidanceBackend` 是基于 [llguidance](https://github.com/guidance-ai/llguidance) 库的 backend，提供与 xgrammar 等价的功能集（JSON / JSON_OBJECT / REGEX / GRAMMAR / CHOICE / STRUCTURAL_TAG 全覆盖）。它通过 `llguidance.LLMatcher` 维护 FSM 状态，`llguidance.torch` 提供 bitmask allocation/fill 工具。

类签名（`vllm/v1/structured_output/backend_guidance.py:87`）：

```python
@dataclass
class GuidanceBackend(StructuredOutputBackend):
    def __post_init__(self): ...
    def compile_grammar(self, request_type, grammar_spec) -> StructuredOutputGrammar: ...
    def allocate_token_bitmask(self, max_num_seqs): ...
    def destroy(self): ...

@dataclass
class GuidanceGrammar(StructuredOutputGrammar):
    ll_matcher: llguidance.LLMatcher
    ll_tokenizer: llguidance.LLTokenizer
    vocab_size: int
    printed_error: bool
    terminated: bool
    rollback_lag: int
    def accept_tokens(self, request_id, tokens) -> bool: ...
    def validate_tokens(self, tokens) -> list[int]: ...
    def rollback(self, num_tokens) -> None: ...
    def fill_bitmask(self, bitmask, idx) -> None: ...
    def is_terminated(self) -> bool: ...
    def reset(self): ...
```

## 为什么

- **JSON 灵活性**：llguidance 提供 `whitespace_flexible` default，让 JSON 输出可任意插空白——比 xgrammar 的 `disable_any_whitespace` 二元控制更细粒度（待核实：xgrammar 后来也类似支持）。
- **automatic additionalProperties**：`_walk_json_for_additional_properties` 自动给所有有 properties / patternProperties 的 object 加 `additionalProperties=False`，避免 schema 允许额外字段时输出膨胀（受 `disable_additional_properties` 控制）。
- **grammar 序列化统一**：所有 spec 类型在 `serialize_guidance_grammar` 中转为 llguidance 内部的"序列化语法"字符串——JSON 走 `LLMatcher.grammar_from_json_schema`，其他走 `grammar_from("regex"|"grammar"|"choice", spec)`，structural_tag 用 `StructTag.to_grammar`。
- **`max_rollback_tokens` 不需要**：llguidance 的 LLMatcher 原生支持任意步数 rollback，不需要 max_rollback_tokens 限制。`rollback_lag` 字段处理"EOS 后多 1 步延迟"（行 161）。
- **lazy import**：`llguidance` / `llguidance.hf` / `llguidance.torch` 都是 LazyLoader（行 28–30），未启用 guidance backend 时不加载。
- **Mistral tokenizer**：用 `self.tokenizer.llg_tokenizer` 而非 `from_tokenizer`——Mistral tokenizer 已自带 llguidance 适配。
- **error reporting**：`check_error` 在每次关键操作（accept / validate / rollback / fill_bitmask）后调一次，把 LLMatcher 内部错误日志出来（仅一次，避免刷屏）。

## 怎么做

### __post_init__（行 88）

```python
def __post_init__(self):
    self.disable_any_whitespace = self.vllm_config.structured_outputs_config.disable_any_whitespace
    self.disable_additional_properties = \
        self.vllm_config.structured_outputs_config.disable_additional_properties
    if is_mistral_tokenizer(self.tokenizer):
        self.ll_tokenizer = self.tokenizer.llg_tokenizer
    else:
        self.ll_tokenizer = llguidance_hf.from_tokenizer(
            self.tokenizer, max(self.vocab_size, len(self.tokenizer)))
```

不同于 xgrammar，guidance 不需要 `GrammarCompiler`——LLMatcher 直接从 serialized grammar + tokenizer 构造。

### compile_grammar（行 103）

```python
def compile_grammar(self, request_type, grammar_spec):
    self.serialized_grammar = serialize_guidance_grammar(
        request_type, grammar_spec,
        self.disable_any_whitespace, self.disable_additional_properties)
    ll_matcher = llguidance.LLMatcher(
        self.ll_tokenizer, self.serialized_grammar,
        log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")))
    r = GuidanceGrammar(
        ll_matcher=ll_matcher, ll_tokenizer=self.ll_tokenizer, vocab_size=self.vocab_size)
    r.check_error()
    return r
```

`serialize_guidance_grammar`（行 219）按 request_type 转 llguidance 内部语法字符串：

- `JSON`：`grammar_from_json_schema(spec, defaults={"whitespace_flexible": not disable_any_whitespace})`，spec 可能 dict 或 str；dict 通过 `process_for_additional_properties` 自动补 `additionalProperties=False`。
- `JSON_OBJECT`：硬编码 `'{"type": "object"}'` schema。
- `REGEX` / `GRAMMAR` / `CHOICE`：`llguidance.grammar_from(tp, spec)`。
- `STRUCTURAL_TAG`：解析 spec JSON，把每个 structure 的 schema 序列化为 JSON grammar，组成 `StructTag.to_grammar(tags)`。

### GuidanceGrammar.accept_tokens（行 153）

```python
def accept_tokens(self, request_id, tokens):
    if self.ll_tokenizer.eos_token in tokens:
        if self.ll_matcher.is_stopped() and not self.terminated:
            self.rollback_lag = 1
        self.terminated = True
    if self.ll_matcher.is_stopped():
        return True
    r = self.ll_matcher.consume_tokens(tokens)
    self.check_error()
    return r
```

`rollback_lag = 1` 让后续 `rollback(n)` 实际只 rollback `n-1`——因为 EOS 已经停止，不应再回滚停止状态。

### validate_tokens（行 181）

```python
def validate_tokens(self, tokens):
    if len(tokens) == 0: return []
    if self.ll_matcher.is_stopped(): return []
    num_tokens = self.ll_matcher.validate_tokens(tokens)
    self.check_error()
    return tokens[:num_tokens]
```

`LLMatcher.validate_tokens` 返回最长合法前缀长度，无需 accept/rollback 操作。

### rollback（行 198）

```python
def rollback(self, num_tokens):
    if num_tokens > 0:
        self.ll_matcher.rollback(num_tokens - self.rollback_lag)
        self.terminated = False
        self.rollback_lag = 0
        self.check_error()
```

### fill_bitmask

```python
def fill_bitmask(self, bitmask, idx):
    llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
    self.check_error()
```

不再调 `matcher` 自身接口，而是用 `llguidance_torch.fill_next_token_bitmask` 全局 helper。

### allocate_token_bitmask

```python
return llguidance_torch.allocate_token_bitmask(max_num_seqs, self.ll_tokenizer.vocab_size)
```

### validate_guidance_grammar（行 288）

前端 processor 校验函数：

```python
def validate_guidance_grammar(sampling_params, tokenizer=None):
    if sampling_params.structured_outputs is None: return
    tp, grm = get_structured_output_key(sampling_params.structured_outputs)
    guidance_grm = serialize_guidance_grammar(tp, grm)
    err = llguidance.LLMatcher.validate_grammar(guidance_grm, tokenizer)
    if err:
        raise ValueError(f"Grammar error: {err}")
```

### has_guidance_unsupported_json_features（行 48）

校验工具：检查 `patternProperties` 不支持；递归检查嵌套 dict / list。

## 与其它模块/系统配合

- [backend-types.md](backend-types.md)：实现 ABC。
- [backend-xgrammar.md](backend-xgrammar.md)：功能等价；xgrammar 是默认，guidance 需要前端 `_backend="guidance"` 显式选。
- [manager.md](manager.md)：SOM 按 `"guidance"` 实例化。
- [utils.md](utils.md)：`apply_token_bitmask_inplace` 是 backend-agnostic 的——guidance 产生的 bitmask 与 xgrammar 格式相同，可互换。
- [tokenizers](../../14-tokenizers-transformers/README.md)：`llguidance_hf.from_tokenizer` 构造 LLTokenizer；Mistral 走 `tokenizer.llg_tokenizer`。
- [配置体系-StructuredOutputsConfig](../../10-config/README.md)（待补充）：`disable_any_whitespace` / `disable_additional_properties`。

## 历史版本演进

- **v0.9.0**：`GuidanceBackend` landfall；llguidance 库作为可选依赖加入。
- **v0.9.5**：`disable_additional_properties` 字段加入，配合 `_walk_json_for_additional_properties` 自动补全 schema。
- **v0.10.0**：structural_tag 支持；`StructTag.to_grammar` 接口稳定。
- **v0.10.5+**：`rollback_lag` 修复 EOS 后多回滚 1 步的 bug；`LLGUIDANCE_LOG_LEVEL` env var 控制 matcher 日志。
- **v0.11.0+**：稳定维护；与 xgrammar 性能基准对比相关 issue 持续优化（待核实具体优化 commit）。

[← 返回结构化输出](../README.md)

## 参见

- [backend-types.md](backend-types.md)
- [backend-xgrammar.md](backend-xgrammar.md)
- [manager.md](manager.md)
