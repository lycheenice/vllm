[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [结构化输出](../README.md) > LMFormatEnforcerBackend

# LMFormatEnforcerBackend（lm-format-enforcer 后端）

> 源码：`vllm/v1/structured_output/backend_lm_format_enforcer.py`
> 依赖：`lmformatenforcer` 库（lazy import）

---

## 是什么

`LMFormatEnforcerBackend` 基于 [lm-format-enforcer](https://github.com/dottxt-ai/outlines) 库（实际上是另一独立项目），提供 character-level parser 的结构化输出。它**仅支持 JSON / JSON_OBJECT / REGEX / CHOICE**——不支持 GRAMMAR 也不支持 STRUCTURAL_TAG，且**显式拒绝 spec decode**（`max_rollback_tokens > 0` 时 raise ValueError）。

类签名（`vllm/v1/structured_output/backend_lm_format_enforcer.py:93`）：

```python
@dataclass
class LMFormatEnforcerBackend(StructuredOutputBackend):
    def __post_init__(self): ...
    def compile_grammar(self, request_type, grammar_spec) -> StructuredOutputGrammar: ...
    def allocate_token_bitmask(self, max_num_seqs) -> torch.Tensor: ...
    def destroy(self): ...

@dataclass
class LMFormatEnforcerGrammar(StructuredOutputGrammar):
    token_enforcer: lmformatenforcer.TokenEnforcer
    current_tokens_prefix: list[int]
    def accept_tokens(self, request_id, tokens) -> bool: ...
    def validate_tokens(self, tokens) -> list[int]: ...
    def rollback(self, num_tokens) -> None: ...
    def fill_bitmask(self, bitmask, batch_index) -> None: ...
    def is_terminated(self) -> bool: ...
    def reset(self): ...
```

## 为什么

- **character-level 解析更宽容**：lm-format-enforcer 工作在字符层而非 token 层，对部分 tokenizer 不规范 token（如不完整 UTF-8 序列）有更宽松处理。这让某些边角 JSON schema 能正确生成。
- **JSON schema / regex 类型支持独立**：库内自带 `JsonSchemaParser` / `RegexParser` / `UnionParser` / `StringParser`，无需依赖外部 grammar compiler。
- **不支持 spec decode 的设计选择**：`max_rollback_tokens > 0` 时 raise `ValueError("LM Format Enforcer backend does not support speculative tokens")`——lm-format-enforcer 的 character-level parser 不易回滚多处状态，作者选择直接拒绝 spec decode。
- **Per-request prefix list**：`current_tokens_prefix` 是简单的 list[int]，accept 时 append、rollback 时 slice——简单但每请求独立。
- **token allowed bitmask 形态不同**：lm-format-enforcer 直接返回"允许的 token ids list"（而非 bit mask），`fill_bitmask` 把它当作 dense bitmask 写入：`bitmask[batch_index] = allowed_tokens.allowed_tokens`（行 79，待核实：是否 `allowed_tokens` 是 bitset 而非 list，名字暗示是 bitset）。

## 怎么做

### __post_init__（行 95）

```python
def __post_init__(self):
    self.tokenizer_data = _cached_build_vllm_token_enforcer_tokenizer_data(
        self.tokenizer, self.vocab_size)
```

`_cached_build_vllm_token_enforcer_tokenizer_data` 是 `@lru_cache` 装饰的，只构造一次 `TokenEnforcerTokenizerData`——lm-format-enforcer 标准的 vLLM 集成入口，内部含 reduced vocabulary 与 byte-level mapping。

### compile_grammar（行 100）

```python
def compile_grammar(self, request_type, grammar_spec):
    if request_type == StructuredOutputOptions.JSON:
        character_level_parser = lmformatenforcer.JsonSchemaParser(json.loads(grammar_spec))
    elif request_type == StructuredOutputOptions.JSON_OBJECT:
        character_level_parser = lmformatenforcer.JsonSchemaParser(None)
    elif request_type == StructuredOutputOptions.REGEX:
        character_level_parser = lmformatenforcer.RegexParser(grammar_spec)
    elif request_type == StructuredOutputOptions.CHOICE:
        choices = ast.literal_eval(grammar_spec)
        character_level_parser = lmformatenforcer.UnionParser(
            [lmformatenforcer.StringParser(choice) for choice in choices])
    else:
        raise ValueError(...)
    max_rollback_tokens = (
        self.vllm_config.speculative_config.num_speculative_tokens
        if self.vllm_config.speculative_config is not None else 0)
    if max_rollback_tokens > 0:
        raise ValueError("LM Format Enforcer backend does not support speculative tokens")
    token_enforcer = lmformatenforcer.TokenEnforcer(
        tokenizer_data=self.tokenizer_data, parser=character_level_parser)
    return LMFormatEnforcerGrammar(token_enforcer)
```

- `JsonSchemaParser(None)` 是 json_object 的写法——传 None 让 parser 用 generic object schema。
- `UnionParser([StringParser(c) for c in choices])` 表达 choice——多 StringParser 联合，等价于 regex `(c1|c2|...)` 但走字符匹配。

### LMFormatEnforcerGrammar.accept_tokens（行 47）

```python
def accept_tokens(self, request_id, tokens):
    original_len = len(self.current_tokens_prefix)
    for token in tokens:
        if not self.token_enforcer.get_allowed_tokens(
            self.current_tokens_prefix).is_token_allowed(token):
            del self.current_tokens_prefix[original_len:]  # rollback partial updates
            return False
        self.current_tokens_prefix.append(token)
    return True
```

每次 accept 前 `get_allowed_tokens(prefix).is_token_allowed(token)`——`current_tokens_prefix` 是 dynamic state，每次调用 `get_allowed_tokens` 都重算 character-level parser 状态（开销通常较大）。

### validate_tokens（行 59）

```python
def validate_tokens(self, tokens):
    for prefix_length in range(len(tokens)):
        prefix = tokens[:prefix_length]
        next_token = tokens[prefix_length]
        if not self.token_enforcer.get_allowed_tokens(
            self.current_tokens_prefix + prefix).is_token_allowed(next_token):
            break
    else:
        return tokens
    return tokens[:prefix_length]
```

逐 prefix 测试；不修改 state。

### rollback（行 72）

```python
def rollback(self, num_tokens):
    self.current_tokens_prefix = self.current_tokens_prefix[:-num_tokens]
```

纯 list slice。

### fill_bitmask（行 75）

```python
def fill_bitmask(self, bitmask, batch_index):
    allowed_tokens = self.token_enforcer.get_allowed_tokens(self.current_tokens_prefix)
    bitmask[batch_index] = allowed_tokens.allowed_tokens
```

`allowed_tokens.allowed_tokens` 是一个 bitset 张量（每 bit 表示对应 token id 是否合法），可直接赋给 bitmask 行。

### is_terminated（行 81）

```python
def is_terminated(self):
    return (len(self.current_tokens_prefix) > 0
            and self.current_tokens_prefix[-1] == self.token_enforcer.eos_token_id)
```

判断最后 token 是否是 EOS——简单且 deterministic。

### validate_structured_output_request_lm_format_enforcer（行 149）

前端校验函数：

- regex / json / choice：直接通过（regex parser 在 runtime 才校验；json 走 `json.loads` / `json.dumps` 检查可序列化）。
- grammar：raise ValueError。

### allocate_token_bitmask

```python
return torch.full((max_num_seqs, (self.vocab_size + 31) // 32), -1,
                  dtype=torch.int32, pin_memory=PIN_MEMORY)
```

与 outlines 相同——手动分配 int32 bitmask，初始全 1。

## 与其它模块/系统配合

- [backend-types.md](backend-types.md)：实现 ABC。
- [backend-xgrammar.md](backend-xgrammar.md) / [backend-guidance.md](backend-guidance.md) / [backend-outlines.md](backend-outlines.md)：其他 backend；lm-format-enforcer 功能最少，且不支持 spec decode。
- [manager.md](manager.md)：SOM 按 `"lm-format-enforcer"` 实例化（lazy import 避免依赖侵入）。
- [utils.md](utils.md)：`apply_token_bitmask_inplace` 是 backend-agnostic；bitmask 格式与 outlines 一致（int32 dense bitset）。
- [tokenizers](../../14-tokenizers-transformers/README.md)：`build_vllm_token_enforcer_tokenizer_data` 内部处理 byte-level mapping；与 tokenizer 强耦合。

## 历史版本演进

- **v0.10.0**：`LMFormatEnforcerBackend` landfall；作为第四个 backend 加入。
- **v0.10.5**：显式 raise 不支持 spec decode（之前 silently 错误，待核实：是否之前未 raise）。
- **v0.11.0**：`@lru_cache _cached_build_vllm_token_enforcer_tokenizer_data` 加入，让同 tokenizer 跨 engine 复用 tokenizer_data。
- **v0.12 / main**：稳定维护；社区反馈 lm-format-enforcer 性能不如 xgrammar，常作为兼容旧 schema 的 fallback。

[← 返回结构化输出](../README.md)

## 参见

- [backend-types.md](backend-types.md)
- [backend-xgrammar.md](backend-xgrammar.md)
- [backend-outlines.md](backend-outlines.md)
- [manager.md](manager.md)
