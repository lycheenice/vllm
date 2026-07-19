[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [结构化输出](../README.md) > OutlinesBackend

# OutlinesBackend（outlines 后端）

> 源码：`vllm/v1/structured_output/backend_outlines.py`
> 依赖：`outlines_core` 库（lazy import）+ `regex` Python module

---

## 是什么

`OutlinesBackend` 基于 [outlines_core](https://github.com/dottxt-ai/outlines-core) 库（Rust 实现 regex-automata DFA），把 JSON Schema / regex / choice 转成 regex pattern，再编译成 DFA。它**不支持 GRAMMAR 与 STRUCTURAL_TAG**——只有 JSON / JSON_OBJECT转 regex / REGEX / CHOICE 三类 option。

类签名（`vllm/v1/structured_output/backend_outlines.py:53`）：

```python
@dataclass
class OutlinesBackend(StructuredOutputBackend):
    def __post_init__(self): ...
    def _compile_index(self, regex_string, vocabulary) -> oc.Index: ...
    def compile_grammar(self, request_type, grammar_spec) -> StructuredOutputGrammar: ...
    def allocate_token_bitmask(self, max_num_seqs) -> torch.Tensor: ...
    def destroy(self): ...

@dataclass
class OutlinesGrammar(StructuredOutputGrammar):
    vocab_size: int
    guide: oc.Guide
    num_processed_tokens: int
    _prev_finished: bool
    def accept_tokens(self, request_id, tokens) -> bool: ...
    def validate_tokens(self, tokens) -> list[int]: ...
    def rollback(self, num_tokens) -> None: ...
    def fill_bitmask(self, bitmask, idx) -> None: ...
    def is_terminated(self) -> bool: ...
    def reset(self): ...
```

## 为什么

- **regex-automata DFA 高性能**：outlines_core 用 Rust regex-automata crate，对纯 regex 场景编译期与运行期都极快；适合纯 regex 约束场景。
- **JSON Schema → regex**：outlines 把 JSON Schema 转为 regex pattern（通过 `outlines_core.json_schema.build_regex_from_schema`），用 regex 表达所有合法 JSON；这与 xgrammar 的 JSON-grammar 路径不同，但兼容性更广。
- **不支持 grammar / structural_tag 的限制**：outlines 仅做 regex，无 EBNF grammar parser；前端 validate 函数 `validate_structured_output_request_outlines` 显式 raise 这两类请求。
- **stateful mapping cache**：`_compile_index` 用 `cache_key = f"{vocab_hash}_{regex}"` 缓存 `oc.Index`——同样 regex 在同 tokenizer 下复用 index；max_rollback_tokens 来自 spec config。
- **`prev_finished` 延迟一步**：outlines_core 的 `is_finished()` 在 DFA 接受态返回 True，但 vLLM 期望"EOS 后才结束"。`_prev_finished` 字段延迟一步让 EOS 仍能被采样（行 119–122）。

## 怎么做

### __post_init__（行 54）

```python
def __post_init__(self):
    self.vocabulary = get_outlines_vocabulary(self.tokenizer)
    self.cache = get_outlines_cache()
```

- `get_outlines_vocabulary`（`utils.py:314`）：构造 `OutlinesVocabulary` 包装的 `oc.Vocabulary`——把 tokenizer 的 vocab 转换为 reduced vocabulary（按 byte 表征，处理 BPE / SentencePiece / GPT2 各种 token 形式）。结果 cache 在 tokenizer 上避免重复计算。
- `get_outlines_cache`（`utils.py:217`）：根据 `VLLM_V1_USE_OUTLINES_CACHE` env var 返回 diskcache（持久）或 `LRUCache(maxsize=128)`（内存）。

### _compile_index（行 58）

```python
def _compile_index(self, regex_string, vocabulary):
    cache_key = f"{vocabulary._hash}_{regex_string}"
    if cache_key in self.cache:
        return self.cache[cache_key]
    index = compile_regex_with_timeout(
        lambda pat: oc.Index(pat, vocabulary.inner),
        regex_string,
    )
    self.cache[cache_key] = index
    return index
```

`compile_regex_with_timeout` 用 `VLLM_REGEX_COMPILATION_TIMEOUT_S` 防 ReDoS——注意 outlines 用 Rust regex-automata，理论上线性时间复杂度不存在 ReDoS，但仍保留 timeout 兜底防止 DFA 状态爆炸。

### compile_grammar（行 73）

```python
def compile_grammar(self, request_type, grammar_spec):
    if request_type == StructuredOutputOptions.JSON:
        regex = json_schema.build_regex_from_schema(grammar_spec)
    elif request_type == StructuredOutputOptions.REGEX:
        regex = grammar_spec
    elif request_type == StructuredOutputOptions.CHOICE:
        choices = ast.literal_eval(grammar_spec)
        choices = [regex_escape(c) for c in choices]
        regex = "(" + "|".join(choices) + ")"
    else:
        raise ValueError(f"Invalid request type for Outlines backend ({request_type!s})")
    index = self._compile_index(regex, self.vocabulary)
    max_rollback_tokens = (
        self.vllm_config.speculative_config.num_speculative_tokens
        if self.vllm_config.speculative_config is not None else 0)
    return OutlinesGrammar(vocab_size=self.vocab_size,
                            guide=oc.Guide(index, max_rollback=max_rollback_tokens))
```

注意：

- JSON_OBJECT 在 outlines 中**不支持**（raise ValueError）；用户用 outlines 时不能用 `json_object=True`，需明确指定 `json=schema`。
- CHOICE 用 regex 交替（`|`）表达；每个 choice 用 `regex_escape` 避免特殊字符。

### allocate_token_bitmask（行 99）

```python
def allocate_token_bitmask(self, max_num_seqs):
    return torch.full(
        (max_num_seqs, (self.vocab_size + 31) // 32),
        -1, dtype=torch.int32, pin_memory=PIN_MEMORY,
    )
```

与 xgrammar 不同：outlines 手动分配 bitmask，初始化全 1（-1 in int32 表示所有 bit 都允许）。

### OutlinesGrammar.accept_tokens（行 123）

```python
def accept_tokens(self, request_id, tokens):
    if self.guide.accepts_tokens(tokens):
        for t in tokens:
            self.guide.advance(t)
            self.num_processed_tokens += 1
        return True
    return False
```

注意 `accepts_tokens` 与 `advance` 是两个独立调用——前者只测试，后者推进状态；advance 可能在 accepts_tokens=True 后仍失败（dead state），但 vLLM 的注释说明"FSM 必须在无 dead state 下被 prepared"才能正常工作（行 130–135）。

### validate_tokens（行 146）

```python
def validate_tokens(self, tokens):
    accepted = []
    for tok in tokens:
        accepted.append(tok)
        if not self.guide.accepts_tokens(accepted):
            accepted.pop()
            break
    return accepted
```

逐 token 累积测试，找最长合法前缀。

### fill_bitmask

```python
def fill_bitmask(self, bitmask, idx):
    mask = bitmask[idx]
    self.guide.write_mask_into(mask.data_ptr(), mask.numel(), mask.element_size())
```

`guide.write_mask_into` 是 outlines_core 提供的 Rust FFI，直接写指针——避免 Python→numpy→torch 拷贝。

### is_terminated（行 159）

```python
def is_terminated(self):
    curr = self.guide.is_finished()
    prev = self._prev_finished
    self._prev_finished = curr
    return prev
```

延迟一步：当前 `is_finished` 不立即视为 terminated，等下一 step 才返回 True，让 EOS 仍可被采样。

### validate_structured_output_request_outlines（行 171）

前端校验函数：

- regex / json / choice 都先转 regex pattern（用 `json_schema.build_regex_from_schema`），再调 `validate_regex_is_buildable(pattern)`。
- grammar 直接 raise。
- `validate_regex_is_buildable` 用 `sre_parse.parse` + `_check_unsupported` + `_prefix_needs_context` 检查 regex 不含 backreferences / look-around / unicode boundaries / anchored prefix needs context。

## 与其它模块/系统配合

- [backend-types.md](backend-types.md)：实现 ABC。
- [backend-xgrammar.md](backend-xgrammar.md) / [backend-guidance.md](backend-guidance.md)：功能更全的替代；outlines 仅适用于纯 regex 场景。
- [manager.md](manager.md)：SOM 按 `"outlines"` 实例化（lazy import 避免依赖侵入）。
- [utils.md](utils.md)：`get_outlines_vocabulary` / `get_outlines_cache` / `compile_regex_with_timeout` 工具函数。
- [tokenizers](../../14-tokenizers-transformers/README.md)：`_reduced_vocabulary` 处理 BPE / SentencePiece / GPT2 / Llama byte token，与 tokenizer 强耦合。
- [配置体系-StructuredOutputsConfig](../../10-config/README.md)（待补充）：outlines cache 路径 `OUTLINES_CACHE_DIR` / `XDG_CACHE_HOME` / `~/.cache/outlines`。

## 历史版本演进

- **v0.10.0**：`OutlinesBackend` landfall（与 outlines_core Rust 库同期）；从 V0 outlines（Python）迁移到 outlines_core（Rust）。
- **v0.10.5**：`_prev_finished` 延迟一步修复 EOS 采样 bug（之前 outlines 在 DFA 接受态直接终止导致 EOS 漏采）。
- **v0.10.5+**：`validate_regex_is_buildable` 加入前端校验，明确列出 outlines 不支持的 regex 特性。
- **v0.11.0**：diskcache 持久化路径受 `VLLM_V1_USE_OUTLINES_CACHE` 控制；默认 LRU 128。
- **v0.12 / main**：与 spec decode 的 max_rollback_tokens 完善；与 Lark→EBNF 不相关（outlines 不支持 grammar）。

[← 返回结构化输出](../README.md)

## 参见

- [backend-types.md](backend-types.md)
- [backend-xgrammar.md](backend-xgrammar.md)
- [backend-guidance.md](backend-guidance.md)
- [utils.md](utils.md)：outlines 共用的 vocabulary / cache / regex timeout helpers
