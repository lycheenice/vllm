[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [结构化输出](../README.md) > XgrammarBackend

# XgrammarBackend（xgrammar 后端）

> 源码：`vllm/v1/structured_output/backend_xgrammar.py`
> 依赖：`xgrammar` 库（`requirements/cuda.txt` 默认安装）

---

## 是什么

`XgrammarBackend` 是 vLLM V1 结构化输出的默认后端，基于 [xgrammar](https://github.com/mlc-ai/xgrammar) 库实现。它支持全部六种 `StructuredOutputOptions`（JSON / JSON_OBJECT / REGEX / GRAMMAR / CHOICE / STRUCTURAL_TAG），并完整支持 spec decode（通过 `max_rollback_tokens`）。`XgrammarGrammar` 是 request-level 实例，封装 `xgr.GrammarMatcher`。

类签名（`vllm/v1/structured_output/backend_xgrammar.py:35`）：

```python
@dataclass
class XgrammarBackend(StructuredOutputBackend):
    def __post_init__(self): ...
    def compile_grammar(self, request_type, grammar_spec) -> StructuredOutputGrammar: ...
    def allocate_token_bitmask(self, max_num_seqs): ...
    def destroy(self): ...

@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    vocab_size: int
    matcher: xgr.GrammarMatcher
    ctx: xgr.CompiledGrammar
    num_processed_tokens: int
    _is_terminated: bool
    def accept_tokens(self, request_id, tokens) -> bool: ...
    def validate_tokens(self, tokens) -> list[int]: ...
    def rollback(self, num_tokens) -> None: ...
    def fill_bitmask(self, bitmask, idx) -> None: ...
    def is_terminated(self) -> bool: ...
    def reset(self): ...
```

## 为什么

- **功能最全**：xgrammar 是当前 vLLM 推荐的默认 backend——支持 JSON Schema 复杂特性（formats、union、array）、structural_tag（条件 grammar 嵌入）、自定义 EBNF grammar、regex、choice。其他 backend 各有局限。
- **编译期 cache**：`GrammarCompiler(cache_enabled=True, cache_limit_bytes=VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024)` 让重复 schema 编译命中 cache；同一 schema 的多请求共享 CompiledGrammar。
- **Mistral tekken tokenizer 支持**：Mistral tokenizer 的 vocab 与 HF general tokenizer 不同，xgrammar 显式构造 `TokenizerInfo`（`VocabType.RAW` for tekken / `VocabType.BYTE_FALLBACK` for others）；`stop_token_ids=[eos_token_id]` 而非默认。
- **grammar 类型转换**：`compile_grammar` 按 `request_type` 分支：
  - `JSON`：`compiler.compile_json_schema(spec, any_whitespace=...)`。
  - `JSON_OBJECT`：硬编码 `{"type": "object"}` schema。
  - `GRAMMAR`：直接 `compile_grammar(spec)`，xgrammar 只支持 EBNF——`validate_xgrammar_grammar` 中 Lark→EBNF 转换 pre-编译（`convert_lark_to_ebnf`）。
  - `REGEX`：`compile_regex_with_timeout(compiler.compile_regex, spec)`——`compile_regex_with_timeout` 用 `ThreadPoolExecutor` + `VLLM_REGEX_COMPILATION_TIMEOUT_S` 防 ReDoS。
  - `STRUCTURAL_TAG`：解析 `grammar_spec` JSON 后调 `compile_structural_tag`——支持新接口（直接传 spec 字符串）与 deprecated 接口（`StructuralTagItem` list）。
- **任何 whitespace 控制**：`disable_any_whitespace` 来自 `structured_outputs_config`，让 JSON 输出最小化空白（紧凑格式）。
- **max_rollback_tokens**：从 `speculative_config.num_speculative_tokens` 取，让 `GrammarMatcher` 内部支持 N 步回滚——配合 spec decode 的 draft accept/reject。

## 怎么做

### __post_init__（行 37）

```python
def __post_init__(self):
    self.disable_any_whitespace = self.vllm_config.structured_outputs_config.disable_any_whitespace
    if is_mistral_tokenizer(self.tokenizer):
        stop_token_ids = [self.tokenizer.eos_token_id]
        self.vocab_size = len(self.tokenizer.vocab)
        tokenizer_info = xgr.TokenizerInfo(
            encoded_vocab=self.tokenizer.vocab,
            vocab_type=xgr.VocabType.RAW if self.tokenizer.is_tekken else xgr.VocabType.BYTE_FALLBACK,
            vocab_size=self.vocab_size,
            stop_token_ids=stop_token_ids,
            add_prefix_space=True,
        )
    else:
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.vocab_size)
    self.compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=8,
        cache_enabled=True,
        cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
    )
    self.num_speculative_tokens = 0
    if self.vllm_config.speculative_config is not None:
        self.num_speculative_tokens = self.vllm_config.speculative_config.num_speculative_tokens
```

- `compiler.max_threads=8` 内部并行编译；`cache_enabled=True` 让 CompileGrammar 自带 LRU cache。
- Mistral tokenizer 特殊处理行 42–59。

### compile_grammar（行 78）

按 `request_type` 分支，前文已述。返回 `XgrammarGrammar(matcher=xgr.GrammarMatcher(ctx, max_rollback_tokens=num_speculative_tokens), vocab_size=..., ctx=ctx)`。

`GrammarMatcher` 是 request-level 状态对象，每次 forward 复用同一实例。

### allocate_token_bitmask

```python
return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)
```

xgrammar 自带 allocator，返回 int32 tensor `[max_num_seqs, ceil(vocab_size / 32)]`。

### XgrammarGrammar.accept_tokens（行 152）

```python
def accept_tokens(self, request_id, tokens):
    if self._is_terminated: return False
    for token in tokens:
        if not self.matcher.accept_token(token):
            logger.error("Failed to advance FSM for request %s for tokens %s.", request_id, token)
            return False
        self.num_processed_tokens += 1
    self._is_terminated = self.matcher.is_terminated()
    return True
```

逐 token accept；任一失败立即 return False。`num_processed_tokens` 用于跟踪 rollback 边界（虽然 matcher 自身也跟踪）。

### validate_tokens（行 173）

```python
def validate_tokens(self, tokens):
    accepted_tokens = []
    for token in tokens:
        if self.matcher.accept_token(token):
            accepted_tokens.append(token)
        else:
            break
    if len(accepted_tokens) > 0:
        self.matcher.rollback(len(accepted_tokens))  # 不推进 FSM
    return accepted_tokens
```

逐 token 测试，找到最长合法前缀；之后 rollback 到测试前的状态。spec decode validator 用此接口判断 draft 是否合法。

### rollback（行 190）

```python
def rollback(self, num_tokens):
    self.matcher.rollback(num_tokens)
    self.num_processed_tokens -= num_tokens
    self._is_terminated = self.matcher.is_terminated()
```

回滚 N 步 accept；spec draft 被拒绝时 SOM 调用。

### fill_bitmask

```python
def fill_bitmask(self, bitmask, idx):
    self.matcher.fill_next_token_bitmask(bitmask, idx)
```

调 xgrammar 内部接口，把 matcher 当前状态的合法 token bit 写入 `bitmask[idx]`。

### has_xgrammar_unsupported_json_features（行 225）

JSON Schema 校验工具：检查 `multipleOf` / `uniqueItems` / `contains` / `patternProperties` / `propertyNames` / 不支持的 string format 等 xgrammar 暂不支持的特性。前端 processor 在 `_validate_structured_output` 阶段调 `validate_xgrammar_grammar`。

### STRING_SUPPORTED_FORMATS

xgrammar 支持的 JSON string format 集合（行 207）：`email` / `date` / `time` / `date-time` / `duration` / `ipv4` / `ipv6` / `hostname` / `uuid` / `uri` / `uri-reference` / `uri-template` / `json-pointer` / `relative-json-pointer`。

## 与其它模块/系统配合

- [backend-types.md](backend-types.md)：实现 ABC。
- [manager.md](manager.md)：`StructuredOutputManager` 第一次 `grammar_init` 时按 `"xgrammar"` 实例化 backend。
- [request.md](request.md)：`compile_grammar` 入参由 `get_structured_output_key` 决定。
- [utils.md](utils.md)：`apply_grammar_bitmask` 调 `xgr.apply_token_bitmask_inplace` 应用 bitmask——这个函数本身不需要 XgrammarBackend 实例，只需要 bitmask + indices。
- [../sampler.md](../sampler.md)：bitmask 应用在 sampler 之前。
- [../rejection-sampler.md](../rejection-sampler.md)：spec decode + structured output 时 `validate_tokens` / `rollback` 与 spec draft 协同。
- [tokenizers](../../14-tokenizers-transformers/README.md)：`cached_tokenizer_from_config` + Mistral 特殊路径。
- [配置体系-StructuredOutputsConfig](../../10-config/README.md)（待补充）：`disable_any_whitespace` / `VLLM_XGRAMMAR_CACHE_MB` 环境变量。

## 历史版本演进

- **v0.6.x（V0）**：V0 xgrammar 是唯一 backend；`GrammarMatcher` 直接绑定到 request。
- **v0.7.0**：V1 `XgrammarBackend` 落地；grammar 编译移到 ThreadPoolExecutor；`max_rollback_tokens` 支持 spec decode 回滚。
- **v0.8.0**：Mistral tekken tokenizer 特殊路径；`VocabType.RAW` / `BYTE_FALLBACK`。
- **v0.9.0**：`VLLM_XGRAMMAR_CACHE_MB` env var 控制 cache 大小；`has_xgrammar_unsupported_json_features` 前端校验加强。
- **v0.10.0**：`STRUCTURAL_TAG` 加入，支持新（字符串）与 deprecated（StructuralTagItem list）两种 compile_structural_tag 接口。
- **v0.10.5+**：`disable_any_whitespace` 字段加入；`compile_regex_with_timeout` 防 ReDoS。
- **v0.11.0+**：Lark→EBNF 转换在 validate 阶段执行而非 compile 阶段（让 convert 错误更早报告）。
- **v0.12 / main**：稳定维护；CPU logits dtype 处理（fp32 强制转换，issue #31901）。

[← 返回结构化输出](../README.md)

## 参见

- [backend-types.md](backend-types.md)
- [backend-guidance.md](backend-guidance.md)：功能等价的替代 backend
- [manager.md](manager.md)
- [utils.md](utils.md)
