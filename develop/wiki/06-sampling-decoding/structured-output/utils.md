[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [结构化输出](../README.md) > Structured Output Utils

# Structured Output Utils（工具函数）

> 源码：`vllm/v1/structured_output/utils.py`

---

## 是什么

`utils.py` 是结构化输出子系统的"杂项工具集"，包含四类函数：

1. **`apply_grammar_bitmask`**（行 85）：worker 端把 scheduler 传来的 bitmask 应用到 logits 的统一入口。
2. **Outlines vocabulary 工具**：`OutlinesVocabulary`、`get_outlines_cache_path`、`get_outlines_cache`、`get_outlines_vocabulary`、`_reduced_vocabulary`——为 OutlinesBackend 服务的 tokenizer 处理。
3. **Grammar 转换工具**：`grammar_is_likely_lark`、`convert_lark_to_ebnf`、`choice_as_grammar`——把用户输入转为后端可处理的形式。
4. **`compile_regex_with_timeout`**（行 47）：用 ThreadPoolExecutor 给 regex 编译加超时，防 ReDoS。

## 为什么

- **worker/scheduler 解耦**：bitmask 在 scheduler 端生成、worker 端应用——两侧都需 import 同一函数。`apply_grammar_bitmask` 作为统一入口让两边交换的数据结构稳定（numpy ndarray 走 MQ，worker 接收后 wrap 成 tensor）。
- **重排序需求**：scheduler 的 `structured_output_request_ids` 顺序与 worker 的 `input_batch.req_ids` 顺序**不一定一致**（scheduler 按 request 优先级 / 加入顺序排，worker 按 batch 槽位排）。`apply_grammar_bitmask` 必须按 worker 顺序重排 bitmask（行 122–140），并用 `out_indices` 让 `apply_token_bitmask_inplace` 只覆盖真正 constraint 的行。
- **Spec decode 多行展开**：spec decode 下每请求占 `1 + K` 行 logits（K 个 draft + 1 bonus），bitmask 也按 `1 + K` 行展开。`cumulative_offset` 跟踪每请求的起始 logit index，让多行 mask 对齐。
- **CPU/GPU 分支**：xgrammar 的 `apply_token_bitmask_inplace` 在 GPU 上需要 `index_tensor`（torch.Tensor），在 CPU 上接受 python list。CPU 路径还要处理 dtype（非 fp32 logits 需先转 fp32 → apply → 转回，issue #31901）。
- **regex ReDoS 防护**：用户提供的 regex 这可能含 nested quantifiers（`(a+)+b`）导致指数级 DFA 状态爆炸；`compile_regex_with_timeout` 用 `VLLM_REGEX_COMPILATION_TIMEOUT_S` 防止 worker 挂死。
- **Outlines vocabulary cache**：reduced vocabulary 计算成本高（遍历 vocab + tokenizer transform），缓存在 tokenizer 实例上避免重复。
- **Lark→EBNF 转换**：xgrammar 只接受 EBNF；用户可能提供 Lark 格式 grammar。`grammar_is_likely_lark` + `convert_lark_to_ebnf` 在前端 validate 阶段转换，让用户透明使用 Lark。

## 怎么做

### apply_grammar_bitmask（行 85）

```python
def apply_grammar_bitmask(scheduler_output, grammar_output, input_batch, logits):
    grammar_bitmask = grammar_output.grammar_bitmask  # numpy ndarray
    
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    struct_out_req_ids = set(grammar_output.structured_output_request_ids)
    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index
    
    out_indices = []
    sorted_bitmask_tensor = torch.full(
        (logits.shape[0], grammar_bitmask.shape[1]), -1,
        dtype=torch.from_numpy(grammar_bitmask[:0]).dtype, pin_memory=PIN_MEMORY,
    )
    sorted_bitmask = sorted_bitmask_tensor.numpy()
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + num_spec_tokens
    
    grammar_bitmask = sorted_bitmask_tensor.to(logits.device, non_blocking=True)
    skip_out_indices = len(out_indices) == logits.shape[0]
    
    if not logits.is_cpu:
        index_tensor = None
        if not skip_out_indices:
            index_tensor = async_tensor_h2d(out_indices, dtype=torch.int32, device=logits.device)
        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
        return
    
    # CPU path
    indices = None if skip_out_indices else out_indices
    if logits.dtype != torch.float32:
        logits_fp32 = logits.to(torch.float32)
        xgr.apply_token_bitmask_inplace(logits_fp32, grammar_bitmask, indices=indices)
        logits.copy_(logits_fp32.to(logits.dtype))
    else:
        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=indices)
```

要点：

- **scheduler → worker 重排序**：`struct_out_req_batch_indices` 建立 `req_id → logit_index` 映射，按 scheduler 顺序写入 `sorted_bitmask`。
- **unconstrained rows 全 1**：不在 `struct_out_req_ids` 的行保留初始 `-1`（所有 bit = 1，全部 token 合法）。
- **`out_indices` 优化**：当不是所有行都约束时，传 `out_indices` 让 xgrammar 只处理这些行；全部约束时 `skip_out_indices=True`，省一次索引传输。
- **CPU dtype 处理**：xgrammar CPU kernel 早期版本只支持 fp32 logits，故非 fp32（如 bf16）需先转换。

### compile_regex_with_timeout（行 47）

```python
def compile_regex_with_timeout(fn, pattern):
    timeout = envs.VLLM_REGEX_COMPILATION_TIMEOUT_S
    if timeout <= 0:
        return fn(pattern)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, pattern)
    try:
        result = future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise ValueError(
            f"Regex compilation timed out after {timeout}s. "
            "The pattern may be too complex or contain constructs that "
            "cause exponential state-space explosion (e.g. nested "
            f"quantifiers). Pattern: {pattern[:200]}"
        ) from None
    else:
        executor.shutdown(wait=False)
        return result
```

每次调用 new 一个 single-thread executor——避免 executor 共享导致 timeout 不准。`VLLM_REGEX_COMPILATION_TIMEOUT_S <= 0` 时禁用 timeout（开发场景）。

### Outlines vocabulary 工具

#### OutlinesVocabulary（行 177）

```python
class OutlinesVocabulary:
    def __init__(self, vocabulary):
        self.inner = vocabulary  # outlines_core.Vocabulary
        hex_str = hashlib.sha256(vocabulary.__repr__().encode("utf-8")).hexdigest()
        self._hash = int(hex_str, 16)
```

包装类，附加 SHA256 哈希作为 cache key——hash 来自 `vocabulary.__repr__()`，让相同 vocab 内容跨进程相同。

#### get_outlines_cache_path（行 193）

按以下优先级返回 cache 路径：

1. `OUTLINES_CACHE_DIR` env var。
2. `$XDG_CACHE_HOME/.cache/outlines`。
3. `~/.cache/outlines`（仅当 home dir 存在且不为 `/`）。
4. `<tempfile>/.cache/outlines`（容器内 fallback）。

#### get_outlines_cache（行 217）

```python
def get_outlines_cache():
    cache_dir = get_outlines_cache_path()
    if envs.VLLM_V1_USE_OUTLINES_CACHE:
        from diskcache import Cache
        logger.warning("Enabling outlines cache. This is an unbounded on-disk "
                       "cache. It may consume a lot of disk space and should "
                       "not be used with untrusted clients.")
        cache = Cache(cache_dir, eviction_policy="none", cull_limit=0)
        outlines_version = importlib.metadata.version("outlines_core")
        cached_version = cache.get("__version__", None)
        if cached_version != outlines_version:
            cache.clear()
        cache.set("__version__", outlines_version)
        return cache
    return LRUCache(maxsize=128)
```

choose diskcache（unbounded）或 in-memory LRU 128。`outlines_core` version 变化时清盘 cache，避免不兼容。

#### _reduced_vocabulary（行 245）

把 tokenizer.get_vocab() 转换为 `{bytes: [token_ids]}` 形式：

- 跳过 `all_special_tokens`。
- 处理 BPE / SentencePiece / Llama byte token (`<0xXX>`) / GPT2 byte token 等多种 token 表征。
- `unicode_to_bytes` map 用 `convert_slow_tokenizer.bytes_to_unicode()` 反转。
- EOS token id 被排除（如果与 vocabulary 重合）。
- `empty_token_ids` 列表收集"text 为空"的 token id（不参与 reduce vocabulary）。

返回的 `dict[bytes, list[int]]` 让 outlines_core 的 `oc.Vocabulary(eos_token_id, reduced_vocab)` 可构造。

#### get_outlines_vocabulary（行 314）

```python
def get_outlines_vocabulary(tokenizer):
    if hasattr(tokenizer, "_outlines_vocabulary"):
        return tokenizer._outlines_vocabulary
    reduced_vocab = _reduced_vocabulary(tokenizer)
    vocabulary = OutlinesVocabulary(oc.Vocabulary(tokenizer.eos_token_id, reduced_vocab))
    tokenizer._outlines_vocabulary = vocabulary
    return vocabulary
```

缓存到 tokenizer 实例——同 tokenizer 跨多请求只算一次。

### Grammar 转换工具

#### grammar_is_likely_lark（行 328）

简单规则：去掉 `#` 与 `//` 注释后，若所有非空行都不含 `::=`，认为是 Lark（Lark 用 `:` 定义规则）。

#### convert_lark_to_ebnf（行 360）

把 Lark grammar 转 EBNF：

- 第一行加 `root ::= <first_rule>`。
- Lark `:` 转 `::=`；`|` 起始的 alternation 行合并到当前 rule。
- `'...'` 字符串字面量转 EBNF 的 `"..."`。
- 收集并校验 `defined_rules ∪ referenced_rules`，未定义的 reference raise ValueError。

#### choice_as_grammar（行 490）

```python
def choice_as_grammar(choice: list[str]) -> str:
    escaped_choices = (escape_ebnf_string(c) for c in choice)
    grammar = "root ::= " + " | ".join(f'"{c}"' for c in escaped_choices)
    return grammar
```

把 vLLM 的 `choice` 参数（list of strings）转为 EBNF grammar。`escape_ebnf_string` 把 `"` 与 `\` 反斜杠转义。

## 与其它模块/系统配合

- [backend-xgrammar.md](backend-xgrammar.md)：`compile_regex_with_timeout`、`convert_lark_to_ebnf`、`grammar_is_likely_lark`、`choice_as_grammar` 都被 xgrammar backend validate / compile 路径调。
- [backend-outlines.md](backend-outlines.md)：`OutlinesVocabulary` / `get_outlines_cache` / `get_outlines_vocabulary` / `_reduced_vocabulary` 独占给 outlines backend 用。
- [manager.md](manager.md)：`grammar_bitmask` 在 manager 端构造返回 ndarray。
- [../sampler.md](../sampler.md)：`apply_grammar_bitmask` 在 sampler 之前调用。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：scheduler 把 `GrammarOutput` 包进 `SchedulerOutput`，与 spec_decode_tokens 一并跨进程传输。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：worker 端 `apply_grammar_bitmask` 在 `execute_model` 流程内调用，紧接 compute_logits 之后、sampler 之前。
- [tokenizers](../../14-tokenizers-transformers/README.md)：`_reduced_vocabulary` 与 `convert_slow_tokenizer.bytes_to_unicode` 紧耦合；`SPIECE_UNDERLINE` 特殊处理。
- [配置体系-envs](../../17-utils-cross-cutting/README.md)（待补充）：`VLLM_REGEX_COMPILATION_TIMEOUT_S` / `VLLM_V1_USE_OUTLINES_CACHE` / `VLLM_XGRAMMAR_CACHE_MB`。

## 历史版本演进

- **v0.7.0**：`apply_grammar_bitmask` landfall；最初仅 GPU 路径，CPU logits 后续补丁加入。
- **v0.8.0**：spec decode + bitmask 多行展开；`cumulative_offset` 与 `cumulative_index` 双指针机制。
- **v0.9.0**：`compile_regex_with_timeout` 加入，防 ReDoS；`VLLM_REGEX_COMPILATION_TIMEOUT_S` env var。
- **v0.9.5**：outlines backend 工具族加入；`_reduced_vocabulary` 处理 BPE / SentencePiece / Llama byte token 多种形式。
- **v0.10.0**：`convert_lark_to_ebnf` + `grammar_is_likely_lark` 加入，让 Lark grammar 透明转 EBNF（xgrammar 兼容）。
- **v0.10.5**：CPU logits dtype 转换路径加入（issue #31901）；`xgr.apply_token_bitmask_inplace` 接口稳定。
- **v0.11.0**：outlines cache 路径选择完善；`diskcache` 持久化路径 + version invalidate 机制。
- **v0.12 / main**：`async_tensor_h2d` 优化 indices 传输；`pin_memory=PIN_MEMORY` 让 CPU 端 bitmask 张量可异步 H2D。

[← 返回结构化输出](../README.md)

## 参见

- [backend-xgrammar.md](backend-xgrammar.md)
- [backend-outlines.md](backend-outlines.md)
- [manager.md](manager.md)
- [../sampler.md](../sampler.md)
