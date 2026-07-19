[← Wiki 首页](../README.md) > [可观测](README.md) > logits-process

# Logits Process（v0 残留 logits processor）

> 源码：`vllm/logits_process.py`（121 行）

## 是什么

`vllm/logits_process.py` 是 v0 时代的 **logits processor** 工具集——在采样前修改 logits 张量的可调用函数链。v1 架构重构后，logits 处理迁入 [06-sampling-decoding/sampler.md](../06-sampling-decoding/sampler.md) 与 `vllm/v1/sample/` 子系统，本文件仅保留 `NoBadWordsLogitsProcessor` 一个具体类与 `LogitsProcessor` 类型别名，**已不在 v1 主采样路径上**。

| 名 | 行号 | 角色 |
|---|---|---|
| `LogitsProcessor` (TypeAlias) | `logits_process.py:10` | `Callable[[list[int], Tensor], Tensor] \| Callable[[list[int], list[int], Tensor], Tensor]` —— 接 previous tokens（+ 可选 prompt tokens）与 logits，返回修改后 logits。**仅类型别名，v1 不调用**。 |
| `get_bad_words_logits_processors(bad_words, tokenizer)` | `logits_process.py:21` | 工厂：把字符串 bad words 编码为 token id 列表，构造 `[NoBadWordsLogitsProcessor(...)]` 列表；处理 `add_prefix_space` 双路径（词开头/中间都禁） |
| `NoBadWordsLogitsProcessor` | `logits_process.py:48` | 实现禁词处理器：1-token 词用预计算 `word_bias` 屏蔽；多 token 词在采样每个位置检查前缀匹配，命中时把对应 last_token logit 设 `-inf` |

`NoBadWordsLogitsProcessor` 算法：

- `_init_word_bias(logits)`：1-token bad words 直接在 `word_bias` 张量设 `-inf`。
- `__call__(past_tokens_ids, logits)`：
  - 跳过 1-token 词（已在 word_bias）。
  - 跳过长度超过 `len(past_tokens_ids) + 1` 的词。
  - 对每词检查 `past_tokens_ids[-prefix_length:]` 是否等于 `bad_word_ids[:prefix_length]`，相等则在 `last_token_bias[bad_word_ids[-1]]` 累加 `-inf`。
  - `logits = logits + word_bias + last_token_bias`。

`get_bad_words_logits_processors` 的 `add_prefix_space` 双路径：

```python
for bad_word in bad_words:
    for add_prefix_space in [False, True]:
        prefix = " " if add_prefix_space else ""
        prompt = prefix + bad_word.lstrip()
        prompt_token_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        # 仅当 prefix space 产生不同首 token 时才追加第二条
        # 避免词开头/词中间同一 token 化重复
```

代码基于 `transformers` 的 `NoBadWordsLogitsProcessor` / `SequenceBiasLogitsProcessor`，注释 `logits_process.py:90-91` 指明出处。

## 为什么

- **v0 logits processor 框架残留**：v0 `SamplingParams.logits_processors: list[LogitsProcessor]` 让用户在采样前注入任意 logits 修改——`NoBadWordsLogitsProcessor` 是 v0 自带的具体实现之一。v1 把这套机制重写为 [06-sampling-decoding/sampler.md](../06-sampling-decoding/sampler.md) 内的更结构化流程，本文件不再被 v1 采样器调用。
- **仍保留的动因**：
  - v0 路径仍可通过 deprecated 入口调用（待核实具体 v0 兼容开关）。
  - 部分 SDK 用户可能在自定义 `LogitsProcessor` 中复用此 helper（`get_bad_words_logits_processors` 是便利工厂）。
  - 历史代码删除需 deprecation 周期；保留作为参考实现。
- **`(待核实)` v1 是否完全弃用**：v1 `SamplingParams.logits_processors` 字段仍存在但语义与 v0 不同（待核实 v1 是否用同一 `LogitsProcessor` 签名）；用户自定义 processor 在 v1 走不同注入路径（具体在 `vllm/v1/sample/sampler.py`，待核实）。
- **bad words 算法限制**：
  - 仅向前看 `len(bad_word_ids) - 1` 个 token——若 bad word 是 5-token 序列且生成第 5 个 token 时前 4 token 已匹配，则屏蔽第 5 token；但不影响已生成的 4 token。
  - 不处理跨请求的 bad word（每请求独立 processor）。
  - `vocab_size` 边界校验防 IndexError。
- **`add_prefix_space` 双路径**：很多 tokenizer（GPT-2/Llama）词首与词中 token 化不同（如 `" hello"` vs `"hello"`），需双路径都禁才能确保词开头与中间都屏蔽。
- **`_SMALLEST_LOGIT = -inf`**：而非大负数 `-1e9`——确保 softmax 后真正为 0；某些数值精度场景 `-1e9` 仍可能保留极小概率。

## 怎么做

**v0 风格用法**（保留兼容；v1 用户应优先看 [06-sampling-decoding/sampler.md](../06-sampling-decoding/sampler.md)）：

```python
from vllm import LLM, SamplingParams
from vllm.logits_process import get_bad_words_logits_processors
from vllm.transformers_utils.tokenizer import get_tokenizer

tokenizer = get_tokenizer(model_path)
bad_words = ["violence", "weapon"]
processors = get_bad_words_logits_processors(bad_words, tokenizer)

sampling_params = SamplingParams(
    logits_processors=processors,  # v0 风格注入
    temperature=0.7, max_tokens=128,
)
llm = LLM(model=model_path)
outputs = llm.generate(prompts, sampling_params)
```

**v1 推荐路径**：v1 用户若需 logits 修改（如 structured output / bad words）应优先考虑：

- 结构化输出：[../../10-config/structured-outputs-config.md](../10-config/structured-outputs-config.md) (`--structured-outputs-config backend`)
- v1 自定义 logits processor：写 `Callable` 满足 v1 `LogitsProcessor` 签名（待核实 v1 签名是否与本文件一致），通过 `SamplingParams.logits_processors` 注入——v1 采样器内调用（`vllm/v1/sample/sampler.py`，待核实）。

本文件的 `get_bad_words_logits_processors` 在 v1 下**可能不工作**——因其返回的 `NoBadWordsLogitsProcessor.__call__` 签名是 `(past_tokens_ids: Sequence[int], logits: Tensor) -> Tensor`，v1 是否传相同参数（待核实）。

## 与其它模块/系统配合

- **[06-sampling-decoding/sampler.md](../06-sampling-decoding/sampler.md)**：v1 采样器是 logits processor 的实际消费方（v0 与 v1 不同路径）。
- **[06-sampling-decoding/logits-processor.md](../06-sampling-decoding/logits-processor.md)**：v1 logits processor 详细文档；本文件是其 v0 历史对照。
- **[06-sampling-decoding/structured-output/](../06-sampling-decoding/README.md)**：v1 结构化输出（更现代的 logits 约束手段）。
- **`vllm/tokenizers.py::TokenizerLike`**：`get_bad_words_logits_processors` 参数类型。
- **`transformers.generation.logits_process`**：算法出处参考。
- **`SamplingParams.logits_processors`**（[10-config/scheduler-config.md](../10-config/scheduler-config.md) 待核实）：注入路径字段。

## 历史版本演进

- **v0.5/v0.6（v0）**：`vllm/logits_process.py` 是 v0 logits processor 主模块；含 `LogitsProcessor` 类型别名与多个具体类（`NoBadWordsLogitsProcessor` 等）；v0 `SamplingParams.logits_processors` 接这些 processor。
- **v0.7（v1 落地）**：v1 重写采样路径，`vllm/v1/sample/sampler.py` 接管 logits 处理；本文件大幅瘦身，仅留 `NoBadWordsLogitsProcessor` 与 `get_bad_words_logits_processors`，**移除其他 processor 类**（如 `TemperatureLogitsProcessor`/`TopPLogitsProcessor` 等，待核实具体移除清单）。
- **v0.8–main**：本文件稳定不变；v1 用户主路径不再调用。具体版本归属（待核实）。

> **(待核实)** 本文件是否在 v1 完全无主路径调用，或仍被 v0 compatibility shim 引用；如完全 dead code 应在某 v0.x 周期 deprecate。

[← 返回可观测首页](README.md)

## 参见

- [../06-sampling-decoding/sampler.md](../06-sampling-decoding/sampler.md) — v1 采样器，logits processor 的现路径。
- [../06-sampling-decoding/logits-processor.md](../06-sampling-decoding/logits-processor.md) — v1 logits processor 主文档。
- [../10-config/structured-outputs-config.md](../10-config/structured-outputs-config.md) — 替代 bad words 的结构化输出方案。
