[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Generate](README.md) > beam search

# generate/beam_search/（beam search 离线 + 在线）

> `beam_search/` 实现 vLLM 的 beam search：`offline.py` 的 `BeamSearchOfflineMixin` 给 `LLM.beam_search`（同步多步、带 structured output bitmask），`online.py` 的 `BeamSearchOnlineMixin` 给 serving 侧提供 ABC 钩子。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `BeamSearchOfflineMixin` | `vllm/entrypoints/generate/beam_search/offline.py:55` | `LLM.beam_search` 实现 |
| `beam_search` | `:58` | 离线入口 |
| `_beam_search_step` | `:193` | 单步 beam 推进 |
| `_init_beam_search_structured_output` | `:327` | 初始化 bitmask 约束 |
| `_build_beam_sampling_params` | `:397` | 构造每步 `SamplingParams` |
| `_bitmask_to_token_ids` | `:41` | bitmask → 允许 token id 列表 |
| `BeamSearchOnlineMixin` | `vllm/entrypoints/generate/beam_search/online.py:22` | serving 侧 ABC |

`beam_search`（`:58`）流程：

1. 多 prompt 并行预处理（renderer）。
2. 初始化每 prompt 的 `beam_width` 个候选 + 每 step 的 `SamplingParams`（`_build_beam_sampling_params`）。
3. 循环 `_beam_search_step`（`:193`）：每步生成 → 选 top beam_width → 应用 bitmask（`_init_beam_search_structured_output` 约束词表/JSON schema）→ 更新 beams。
4. 返回最终 beams。

`BeamSearchOnlineMixin`（`online.py:22`）是 ABC，让 `GenerateBaseServing` 子类可选实现 beam search 端点。

## 为什么

- **精确解码**：beam search 保留多候选、按累积 logprob 选优，比贪心/采样更精确，适合结构化任务（grammar/JSON）。
- **structured output 联动**：`_init_beam_search_structured_output` + bitmask 让每步 beam 只在允许 token 集；`_bitmask_to_token_ids` 反查。
- **离线优先**：beam search 计算密集、延迟高，主要离线评测用（`LLM.beam_search`）；在线仅留 ABC 钩子，不默认暴露。

## 怎么做

```python
from vllm import LLM, BeamSearchParams
llm = LLM(model="...")
outs = llm.beam_search(
    ["prompt1","prompt2"],
    BeamSearchParams(beam_width=4, max_tokens=64),
)
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| BeamSearchOfflineMixin | `vllm/entrypoints/generate/beam_search/offline.py:55` |
| beam_search | `vllm/entrypoints/generate/beam_search/offline.py:58` |
| _beam_search_step | `vllm/entrypoints/generate/beam_search/offline.py:193` |
| _init_beam_search_structured_output | `vllm/entrypoints/generate/beam_search/offline.py:327` |
| _build_beam_sampling_params | `vllm/entrypoints/generate/beam_search/offline.py:397` |
| _bitmask_to_token_ids | `vllm/entrypoints/generate/beam_search/offline.py:41` |
| BeamSearchOnlineMixin | `vllm/entrypoints/generate/beam_search/online.py:22` |

## 与其它模块/系统配合

- [llm.md](../llm.md)：`LLM` 继承 `BeamSearchOfflineMixin`。
- [base-serves.md](base-serves.md)：`GenerateBaseServing` 继承 `BeamSearchOnlineMixin`。
- [采样-结构化输出](../../06-sampling-decoding/structured-output/README.md)：bitmask、grammar、JSON schema。
- [引擎核心-LLMEngine](../../01-engine-core/engine-core-process.md)：`_beam_search_step` 经 `LLMEngine` 同步推进。

## 历史版本演进

- **v0.7（beam search 离线）**：`LLM.beam_search` 首版。
- **v0.8（在线 mixin）**：`BeamSearchOnlineMixin` ABC。
- **v0.11/main**：structured output bitmask（`_bitmask_to_token_ids`/`_init_beam_search_structured_output`）；`BeamSearchParams` 顶层别名。

## 参见

- [← 返回 Generate 首页](README.md)
- [llm.md](../llm.md)
- [base-serves.md](base-serves.md)
- [采样-结构化输出](../../06-sampling-decoding/structured-output/README.md)
