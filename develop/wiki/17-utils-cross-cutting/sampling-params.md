# 采样参数（SamplingParams）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 采样参数

本页覆盖 `vllm/sampling_params.py`（1103 行），定义生成类请求的公共入参类型 `SamplingParams`，以及配套枚举/数据类 `SamplingType`、`StructuredOutputsParams`、`RepetitionDetectionParams`、`RequestOutputKind`、`BeamSearchParams`。

## 是什么

### `SamplingParams`（`vllm/sampling_params.py:199`）

继承 `msgspec.Struct`（`omit_defaults=True`、`array_like=True`），是面向 API 用户的采样配置。关键字段（按源码顺序）：

- **基本采样**：`n`、`best_of`、`temperature`、`top_p`、`top_k`、`min_p`、`seed`。
- **停止条件**：`stop`（str 或 list[str]）、`stop_token_ids`、`ignore_eos`、`max_tokens`、`min_tokens`。
- **logprobs**：`logprobs`、`prompt_logprobs`；支持 `logprobs=-1` 仅对少量 token 计算。
- **输出形态**：`output_kind: RequestOutputKind`（默认 `CUMULATIVE`），决定流式 vs 累积 vs 仅最终。
- **重复惩罚**：`repetition_penalty`、`repetition_detection_params`（动态窗口检测）、`frequency_penalty`、`presence_penalty`。
- **结构化输出**：`guided_decoding`（`StructuredOutputsParams`），可在 `extra_kwargs` 中携带 regex/json/schema/choice/structural_tag。
- **词级控制**：`bad_words`、`allowed_token_ids`。
- **位置偏置/logits 处理**：`logits_processors`（用户自定义函数列表，签名见 `vllm/logits_process.py:10` 的 `LogitsProcessor` TypeAlias 与 `NoBadWordsLogitsProcessor`）、`prepend_tokens`、`skip_pruned_tokens` 等内部字段。
- **内部字段**：`request_id` 等由引擎注入的 hidden 字段（不在公开 API）。

### 配套类型

- `SamplingType`（`vllm/sampling_params.py:64`，`IntEnum`）：`GREEDY`/`SAMPLING`/`SAMPLING_LOGITS` 等，供采样器分派。
- `RequestOutputKind`（`:182`，`Enum`）：`FINAL_ONLY`/`CUMULATIVE`/`UNTOUCHED`/`DELTA`；`PoolingParams` 强制为 `FINAL_ONLY`。
- `StructuredOutputsParams`（`:72`，dataclass）：`regex`/`json`/`json_object`/`schema`/`choice`/`guidance`/`structural_tag`/`backend` 等结构化输出开关。
- `RepetitionDetectionParams`（`:146`）：动态重复检测窗口配置。
- `BeamSearchParams`（`:1089`）：独立 msgspec 结构，承载 beam search 专属参数（`max_tokens`/`ignore_eos`/`temperature` 等），用于非 sampling 路径。
- `validate_thinking_token_budget`（`:35`）：校验 thinking token 预算的工具函数。

## 为什么

- **统一 API 与引擎内部表示**：用户传 `dict`/OpenAI 参数 → API 层构造 `SamplingParams` → msgspec 序列化跨 ZMQ 进程到 EngineCore；`omit_defaults`+`array_like` 降低 IPC 体积与 GC 压力。
- **校验集中**：`_verify_args`/`derived` 等方法在入引擎前一次性校验语义（如 `top_k>=-1`、`min_tokens<=max_tokens`），并计算 `SamplingType`，避免采样器分支里重复判断。
- **结构化输出统一入口**：把 xgrammar/outlines/guidance 各后端差异收敛到 `StructuredOutputsParams`，由 [结构化输出](../06-sampling-decoding/README.md) 子系统读取。
- **与 `PoolingParams` 共享 `output_kind`**：复用枚举减少分歧。

## 怎么做

```python
from vllm.sampling_params import SamplingParams, RequestOutputKind
sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512,
                    guided_decoding={"json": schema_dict})
```

- 引擎内部经 `SamplingParams.from_engine_core_request` 恢复（msgspec），再 `clone()`/`verify()`。
- `all_parameters`/`valid_parameters`（对 `PoolingParams` 重要）模式也部分适用于此处（待核实具体字段列表）。
- 自定义 `logits_processors` 函数签名见 `LogitsProcessor` 别名（`vllm/logits_process.py:10`）：`(token_ids, logits) -> logits` 或带 prompt 版本。

## 与其它模块/系统配合

- [采样与解码](../06-sampling-decoding/README.md)：采样器消费 `SamplingParams`，按 `SamplingType` 分派。
- [结构化输出](../06-sampling-decoding/README.md)：读 `guided_decoding` 构造 grammar。
- [引擎核心](../01-engine-core/README.md)：`EngineCoreRequest` 携带 `SamplingParams` 跨进程；`InputProcessor` 与 `OutputProcessor` 读 `output_kind` 决定流式行为。
- [API 入口](../13-entrypoints/README.md)：OpenAI/Anthropic/LLM serve 把 HTTP body 转成 `SamplingParams`。
- [pooling-params.md](pooling-params.md)：复用 `RequestOutputKind`，校验为 `FINAL_ONLY`。
- [logits_process](outputs.md)：`bad_words` 经 `NoBadWordsLogitsProcessor` 实现。

## 历史版本演进

- **v0.5–v0.6**：`SamplingParams` 为 dataclass，字段较少；`best_of`/`n` 区分尚在；无结构化输出。
- **v0.7–v0.8**：改为 `msgspec.Struct` 以支持 v1 跨进程；引入 `guided_decoding`（最初为 `GuidedDecodingParams`）；`RequestOutputKind` 取代旧的 streaming 标志。
- **v0.9–v0.10**：加入 `min_p`、`repetition_detection_params`、`structural_tag`、thinking token budget、`bad_words` 重构；`BeamSearchParams` 独立化。
- **v0.11–main**：`StructuralOutputsParams` 合并多后端开关；`skip_pruned_tokens` 等 spec-decode 相关字段细化；具体字段引入版本（待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [pooling-params.md](pooling-params.md)
- [outputs.md](outputs.md)
- [采样与解码子系统](../06-sampling-decoding/README.md)
