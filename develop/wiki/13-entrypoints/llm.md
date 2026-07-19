[← Wiki 首页](../README.md) > [API 入口](README.md) > LLM

# LLM 离线类

> `vllm.LLM` 是 vLLM 的同步离线推理 API：一次构造一个引擎（含分词器、模型、KV 缓存），用智能批处理把一批 prompt 转成 `RequestOutput`。它是脚本、评测、批处理的首选入口；在线服务请改用 `AsyncLLM`。

## 是什么

`LLM` 定义于 `vllm/entrypoints/llm.py:66`，继承自三个 mixin：

```python
class LLM(BeamSearchOfflineMixin, PoolingOfflineMixin, OfflineInferenceMixin):
```

| Mixin | 来源 | 提供方法 |
|---|---|---|
| `OfflineInferenceMixin` | `vllm/entrypoints/offline_utils.py:49` | `generate`、`chat`、`_preprocess_cmpl`、`_run_engine` |
| `PoolingOfflineMixin` | `vllm/entrypoints/pooling/offline.py:31` | `encode`、`embed`、`classify`、`score` |
| `BeamSearchOfflineMixin` | `vllm/entrypoints/generate/beam_search/offline.py:55` | `beam_search`、`_beam_search_step` |

`__init__`（`vllm/entrypoints/llm.py:176`）把构造参数转成 `EngineArgs`，再调 `LLMEngine.from_engine_args`（`vllm/entrypoints/llm.py:349`）拿到同步引擎核心 `LLMEngine`（`vllm/v1/engine/llm_engine.py`）。实例上挂了：

- `self.llm_engine`：`LLMEngine`
- `self.model_config` / `self.runner_type` / `self.supported_tasks`
- `self.renderer`：`OnlineRenderer`（同步路径）
- `self.input_processor`
- `self.chat_template`：经 `load_chat_template` 解析
- `self.request_counter` / `self.default_sampling_params`

`generate`（`vllm/entrypoints/llm.py:422`）签名：`generate(prompts, sampling_params=None, *, use_tqdm=True, lora_request=None, priority=None, tokenization_kwargs=None, mm_processor_kwargs=None)`。

## 为什么

- **离线与在线同源**：`LLM` 直接持有 `LLMEngine`（同步、同进程），而 `AsyncLLM` 持有跨进程的 `EngineCore`。二者共用同一套 renderer/input_processor/scheduler 代码，确保脚本里调 `llm.generate` 与 HTTP `v1/chat/completions` 行为一致。
- **多任务一个类**：通过 mixin 组合，`LLM` 同时承载 generate/chat（`OfflineInferenceMixin`）、池化任务（`PoolingOfflineMixin`，对 `runner_type="pooling"`）、beam search（`BeamSearchOfflineMixin`）。构造时按 `model_config.runner_type` 决定可用方法。
- **拒绝单进程 DP**：`__init__` 检测 `data_parallel_size>1` 且非 `external_launcher`/TPU 时抛错（`vllm/entrypoints/llm.py:298`），引导用户用多进程示例，避免挂起。
- **配置对象化**：`compilation_config`/`structured_outputs_config`/`profiler_config`/`attention_config` 既可传 dict 也可传实例，由 `_make_config`（`vllm/entrypoints/llm.py:267`）归一化，仅保留 `is_init_field` 字段，避免传非法键。
- **worker 序列化**：若 `worker_cls` 传的是类对象，用 `cloudpickle.dumps` 序列化（`vllm/entrypoints/llm.py:243`），规避跨进程 pickle 限制。

## 怎么做

### 基本生成

```python
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3-8B")
outs = llm.generate(["A: Hello", "B: Hi"], SamplingParams(max_tokens=16))
```

### Chat 接口

```python
from vllm import LLM
llm = LLM(model="...", chat_template="{...}")
msgs = [{"role": "user", "content": "讲个笑话"}]
outs = llm.chat(msgs)
```

`chat` 经 `OfflineInferenceMixin._preprocess_chat`（`vllm/entrypoints/offline_utils.py:158`）→ `OnlineRenderer.preprocess_chat` 把消息转 token，再 `_add_chat_requests` 入队。

### 池化任务

```python
llm = LLM(model="BAAI/bge-m", runner="pooling")  # runner=auto 也行
emb = llm.embed(["文本"])
```

`embed`/`classify`/`score` 由 `PoolingOfflineMixin`（`vllm/entrypoints/pooling/offline.py:199/244/289`）实现，统一入口 `encode`（`:51`）按 `pooling_task` 分派。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| 类定义 | `vllm/entrypoints/llm.py:66` |
| 构造函数 | `vllm/entrypoints/llm.py:176` |
| EngineArgs 组装 | `vllm/entrypoints/llm.py:305` |
| 引擎创建 | `vllm/entrypoints/llm.py:349` |
| `generate` | `vllm/entrypoints/llm.py:422` |
| `get_default_sampling_params` | `vllm/entrypoints/llm.py:415` |
| `reset_mm_cache` | `vllm/entrypoints/llm.py:411` |
| `from_engine_args` | `vllm/entrypoints/llm.py:386` |
| `get_world_size` | `vllm/entrypoints/llm.py:394` |
| DP 校验 | `vllm/entrypoints/llm.py:291` |

## 与其它模块/系统配合

- [offline-utils.md](offline-utils.md)：`generate`/`chat` 的实际实现。
- [chat-utils.md](chat-utils.md)：`load_chat_template`、`ChatCompletionMessageParam` 类型。
- [引擎核心-LLMEngine](../01-engine-core/engine-core-process.md)：`self.llm_engine` 即 V1 同步引擎。
- [多模态](../11-multimodal/README.md)：`mm_processor_kwargs` 透传给 MM processor；`reset_mm_cache` 清理。
- [LoRA](../12-lora/README.md)：`generate(..., lora_request=...)` 接受 `LoRARequest`。
- [pooling/offline.md](pooling/README.md)：`PoolingOfflineMixin`。
- [generate/beam-search.md](generate/beam-search.md)：`BeamSearchOfflineMixin`。
- [tokenizers-transformers](../14-tokenizers-transformers/README.md)：`get_tokenizer`、renderer。

## 历史版本演进

- **v0.5（V0 时代）**：`LLM` 持有 `LLMEngine`（V0），`generate` 同步驱动 step。
- **v0.7–v0.8（V1 迁移）**：切到 `vllm.v1.engine.llm_engine.LLMEngine`；引入 `runner`/`convert` 顶层参数；`swap_space` 被弃用并告警（`vllm/entrypoints/llm.py:224`）。
- **v0.9（mixin 拆分）**：把 generate/chat 实现挪到 `OfflineInferenceMixin`，pooling 挪到 `PoolingOfflineMixin`，beam search 挪到 `BeamSearchOfflineMixin`，`LLM` 类只剩构造与薄包装。
- **v0.10（spec 顶层别名）**：新增 `spec_method`/`spec_model`/`spec_tokens` 作为 `speculative_config` 的顶层快捷方式（`vllm/entrypoints/llm.py:217`）。
- **v0.11/main**：`kv_cache_memory_bytes` 字段（`:213`）允许显式指定 KV 显存；`renderer_num_workers` 在离线路径告警为 no-op（`:370`）；`enable_return_routed_experts` 支持 MoE 路由专家返回；`logits_processors` 接受类名字符串或类对象。

## 参见

- [← 返回 API 入口首页](README.md)
- [offline-utils.md](offline-utils.md)
- [chat-utils.md](chat-utils.md)
- [引擎核心-AsyncLLM 前端](../01-engine-core/async-llm-frontend.md)
- [pooling/README.md](pooling/README.md)
- [generate/beam-search.md](generate/beam-search.md)
