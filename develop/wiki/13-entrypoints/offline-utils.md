[← Wiki 首页](../README.md) > [API 入口](README.md) > offline_utils

# offline_utils（OfflineInferenceMixin）

> `vllm/entrypoints/offline_utils.py` 实现 `LLM` 类的生成/对话执行管线：把 `prompts` 与 `SamplingParams` 预处理成 `EngineInput`、入队 `LLMEngine`、轮询 `RequestOutput`、再交由 renderer 后处理。它是 `llm.generate`/`llm.chat` 的真正实现。

## 是什么

`OfflineInferenceMixin`（`vllm/entrypoints/offline_utils.py:49`）是一组私有方法的 mixin，被 `LLM`（与 `PoolingOfflineMixin`/`BeamSearchOfflineMixin`）继承。核心方法链：

| 方法 | 位置 | 职责 |
|---|---|---|
| `_resolve_mm_lora` | `:57` | 解析多模态 LoRA 默认适配 |
| `_preprocess_cmpl` / `_preprocess_cmpl_one` | `:109` / `:145` | completion 模式预处理（prompt/token ids） |
| `_preprocess_chat` / `_preprocess_chat_one` | `:158` / `:216` | chat 模式预处理（messages → tokens） |
| `_params_to_seq` / `_lora_request_to_seq` / `_priority_to_seq` | `:242` / `:258` / `:274` | 把参数展开到每个序列 |
| `_add_completion_requests` / `_add_chat_requests` | `:290` / `:387` | 批量打包请求入队 |
| `_run_completion` / `_run_chat` | `:326` / `:351` | 顶层调度（预处理 + `_render_and_run_requests`） |
| `_adjust_params_for_parsing` | `:447` | 按 parser 关系调整采样参数（如 tool parser 关 `temperature`） |
| `_render_and_run_requests` | `:494` | 渲染 + 加请求 + `_run_engine` |
| `_render_and_add_requests` | `:523` | 同步 renderer 渲染一批 |
| `_add_request` | `:552` | 单请求入队，绑 `request_id` |
| `_run_engine` | `:573` | 轮询引擎直到所有请求完成 |

类型别名 `_O`/`_R`（文件顶部）描述输出/渲染器泛型。

## 为什么

- **completion/chat 双管线**：completion（`generate`）接受 `str`/`TokensPrompt`/`MessagesPrompt`，chat（`chat`）接受 `list[message]`。两条路各自预处理，但收敛到同一 `_render_and_run_requests`，避免重复。
- **批量优先**：`_preprocess_cmpl` 一次性处理整批 prompt，renderer 同步并行多模态预处理；`_add_*_requests` 按参数对齐展开成逐序列请求，单次 `engine.add_request` 循环入队后由调度器集中批处理。
- **逐序列参数对齐**：`_params_to_seq` 把单一 `SamplingParams` 广播到所有序列，或把 list 按索引对齐；`_lora_request_to_seq`、`_priority_to_seq` 同理，保证长度不匹配即报错。
- **parser 参数纠偏**：`_adjust_params_for_parsing`（`:447`）若启用了 tool/reasoning parser，会按需关闭 `temperature`/`top_p` 等以保证结构化输出可解析。
- **同步轮询**：`_run_engine` 用 `while engine.has_unfinished_requests()` 调 `engine.step()` 收割输出，匹配 V1 `LLMEngine` 的同步推进语义。

## 怎么做

### generate 内部流（`llm.generate` → 本 mixin）

```
generate(prompts, sampling_params, ...)
  └─ _run_completion
       ├─ _preprocess_cmpl          # 一批 prompt → EngineInput
       │    └─ _preprocess_cmpl_one
       ├─ _add_completion_requests  # 展开 + _add_request
       └─ _render_and_run_requests
            ├─ _render_and_add_requests  # renderer.process
            └─ _run_engine               # engine.step 轮询
```

### chat 内部流（`llm.chat` → 本 mixin）

```
chat(messages, sampling_params, ...)
  └─ _run_chat
       ├─ _preprocess_chat          # messages → conversations + EngineInput
       │    └─ _preprocess_chat_one      # OnlineRenderer.preprocess_chat
       ├─ _add_chat_requests
       └─ _render_and_run_requests
            └─ _run_engine
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| Mixin 类 | `vllm/entrypoints/offline_utils.py:49` |
| completion 预处理 | `vllm/entrypoints/offline_utils.py:109` |
| chat 预处理 | `vllm/entrypoints/offline_utils.py:158` |
| 参数对齐 | `vllm/entrypoints/offline_utils.py:242` |
| completion 入队 | `vllm/entrypoints/offline_utils.py:290` |
| chat 入队 | `vllm/entrypoints/offline_utils.py:387` |
| 渲染+执行 | `vllm/entrypoints/offline_utils.py:494` |
| 引擎轮询 | `vllm/entrypoints/offline_utils.py:573` |
| 参数纠偏 | `vllm/entrypoints/offline_utils.py:447` |

## 与其它模块/系统配合

- [llm.md](llm.md)：`LLM` 继承本 mixin，`generate`/`chat` 委托给它。
- [chat-utils.md](chat-utils.md)：`_preprocess_chat_one` 调 `OnlineRenderer.preprocess_chat` → `parse_chat_messages`。
- [引擎核心-LLMEngine](../01-engine-core/engine-core-process.md)：`_run_engine` 推进的就是 `LLMEngine`。
- [多模态](../11-multimodal/README.md)：`_preprocess_cmpl` 处理 `MessagesPrompt` 的 `multi_modal_data`。
- [LoRA](../12-lora/README.md)：`_lora_request_to_seq` 把 `LoRARequest` 绑到每个序列。
- [采样-结构化](../06-sampling-decoding/structured-output/README.md)：`_adjust_params_for_parsing` 关联 parser 与采样参数。

## 历史版本演进

- **v0.5–v0.6（V0 内联）**：generate/chat 逻辑内联在 `LLM` 类，约 300 行。
- **v0.7–v0.8（V1 + mixin 抽离）**：迁到 V1 `LLMEngine`；把执行管线抽到 `OfflineInferenceMixin`，`LLM` 类瘦身。
- **v0.9（renderer 接管）**：把 tokenize/多模态准备工作从 mixin 移到 `OnlineRenderer`，mixin 只做"参数对齐 + 入队 + 轮询"。
- **v0.10（priority + lora 序列化）**：`_priority_to_seq` 支持按优先级入队；`_lora_request_to_seq` 对齐 list/单值。
- **v0.11/main**：`_preprocess_cmpl`/`_preprocess_chat` 拆出 `_one` 变体以便单条错误隔离；`_adjust_params_for_parsing` 加入 reasoning parser 分支。

## 参见

- [← 返回 API 入口首页](README.md)
- [llm.md](llm.md)
- [chat-utils.md](chat-utils.md)
- [引擎核心-LLMEngine](../01-engine-core/engine-core-process.md)
