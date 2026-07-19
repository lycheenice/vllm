[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [serve/](README.md) > engine serving

# engine/serving.py（BaseServing 基类）

> `serve/engine/serving.py` 定义 `BaseServing`——所有具体 serving 类（`GenerateBaseServing`/`PoolingBaseServing`/`SpeechToTextBaseServing`/`ServingTokenization`）的共同祖先。它封装"模型校验 + 错误响应构造 + 引擎 client 持有"，把跨 task 的公共行为下沉。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `BaseServing` | `vllm/entrypoints/serve/engine/serving.py:29` | 基类 |
| `__init__` | `:30` | 持 `models`/`model_config`/`request_logger` |
| `_check_model` | `:40` | 模型/LoRA 校验（含动态 LoRA） |
| `_is_model_supported` | `:70` | base model + `VLLM_SKIP_MODEL_NAME_VALIDATION` |
| `create_error_response` | `:77` | 静态方法：构造 `ErrorResponse` |

`_check_model`（`:40`）流程：

1. `_is_model_supported(request.model)` → 通过返回 None。
2. `request.model in models.lora_requests` → 通过。
3. `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 且 `request.model` 非空 → `models.resolve_lora(model)`：
   - 返回 `LoRARequest` → 通过。
   - 返回 `ErrorResponse` 且 code==400 → 暂存为 `error_response`（继续报 400 而非 404）。
4. 否则返 404 `NotFoundError`。

`_is_model_supported`（`:70`）：`model_name` 空→True；`VLLM_SKIP_MODEL_NAME_VALIDATION`→True；否则 `models.is_base_model(model_name)`。

## 为什么

- **统一模型校验**：generate/pooling/speech 共用同一 `_check_model`，保证 LoRA 路由与 404 行为一致；子类只加 task 专属校验。
- **动态 LoRA 容错降级**：`resolve_lora` 返回 400（如 adapter 加载失败）时优先报该 400，而非笼统 404，提升调试体验。
- **`VLLM_SKIP_MODEL_NAME_VALIDATION`**：代理/网关场景下 `model` 字段可能被改写，跳过校验让请求透传。
- **静态 `create_error_response`**：所有 serving 类共享错误构造逻辑，与 `serve/utils/error_response.py` 工厂对齐 OpenAI schema。

## 怎么做

子类典型用法：

```python
class MyServing(BaseServing):
    async def handle(self, request):
        err = await self._check_model(request)
        if err: return err
        ...
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| BaseServing | `vllm/entrypoints/serve/engine/serving.py:29` |
| __init__ | `vllm/entrypoints/serve/engine/serving.py:30` |
| _check_model | `vllm/entrypoints/serve/engine/serving.py:40` |
| _is_model_supported | `vllm/entrypoints/serve/engine/serving.py:70` |
| create_error_response | `vllm/entrypoints/serve/engine/serving.py:77` |

## 与其它模块/系统配合

- [generate/base-serves.md](../generate/base-serves.md)：`GenerateBaseServing(BaseServing, BeamSearchOnlineMixin)`。
- [pooling/base.md](../pooling/base.md)：`PoolingBaseServing(BaseServing)`。
- [speech-to-text/base.md](../speech-to-text/base.md)：`SpeechToTextBaseServing(BaseServing)`。
- [openai/models.md](../openai/models.md)：`models.lora_requests`、`resolve_lora`。
- [utils.md](utils.md)：`create_error_response` 与 `error_response.py` 一致。

## 历史版本演进

- **v0.5–v0.6（内联）**：`_check_model` 散在各 handler。
- **v0.7（BaseServing 抽出）**：统一基类；`VLLM_ALLOW_RUNTIME_LORA_UPDATING` 分支。
- **v0.9（serve/engine 子包）**：移到 `serve/engine/serving.py`。
- **main**：`VLLM_SKIP_MODEL_NAME_VALIDATION`；`resolve_lora` 400 降级分支（`:58`）。

## 参见

- [← 返回 serve/ 首页](README.md)
- [generate/base-serves.md](../generate/base-serves.md)
- [openai/models.md](../openai/models.md)
