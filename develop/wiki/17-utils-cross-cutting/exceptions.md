# 异常类型（exceptions）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 异常类型

本页覆盖 `vllm/exceptions.py`（共 100 行），描述 vLLM 定义的自定义异常树。

## 是什么

`vllm/exceptions.py` 提供四个异常类，构成一棵小树：

```
ValueError
├── VLLMValidationError              # 请求级参数校验失败
└── VLLMUnprocessableEntityError     # 请求实体无法处理（如媒体 URL 404）
Exception
└── VLLMNotFoundError
    └── LoRAAdapterNotFoundError     # LoRA 适配器未找到
```

- `VLLMValidationError`（`vllm/exceptions.py:9`）：继承 `ValueError`，携带 `parameter`、`value` 两个 keyword-only 字段；`__str__` 追加 ` (parameter=..., value=...)`。
- `VLLMNotFoundError`（`vllm/exceptions.py:39`）：继承普通 `Exception`，语义上的"资源未找到"基类。
- `LoRAAdapterNotFoundError`（`vllm/exceptions.py:45`）：继承 `VLLMNotFoundError`，构造时按 `lora_name`/`lora_path` 生成固定 message。
- `VLLMUnprocessableEntityError`（`vllm/exceptions.py:69`）：继承 `ValueError`，API 层对应 HTTP 422；用于媒体 URL 不可达等场景，结构与 `VLLMValidationError` 一致。

## 为什么

- **可被 API 层精准映射 HTTP 状态码**：`VLLMValidationError` → 400，`VLLMUnprocessableEntityError` → 422，`VLLMNotFoundError` → 404，`LoRAAdapterNotFoundError` → 404；避免在 API 边界写大量 `isinstance` 字符串匹配。
- **同时继承 stdlib 基类**（`ValueError`/`Exception`）保证现有 `except ValueError` 代码不破。
- **关键字段结构化**：`parameter`/`value` 让前端能精确指出哪个入参错，而不仅是一段文字。

## 怎么做

```python
from vllm.exceptions import VLLMValidationError
raise VLLMValidationError(
    "max_tokens must be positive",
    parameter="max_tokens", value=-1,
)
```

API server（[API 入口](../13-entrypoints/README.md)）捕获后转为对应的 OpenAI 风格 error 对象。新增异常类时应：选择合适的 stdlib 基类、在 `__init__` 里固定结构化字段、重写 `__str__` 以便于日志检索。

## 与其它模块/系统配合

- [API 入口](../13-entrypoints/README.md)：异常 → HTTP 状态码映射的主消费者。
- [LoRA](../12-lora/README.md)：`LoRAAdapterNotFoundError` 由 LoRA resolver/manager 在加载失败时抛出。
- [多模态](../11-multimodal/README.md)：媒体拉取失败时由 input processor 转 `VLLMUnprocessableEntityError`。
- 引擎内部错误（调度/执行）一般不使用本文件类型，而是直接 `RuntimeError`/`AssertionError` 或 `EngineCoreError`（待核实具体类名）。

## 历史版本演进

- **早期 v0**：仅有零散 `ValueError`，无统一异常基类。
- **v0.6–v0.7**：引入 `VLLMValidationError` 与 `VLLMNotFoundError` 以支撑 OpenAI API 错误规范。
- **v0.8–v0.9**：`LoRAAdapterNotFoundError` 随 LoRA 子系统就位；`VLLMUnprocessableEntityError` 为多模态 URL 拉取失败引入（422 语义）。
- **v0.10–main**：结构基本稳定，新增字段以兼容方式扩展现有类；未引入新顶层异常（待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [API 入口异常处理](../13-entrypoints/README.md)
- [LoRA 子系统](../12-lora/README.md)
