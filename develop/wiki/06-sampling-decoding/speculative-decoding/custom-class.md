[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > Custom Class Proposer

# Custom Class Proposer（自定义 drafter）

> 源码：`vllm/v1/spec_decode/custom_class_proposer.py`

---

## 是什么

`create_custom_proposer(vllm_config)` 是一个工厂函数，让用户用 FQCN（fully-qualified class name）字符串注册自定义 drafter 类。drafter 类的构造函数接受 `VllmConfig`，必须提供 `propose` 方法。这是 V1 spec decode 的"扩展点"——任何不内置的 drafter 算法都可通过此机制接入，无需修改 vLLM 主代码。

函数签名（`vllm/v1/spec_decode/custom_class_proposer.py:12`）：

```python
def create_custom_proposer(vllm_config: VllmConfig):
    """Load and instantiate a user-provided proposer class.
    The class path is read from ``speculative_config.model``
    (e.g., ``"my_module.MyCustomProposer"``). The class is
    imported, instantiated with *vllm_config*, and returned
    directly so the caller can use it without any wrapper.
    """
```

## 为什么

- **零侵入扩展**：用户无需 fork vLLM 即可加 drafter。只需在 `speculative_config.model = "my_pkg.MyProposer"` 即可让 ModelRunner 使用自定义类。
- **抽象兼容**：vLLM 不要求自定义 drafter 继承任何 base 类，只检查 `propose` 是 callable。让第三方库（如 Arctic Inference 的 SuffixDecodingProposer、第三方 EAGLE 训练框架等）能直接接入。
- **FQCN 解析**：用 `importlib.import_module(module_path)` + `rsplit(".", 1)` 解析 class name；支持多级 qualname（如 `pkg.subpkg.Module.Class`）的逐级 `getattr`。
- **构造签名**：用户类构造必须 `__init__(self, vllm_config: VllmConfig)`（不接受 device/runner 等额外参数）；这条限制让接口简单，不强制用户处理 ModelRunner 内部状态。
- **错误友好**：模块导入失败、类未找到、构造异常、缺 propose 方法——每种错误都有清晰的 ValueError/ImportError/AttributeError 包装，含原始异常 + 用户提示。

## 怎么做

### 创建流程

```python
def create_custom_proposer(vllm_config):
    spec_config = vllm_config.speculative_config
    backend = spec_config.model
    if "." not in backend:
        raise ValueError(...)
    module_path, class_name = backend.rsplit(".", 1)
    module = importlib.import_module(module_path)
    user_class = getattr(module, class_name, None)
    if user_class is None:
        raise AttributeError(...)
    instance = user_class(vllm_config)
    if not hasattr(instance, "propose") or not callable(instance.propose):
        raise AttributeError(...)
    return instance
```

注意 `rsplit(".", 1)` 只切最右一个 dot——这意味着模块路径中可以有任意多个点（如 `my_pkg.subpkg.module`），class name 是最后一段。如果用户类是嵌套类（`Module.OuterClass.InnerClass`），需要逐级 `getattr` 走 qualname；当前的 `rsplit(".", 1)` 不支持，但实际使用中嵌套类极罕见。

### 配置

`SpeculativeConfig.method = "custom_class"` + `SpeculativeConfig.model = "module.path.ClassName"`。在 `gpu_model_runner.py:583` 触发：

```python
if self.speculative_config.method == "custom_class":
    self.drafter = create_custom_proposer(self.vllm_config)
```

drafter 实例存入 `self.drafter`，与内置 drafter 走相同的 target verify + rejection sampler 流程。

### 实例化约束

- **构造**：`user_class(vllm_config: VllmConfig)`。
- **必选方法**：`propose`（callable）。
- **可选方法**：`load_model`、`dummy_run`、`initialize_attn_backend`、`prepare_next_token_ids_padded`、`prepare_inputs_padded` 等——若实现则 ModelRunner 会调用；若不实现则走默认 / 报错。

### 典型用例

- **Suffix decoding**：早期实现可能通过 custom_class 接入；现在已内置（见 [suffix.md](suffix.md)）。
- **第三方 EAGLE 变体**：研究机构训练的特殊 drafter 可走此路径。
- **基于规则的 drafter**：如 template-aware drafter、structured-output-aware drafter 等。

## 与其它模块/系统配合

- [suffix.md](suffix.md)：第三方库 `arctic_inference` 集成同样走 lazy import 模式。
- [ngram.md](ngram.md) / [ngram-gpu.md](ngram-gpu.md) / [eagle.md](eagle.md)：均不经过 custom_class 路径，但它们的接口设计可被自定义类参考。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：实例化后与内置 drafter 完全相同对待。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：scheduler 不感知 drafter 自定义与否；`SpecDecodeMetadata` 作统一接口。
- [配置体系-SpeculativeConfig](../../10-config/README.md)（待补充）：`method="custom_class"` + `model=...` 的 schema。

## 历史版本演进

- **v0.9.0**：`create_custom_proposer` 函数 landfall，作为 spec decode 的标准扩展点。
- **v0.9.5**：错误信息完善；明确 `propose` 必须是 callable 的检查。
- **v0.10.0**：与 padded drafter batch + cudagraph 不完全兼容（无 `prepare_inputs_padded` 实现时禁用 pad 模式）；用户文档说明使用限制。
- **v0.11.0+**：稳定维护；用户文档建议优先使用内置 method。

[← 返回投机解码](../README.md)

## 参见

- [suffix.md](suffix.md)
- [extract-hidden-states.md](extract-hidden-states.md)：另一特殊drafter（不真正 propose spec token）
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)
