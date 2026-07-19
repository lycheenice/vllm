# ModelExpressModelLoader：ModelExpress 桥接

[← Wiki 首页](../../README.md) > [模型执行](../README.md) > [模型加载器](./README.md) > **ModelExpress**

> 源码：`vllm/model_executor/model_loader/modelexpress_loader.py`

---

## 是什么

`ModelExpressModelLoader`（`modelexpress_loader.py:33`）对应 `load_format="modelexpress"`，是一个**薄包装**：它本身不实现加载逻辑，而是在 `__init__` 时 `importlib.import_module("modelexpress.engines.vllm.loader")` 拿到外部 `MxModelLoader` 类，把所有方法调用委托给它。这使得 vLLM 能原生支持 ModelExpress 生态（不同的权重存储/转换后端）而无需把它的代码内联进 vLLM。

---

## 为什么

vLLM 的 loader 体系对所有 format 统一走 `BaseModelLoader` 契约，外部加载生态若想接入最干净的方式是写一个 `BaseModelLoader` 子类。但要求外部包把类搬进 vLLM 仓库不现实。ModelExpress loader 用"vLLM 内只放桥接、实现在外部包"的模式：vLLM 提供原生 `load_format` 入口（出现在 `LoadFormats` Literal 与分派字典里），用户 `pip install modelexpress` 后即可用 `--load-format modelexpress`。这是 out-of-tree loader 与 in-tree 入口共存的折中方案。

---

## 怎么做

### 模块定位

`modelexpress_loader.py:15` 定义 `_MODELEXPRESS_LOADER_MODULE = "modelexpress.engines.vllm.loader"`，并枚举 `_MISSING_MODELEXPRESS_MODULES` 集合。`importlib.import_module` 失败时，只有当 `ModuleNotFoundError.name` 落在该集合内才报"需 pip install modelexpress"的友好错误；否则透传原异常（避免吞掉真正的代码 bug）。

### 委托

`_load_modelexpress_loader`（`modelexpress_loader.py:40`）取 `module.MxModelLoader(load_config)` 存为 `self._loader`。后续 `download_model` / `load_weights` / `load_model` 全部 `self._loader.xxx(...)`：

```python
@instrument(span_name="Load model")
def load_model(self, vllm_config, model_config, prefix=""):
    model = self._loader.load_model(vllm_config=..., model_config=..., prefix=...)
    return model.eval()
```

注意 `load_model` 在 `BaseModelLoader` 里是模板方法，这里**重写**了它（不再走 base 的 dtype/device/`process_weights_after_loading` 模板），把控制权完全交给 `MxModelLoader`。`MxModelLoader` 自身需是 `BaseModelLoader` 子类（由 `register_model_loader` 的设计约束，但此处桥接不做 `issubclass` 校验，依赖外部包自觉）。

> `(待核实)` `MxModelLoader` 是否在 `process_weights_after_loading` 等收尾上与 vLLM base 模板一致，取决于 `modelexpress` 包实现，本仓库不持有其源码。

---

## 与其它模块/系统配合

| 协作方 | 关系 |
|---|---|
| `modelexpress`（外部包） | 提供 `MxModelLoader`；缺失时 `ImportError` |
| `model_loader/__init__.py` | `"modelexpress" → ModelExpressModelLoader` 注册在分派字典 |
| `vllm/tracing` | `@instrument(span_name="Load model")` 委托调用也被 trace |

---

## 历史版本演进

| 时间锚 | 变更要点 |
|---|---|
| main（#43105） | 新增 native `modelexpress` load format 与本桥接 loader |

---

## 参见

- [`dispatch.md`](dispatch.md) —— 分派表里 `modelexpress` 一行
- [`../README.md`](../README.md) —— 返回模型执行首页
