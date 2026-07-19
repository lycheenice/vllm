[← Wiki 首页](../../README.md) > [可观测](../README.md) > logging_utils/torch-tensor

# Torch Tensor（tensor 字符串截断打印）

> 源码：`vllm/logging_utils/torch_tensor.py`（10 行）

## 是什么

`torch_tensor.py` 提供一个极简 helper `tensors_str_no_data(arg)`，利用 `torch._tensor_str.printoptions` 上下文管理器把 tensor 的字符串表示**只出 shape/dtype 不出数据**，让排障日志不刷巨型 tensor。

实现全文：

```python
from typing import Any

def tensors_str_no_data(arg: Any):
    from torch._tensor_str import printoptions

    with printoptions(threshold=1, edgeitems=0):
        return str(arg)
```

## 为什么

- **避免日志洪水**：`logger.debug("logits=%s", logits)` 当 logits 是 `[1, 32000]` float tensor 会输出数万行数字；用户排障一次日志就把 stdout撑爆。本函数让输出形如 `Tensor(shape=[1, 32], device=cuda:0, dtype=torch.float32)`（具体格式视 torch 版本与 tensor size，待核实）。
- **`printoptions(threshold=1, edgeitems=0)`**：torch 内部 `printoptions` 是 `numpy` 风格的 context manager（`torch._tensor_str`）：
  - `threshold=1` — tensor 元素总数超过 1 即触发"summarize"（显示 `...`）。
  - `edgeitems=0` — 每边显示 0 个元素，即完全省略数据，仅保留 `Tensor(...)` 外壳与 metadata。
- **作用 `str(arg)` 不只是 tensor**：函数接受 `Any`，作用范围由 `printoptions` 决定——`str(tensor)` 受影响；`str(list_of_tensors)` 内部 tensor 也受影响（因 list `__str__` 调子项 `__repr__` 进入 tensor 行为）；非 tensor 对象不受影响。这让 helper 适合 `dict[str, Tensor]` / `list[Tensor]` 等容器。
- **lazy import**：`from torch._tensor_str import printoptions` 在函数内而非模块顶，避免 `import vllm.logging_utils` 必须先 `import torch`——vLLM 某些早期初始化阶段可能未导入 torch。
- **依赖 torch 私有 API**：`torch._tensor_str` 是私有模块（`_` 前缀），torch 版本升级可能改 API；vLLM 选择软依赖（lazy import），若 torch 改 API 此函数会 ImportError（影响调试而非运行）——风险局限在 DEBUG 场景。
- **不求值数据**：仅修改 `str()` 显示行为，不分配新内存、不读 device data——故 GPU tensor 也无需 host 拷贝即可打印外壳。

## 怎么做

**典型用法（排障时）**：

```python
from vllm.logger import init_logger
from vllm.logging_utils import tensors_str_no_data, lazy

logger = init_logger(__name__)

# 直接用：logits 外壳
logger.debug("logits: %s", tensors_str_no_data(logits))

# 配合 lazy：仅 DEBUG 输出时 str
logger.debug("logits: %s", lazy(lambda: tensors_str_no_data(logits)))

# 容器场景
state = {"logits": logits, "hidden": hidden_states}
logger.debug("state: %s", tensors_str_no_data(state))
# state: {'logits': Tensor(shape=[1, 32], ...), 'hidden': Tensor(shape=[1, 128, 4096], ...)}
```

**注意**：
- `printoptions` 上下文退出后恢复原 printoption；连续调用安全。
- 输出的 `Tensor(shape=...)` 格式由 torch 决定（不同 torch 版本可能含/不含 `device`/`dtype`，待核实）。
- 若 vLLM 未来升级 torch 到无 `_tensor_str.printoptions` 的版本，本函数会 ImportError——届时需切到 `torch.set_printoptions` 全局 API + try/finally restore。
- 仅在 logger 实际输出时输出——和 `lazy` 配合让非 DEBUG 模式零开销。

## 与其它模块/系统配合

- **[../logger.md](../logger.md)**：logger 实际 format 时调 `tensors_str_no_data(...)` 的返回值——通过 `%s` 自动 `__str__`。
- **[lazy.md](lazy.md)**：常用搭配 `lazy(lambda: tensors_str_no_data(t))`。
- **[dump-input.md](dump-input.md)**：`prepare_object_to_dump` 对 tensor 走单独分支 `f"Tensor(shape={obj.shape}, device={obj.device}, dtype={obj.dtype})"`——与本函数目的相同但实现独立（不依赖 torch 私有 API），避免 dump 路径在 torch 升级时崩。
- **`torch._tensor_str.printoptions`**：依赖私有 API，torch 版本兼容性待核实（torch >= 1.x 起稳定提供）。

## 历史版本演进

- **v0.5/v0.6（v0）**：无此模块（v0 排障多靠 user 自己写 `logits.shape`）。
- **v0.7（v1 落地）**：引入 `vllm/logging_utils/torch_tensor.py`；与 v1 worker 调试需求配套（v1 model_forward 多个 tensor 流转）。
- **v0.8–main**：未变。具体版本归属（待核实）。

[← 返回可观测首页](../README.md)

## 参见

- [logging-readme.md](logging-readme.md) — 子目录总览。
- [lazy.md](lazy.md) — 配合惰性求值。
- [dump-input.md](dump-input.md) — dump 路径对 tensor 的独立处理（不依赖此函数）。
