# 序列残留类型（sequence）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 序列残留类型

本页覆盖 `vllm/sequence.py`（62 行）。在 v0 时代它是引擎中枢类型的栖身之所；进入 v1 后绝大多数类型已迁出，本文件仅保留 `IntermediateTensors` 一个容器。

## 是什么

`IntermediateTensors`（`vllm/sequence.py:12`）：一个轻量 dataclass，本质是 `tensors: dict[str, torch.Tensor]` 的字典包装。提供：

- `__getitem__`：按 `str` 键取张量，或按 `slice` 切片返回同型 `IntermediateTensors`（对每个张量同步切片）。
- `__setitem__`、`items()`、`__len__`、`__eq__`（逐张量 `torch.equal`）、`__repr__`。
- `empty_like(intermediate_tensors)`：静态方法，按同形状新建空张量字典。

设计上的特殊点：作者**手写** `__init__`（而非用 dataclass 自动生成），并在注释（`vllm/sequence.py:24`）说明原因——为了让 Dynamo 知道 `IntermediateTensors()` 来自本文件；dataclass 生成的 `__init__` 由字符串 `eval` 而来，会丢失源文件信息，破坏 torch.compile 的守门。同时注释指出"不能用 `msgspec.Struct`，因为 Dynamo 不支持"。

## 为什么

- **Pipeline Parallelism（PP）跨阶段传递隐状态**：除最后一阶段外，模型前向需要把 hidden states / residuals 送往下一 stage，`IntermediateTensors` 就是这个张量包的标准容器。
- **必须被 torch.compile 识别**：PP 场景下图是分段编译的，容器构造若被 Dynamo 误判为 opaque，会丢失追踪能力，因此手工 `__init__`。
- **留在 `vllm/sequence.py` 而非 `vllm/v1/`**：被编译器-side 与多处模型代码直接 import，路径稳定可减小改动面（待核实是否有迁移计划）。

## 怎么做

```python
from vllm.sequence import IntermediateTensors
it = IntermediateTensors({"hidden": h, "residual": r})
h2 = it["hidden"]
slice_it = it[:tok]              # 切片
empty = IntermediateTensors.empty_like(it)
```

PP 执行器（[执行层](../02-execution/README.md)）在中间 stage 产出、末 stage 消费；`empty_like` 用于为通信缓冲预分配。

## 与其它模块/系统配合

- [执行层](../02-execution/README.md)：PP Executor / Worker 在 stage 边界交换 `IntermediateTensors`。
- [编译与 IR](../09-compilation-ir/README.md)：手工 `__init__` 让 Dynamo 正确追踪。
- [分布式](../07-distributed/README.md)：PP 通信（NCCL/all-reduce）的载荷载体。
- 与 v1 内部 `Request`（`vllm/v1/request.py`，见 [引擎核心·数据模型](../01-engine-core/data-model.md)）**无直接关系**——`Request` 是请求状态机，`IntermediateTensors` 是张量容器，二者分层。

## 历史版本演进

- **v0.5–v0.6**：`vllm/sequence.py` 体量大，定义 `Sequence`/`SequenceGroup`/`SequenceGroupMetadata`/`SequenceOutput`/`SequenceStatus` 等十余类，是 v0 引擎中枢。
- **v0.7（v1 引入）**：v1 用 `vllm/v1/request.py` 的 `Request` 替代 `SequenceGroup`；`sequence.py` 大规模瘦身，仅留 `IntermediateTensors`。
- **v0.8–v0.10**：因 torch.compile 追踪问题改手写 `__init__`；其余接口稳定。
- **v0.11–main**：无大改；`IntermediateTensors` 仍在 PP 路径关键链路上（待核实是否有进一步迁移到 `vllm/v1/` 的计划）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [引擎核心 · 数据模型](../01-engine-core/data-model.md)（v1 `Request`/`ModelRunnerOutput`）
- [执行层](../02-execution/README.md)
- [编译与 IR](../09-compilation-ir/README.md)
