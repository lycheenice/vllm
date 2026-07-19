# communication_op.py — TP 通信算子薄封装

[← Wiki 首页](../README.md) > [分布式](../README.md) > communication-op

源码：`vllm/distributed/communication_op.py`（约 43 行）。这是 `parallel_state` 与模型层之间的一层极薄封装，把"tensor model parallel"语义映射到 `get_tp_group()` 的对应方法，让模型代码不用直接拿到 GroupCoordinator。

## 是什么

五个公开函数（`communication_op.py:12` 起）：

| 函数 | 委派目标 | 用途 |
|---|---|---|
| `tensor_model_parallel_all_reduce(input_)` | `get_tp_group().all_reduce(input_)` | TP 全规约 |
| `tensor_model_parallel_all_gather(input_, dim=-1)` | `get_tp_group().all_gather(input_, dim)` | TP 全收集 |
| `tensor_model_parallel_reduce_scatter(input_, dim=-1)` | `get_tp_group().reduce_scatter(input_, dim)` | TP 规约分散 |
| `tensor_model_parallel_gather(input_, dst=0, dim=-1)` | `get_tp_group().gather(input_, dst, dim)` | TP 收集到 dst |
| `broadcast_tensor_dict(tensor_dict, src=0)` | `get_tp_group().broadcast_tensor_dict(tensor_dict, src)` | 广播 tensor dict；无 torch dist 时直接返回原 dict |

`__init__.py` 用 `from .communication_op import *` 把它们重新导出到 `vllm.distributed` 顶层，因此模型层既可写 `from vllm.distributed import tensor_model_parallel_all_reduce`。

## 为什么

- **语义命名**：模型/算子代码里写 `tensor_model_parallel_all_reduce` 比 `get_tp_group().all_reduce` 更可读，且与 Megatron/HF 习惯一致。
- **单一委派点**：所有 TP 通信集中在此文件，未来若要加 trace/metric/async 包装只改一处。
- **`broadcast_tensor_dict` 短路**：单卡或未初始化 distributed 时（`not torch.distributed.is_initialized()`）直接返回传入 dict，避免 `get_tp_group()` 断言失败。
- **与 PP 分离**：PP 通信（`send_tensor_dict`/`recv_tensor_dict`）不在此抽象，因为 PP 不归"all-reduce 一类"语义，由 `get_pp_group()` 直接调用。

## 怎么做

调用链示例（模型 `RowParallelLinear` forward）：

```python
# vllm/model_executor/layers/linear.py 内近似调用
out = tensor_model_parallel_all_reduce(matmul_out)
# 等价于
out = get_tp_group().all_reduce(matmul_out)
# 最终落到 CudaCommunicator.all_reduce，由其调度链选 NCCL/custom AR/symm_mem
```

`broadcast_tensor_dict` 的典型用法是 worker 把 `forward_context` 内某些小张量/buffer 同步给 TP 组其它 rank（如 samplers 间 logits 对齐）：

```python
if not torch.distributed.is_initialized():
    return tensor_dict           # 单卡短路
return get_tp_group().broadcast_tensor_dict(tensor_dict, src=0)
```

## 与其它模块/系统配合

- **[parallel-state](parallel-state.md)**：唯一依赖；`get_tp_group` 是其单例 `_TP`。
- **[03-model-execution](../03-model-execution/README.md)**：`RowParallelLinear`/`ColumnParallelLinear`/`VocabParallelEmbedding`/`QKVParallelLinear` 等层都调用本文件函数。
- **[09-compilation-ir](../09-compilation-ir/README.md)**：`all_reduce`/`all_gather` 经 `parallel_state.all_reduce`（带 fake）注册为 custom op，使 `torch.compile` 能正确 trace；本文件是其入口。
- **[device-communicators/cuda](device-communicators/cuda.md)**：实际算子语义在 `CudaCommunicator.all_reduce` 内部走 dispatch 链。

## 历史版本演进

- **早期**：函数集合与 Megatron `mpu` 一一对应；`broadcast_tensor_dict` 在 v0 早期就有。
- **v0.5/v0.6**：随 `GroupCoordinator` 重构后，这里的委派从手写 `dist.all_reduce(group=...)` 改为 `get_tp_group().all_reduce(...)`，便于 device_communicator 接管。
- **v0.7（v1）**：`tensor_model_parallel_all_gather` 用 concat-style all-gather（见 `base_device_communicator.all_gather`），因 stack-style 与 torch.compile 不兼容（`base_device_communicator.py:199` 注释）。
- **v0.8–main**：函数集稳定，未新增；可作为 stable API。（具体版本待核实）

[← 返回分布式首页](../README.md)

## 参见

- [parallel-state.md](parallel-state.md) — `get_tp_group` 的来源。
- [device-communicators/cuda.md](device-communicators/cuda.md) — `all_reduce` 真实执行链。
- [device-communicators/base.md](device-communicators/base.md) — `all_gather` 的 concat-style 实现。
