# mnnvl_compat.py — CustomCommunicator (FlashInfer MNNVL 适配)

[← Wiki 首页](../../README.md) > [分布式](../../README.md) > [device-communicators](README.md) > mnnvl-compat

源码：`vllm/distributed/device_communicators/mnnvl_compat.py`（约 38 行）。一个极小的适配器类 `CustomCommunicator`，让 vLLM 的 `torch.distributed.ProcessGroup` 适配 FlashInfer `flashinfer.comm.mnnvl.CommBackend` 协议，使 FlashInfer 的 MNNVL all2all/allreduce 内核能复用 vLLM 已建好的 group。

## 是什么

`CustomCommunicator(CommBackend)`（`:13`，继承 `flashinfer.comm.mnnvl.CommBackend`）。构造 `__init__(group)`（`:14`）：仅 `self._group = group`。

实现的方法（满足 FlashInfer `CommBackend` 接口）：
- `Get_rank() -> int`（`:17`）：`self._group.rank()`。
- `Get_size() -> int`（`:20`）：`self._group.size()`。
- `allgather(data: int)`（`:23`）：`dist.all_gather_object([None]*size, data, group=self._group)`。
- `bcast(data, root)`（`:28`）：`dist.broadcast_object_list([data], src=root, group=self._group)`。
- `barrier()`：`dist.barrier(group=self._group)`。
- `Split(color, key) -> self`（`:37`）：返回 `self`（不真正 split，因 MNNVL workspace 已建在原 group 上）。

模块顶部 `assert has_flashinfer_nvlink_two_sided()`（`:10`）保证导入即检查依赖。

## 为什么

- **FlashInfer 接口差异**：FlashInfer 的 `MnnvlConfig`/`MoeAlltoAll`/`MnnvlMoe` 需要一个 `CommBackend` 对象做内部 collective（rank 查询、对象 allgather、bcast、barrier），但 vLLM 用的是 torch `ProcessGroup`。`CustomCommunicator` 做协议转换。
- **PascalCase 方法名**：FlashInfer 内部用 `Get_rank`/`Get_size`（Google 风格），与 Python 习惯不同；适配器封装此差异。
- **Split 返回 self**：FlashInfer 在某些路径会调 `Split(color, key)` 想拆子组，但 MNNVL workspace 已按原 EP group 分配（见 [all2all.md](all2all.md) 的 `FlashInferNVLinkTwoSidedManager` 注释 `:604`），不能再 split；返 self 保留原 group 即可。
- **allgather 整数**：FlashInfer bootstrap 阶段会 allgather 一些小整数（如各 rank 的 GPU 数、节点数），用 `all_gather_object` 足够，无需 tensor API。
- **导入即 assert**：减少运行时延迟暴露依赖缺失。

## 怎么做

### 使用链路

```mermaid
flowchart LR
    FI[FlashInferNVLink*Manager.__init__] -->|"given cpu_group (EP 组)"| AD[CustomCommunicator(cpu_group)]
    AD --> MC[MnnvlConfig comm_backend=AD]
    MC --> WS[MnnvlMoe.get_moe_workspaces mapping, mc]
    WS --> RUN[FlashInfer all2all/allreduce 内部 bcast/allgather/barrier]
```

### 集成点

`FlashInferNVLinkTwoSidedManager.initialize`（[all2all.md](all2all.md) / `all2all.py:600`）：`CustomCommunicator(self.cpu_group)` 传给 `MnnvlConfig`。`is_single_group` 上 MNNVL path 需要 `EP 组` 的 `cpu_group`（注释 `:604`）。

## 与其它模块/系统配合

- **[all2all](all2all.md)**：`FlashInferNVLinkTwoSidedManager`/`FlashInferNVLinkOneSidedManager` 直接消费。
- **[flashinfer-all-reduce](flashinfer-all-reduce.md)**：FlashInfer allreduce 路径同样需要 `TorchDistBackend(group)`（与本适配器概念同源，但 allreduce 走 `TorchDistBackend` 而非 `CustomCommunicator`）。
- **[cuda](cuda.md)**：通过 all2all manager 间接使用。
- **[08-platforms](../../08-platforms/README.md)**：仅 CUDA/MNNVL（GB200 NVL72）路径会用。

## 历史版本演进

- **v0.9**：随 `FlashInferNVLinkTwoSidedManager` 引入，初版仅 `Get_rank`/`Get_size`/`allgather`/`bcast`/`barrier`。
- **v0.10**：`Split` 返 self 明确化；`has_flashinfer_nvlink_two_sided` 导入即 assert。
- **v0.11/main**：随 FlashInfer 上游 API 变化微调（待核实）。

[← 返回 device-communicators 首页](README.md)

## 参见

- [all2all.md](all2all.md) — 主要消费方。
- [flashinfer-all-reduce.md](flashinfer-all-reduce.md) — 同族 FlashInfer all-reduce。
