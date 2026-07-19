# custom_all_reduce.py — CustomAllreduce

[← Wiki 首页](../../README.md) > [分布式](../../README.md) > [device-communicators](README.md) > custom-all-reduce

源码：`vllm/distributed/device_communicators/custom_all_reduce.py`（约 323 行）。vLLM 自研的低延迟 all-reduce，针对单节点 NVLink/PCIe P2P + 小到中等张量，使用 `vllm._custom_ops`（ CUDA 内核）+ IPC 句柄共享 GPU buffer，绕开 NCCL 的 launch 开销。由 [CudaCommunicator](cuda.md) 在 TP 组持有 `ca_comm` 字段。

## 是什么

### 启用判定与构造

- `_can_p2p(rank, world_size)`（`:31`）：逐对 `gpu_p2p_access_check(rank, i)`（或 `VLLM_SKIP_P2P_CHECK` 时退化为 `torch.cuda.can_device_access_peer`），决定能否走 P2P。
- `CustomAllreduce._SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]`（`:52`）。
- `__init__(group, device, max_size=8192*1024, symm_mem_enabled=False)`（`:55`）：
  - `dist.get_world_size(group)`/`rank` 等装定；
  - `full_group_ranks` + `in_the_same_node_as` 判同节点；
  - 全组同节点 + P2P 可用 + world_size 在支持列表 → 建自定义 communicator；否则 `disabled=True`。
  - `self.max_size` 来自 `CUSTOM_ALL_REDUCE_MAX_SIZES[cap][ws]`（[all-reduce-utils](all-reduce-utils.md)）或 `max_size` 参数。
  - 分配 GPU buffer + `cudaIpcMemHandle_t`（[cuda-wrapper](cuda-wrapper.md)）+ 跨进程交换 handle + `ops.meta_size()`/`ops_REGISTER` 等 C 扩展调用。
  - `symm_mem_enabled=True` 时优先走 symmetric memory 路径。

### 主要方法

- `should_custom_ar(inp)`：判 world_size/dtype/size/contiguity，复合 `is_weak_contiguous`（[utils](../utils.md)）。
- `custom_all_reduce(inp) -> torch.Tensor | None`：调 `vllm._custom_ops` 内核执行 all-reduce；若 size/dtype 不满足返回 None 让上层 fall through。
- `all_reduce(inp, *, out=None)`：包装 `custom_all_reduce`。
- `capture()`：CUDA graph 捕获前的 warmup，注册 buffer。
- `destroy()`：释放 IPC buffer、关闭 handle。

## 为什么

- **NCCL launch 开销**：小张量 all-reduce 受 NCCL kernel launch + barrier 开销主导；自研内核用 IPC 共享 buffer + 直接 load/store，把延迟从 ~数十 μs 压到 ~μs 级。
- **同节点 NVLink P2P**：`gpu_p2p_access_check` 保证 1-hop 直接读写对端 GPU 显存；NCCL 仍要走 communicator 抽象，自研内核可省。
- **世界_size 限制**：`[2,4,6,8]` 是内核 hard-coded warp/thread block 配置；其它 world_size 走 NCCL。
- **CUDA graph 友好**：自研内核无 NCCL 内部 stream 同步复杂度，捕获更稳定；`capture()` 保证 graph 内指针有效。
- **`symm_mem_enabled` 选项**：当上层也开了 `SymmMemCommunicator`，custom AR 复用其 symmetric buffer 减少分配（`cuda_communicator.py:120` 传 `symm_mem_enabled`）。
- **max_size 表**：不同 SM 架构 + world_size 有不同 buffer 上限，表化避免硬编码到内核。
- **非同节点 disable**：跨节点 P2P 不可用，IPC handle 无意义，直接 `disabled=True` 让调度链跳过。

## 怎么做

### 启用 + 调度

```mermaid
flowchart LR
    INIT[CudaCommunicator TP 组] -->|use_custom_allreduce + ws in [2,4,6,8]| P2P[_can_p2p check]
    P2P -->|全组同节点 + P2P ok| CA[CustomAllreduce enabled]
    P2P -->|否则| DIS[disabled=True]
    CA --> RUN[all_reduce 链路中: CudaCommunicator.all_reduce 调 ca_comm.should_custom_ar + custom_all_reduce]
```

### IPC 句柄交换

每个 rank 分配本地 GPU buffer，取 `cudaIpcMemHandle_t`（128 字节），经 `cpu_group`（gloo）`broadcast_object_list`/`all_gather_object` 交换给全员；各 rank `ops.register_buffer(handle)` 把对端 buffer 映射到自己地址空间。运行时内核直接读写对端 buffer。

### CUDA graph 集成

`capture()` 在 graph 捕获前调一次内核 warmup，让 NCCL/custom AR 的内部状态机初始化、buffer 指针固定；之后 `torch.cuda.graph` 内调 `custom_all_reduce` 可重放。

## 与其它模块/系统配合

- **[cuda](cuda.md)**：`CudaCommunicator.ca_comm` 字段（`:91` 初始化）；TP-only。
- **[cuda-wrapper](cuda-wrapper.md)**：`cudaIpcMemHandle_t` + `CudaRTLibrary` IPC API。
- **[all-reduce-utils](all-reduce-utils.md)**：`CUSTOM_ALL_REDUCE_MAX_SIZES` 表 + `gpu_p2p_access_check`。
- **[pynccl](pynccl.md)**：fall-through 兜底（custom AR 拒绝时走 PyNccl）。
- **[symm-mem](symm-mem.md)**：`symm_mem_enabled` 共享 buffer 选项。
- **[quick-all-reduce](quick-all-reduce.md) / [aiter-custom-all-reduce](aiter-custom-all-reduce.md)**：ROCm 上的对应物。
- **[09-compilation-ir](../../09-compilation-ir/README.md)**：`capture()` + custom op 注册使其可入 piecewise CUDA graph。
- **[03-model-execution](../../03-model-execution/README.md)**：`RowParallelLinear` all-reduce 受益。

## 历史版本演进

- **早期（v0.4/v0.5）**：`CustomAllreduce` 引入，支持 ws `[2,4,8]`，NVLink only。
- **v0.6**：`_SUPPORTED_WORLD_SIZES` 加 6；PCIe + `can_device_access_peer` 回退路径。
- **v0.7（v1）**：`VLLM_SKIP_P2P_CHECK` 加入避免启动时逐对探测慢；CUDA graph `capture()` 稳定。
- **v0.8**：`symm_mem_enabled` 选项接入，与 `SymmMemCommunicator` 共享 buffer。
- **v0.9**：`CUSTOM_ALL_REDUCE_MAX_SIZES` 按 SM cap（9.0/10.0/10.3）细化。
- **v0.10/main**：max_size 表随 GB200 调优；与 NCCL symm mem 调度链配合持续调整（待核实）。

[← 返回 device-communicators 首页](README.md)

## 参见

- [all-reduce-utils.md](all-reduce-utils.md) — max_size 表与 P2P 探测。
- [quick-all-reduce.md](quick-all-reduce.md) — ROCm 对应实现。
- [pynccl.md](pynccl.md) — 兜底 NCCL 路径。
