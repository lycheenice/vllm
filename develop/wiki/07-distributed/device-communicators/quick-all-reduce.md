# quick_all_reduce.py — QuickAllReduce

[← Wiki 首页](../../README.md) > [分布式](../../README.md) > [device-communicators](README.md) > quick-all-reduce

源码：`vllm/distributed/device_communicators/quick_all_reduce.py`（约 367 行）。ROCm MI300 系列上的量化 all-reduce，源自 [quickreduce](https://github.com/mk1-project/quickreduce) 项目。与 `CustomAllreduce`（vLLM 自研）或 AITER custom AR 互补——同组内可同时启用，由调度链按 size/dtype 选择。由 [CudaCommunicator](cuda.md) 在 ROCm + `use_custom_allreduce` 时持有 `qr_comm` 字段。

## 是什么

### 常量与配置

- `QuickReduceRegime(Enum)`（`:30`）：`FP=0`/`INT8=1`/`INT6=2`/`INT4=3`/`INT3=4`/`NONE=5`，与 `csrc/quickreduce/quick_reduce.h` 对齐。
- `QuickAllReduce._SUPPORTED_WORLD_SIZES = [2, 4, 8]`（`:45`）。
- `_SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]`（`:46`）。
- `_QR_MIN_SIZE: dict[(dtype, world_size), list[int]]`（`:49`）：FP/INT8/INT6/INT4/INT3 各 regime 的最小张量尺寸（KB/MB）。例如 bf16+ws8 各 INT regime 都需 2048 MB（几乎不可达）。

### 构造（`:64`）

`__init__(group, device)`：
- `dist.get_world_size(group)`/rank 装定。
- `in_the_same_node_as` 判同节点。
- 全组同节点 + ws 在 `[2,4,8]` + platform.is_rocm() → `disabled=False`，否则 disable。
- 调 `ops.qr_max_size()`（`vllm._custom_ops` 自研 quickreduce 内核）分配 buffer + IPC handle 交换。

### 主要方法

- `should_quick_allreduce(inp)`：判 ws/dtype/size/`QuickReduceRegime` 匹配；选满足 `inp_size >= _QR_MIN_SIZE[(dtype,ws)][regime]` 的最优 regime（FP 优先，量化降级）。
- `quick_all_reduce(inp) -> torch.Tensor`：选 regime 后调 `vllm._custom_ops` 内核执行；INT regime 在传输前量化、传输后反量化，以带宽换精度损失可接受下的高吞吐。
- `capture()`：CUDA graph warmup。
- `destroy()`：释放 buffer。

## 为什么

- **ROCm MI300 NVLink 不如 NV 主导**：MI300 上 NCCL all-reduce 在大数据吞吐不差但 small-medium 延迟较高；quickreduce 用自研内核 + IPC 直连降延迟。
- **量化降带宽**：MI300 显存带宽虽高但互联受限于 Infinity Fabric；INT8/INT6/INT4 把传输量减半到 1/8，对大张量收效显著。`_QR_MIN_SIZE` 表让小张量仍走 FP（量化收益不足）。
- **INT3 仅 ws2**：注释 `:72` 说明 INT3 在更大 ws 上性能差，故 `_SUPPORTED_WORLD_SIZES` + `_QR_MIN_SIZE` 双向约束。
- **与 AITER 互补**：AITER custom AR 是 vLLM 调 AITER 库；quickreduce 是 vLLM 自有 C 扩展。两者启用条件不同（AITER 走 `VLLM_ROCM_USE_AITER_CUSTOM_AR`），调度链让 quickreduce 优先尝试小-中数据 niche。
- **同节点限定**：跨节点 Infinity Fabric 延迟高且无 IPC P2P，直接 disable。

## 怎么做

### regime 选择（伪码）

```python
def should_quick_allreduce(inp):
    if disabled or ws not support or dtype not support: return False
    inp_size = inp.numel() * inp.element_size()
    for regime_idx, regime in enumerate([FP, INT8, INT6, INT4, INT3]):
        if inp_size >= _QR_MIN_SIZE[(dtype,ws)][regime_idx]:
            self._chosen_regime = regime
            return True
    return False
```

实际选择倾向于前缀（FP 优先），即"够最小尺寸就走该 regime"。

### 调度链位置

`CudaCommunicator.all_reduce`（`cuda_communicator.py:285`）：NCCL symm mem 之后、FlashInfer/AITER/CUSTOM 之前尝试 quickreduce。MI300 上它是优先的小-中数据后端。

### CUDA graph 集成

`capture()` 预热内核；INT regime 的量化/反量化 kernel 也在 graph 内重放。

## 与其它模块/系统配合

- **[cuda](cuda.md)**：`CudaCommunicator.qr_comm`（`:92`），仅 ROCm + `use_custom_allreduce`（`:125`）。
- **[aiter-custom-all-reduce](aiter-custom-all-reduce.md)**：ROCm 另一 AR 后端，互补。
- **[all-reduce-utils](all-reduce-utils.md)**：P2P 探测复用。
- **[08-platforms](../../08-platforms/README.md)**：`current_platform.is_rocm()` + MI300 判定（`rocm_aiter_ops.is_custom_all_reduce_enabled`）。
- **[18-build-ci-testing](../../18-build-ci-testing/README.md)**：`csrc/quickreduce/` 是 C 扩展源。

## 历史版本演进

- **v0.7**：`QuickAllReduce` 引入，初版仅 FP + INT8，ws `[2,4,8]`。
- **v0.8**：INT6/INT4/INT3 加入；`_QR_MIN_SIZE` 表细化；与 AITER custom AR 互补关系明确化。
- **v0.9/v0.10**：MI300 调优；CUDA graph `capture()` 与调度链顺序调整。
- **v0.11/main**：`_SUPPORTED_DTYPES`/regime 表持续微调（待核实）。

[← 返回 device-communicators 首页](README.md)

## 参见

- [aiter-custom-all-reduce.md](aiter-custom-all-reduce.md) — ROCm AITER 对应。
- [custom-all-reduce.md](custom-all-reduce.md) — CUDA 上对应自研。
- [cuda.md](cuda.md) — 调度链总览。
