[← Wiki 首页](../../README.md) > [执行层](../README.md) > [Worker](./README.md) > UBatching

# UBatching：Disaggregated Batched Overlap（DBO）

源码：`vllm/v1/worker/ubatching.py`（241 行）+ `vllm/v1/worker/ubatch_utils.py`（265 行）+ `vllm/v1/worker/gpu_ubatch_wrapper.py`（537 行）

## 是什么

UBatching 是 vLLM 的**微批流水线**机制：在一步 decode 中把一个大 batch 切成 2 个（默认）"micro-batch"，让**通信与计算在两个 CUDA stream + 两个 Python 线程上交错执行**——A 在算的时候 B 在做 All-to-All / All-Reduce，反之亦然，把 EP/TP 通信延迟部分隐藏在计算后面。开关为 `parallel_config.enable_dbo`，触发后 `gpu_worker` 会把 `num_ubatches=2` 传给 `init_workspace_manager`。

三个文件的分工：

- **`ubatching.py`**：纯"并发原语"层。`UBatchContext` + 全局 `_THREAD_ID_TO_CONTEXT`/`_CURRENT_CONTEXTS`，定义 `dbo_yield` / `dbo_switch_to_comm/compute` / `dbo_register_recv_hook` 等线程级原语。
- **`ubatch_utils.py`**：纯"切批数据结构"。`UBatchSlice` + `maybe_create_ubatch_slices` + `split_attn_metadata`，把 token/请求切成 2 份并切片 attention metadata。
- **`gpu_ubatch_wrapper.py`**：GPU 上把前两个组合成"模型 forward 的 wrapper"。`UBatchWrapper` 包住模型 runnable，捕获/重放 per-ubatch cudagraph，并通过 `SMControlContextManager` 控制 DeepEP/DeepGEMM 的 SM 分配。

## 为什么

- **隐藏 MoE 通信**：DeepEP/DeepGEMM 的 all-to-all 在 EP>1 时延迟可观，DBO 让通信与计算重叠，TPS 显著提升（尤其大 batch decode）。
- **不破坏 cudagraph**：每个 ubatch 形状固定，`UBatchWrapper.cudagraphs: dict[int, CUDAGraphMetaData]` 按 `num_tokens` 缓存整张 combined graph，捕获期就把两个 ubatch 串起来一次录制。
- **CPU 线程级互斥**：`UBatchContext.__enter__/__exit__` + `threading.Barrier` + `cpu_wait_event/cpu_signal_event` 保证**任一时刻只有一个线程在跑 Python**（CUDA stream 是异步的，CPU 这边切换所有权）。
- **自适应门槛**：`ParallelConfig.dbo_{prefill,decode}_token_threshold` 控制只有足够大的 batch 才切微批；小 batch 直接 eager。
- **与 DP 强耦合**：`UBatchWrapper.__call__` assert `dp_metadata is not None`——DBO 当前只在 DP>1 且 EP>1 时启用（见 `_create_sm_control_context`）。

## 怎么做

### UBatchContext（`ubatching.py:20`）

每个 micro-batch 一个 `UBatchContext`，成员：

| 字段 | 作用 |
|---|---|
| `id` | 0 / 1 |
| `comm_stream` / `compute_stream` | 两条 CUDA stream（DBO 把通信放 comm_stream，计算放 compute_stream） |
| `forward_context` | 本 ubatch 的 `ForwardContext` 快照 |
| `ready_barrier` | `threading.Barrier(num_ubatches+1)`，主线程 + 2 ubatch 线程一起放行 |
| `cpu_wait_event` / `cpu_signal_event` | `threading.Event` 环形：ctx i 的 `signal` = ctx (i+1)%N 的 `wait` |
| `gpu_comm_done_event` / `gpu_compute_done_event` | CUDA event，跨 stream 同步 |
| `recv_hook` | 在 `__exit__` 末尾调用（用于 NCCL `irecv` 的延迟启动） |

生命周期：

- `__enter__`：注册 `_THREAD_ID_TO_CONTEXT[get_ident()]=self.id`；`ready_barrier.wait()`（同步所有线程就位）；`cpu_wait_event.wait()`（等上一环让自己跑）；`_restore_context()`（恢复 `forward_context._forward_context`）；切 stream 到 compute。
- `__exit__`：`maybe_run_recv_hook()`；`cpu_signal_event.set()`（唤醒下一环）；清理 `_THREAD_ID_TO_CONTEXT`。

控制流：

- `switch_to_comm_sync()` / `switch_to_compute_sync()`：切 stream + 记 event + 等 prev stream 事件。
- `yield_and_switch_from_compute_to_comm()` / `yield_and_switch_from_comm_to_compute()`：记 done-event → CPU yield（让另一线程跑）→ 恢复后切 stream → 等 prev done-event。
- `yield_()`：保留当前 stream 不切换的让出。

### 全局原语（`ubatching.py:160-200`）

`_register_ubatch_function(func)` 把 `UBatchContext` 方法包装成"看当前线程是否有 ctx"的函数：

- `dbo_yield` / `dbo_yield_and_switch_from_compute_to_comm` / `dbo_yield_and_switch_from_comm_to_compute`
- `dbo_switch_to_comm` / `dbo_switch_to_compute` / `dbo_switch_to_comm_sync` / `dbo_switch_to_compute_sync`
- `dbo_maybe_run_recv_hook` / `dbo_register_recv_hook`
- `dbo_get_previous_event(func, *args)`：在 ubatch compute stream 上执行 `func`（注册/等待 cuda event）。
- `dbo_enabled()` / `dbo_current_ubatch_id()`：当前线程是否在 ubatch 上下文中。

### make_ubatch_contexts（`ubatching.py:202`）

```python
cpu_events = [threading.Event() for _ in range(num_micro_batches)]
gpu_comm_done_events = [torch.Event() for _ in range(num_micro_batches)]
gpu_compute_done_events = [torch.Event() for _ in range(num_micro_batches)]
for i in range(num_micro_batches):
    ctx = UBatchContext(
        compute_stream=compute_stream, comm_stream=comm_stream,
        forward_context=forward_contexts[i],
        ready_barrier=ready_barrier,
        cpu_wait_event=cpu_events[i],
        cpu_signal_event=cpu_events[(i+1) % num_micro_batches],
        gpu_comm_done_event=gpu_comm_done_events[i],
        gpu_compute_done_event=gpu_compute_done_events[i],
        ...)
```

`_NUM_UBATCHES` 全局变量在此设置（默认 2）。要求 `num_micro_batches > 1`。

### ubatch_utils：切片工具

- **`UBatchSlice(request_slice, token_slice)`** (`ubatch_utils.py:13`)：一个 ubatch 覆盖的请求/ token 范围。
- **`is_last_ubatch_empty`** (`:32`)：判断切完两份后第二份是否空（避免空 ubatch）。
- **`check_ubatch_thresholds`** (`:38`)：根据 `dbo_{prefill,decode}_token_threshold` 决定是否应该切。
- **`maybe_create_ubatch_slices(should_ubatch, num_scheduled_tokens, num_tokens_padded, num_reqs_padded, num_ubatches, split_point)`** (`:63`)：按 `split_point` 切 token，用 `np.searchsorted` 找出每个 token 切点对应的请求切片；处理跨切片的请求；`_pad_out_ubatch_slices` 把最后一份 pad 到总 token 数。返回 `(ubatch_slices, ubatch_slices_padded)`。
- **`split_attn_metadata(ubatch_slices, common_attn_metadata)`** (`:251`)：对每个 UBatchSlice 调 `_make_metadata_with_slice`，调整 `query_start_loc`/`seq_lens`/`block_table`/`slot_mapping`，处理"splits_first_request"（同一请求跨 ubatch）和"splits_last_request"。

### UBatchWrapper（`gpu_ubatch_wrapper.py:113`）

构造期：

- `comm_stream = torch.cuda.Stream(device)`（与 compute_stream 分离）。
- `ready_barrier = Barrier(num_ubatches+1)`。
- `cudagraphs: dict[int, CUDAGraphMetaData]`：按 `num_tokens` 缓存 combined cudagraph。
- `cudagraph_wrapper = CUDAGraphWrapper(runnable, vllm_config, runtime_mode)`（非 DBO 路径回退）。
- `sm_control = _create_sm_control_context(vllm_config)`：DeepEP `set_num_sms` + DeepGEMM `set_num_sms`，DBO 时给 comm 分配固定 SM 数、compute 用剩余 SM；ROCm + DeepEP HT DBO 特例 `comm_sms=0`。
- `__getattr__` 透传到 `runnable`。

`__call__` 分三种路径：

1. **无 ubatch_slices**（`forward_context.ubatch_slices is None`，DBO 中止或非 DBO）：
   - 若 `cudagraph_runtime_mode == FULL` 且该 `num_tokens` 已有 ubatch graph 缓存：强制降为 `NONE`（避免无 ubatch 形状误触发非 ubatch 捕获）。
   - NONE/PIECEWISE：直接 `self.runnable(...)`；FULL：`self.cudagraph_wrapper(...)`。
2. **FULL + 形状未缓存**：`_make_ubatch_metadata(cudagraph_runtime_mode=NONE)` → `_capture_ubatches(ubatch_metadata, runnable)`，在两个线程里跑两个 ubatch 传入同一 `torch.cuda.graph()` 捕获。
3. **FULL + 已缓存**：`get_offloader().sync_prev_onload()` → `cudagraph_metadata.cudagraph.replay()` → 返回 `outputs`。
4. **非 FULL（PIECEWISE/NONE）+ ubatch**：`_make_ubatch_metadata(...)` → `_run_ubatches`，运行时双线程执行（无 cudagraph）。

`_run_ubatches` / `_capture_ubatches` 在两个 daemon 线程里 `with ubatch_contexts[i]:` 进入 `UBatchContext`，调 `runnable(**sliced_kwargs)` 或在 capture 路径加 `torch.cuda.graph(cudagraph)` context；线程末把输出 cat 回来。

`_make_ubatch_metadata` (`:343`) 构造每个 ubatch 的 `UbatchMetadata`（`context` + `input_ids`/`positions`/...切片 + `num_tokens`），并复制 `dp_metadata` 单 ubatch 版。

### 在 V1 GPUModelRunner 里的接入

- `_determine_batch_execution_and_padding`（见 [gpu-model-runner.md](gpu-model-runner.md)）在 DP>1 时调 `coordinate_batch_across_dp` 决定 `should_ubatch`。
- `maybe_create_ubatch_slices(should_ubatch, ...)` 切片。
- 切出的 `ubatch_slices` 写到 `forward_context.ubatch_slices`，由 `UBatchWrapper` 消费。
- V1 `gpu_ubatch_wrapper.UBatchWrapper` 替换原 `CUDAGraphWrapper` 作为模型 forward 的入口。

## 与其它模块/系统配合

- [GPU Worker](gpu-worker.md)：`enable_dbo` → `num_ubatches=2` 传给 `init_workspace_manager`，workspace 才会按 2 份缓冲预留。
- [GPU Model Runner V1](gpu-model-runner.md)：`_determine_batch_execution_and_padding` / `maybe_create_ubatch_slices` / `ubatch_slices_attn`。
- [cudagraph 捕获/重放](cudagraph-capture.md)：`UBatchWrapper` 内嵌 `CUDAGraphWrapper`，捕获期一次录制两个 ubatch。
- [分布式](../../07-distributed/README.md)：`get_ep_group().device_communicator.all2all_manager`、`DeepEP_high_throughput`、DP rank 协调；`DPMetadata.make` 每微批单独构造。
- [模型执行](../../03-model-execution/README.md)：MoE all-to-all 与 compute 在 comm_stream / compute_stream 上交错。
- [编译子系统](../../09-compilation-ir/README.md)：`CUDAGraphWrapper` 共享 graph pool；`BatchExecutionDescriptor` 不感知 ubatch 维度。
- [分布式 DeepEP/DeepGEMM](../../07-distributed/README.md)：`SMControlContextManager` 控制 all-to-all 与 GEMM 各自可用 SM 数。

## 历史版本演进

- **v0.10.0（实验）**：DBO 框架首次合入，仅 EP+DP+大 batch decode 场景；`ubatching.py` 的线程级原语落地。
- **v0.11.0**：`UBatchWrapper.cudagraphs` 全量捕获两个 ubatch 合一；`SMControlContextManager` 接入 DeepEP/DeepGEMM SM 分流；`coordinate_batch_across_dp` 协调 DP。
- **v0.12.0**：`dbo_{prefill,decode}_token_threshold` 可配；`is_last_ubatch_empty` 防空；RoCm + DeepEP HT 特例 `comm_sms=0`；`get_offloader().sync_prev_onload` 在 replay 前 sync。
- **main**：`UBatchContext.recv_hook` 延迟 NCCL irecv；`_pad_out_ubatch_slices` 处理 DP padding 后的最后一份；与 ModelRunner V2 的集成仍在推进 `(待补充)`。

[← 返回执行层首页](../README.md)

## 参见

- [GPU Model Runner V1](gpu-model-runner.md)
- [Model Runner V2](model-runner-v2.md)
- [cudagraph 捕获/重放](cudagraph-capture.md)
- [GPU Worker](gpu-worker.md)
- [分布式子系统](../../07-distributed/README.md)
