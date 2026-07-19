# sleep-mode.md — sleep mode 与 KV 卸载协作

[← Wiki 首页](../README.md) > [KV 卸载](README.md) > sleep-mode

源码：`vllm/device_allocator/sleep_mode_backend.py`（约 196 行）+ `vllm/v1/worker/gpu_worker.py` 的 sleep/wake_up 方法 + `vllm/v1/kv_offload/cpu/manager.py` 与 `tiering/manager.py` 的 `reset_cache`。本页描述 KV 卸载子系统如何与 sleep mode、`device_allocator` 协作——sleep 触发时腾空 GPU 显存，KV 卸载的 in-flight transfer 必须先 drain，避免 `_free_block` 与在飞 IO 抢内存。

> `vllm/device_allocator/` 与本子系统是**两个独立子系统**——sleep mode 主要处理 weight tensor 的 offload/restore，但 KV cache 块的清理由本子系统的 `reset_cache()` 负责。两者通过 `Worker.sleep/wake_up` 的过程串起来。

---

## 是什么

### 1. `SleepModeBackend` ABC 与 `SleepModeBackendFactory`

`SleepModeBackend`（`sleep_mode_backend.py:37`）抽象 sleep 机制。capability flag 全部 `@classmethod`，允许在不实例化的前提下 introspect——与 attention backend 约定一致。

| 成员 | 默认 | 含义 |
|---|---|---|
| `suspend(level=1)` | abstract | 释放 GPU 资源；level=1 仅 weights offload 到 host RAM、level=2 丢弃 weights（resume 时从 model source 重读） |
| `resume(tags=None)` | abstract | 恢复；`tags` 可选限定哪些 offload_tag 被恢复（如 `["weights"]`/`["kv_cache"]`） |
| `state()` | `"RUNNING"` | 返回 `"RUNNING"/"SUSPENDED"/"RESUMING"` |
| `is_supported()` | True | 当前平台/驱动是否支持 |
| `preserves_communicators()` | False | 集合通信（NCCL）是否跨 suspend/resume 保留 |
| `preserves_compiled_artifacts()` | False | torch.compile/JIT kernels 是否保留 |
| `preserves_graphs_with_communicators()` | False | 含 NCCL 的 CUDA graph 是否保留 |
| `supports_durable_storage()` | False | 是否能把 suspended state 持久化到盘/对象存储并在新进程恢复 |

### 2. `CuMemBackend`（默认实现）

`sleep_mode_backend.py:109`，包装 `vllm.device_allocator.get_mem_allocator_instance()`——在 CUDA 上返回 `CuMemAllocator`、在 XPU 上返回 `XpuMemAllocator`。

```python
def suspend(self, level=1):
    allocator = get_mem_allocator_instance()
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())

def resume(self, tags=None):
    allocator = get_mem_allocator_instance()
    allocator.wake_up(tags)
```

- level=1：只 offload `weights` tag，保留 KV cache（如果也 tag 了 kv_cache，可以靠 tags 选择性恢复）。
- level=2：`offload_tags=tuple()` = 全部。weights 在 resume 时从 model source 重读，最省 host RAM。
- `preserves_communicators=True`：NCCL buffers 在 allocator pool 外分配，suspend 不动它们。

### 3. `SleepModeBackendFactory`

`sleep_mode_backend.py:142`，与 [factory.md](factory.md) `OffloadingSpecFactory` 同构的注册表：

- `register_backend(name, module_path, class_name)` 懒加载。
- `get_backend_class(name)` name 不在注册表抛 ValueError。
- `create_backend(model_config)` 从 `model_config.sleep_mode_backend` 选并 `is_supported()` 校验。

内置注册（`:192`）：

```python
SleepModeBackendFactory.register_backend("cumem",
    "vllm.device_allocator.sleep_mode_backend", "CuMemBackend")
```

第三方 backend（CUDA process checkpoint、CRIU、durable snapshot/restore）经 `vllm.general_plugins` entry point 注册——RFC #34303 提出多 backend 并存，dispatch 路径不变（`/sleep` → engine → executor → worker）。

### 4. `Worker.sleep` / `Worker.wake_up`（`vllm/v1/worker/gpu_worker.py:187`）

```python
def sleep(self, level: int = 1) -> None:
    torch.accelerator.synchronize()
    free_bytes_before = torch.accelerator.get_memory_info()[0]
    if level == 2:
        # save non-persistent buffers before discard
        self._sleep_saved_buffers = {name: b.cpu().clone() for name, b in model.named_buffers()}
    self._get_sleep_mode_backend().suspend(level)
    torch.accelerator.synchronize()
    # 等待 free_bytes 增长（CUDA 异步释放）；ROCm 给 5s 宽限
    ...

def wake_up(self, tags=None) -> None:
    self._get_sleep_mode_backend().resume(tags)
    if len(self._sleep_saved_buffers):
        # restore non-persistent buffers
        for name, buffer in model.named_buffers():
            if name in self._sleep_saved_buffers:
                buffer.data.copy_(self._sleep_saved_buffers[name].data)
        self._sleep_saved_buffers = {}
    if tags is None or "kv_cache" in tags:
        self.model_runner.post_kv_cache_wake_up()
```

`_sleep_mode_backend` 懒初始化（`gpu_worker.py:174`）——首次 sleep 时 `SleepModeBackendFactory.create_backend(self.vllm_config.model_config)` 持久化到 worker。

### 5. KV 卸载侧：`reset_cache`

`OffloadingManager.reset_cache()` 是 [base.md](base.md) 接口的方法，由 `OffloadingConnectorScheduler` 在 sleep / weight update / resume 前**显式调用**。语义：清空所有 offloaded block，让下一 step 干净启动。

**`CPUOffloadingManager.reset_cache`**（[cpu.md](cpu.md) `cpu/manager.py:273`）：

```python
self._policy.clear()
self._num_evictable_cache_blocks = 0
self._free_list.clear()
self._num_allocated_blocks = 0
```

注释明确：scheduler 的 `_stale_job_threshold` 保证此时不会有 `complete_load/store` 进来，无需 lazy 清理；scheduler 也已 flush in-flight load job IDs 给 worker 后才让新 store 启动，避免跨方向竞态。

**`TieringOffloadingManager.reset_cache`**（[tiering.md](tiering.md) `tiering/manager.py:643`）更复杂：

1. 对每个 secondary tier `drain_jobs()`：阻塞至所有在飞 transfer 结束。FS tier 调 `pool.wait_idle()`、Obj tier 阻塞轮询 `check_xfer_state`、P2P tier 阻塞轮询 `_poll_once`。
2. `_process_finished_jobs()`：消费 drain 期间产生的完成事件，让 primary 的 ref_cnt 释放。
3. `_pending_load_submissions.clear()`：未提交的 promotion（lookup 期延后的）作废——它们的 `submit_load` 还没发，IO 没碰过内存，安全。
4. **保留 finished 请求状态**：遍历 `_req_state` 把 `pending_primary_stores=0`，is_finished 的请求让 secondary tier 转发 `on_request_finished`，然后从 dict 删——这样这些请求的 bookkeeping 不留到下个 run。
5. `primary_tier.reset_cache()`：清 primary 块表与 free_list。
6. **不 reset secondary tier**：FS/Obj/P2P 的持久数据要跨 reset 保留（盘/对象/远端 peer 那边的数据不变）；只是 supervisor 端的 in-flight jobs 被 drain。

注释强调："A stuck tier will block here visibly — preferable to silent corruption from reusing primary slots while a transfer is mid-copy."

---

## 为什么

- **正交但需要协作**：sleep mode 主要服务于 weight（CuMemAllocator pool 里的 weight tensor），但 KV 卸载的 CPU 槽与 GPU block 引用如果不清，sleep 后这些"幽灵引用"会让 wake_up 后的调度产生不一致（block_ids 复用撞在飞 transfer）。
- **`reset_cache` 显式调用而非 hook**：connector scheduler 在 sleep 路径上主动调 `manager.reset_cache()`，由 connector 控制时序——`sleep()` 在 worker、`reset_cache()` 在 scheduler，需在 IPC 顺序上对齐。
- **drain over abort**：sleep 时 in-flight RDMA/IO transfer 中途 abort 会留下"半写"内存——既影响 primary memoryview 一致性也影响 secondary tier 的对象状态。等完成 + 完成后清表更安全，代价是 stuck transfer 会阻塞 sleep，但 stuck 本身需要被注意到。
- **保留 secondary tier 持久数据**：FS/Obj/P2P tier 跨 sleep 应当保留——盘不会被 sleep 清掉、远端 peer 那边的 KV 仍有用；只有 primary CPU 槽（worker 私有）才清。
- **backend factory 抽象**：把 sleep 机制从 hard-wired CuMemAllocator 解耦——未来 CUDA process checkpoint（保留 graph + comm，跨进程 durable）等可平替而不改 `/sleep` API。
- **level 区分**：level=1 让 weights 仍在 host RAM、resume 快（适合同机多模型 swap）；level=2 完全丢弃、resume 慢但 host RAM 占用最低（适合长 idle 释放资源）。

---

## 怎么做

### 启用 sleep mode

`model_config.enable_sleep_mode=True` + `model_config.sleep_mode_backend="cumem"`（默认）。

通过 API：

```python
engine.sleep(level=1)   # weights to host RAM
# 或
engine.sleep(level=2)   # discard weights
```

engine 内部：scheduler 先 `connector.scheduler_manager` 调 `reset_cache()`（通过 `OffloadingConnectorScheduler` 走 `manager.reset_cache()`），executor 通知 worker 调 `worker.sleep(level)`。

### `<offload_tag>` 选择性 resume

CuMemAllocator 支持 `offload_tags`：在 weight 注册时打 `"weights"` tag、KV cache 注册时打 `"kv_cache"` tag。`resume(tags=["weights"])` 只取回 weights，KV cache 保持 sleep 状态——配合 KV 卸载的"零状态启动"场景：sleep 后 KV 留在 CPU/secondary tier，weights 恢复，按需通过 promotion 路径逐步回 GPU。

### 自定义 sleep backend

```python
class MyBackend(SleepModeBackend):
    def suspend(self, level=1): ...
    def resume(self, tags=None): ...
    @classmethod
    def preserves_communicators(cls): return True
    @classmethod
    def supports_durable_storage(cls): return True

SleepModeBackendFactory.register_backend("myck", "my_pkg.sleep", "MyBackend")

# 用户配 model_config.sleep_mode_backend="myck"
```

---

## 与其它模块/系统配合

- 调用入口：`vllm/v1/worker/gpu_worker.py:187` `Worker.sleep/wake_up`；scheduler 侧 sleep 路径（待核实：EngineCore sleep 命令转 scheduler reset_cache 调用顺序，约在 `vllm/v1/engine/core.py` 或 `vllm/v1/executor/`）。
- KV 卸载 reset：[cpu.md](cpu.md) `CPUOffloadingManager.reset_cache`、[tiering.md](tiering.md) `TieringOffloadingManager.reset_cache`（含 `drain_jobs`）。
- [平台-device_allocator](../08-platforms/device-allocator.md)（待补充）：`CuMemAllocator`/`XpuMemAllocator` 详细行为。
- [配置-model-config](../10-config/model-config.md)：`enable_sleep_mode`/`sleep_mode_backend` 字段。
- `vllm.device_allocator.cumem.CuMemAllocator` / `xpumem.XpuMemAllocator`：内部 allocator，sleep/wake_up 在这里实现。
- 跨子系统锚：[03-model-execution sleep/预热](../03-model-execution/README.md) 的 sleep/restore 模型加载路径。

---

## 历史版本演进

| 版本 | 变化 |
|---|---|
| v0.6（待核实） | sleep mode 引入，硬编码 `CuMemAllocator.sleep/wake_up`；KV 卸载（旧 `simple_kv_offload`）的 reset 状态由 scheduler 的 connector 在 sleep 前调用 |
| v0.7–v0.8 | level 1/2 区分；`offload_tags` 让选性恢复成为可能（"weights" / "kv_cache" 分离） |
| v0.9 | 新管线 `kv_offload/` 出现，`OffloadingManager.reset_cache()` 成为 sleep 协作的标准接口 |
| v0.10 | `TieringOffloadingManager.reset_cache` 加入：调每 secondary tier `drain_jobs()` 防止 cross-tier IO 与 primary 槽释放竞态——这是 sleep mode 与多层 tiering 联动后必须加的同步屏障 |
| v0.10.x | RFC #34303 启动：`SleepModeBackend` ABC + `SleepModeBackendFactory` 注册表抽象落地，原 `CuMemAllocator` 包装成 `CuMemBackend`；保留 `preserves_communicators=True` 等 capability flag 让 executor 决定是否 reinit NCCL |
| main | `tags` 选择性 resume（"kv_cache" tag）；第三方 backend 通过 `vllm.general_plugins` entry point 注册（CUDA checkpoint / CRIU / durable snapshot 路线，待核实主线落地时间） |

---

[← 返回 KV 卸载首页](README.md)

## 参见

- [cpu.md](cpu.md)：`CPUOffloadingManager.reset_cache`。
- [tiering.md](tiering.md)：`TieringOffloadingManager.reset_cache` 与 `drain_jobs`。
- [base.md](base.md)：`OffloadingManager` 抽象接口含 `reset_cache` 默认实现。
- [10-config/model-config](../10-config/model-config.md)：`enable_sleep_mode`/`sleep_mode_backend`。
