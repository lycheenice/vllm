# tiering/fs/ — 本地文件系统二级层

[← Wiki 首页](../README.md) > [KV 卸载](README.md) > tiering-fs

源码目录：`vllm/v1/kv_offload/tiering/fs/`（3 文件 + `__init__.py`）。把 KV 块作为 `.bin` 文件写到本地目录（SSD/HDD/NFS/PVC），用 `O_DIRECT` + `readv/write` 直通 IO；后台 `DualQueueThreadPool` 让读优先与写优先两组线程互不饿死。可作"廉价跨实例共享 KV"手段——多个 vLLM 实例挂同一 PVC 即共享前缀缓存。

```
fs/
├── __init__.py
├── manager.py     # FileSystemTierManager + FsAsyncLookupManager
├── io.py          # store_block / load_block（O_DIRECT + 原子 replace）
└── thread_pool.py # DualQueueThreadPool + JobState
```

---

## 是什么

### 1. `FileSystemTierManager`（`manager.py`）

继承 [tiering.md](tiering.md) 的 `SecondaryTierManager`。构造参数（通过 tier_config 透传）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `root_dir` | 必填 | 根目录；由 `FileMapper.from_offloading_spec` 切出 `<root_dir>/<safe_model>_<hash>/` |
| `n_read_threads` | 16 | 读优先线程数 |
| `n_write_threads` | 16 | 写优先线程数 |

构造逻辑：

1. `_block_size = primary_kv_view.strides[0]`：从 primary tier memoryview 取每块字节数。
2. `FileMapper.from_offloading_spec(root_dir, spec, gpu_blocks_per_file=block_size_factor, parallel_agnostic=True)`：详见 [file-mapper.md](file-mapper.md)。
3. 若 `config.json` 不存在则写一份 `file_mapper.get_run_config()` 进去——给同目录的后续实例做 sanity check。
4. `DualQueueThreadPool(n_read_threads, n_write_threads, "vllm_kv_py_fs")`。
5. `FsAsyncLookupManager(tier=self, tier_type=self.tier_type)`：异步存在性查询。

#### `submit_store`（`:159`）

对每个 (key, bid) 生成 `functools.partial(store_block, file_name, primary_kv_view, bid*block_size, block_size)` 任务，`pool.enqueue_store(job_id, n, tasks)`。任务里 `bid * block_size` 是 primary memoryview 的字节偏移——IO 线程从这位置读到 FS。

#### `submit_load`（`:173`）

反向，`load_block(file_name, primary_kv_view, bid*block_size, block_size)` 写回 primary。

#### `lookup`（`:152`）

委托给 `FsAsyncLookupManager`：返回 None → `RETRY`；True → `HIT`；False → `MISS`。

#### 其它

- `get_finished_jobs`：拉 `pool.get_finished()` 转 `JobResult`。
- `drain_jobs`：`pool.wait_idle()` 阻塞至无在飞任务。
- `on_schedule_end` → `lookup_manager.flush()`：本期 batch 投递给后台线程。
- `on_request_finished` → `lookup_manager.cleanup(req_id)`：清只属于此请求的 lookup 状态。
- `shutdown`：lookup_manager.shutdown + pool.shutdown(wait=True)。

### 2. `FsAsyncLookupManager`（`manager.py:53`）

继承 [tiering.md](tiering.md) `AsyncLookupManager`，实现 `batch_lookup`：

```python
paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
if _HAS_BATCH_LOOKUP_C:
    return batch_lookup_C(paths)  # C 扩展，GIL 释放，批量 faccessat(2)
return (os.path.exists(p) for p in paths)
```

`vllm.fs_io_C` 是可选 C 扩展（待核实 build 配置）；存在性查询是 lookup 的瓶颈点，C 扩展比 Python `os.path.exists` 在大批量时快数倍。

### 3. `DualQueueThreadPool`（`thread_pool.py`）

双队列双线程组的线程池：

- `_load_q` 与 `_store_q` 两个 `deque`，共享一个 `threading.Condition`。
- `n_read_threads` 个线程 `load_priority=True`：先 drain `_load_q`，空了 fall back 到 `_store_q`。
- `n_write_threads` 个线程 `load_priority=False`：反过来，先 `_store_q` 后 `_load_q`。
- 默认 16+16=32 线程；load 与 store 都不会互相饿死（任一队列空了就帮另一边）。

#### `JobState`（`:21`）

`__slots__ = ("_job_id", "_n_tasks", "_completed", "_success", "_lock")`。每个 task 完成时调 `task_done(success)`，加锁递增 `_completed`；`_completed == _n_tasks` 时返回 `(True, overall_success)`。

#### 关键方法

- `enqueue_load(job_id, n_tasks, tasks)` / `enqueue_store(job_id, n_tasks, tasks)`：put 任务到对应队列，`notify(n_tasks)` 唤醒等待线程。
- `_worker(load_priority)`：循环 `condition.wait_for(lambda: stop or load_q or store_q)` → 从 primary 队列 popleft（空则 secondary）→执行 task → `task_done` → 若 job 完成，`_finished_q.append((job_id, success))` 并 `notify_all` 让 `wait_idle` 复检。
- `get_finished`：scheduler 唯一 popper，安全 `popleft`。
- `wait_idle`：`condition.wait_for(lambda: _inflight_jobs == 0)` 阻塞到所有 job 完成。
- `shutdown(wait=True)`：清队列、`_inflight_jobs=0`（取消的 task 不再 decrement，需手动复位以免 `wait_idle` 死锁）、`notify_all`、`join` 所有线程。

### 4. `io.py` 中的 `store_block` / `load_block`

两个底层 IO 回调，**在 worker 线程里**调用，直接对 `primary_kv_view` 切片读写。

#### `store_block(dest_path, buffer, offset, block_size)`

1. `os.path.exists(dest_path)`：已存在直接返回，跳过冗余写——多实例共享目录时的隐式去重。
2. `tmp_path = dest_path + _get_tmp_suffix()`：每个线程有 thread-local 唯一后缀（`_{random}.tmp`），避免多线程写同一 path 互踩。
3. `_ensure_dirs(dest_path)`：`os.makedirs(dirname, exist_ok=True)`。
4. `os.open(tmp_path, O_CREAT | O_EXCL | O_WRONLY | O_TRUNC | O_DIRECT, 0o644)`：`O_EXCL` 防止 tmp 名字撞车；`O_DIRECT` 绕过 page cache（KV 块就是大块冷数据，缓存它无意义且拖慢系统）。
5. `os.write(fd, view_slice)`：`view_slice = buffer.cast("B")[offset:offset+block_size]`——`cast("B")` 把多维 memoryview 平铺成字节 view，使切片用字节索引。短写 raise。
6. `os.replace(tmp_path, dest_path)`：原子 rename，让其他实例只在文件完整时看到它。
7. 异常时 `os.remove(tmp_path)` 清理。

#### `load_block(source_path, view, offset, block_size)`

1. `os.open(source_path, O_RDONLY | O_DIRECT)`。
2. `os.readv(fd, [view_slice])`：scatter-gather read 一次填满切片。短读 raise。
3. 异常时 `os.remove(source_path)`——不可读文件视作损坏直接删（下次会重算）。
4. `finally` 关 fd。

`O_DIRECT` 在 macOS 不存在（`io.py:12` 用 `getattr(os, "O_DIRECT", 0)` 兜底）。

---

## 为什么

- **持久化 + 跨实例共享**：FS tier 是把 KV 落到磁盘的最便宜手段；多实例挂同一 NFS/PVC 时通过 `FileMapper` 的内容寻址布局互相复用——比 P2P 简单，比 Obj tier 延迟低。
- **O_DIRECT 与原子 replace**：KV 块是 GB 级冷数据，过 page cache 既浪费内存又招致写回抖动；`O_DIRECT` 直接落盘。`O_EXCL` tmp + `os.replace` 保证其他实例只看到完整文件。
- **双队列线程池**：纯单队列会让批量 store 把 load 拖到队尾；纯分开两个池要么有一边闲、要么总线程数翻倍。`DualQueueThreadPool` 的"primary 队列优先、空了帮 secondary"是经典 trade-off。
- **异步 lookup**：FS 的存在性查询要 `faccessat(2)`，单 step 几百块串行查会阻塞 scheduler；后台线程批量查 + cache 结果是必然选择——同 step 内只查 dict。
- **`view.cast("B")` 字节切片**：`primary_kv_view` 可能 itemsize > 1（如 int64），但 IO 需要 byte 范围；`cast("B")` 零拷贝转字节 view。
- **C 扩展 `vllm.fs_io_C`**：`os.path.exists` 每次调一次 syscall 还需 Python/C 边界开销；批量 `faccessat` 一次 syscall 数百路径、释 GIL，是热 path 显著优化。

---

## 怎么做

### 配置示例

```json
{
  "spec_name": "TieringOffloadingSpec",
  "cpu_bytes_to_use": 8589934592,
  "secondary_tiers": [
    {
      "type": "fs",
      "root_dir": "/mnt/shared_pvc/kv",
      "n_read_threads": 16,
      "n_write_threads": 16
    }
  ]
}
```

### 启用跨实例共享

1. 所有 vLLM 实例 `PYTHONHASHSEED=0`（见 [file-mapper.md](file-mapper.md) 跨实例 hash 一致性章节）。
2. `root_dir` 指向所有实例可读写的共享存储。
3. model / `block_size` / dtype 等参数一致；若 tp 不同需保证 `parallel_agnostic` 条件成立（MLA / V2 model runner 会自动关闭）。
4. `config.json` 由首实例写入，后续实例看到即校验。

### 调优要点

- `n_read_threads + n_write_threads` 不要超过磁盘 NVMe 队列深度（一般 NVMe 32–64 队列合理）；HDD 应大幅减少。
- NFS 场景下 `O_DIRECT` 可能无效或被忽略，元数据 `getattr` 会慢——`n_read_threads` 调更高以掩盖 RTT。
- `block_size_factor > 1` 让单文件更大、IO 次数减少；FS tier 受益最明显。
- 跨实例共享时启动顺序无关——`config.json` race 通过 `O_EXCL` 写盘避免（但首启两实例同时写盘可能各自写一份相同内容，无数据损坏）。

---

## 与其它模块/系统配合

- 编排：[tiering.md](tiering.md) `TieringOffloadingManager.complete_store` 调本 tier `submit_store`；`lookup` 经 `AsyncLookupManager` 异步。
- 路径布局：[file-mapper.md](file-mapper.md) `FileMapper`。
- primary memoryview：[cpu.md](cpu.md) `SharedOffloadRegion.create_kv_memoryview()`。
- 跨子系统锚：[引擎核心-KV管理](../01-engine-core/kv-cache-management/README.md)（block_hashes 来源）、[配置-offload](../10-config/offload-config.md)（FS root_dir 概念上类似但不冲突）。

---

## 历史版本演进

| 版本 | 变化 |
|---|---|
| v0.10 | FS tier 首发：`FileSystemTierManager` + `DualQueueThreadPool` + `store_block/load_block`（O_DIRECT + 原子 replace）；同期 `AsyncLookupManager` 框架落地，FS 即接入 |
| v0.10.x | 加入 `parallel_agnostic` 与 `config.json` 写盘，正式支持跨实例共享 |
| v0.11 | P2P tier 出现后，FS tier 仍保留作为"持久/pvc 共享"场景的默认选项；与 P2P 在 `secondary_tiers` 列表里可共存 |
| main | C 扩展 `vllm.fs_io_C` 引入 `batch_lookup_C`（待核实 build 选项与内置分发策略）；`store_block` 加 multithread-safe tmp 后缀（之前用固定 `.tmp` 在多线程下会撞） |

---

[← 返回 KV 卸载首页](README.md)

## 参见

- [tiering.md](tiering.md)：本 tier 的注册位置与编排上下文。
- [file-mapper.md](file-mapper.md)：文件路径布局。
- [tiering-obj.md](tiering-obj.md)：类似的存储后端，但用对象存储。
