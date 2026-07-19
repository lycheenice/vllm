# gpu_ipc_memory.py · 前端 GPU 多模态解码内存池

[← Wiki 首页](../README.md) > [多模态](../README.md) > gpu-ipc-memory

## 是什么

`vllm/multimodal/gpu_ipc_memory.py` 实现前端（API server）进程 GPU 侧多模态解码的准入控制：`MultiModalGPUMemoryPool`（字节计数信号量）、`MultiModalGPUMemoryLease`（租约 handle + context manager）、`set_mm_gpu_ipc_pool` / `get_mm_gpu_ipc_pool`（进程全局单例）、`maybe_init_mm_gpu_ipc_pool`（按 `mm_ipc_gpu_memory_gb` 初始化）。仅当前端进程在 GPU 上做媒体解码（如 PyNvVideoCodec 视频解码）时启用，引擎进程会预先从 KV cache 预算里划出对应显存让"前端留出的空间物理存在"。

## 为什么

常规多模态解码（PIL 读图、PyAV 解码视频）在 CPU 上做，不占 GPU。但部分场景需要 GPU 加速：

- **PyNvVideoCodec 视频后端**：在 `V100/A100/H100` 上跑 NVDEC，吞吐远超 CPU，但解码出的帧直接落在 GPU 显存。
- **GPU 侧预处理**：`resize` / `normalize` 用 CUDA kernel 加速。
- **GPU 直接 IPC**：解码后的张量通过 `tensor.share_memory_()` 零拷贝传给 engine，无需 `host→device`。

问题是前端进程的 GPU 显存与 engine 进程共享同一物理卡，engine 已经为权重 / 激活 / KV cache 做了精细预算。若前端不加节制地 decode 大视频，会撑爆显存触发 OOM 影响全局。`MultiModalGPUMemoryPool` 用一个简单的字节计数信号量把前端 GPU 多模态用量圈在 `mm_ipc_gpu_memory_gb` 内部，并发请求 serialize 而非 oversubscribe；engine 侧在初始化时按 `mm_ipc_gpu_memory_gb` 等额减少 KV cache 预算，使总显存不超标。

## 怎么做

### MultiModalGPUMemoryLease

`MultiModalGPUMemoryLease`（`:27`）持有 `(pool, lease_id, nbytes)`：

- `release()`（`:39`）调 `pool._release(self)`，幂等。
- `__enter__` / `__exit__` 支持 `with` 语义，即使解码 raise 也能归还预算。

### MultiModalGPUMemoryPool

`MultiModalGPUMemoryPool`（`:49`）：

- `__init__(total_bytes)`：`total_bytes <= 0` raise。`_available = total_bytes`，`_cond = threading.Condition()`，`_outstanding: set[int]` 记录未归还 lease_id（双 release 时跳过）。
- `acquire(nbytes)`（`:75`）：负数 raise；超过 `total_bytes` raise（带"raise --mm-ipc-gpu-memory-gb or reduce input size"提示）；`with self._cond: while self._available < nbytes: self._cond.wait()` 阻塞；扣减 `_available`，分配 `lease_id`，加入 `_outstanding`。
- `_release(lease)`（`:98`）：`with self._cond: if lease_id not in _outstanding: return`（幂等）；discard、回加 `_available`、`notify_all` 唤醒等待者。
- `available_bytes`（`:71`）：当前可用，供可观测性读取。

线程安全：`acquire` 与 `release` 都来自 renderer 的多模态执行线程池，`Condition` 保证不丢通知。

### 全局单例与初始化

- `_GLOBAL_POOL`（`:108`）：module-level singleton。
- `set_mm_gpu_ipc_pool(pool)` / `get_mm_gpu_ipc_pool()`（`:111`/`:117`）：直接读写全局。
- `maybe_init_mm_gpu_ipc_pool(mm_ipc_gpu_memory_gb, api_process_count=1)`（`:122`）：
  1. `mm_ipc_gpu_memory_gb <= 0` → `set_mm_gpu_ipc_pool(None)` 返回 None（gating 禁用）。
  2. `api_process_count <= 0` raise。
  3. `total_bytes = int(mm_ipc_gpu_memory_gb * GiB_bytes) // api_process_count`——多 API 进程时均分预算（每进程独立 pool，但 engine 侧已扣总额）。
  4. 构造 `MultiModalGPUMemoryPool`，`set_mm_gpu_ipc_pool`，info log 含"per-process bytes / total GiB / API process count"。

## 与其它模块/系统配合

- **renderers / API server**：在媒体 URL → GPU 解码路径上，调用方先 `pool = get_mm_gpu_ipc_pool()`；若非 None，`lease = pool.acquire(estimated_nbytes)`，解码完成或异常时 `lease.release()`。具体调用点 `(待补充)`，应在 `media/video.py` 的 PyNvVideoCodec 后端或 renderer 的 multimodal executor 中。
- **engine 侧显存预算**：`vllm/engine/...` 在算 KV cache 大小时按 `mm_ipc_gpu_memory_gb` 扣减，使前端 pool 用的字节"物理空出来"。详细位置 `(待核实)`，应在 `vllm/v1/engine` 或 `vllm/platforms` 的显存探测路径。
- **配置**：`MultiModalConfig.mm_ipc_gpu_memory_gb`（默认 0，禁用）；`parallel_config.api_process_count` 决定均分份数。详见 [`10-config/multimodal-config.md`](../10-config/multimodal-config.md)。
- **video.py**：GPU 视频后端（`PyNvVideoCodecVideoBackend`）是主要消费方，详见 [video.md](video.md)。
- **logger / observability**：`available_bytes` 可被 metrics 抓取（`(待补充)` 是否已有 metric）。

## 历史版本演进

- **v0.10**：本文件首次出现，配合 GPU 视频解码后端（PyNvVideoCodec）合入。仅 `acquire`/`release` 阻塞语义 + 单进程单 pool。
- **v0.11（EVS）**：`api_process_count` 参数加入，支持多 API 进程均分预算；lease 加入 `_outstanding` 集合实现幂等 release，配合 `with` 块防 raise 泄漏。
- **main**：注释明确"engine carves the matching amount out of its KV-cache budget"；`acquire` 错误信息提示 `--mm-ipc-gpu-memory-gb`；`maybe_init_mm_gpu_ipc_pool` info 日志含 per-process bytes 与 total budget 对比。

[← 返回多模态首页](../README.md)

## 参见

- [video.md](video.md)：GPU 视频后端是 pool 的主要使用方。
- [media.md](media.md)：`MediaConnector` 的视频路径会触发 GPU 解码。
- [10-config/multimodal-config.md](../10-config/multimodal-config.md)：`mm_ipc_gpu_memory_gb` 配置。
- [08-platforms](../08-platforms/README.md)：GPU 显存预算与 KV cache 分配。
