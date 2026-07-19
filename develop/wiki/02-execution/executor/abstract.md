[← Wiki 首页](../../README.md) > [执行层](../README.md) > [Executor](./README.md) > Abstract Executor

# Executor 抽象基类（abstract.py）

源码：`vllm/v1/executor/abstract.py`（380 行）

## 是什么

`Executor` 是所有 Executor 的抽象基类（ABC），定义了"控制面"的全部公共 API。它本身不持有 Worker 进程，只规定每个子类必须实现 `_init_executor()` 与 `collective_rpc()`，其余方法（`execute_model` / `sample_tokens` / `sleep` / `wake_up` / `add_lora` / `check_health` …）都建立在 `collective_rpc` 之上。

类属性：

- `uses_ray: bool = False` — 是否用 Ray 编排（`RayDistributedExecutor`/`RayExecutorV2` 置 True）。
- `supports_pp: bool = False` — 是否支持流水并行（`mp`/`ray`/`ray-v2` 置 True）。

## 为什么

- **统一控制面**：把"在某设备上跑一次方法"这件事抽象成 `collective_rpc(method, args, kwargs)`，无论下游是同进程直调、`Pipe.send`、`MessageQueue.enqueue` 还是 `ray.get` 都套同一套调用代码。
- **工厂分发**：`get_class()` (`abstract.py:48`) 根据 `parallel_config.distributed_executor_backend` 字符串/类型，惰性 import 对应子类，避免让上层强制依赖 Ray。
- **生命周期钩子**：`register_failure_callback` / `check_health` / `shutdown` / `sleep` / `wake_up` 让 `EngineCore` 不用关心具体后端的故障恢复语义。

## 怎么做

### 选择子类

`Executor.get_class(vllm_config)` 的判定顺序（`abstract.py:48-92`）：

| `distributed_executor_backend` | 选中类 |
|---|---|
| `type` 且是 `Executor` 子类 | 直接用 |
| `"ray"` | `RayExecutorV2`（若 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`）否则 `RayDistributedExecutor` |
| `"mp"` | `MultiprocExecutor` |
| `"uni"` | `UniProcExecutor` |
| `"external_launcher"` | `ExecutorWithExternalLauncher` |
| 其它字符串 | `resolve_obj_by_qualname` 反射 |

### 构造与初始化

`__init__` (`abstract.py:94`) 把 `VllmConfig` 各子配置暴露为属性，然后调用 `_init_executor()`（子类实现，负责拉起 Worker）。`is_sleeping` / `sleeping_tags` / `kv_output_aggregator` 在此预置。

### 控制面 API

`collective_rpc` 是真正的"广播+回收"原语，带两个 overload（block / non-block 返回 `Future`）(`abstract.py:152-202`)。子类必须重写。建议只传控制消息，张量走单独数据面（见 docstring `Note`）。

其上构建的高层 API：

- `initialize_from_config(kv_cache_configs)` (`abstract.py:118`)：先 `collective_rpc("initialize_from_config")`，再 `collective_rpc("compile_or_warm_up_model")`，并把每个 Worker 上报的 `CompilationTimes` 取 `max` 回写到 `vllm_config.compilation_config.compilation_time`，让主进程与 Worker 进程编译耗时一致。
- `execute_model(scheduler_output, non_block)` (`abstract.py:221`)：调 `collective_rpc("execute_model")`，取 `output[0]`。
- `sample_tokens(grammar_output, non_block)` (`abstract.py:241`)：同上，secondary path（用于 structured outputs 拆分）。
- `sleep(level)` / `wake_up(tags)` (`abstract.py:318`/`331`)：维护 `is_sleeping` 与 `sleeping_tags` 集合，`tags` 控制只恢复部分（如只 wake `weights` 不 wake `kv_cache`）。
- `add_lora` / `remove_lora` / `pin_lora` / `list_loras` (`abstract.py:292-308`)：用 `all(...)` / `assert sets[0]==...` 保证全部 Worker 一致。
- `profile` / `save_sharded_state` / `execute_dummy_batch` / `take_draft_token_ids` / `reset_mm_cache` / `reset_encoder_cache`：透传。
- `init_kv_output_aggregator(connector)` (`abstract.py:280`)：当存在 KV connector 时构造 `KVOutputAggregator`，在 `execute_model` 聚合多 Worker 输出。
- `supports_async_scheduling()` 类方法默认 `False`，仅 `UniProcExecutor`/`MultiprocExecutor`/`RayExecutorV2` 返回 True。

### 向后兼容

文件末尾（`abstract.py:371-380`）再次从 `uniproc_executor` import `UniProcExecutor` 与 `ExecutorWithExternalLauncher`，仅用于向后兼容外部 import 路径。

## 与其它模块/系统配合

- [引擎核心](../../01-engine-core/README.md)：`EngineCore` 直接持有 `Executor`，调 `execute_model` / `sample_tokens` / `sleep`。
- [Worker 数据面](../worker/worker-base.md)：`collective_rpc` 的目标方法是 `WorkerBase`/`WorkerWrapperBase` 上的方法名。
- [KV connector](../worker/kv-connector-mixin.md)：`init_kv_output_aggregator` 在执行时聚合跨 Worker 的 `KVConnectorOutput`。
- [编译子系统](../../09-compilation-ir/README.md)：`compile_or_warm_up_model` 聚合 `CompilationTimes`，回写主配置。
- [分布式](../../07-distributed/README.md)：`reinitialize_distributed` 默认 `NotImplementedError`，只有 `RayDistributedExecutor` 重写以支持弹性 EP 扩缩容。

## 历史版本演进

- **v0.7.0**：V1 引入 `Executor` ABC 与 `collective_rpc` 单一原语，取代 V0 的多种 `execute_model` 实现并存。
- **v0.8.0**：新增 `register_failure_callback` 钩子，把 Worker 进程死亡时的恢复逻辑下沉到 Executor。
- **v0.9.0**：`sleep`/`wake_up` 引入 `tags` 语义，支持部分恢复（weights vs kv_cache）。
- **v0.10.0**：`sample_tokens` 作为独立 API（与 `execute_model` 解耦），服务 structured outputs 并行化重构。
- **v0.11.0**：`init_kv_output_aggregator` + `KVOutputAggregator` 让 `execute_model` 在多 Worker 场景统一聚合 connector 输出。
- **v0.12 / main**：`get_class()` 增加 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` 分支、`external_launcher` 显式分支；`supports_async_scheduling` 类方法加入。

[← 返回执行层首页](../README.md)

## 参见

- [UniProc 单进程执行器](uniproc.md)
- [Multiproc 多进程执行器](multiproc.md)
- [Ray 经典执行器](ray.md)
- [Ray V2 执行器](ray-v2.md)
- [Worker 基类](../worker/worker-base.md)
