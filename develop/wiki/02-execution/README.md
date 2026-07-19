[← Wiki 首页](../README.md)

# 执行层子系统（02-execution）

> 执行层是 vLLM V1 引擎中"调度器输出 → 设备算子"之间的承上启下层。它把每个调度步（`SchedulerOutput`）翻译成一组真实设备上的前向/采样调用，并把 `ModelRunnerOutput` 回流给引擎核心。整个子系统由两条互补轴组成：**Executor（控制面）** 与 **Worker（数据面）**。

---

## 是什么

执行层负责"在哪里、用谁、怎么"地把一次 `SchedulerOutput` 跑出来：

- **Executor**：在引擎主进程里决定"几个设备、用什么进程/Actor 拉起、控制消息怎么广播"。它本身**不碰张量**，只通过 `collective_rpc` 把方法名+参数发给所有 Worker。
- **Worker**：每个设备一个进程（或 Ray actor），持有模型、KV cache、CUDA graph，真正跑前向+采样。Worker 内部又委托给一个 **ModelRunner** 完成 per-step 的输入组装、cudagraph 重放、采样组装。

```mermaid
flowchart TD
    Eng["EngineCore 主进程<br/>(调度器)"] -->|"SchedulerOutput"| Exec["Executor<br/>(控制面)"]
    Exec -->|"collective_rpc<br/>+ MessageQueue 广播"| W0["Worker #0 (rank0 driver)"]
    Exec -->|"collective_rpc"| W1["Worker #1"]
    Exec -->"|collective_rpc|" Wn["Worker #N-1"]
    W0 --> MR0["GPUModelRunner / CPUModelRunner / XPUModelRunner"]
    W1 --> MR1["ModelRunner"]
    Wn --> MRn["ModelRunner"]
    MR0 -->|"ModelRunnerOutput"| Exec
```

## 为什么独立成层

- 解耦调度语义（引擎核心）与设备语义（硬件相关）。调度器只看到"请求/token"，不必关心 NCCL/Ray/CUDA graph 细节。
- 让一项 `collective_rpc` 抽象同时承载单进程（`uni`）、多进程（`mp`）、Ray（`ray`/`ray-v2`）、外部启动器（`external_launcher`）四种部署形态，配置项 `parallel_config.distributed_executor_backend` 一键切换。
- 把"控制消息广播"与"张量数据交换"分离：控制面走 `MessageQueue`/Ray RPC，数据面走 NCCL/Ray Compiled Graph，互不阻塞。

## 怎么选 Executor × Worker

| `distributed_executor_backend` | Executor 类 | 典型 Worker | 适用场景 |
|---|---|---|---|
| `"uni"` | `UniProcExecutor` | `Worker`（GPU 单卡 / CPU / XPU） | TP=1 单设备推理、开发联调 |
| `"mp"` | `MultiprocExecutor` | `Worker` × N（子进程） | 单机多卡 TP/PP，无 Ray 依赖 |
| `"ray"` | `RayDistributedExecutor` | `RayWorkerWrapper`（Ray actor） | 多机多卡、需 Ray 编排 |
| `"ray"` + `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` | `RayExecutorV2` | `RayWorkerProc`（Ray actor + MQ） | 新版 Ray 后端，复用 `MultiprocExecutor` 控制面 |
| `"external_launcher"` | `ExecutorWithExternalLauncher` | `Worker`（torchrun 拉起） | SPMD 离线推理、多 engine 协作 |

选择逻辑集中在 `Executor.get_class()` (`vllm/v1/executor/abstract.py:48`)；Worker 类通过 `parallel_config.worker_cls` 字符串全名由 `WorkerWrapperBase.init_worker()` 反射实例化 (`vllm/v1/worker/worker_base.py:230`)。

## 与其它子系统的协作

- [引擎核心](../01-engine-core/README.md)：`EngineCore` 持有 `Executor` 实例，每步调用 `executor.execute_model(scheduler_output)` / `executor.sample_tokens(grammar_output)`。
- [模型执行](../03-model-execution/README.md)：Worker 通过 `ModelRunner.load_model()` 调用 ModelLoader 加载权重，前向时调用层库/算子。
- [注意力后端](../05-attention/README.md)：`ModelRunner` 在 `_build_attention_metadata` / `prepare_attn` 中构造 `AttentionMetadata` 并注入 forward_context。
- [分布式](../07-distributed/README.md)：TP/PP/DP/EP/PCP/DCP 通信组在 `init_worker_distributed_environment()` 里建立，KV/EC connector 也在此初始化。
- [编译与 IR](../09-compilation-ir/README.md)：`compile_or_warm_up_model()` 触发 `torch.compile` + cudagraph 捕获，捕获结果由 `CUDAGraphWrapper` 在前向时重放。

## 子目录导航

```
02-execution/
├── README.md                     （本页：执行层总览）
├── executor/                     （控制面：Executor 家族）
│   ├── README.md
│   ├── abstract.md               （Executor ABC + get_class 工厂）
│   ├── uniproc.md                （UniProcExecutor + ExternalLauncher）
│   ├── multiproc.md              （MultiprocExecutor + WorkerProc + 共享工具）
│   ├── ray.md                    （RayDistributedExecutor + vllm_net_devices）
│   └── ray-v2.md                 （RayExecutorV2：MQ 控制面 + Ray actor）
└── worker/                       （数据面：Worker × ModelRunner）
    ├── README.md
    ├── worker-base.md            （WorkerBase / WorkerWrapperBase）
    ├── gpu-worker.md             （GPU Worker：设备/内存/睡眠/PP）
    ├── gpu-model-runner.md       （GPUModelRunner V1：步前向/cudagraph/采样）
    ├── cpu-worker.md             （CPUWorker + CPUModelRunner + cpu/ 子包）
    ├── xpu-worker.md             （XPUWorker + XPUModelRunner）
    ├── model-runner-v2.md        （vllm/v1/worker/gpu/：MRv2 设计）
    ├── ubatching.md              （DBO 微批：UBatchContext + UBatchWrapper）
    ├── cudagraph-capture.md      （cudagraph 捕获/重放 + cuda_graph/breakable）
    ├── lora-mixin.md             （LoRAModelRunnerMixin + gpu/lora_utils）
    ├── kv-connector-mixin.md     （KV connector 在 ModelRunner 中的钩子）
    └── ec-connector-mixin.md     （EC connector mixin：编码器缓存迁移）
```

## 历史版本演进

- **v0.5–v0.6**：V0 时代 Executor/Worker 抽象建立，`RayDistributedExecutor` 与 `MultiprocExecutor` 并存，控制面走 `ray.get` / `Process.exec`。
- **v0.7.0**：V1 引擎落地，统一 `collective_rpc` 抽象，引入 `UniProcExecutor` 作为 TP=1 快路径；`WorkerWrapperBase` 提供"延迟初始化"两段式。
- **v0.8–v0.9**：`MultiprocExecutor` 切换到 `MessageQueue`（`shm_broadcast`）广播 `SchedulerOutput`，异步调度（`async_scheduling`）落地；KV connector mixin 接入。
- **v0.10–v0.11**：`RayExecutorV2` 进入代码库（继承 `MultiprocExecutor`），Ray 后端从 Compiled DAG 转向 MQ 控制面；UBO（`ubatching.py`）+ breakable cudagraph 引入。
- **v0.12 / main**：`vllm/v1/worker/gpu/` 下 MRv2 进入活跃开发（标记 Experimental）；`worker_cls` 全名反射、`use_v2_model_runner` 开关、EPLB 控制器、DBO 微批 SM 分流等持续打磨。

[← 返回 Wiki 首页](../README.md)
