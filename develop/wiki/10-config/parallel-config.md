# ParallelConfig + EPLBConfig（parallel.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > ParallelConfig

源码：`vllm/config/parallel.py`（约 1001 行）。`ParallelConfig` 描述分布式拓扑与并行策略：TP/PP/DP/EP/DCP/PCP、EPLB、all2all backend、NUMA 绑定、executor 后端与 worker 类。`EPLBConfig` 是其内嵌子配置，控制专家并行负载均衡。它是 `VllmConfig.parallel_config`，被 Executor、Worker、分布式 init、调度器 DP 协调消费。

## 是什么

### `ParallelConfig`（`parallel.py:116`）

**并行维度**

| 字段 | 默认 | 含义 |
|---|---|---|
| `tensor_parallel_size` | `1` | TP 组数 |
| `pipeline_parallel_size` | `1` | PP 组数 |
| `prefill_context_parallel_size` | `1` | prefill context parallel |
| `data_parallel_size` | `1` | DP 组数；MoE 按 `TP*DP` 切专家 |
| `data_parallel_size_local` | `1`(0=外部指定) | 本机 DP 数 |
| `data_parallel_rank` | `0` | DP rank |
| `data_parallel_rank_local` | `None` | 本地 DP rank（SPMD 模式） |
| `decode_context_parallel_size` | `1` | DCP 组数（复用 TP GPU，须 `tp_size % dcp_size == 0`） |
| `cp_kv_cache_interleave_size` | `1` | DCP/PCP 的 KV 交错大小（1=token 级，block_size=块级） |
| `dcp_kv_cache_interleave_size` | `1` | （将被 PCP 取代） |
| `dcp_comm_backend` | `"ag_rs"` | DCP 通信后端：`ag_rs`(AllGather+ReduceScatter)/`a2a` |

**DP 协调与 LB**

| 字段 | 默认 | 含义 |
|---|---|---|
| `data_parallel_master_ip` | `"127.0.0.1"` | DP master IP |
| `data_parallel_master_port`/`data_parallel_rpc_port` | `29500`/`29550` | 端口 |
| `data_parallel_backend` | `"mp"` | `mp`/`ray` |
| `data_parallel_external_lb` | `False` | 外部 LB 模式（K8s 一 pod 一 rank，仅 MoE） |
| `data_parallel_hybrid_lb` | `False` | 混合 LB（节点内 vLLM 自 LB，节点间外部 LB） |
| `disable_nccl_for_dp_synchronization` | `None`(三态) | DP 同步用 Gloo 而非 NCCL，async scheduling 默认 True |
| `is_moe_model` | `None`(派生) | 是否 MoE（由 `VllmConfig` 从 `ModelConfig` 回填） |

**专家并行 / DBO / ubatching**

| 字段 | 默认 | 含义 |
|---|---|---|
| `enable_expert_parallel` | `False` | MoE 用专家并行而非 TP |
| `enable_ep_weight_filter` | `False` | 加载时跳过非本 rank 专家权重 |
| `expert_placement_strategy` | `"linear"` | `linear`/`round_robin` |
| `all2all_backend` | `"allgather_reducescatter"` | MoE all2all：`naive`/`pplx`/`deepep_high_throughput`/`deepep_low_latency`/`deepep_v2`/`mori_*`/`nixl_ep`/`flashinfer_*` |
| `enable_eplb` | `False` | 启用专家并行负载均衡 |
| `eplb_config` | `EPLBConfig()` | EPLB 子配置 |
| `enable_elastic_ep` | `False` | 弹性 EP（无状态 NCCL 组） |
| `enable_dbo` | `False` | dual batch overlap |
| `ubatch_size` | `0` | ubatch 大小（`use_ubatching = enable_dbo or ubatch_size>1`） |
| `dbo_decode_token_threshold` | `32` | 纯 decode 批次 DBO 阈值 |
| `dbo_prefill_token_threshold` | `512` | 含 prefill 批次 DBO 阈值 |

**executor / worker / 拓扑**

| 字段 | 默认 | 含义 |
|---|---|---|
| `distributed_executor_backend` | `None` | `ray`/`mp`/`uni`/`external_launcher`/Executor 类 |
| `worker_cls` | `"auto"` | worker 类全名 |
| `sd_worker_cls` | `"auto"` | 投机解码 worker 类 |
| `worker_extension_cls` | `""` | worker 扩展类（动态继承注入） |
| `max_parallel_loading_workers` | `None` | 顺序加载时的并行 worker 上限 |
| `disable_custom_all_reduce` | `False` | 关自定义 all-reduce，回退 NCCL |
| `placement_group` | `None` | Ray placement group |
| `ray_workers_use_nsight`/`ray_runtime_env` | `False`/`None` | Ray profiling/runtime env |
| `master_addr`/`master_port`/`node_rank`/`nnodes` | `127.0.0.1`/`29501`/`0`/`1` | mp 多节点拓扑 |
| `numa_bind`/`numa_bind_nodes`/`numa_bind_cpus` | `False`/`None`/`None` | NUMA 绑定（GPU↔NUMA 自动检测或手填 CPU 列表） |
| `assigned_physical_gpu_ids` | `None` | 逻辑 GPU ID→物理 GPU ID 映射 |
| `distributed_timeout_seconds`/`cpu_distributed_timeout_seconds` | `None`/`None` | NCCL/gloo 超时 |
| `world_size` | init=False |=`TP*PP` |
| `rank` | `0` | 全局 rank |
| `data_parallel_index` | init=False | DP index（dense 不覆盖） |

**API scale-out 内部字段**（`_api_process_count`/`_api_process_rank`）：仅 API server scale-out 设置，`_api_process_rank` 须 `< _api_process_count` 或 `-1`。

### `EPLBConfig`（`parallel.py:57`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `window_size` | `1000` | 专家负载记录窗口 |
| `step_interval` | `3000` | 重排专家间隔 |
| `num_redundant_experts` | `0` | 冗余专家数 |
| `log_balancedness`/`log_balancedness_interval` | `False`/`1` | 平衡度日志 |
| `use_async` | `True` | 非阻塞 EPLB |
| `policy` | `"default"` | 策略（async 仅支持 default） |
| `communicator` | `None` | `torch_nccl`/`torch_gloo`/`nixl`/`pynccl`/`None`(自动) |

校验：`use_async` 仅配 `policy="default"`；`torch_nccl`/`pynccl` 与 async 不兼容（NCCL 多流冲突）。

### 关键属性与方法

- `world_size_across_dp` =`TP*PP*DP`；`use_ubatching`/`num_ubatches`；`local_engines_only`（外部/混合 LB 时 True）。
- `use_sequence_parallel_moe`/`use_batched_dp_moe`：按 all2all backend + EP + TP>1 + DP>1 判定。
- `nnodes_within_dp`/`node_rank_within_dp`/`local_world_size`：多节点 DP 拓扑派生。
- `stateless_init_dp_group(return_store)`（`parallel.py:590`）：用 gloo stateless init DP 进程组，带 `EADDRINUSE` 重试。
- `has_unfinished_dp`/`sync_dp_state`/`sync_kv_cache_memory_size`：DP 间 all-reduce 同步状态（OR/SUM/MIN）。
- `get_next_dp_init_port`/`_pick_stateless_dp_port`：DP 端口管理（coord store 模式下 rank 0 bind+发布）。

`compute_hash`（`parallel.py:735`）：排除 rank/端口/网络/启动详情等派生项，把 `tensor_parallel_size`/`pipeline_parallel_size`/`data_parallel_size`/`enable_expert_parallel`/`all2all_backend`/`enable_dbo`/`enable_eplb`/`eplb_config`/`dcp_*`/`cp_kv_cache_interleave_size` 等图形状相关项纳入。该哈希也用于 **DP worker 配置一致性校验**防 hang。

## 为什么

- **多并行维度统一**：TP/PP/DP/EP/DCP/PCP 共存，互斥与整除关系复杂（如 `tp_size % dcp_size == 0`、`dcp_comm_backend='a2a'` 须 `dcp_size>1`）。`_validate_parallel_config` 集中校验。
- **DP 三态 LB**：`external_lb`/`hybrid_lb`/内部 LB 决定 `DPCoordinator` 是否需要、`local_engines_only` 行为；MoE 与 dense 模型策略不同（`needs_dp_coordinator`）。
- **stateless init**：弹性 EP 与多次 DP group init 需要无状态初始化（不依赖全局 rank），`stateless_init_dp_group` + coord store 端口发布解决端口竞争。
- **DBO / ubatching**：`enable_dbo` 与 `use_ubatching` 把批次切成 2 个 microbatch 重叠，`VllmConfig` 中强制关 cascade attention 与限制 all2all backend。
- **EPLB async + communicator 约束**：NCCL 多流与 async 冲突，强制 gloo/nixl，避免静默数据损坏。

## 怎么做

- **TP**：`--tensor-parallel-size 4`，executor 自动按 TP*PP 选 `mp`/`ray`。
- **MoE EP**：`--enable-expert-parallel --all2all-backend deepep_low_latency`；`enable_ep_weight_filter` 减加载 I/O。
- **DP**：`--data-parallel-size 2 --data-parallel-backend mp`；`VllmConfig` 自动决定 `disable_nccl_for_dp_synchronization` 与 `DPCoordinator`。
- **DCP**：`--decode-context-parallel-size 2`（须整除 TP）；`--dcp-comm-backend a2a` 减 NCCL 调用。
- **EPLB**：`--enable-eplb --eplb-config.num-redundant-experts 8`；async 默认开，communicator 自动选 nixl/gloo。
- **NUMA 绑定**：`--numa-bind`（自动按 GPU↔NUMA 拓扑）或 `--numa-bind-nodes '[0,0,1,1]'`/`--numa-bind-cpus '["0-3","4-7","8-11","12-15"]'`。

## 与其它模块/系统配合

- **Executor（[`02-execution/executor/`](../02-execution/executor/README.md)）**：`distributed_executor_backend`/`world_size`/`worker_cls` 决定 `UniProcExecutor`/`MultiprocExecutor`/`RayDistributedExecutor`/`external_launcher`。
- **Worker（[`02-execution/worker/`](../02-execution/worker/README.md)）**：TP/PP rank 初始化 NCCL；`worker_extension_cls` 动态注入 mixin；`numa_bind` 在 fork 前设 numactl。
- **DP 协调（[`01-engine-core/dp-coordinator.md`](../01-engine-core/dp-coordinator.md)）**：`needs_dp_coordinator` 决定起独立 `DPCoordinatorProc`；`prefill_schedule_interval` 对齐 prefill 节拍。
- **调度器（[`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)）**：`disable_nccl_for_dp_synchronization` + `async_scheduling` 影响 DP 同步路径；`use_ubatching` 触发 microbatch 调度。
- **KV 迁移（[kv-transfer-config.md](kv-transfer-config.md) 与 [`07-distributed/`](../07-distributed/README.md)）**：PD disagg 依赖 `kv_parallel_size` 与 connector；`enable_elastic_ep` 与无状态 NCCL 组配合。
- **编译（[compilation-config.md](compilation-config.md)）**：`all2all_backend`/`data_parallel_size` 决定 `set_splitting_ops_for_v1`；`enable_sp`/`fuse_gemm_comms`（async TP）依赖 `tensor_parallel_size>1`。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`async_scheduling` 兼容性校验依赖 `all2all_backend`/`enable_dbo`；`use_ubatching` 强制 all2all backend 与关 cascade attention；`dbob` ROCm 与 async 互斥。

## 历史版本演进

- **v0.5/v0.6（v0）**：`ParallelConfig` 含 TP/PP/`distributed_executor_backend`/`worker_cls`/`placement_group`；`data_parallel_size` 尚无。
- **v0.7（v1 落地）**：DP 概念引入；`data_parallel_*` 字段铺开；`enable_expert_parallel`/`all2all_backend`（初版 `naive`/`pplx`）。
- **v0.8（v1 默认）**：EPLB + `EPLBConfig` 子配置；`enable_elastic_ep`；`dcp_*`/`cp_kv_cache_interleave_size`（DCP）；Mooncake/NIXL KV connector。
- **v0.9**：`external_launcher` executor；`data_parallel_external_lb`/`data_parallel_hybrid_lb`；`stateless_init_dp_group` + coord store；`enable_dbo`/`ubatch_size`/`dbo_*_threshold`；`mori_*`/`nixl_ep`/`flashinfer_*` all2all。
- **v0.10**：`assign_physical_gpu_ids`；`numa_bind_nodes`/`numa_bind_cpus` 精细 NUMA 控制；`dcp_comm_backend='a2a'`；`prefill_context_parallel_size`；`_api_process_*` scale-out 内部字段。
- **v0.11 / v0.12 / main**：`deepep_v2` all2all；`prefill_context_parallel_size`；`disable_nccl_for_dp_synchronization` 三态与 async 联动；`use_sequence_parallel_moe`/`use_batched_dp_moe` 属性；MNNVL `flashinfer_nvlink_*`。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [vllm-config.md](vllm-config.md) — `async_scheduling`/DBO/ubatching 校验与 `is_moe_model` 回填。
- [compilation-config.md](compilation-config.md) — `all2all_backend` 影响 splitting_ops。
- [kv-transfer-config.md](kv-transfer-config.md) — PD disagg 的并行面协同。
- [../07-distributed/README.md](../07-distributed/README.md) — EP/EPLB/elastic-ep/权重迁移。
- [../02-execution/executor/README.md](../02-execution/executor/README.md) — executor 选择。
