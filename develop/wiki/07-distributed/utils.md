# utils.py — StatelessProcessGroup 与 PP 分层工具

[← Wiki 首页](../README.md) > [分布式](../README.md) > utils

源码：`vllm/distributed/utils.py`（约 733 行）。本文件是 distributed 包的"工具箱"：提供不依赖 torch WORLD 的元数据通信（`StatelessProcessGroup`）、Pipeline 分层切分、TCP store 包装、gloo/NCCL 无状态 PG 构造、以及若干张量切分纯函数。`__init__.py` 把它 `import *` 到 `vllm.distributed`。

## 是什么

### 张量切分纯函数

- `ensure_divisibility(numerator, denominator)`/`divide(n, d)`/`verify_group_size_divides_partition`（`:53`/`:60`/`:67`）：并行 size 校验常用。
- `is_weak_contiguous(inp)`（`:85`）：放宽 `is_contiguous`，允许 channel-last 等情形，被 all-reduce 后端用作输入合法性闸门。
- `split_tensor_along_last_dim(input, num_partitions, contiguous_split_chunks=False)`（`:99`）：TP 切权重/激活的标准切法。
- `TensorMetadata` namedtuple（在 `parallel_state.py:70` 定义，本文件复用）。

### get_pp_indices（`utils.py:127`）

把 `num_hidden_layers` 均分到 `pp_size` 份，返回 `(start_layer, end_layer)`。规则：

1. 若设了 `VLLM_PP_LAYER_PARTITION` 环境变量（逗号分隔），直接按用户列表，校验 `len==pp_size` 且 `sum==num_hidden_layers`。
2. 否则均分 `num_hidden_layers // pp_size`；余数 `r` 给最后 `r` 个分区各 +1（倒序 `partitions[-i] += 1`，`i=2..r+1`），让含输出 embedding 的最后分区更轻、含输入 embedding 的首分区也避开余数（注释 `:138`）。

### StatelessProcessGroup（`utils.py:199`）

`@dataclass`，属性 `rank`/`world_size`/`store: torch._C._distributed_c10d.Store`（通常 TCPStore）+ 一组发送/接收计数器 + `entries: deque[(key, timestamp)]`。**用于元数据通信**——docstring 明确"For data-plane communication, create NCCL-related objects"。

提供 `send_obj`/`recv_obj`/`broadcast_obj`/`all_gather_obj`/`broadcast(tensor)`/`send(tensor)`/`recv(tensor)`/`all_reduce`/`barrier`/`create`(classmethod) 等。所有方法基于 `store.set/get` + pickle；通过 `send_to/{dst}/{counter}`、`broadcast_from/{src}/{counter}` 等键名 + 单调计数器保证 FIFO。`expire_data()`（`:235`）按 `data_expiration_seconds`（默认 1 小时）清理老 key，避免 store 无限增长。

`create`（`:468`）是工厂：rank 0 创建 TCPStore（持 listen socket），其它 rank 连接。

### PG 辅助

- `create_tcp_store(host, port, listen_socket=None, **kwargs)`（`:175`）：可选接管一个已绑定 socket（`master_listen_fd`），用于端口复用。
- `get_cached_tcp_store_client(host, port)`（`:517`）：模块级缓存，避免重复建 TCPStore。
- `get_cpu_distributed_timeout_or_none`/`get_distributed_timeout_or_none`（`:526`/`:536`）：从 envs 读超时。
- `init_gloo_process_group`（`:546`）：建一个独立 gloo PG。
- `stateless_init_torch_distributed_process_group`（`:576`）/`stateless_destroy_torch_distributed_process_group`（`:687`）：不污染全局 `_world` 地构造/销毁一个 torch PG——Elastic EP/StatelessGroupCoordinator/NIXL 握手都靠它。
- `get_worker_rank_suffix(global_rank=None)`（`:696`）：用于日志区分多 worker。
- `sched_yield()`（`:46`）：os 调度让步，`barrier` 和 SHM 自旋等待里复用。

## 为什么

- **跨实例元数据**：P/D 分离、NIXL/Mooncake 握手要传 NCCL unique id、port、agent 元数据；这些对象小且非吞吐关键，TCPStore + pickle 即可，开销远低于建独立 NCCL PG。
- **不污染 torch WORLD**：`StatelessProcessGroup` 与 `stateless_init_torch_distributed_process_group` 让"动态、可重建的副 PG"成为可能，是 Elastic EP scale up/down 的前提。
- **barrier 三段式实现**（`:333`）：rank0 广播 `barrier_id` → 各 rank 写 `arrival_*` → rank0 等所有 arrival → 各 rank 写 `departure_*` → rank0 等所有 departure。保证下一动作（尤其 teardown TCPStore）时所有 client 已离开，避免 rank0 端口被吊销后再有人 connect。
- **VLLM_PP_LAYER_PARTITION**：自动均分会把余数堆在尾端，但尾端含 LM head + norm，显式 env 让用户手动平衡（如 `[8,8,8,6]`）以均匀显存/算力。
- **`is_weak_contiguous`**：all-reduce 后端对 non-contiguous 输入会 fallback 或慢路径，弱连续判定减少不必要的 `.contiguous()` 拷贝。
- **缓存 TCPStore**：多 connector 共用同一协调地址时避免各建一份。

## 怎么做

### StatelessProcessGroup 典型时序

```mermaid
sequenceDiagram
    participant R0 as Rank 0 (src)
    participant Store as TCPStore
    participant R1 as Rank 1 (dst)

    R0->>Store: create(host, port, is_server=True)
    R1->>Store: connect
    R0->>Store: set("send_to/1/0", pickle(obj))
    R0->>R0: send_dst_counter[1]+=1; entries.append((key, now))
    R1->>Store: get("send_to/1/0")  ; recv_src_counter[0]+=1
    Note over Store: counter 单调 => FIFO
    loop 周期
        R0->>R0: expire_data() 删超时 key
    end
```

### stateless PG 构造

`stateless_init_torch_distributed_process_group(host, port, rank, world_size, backend, group_name, listen_socket=None)`（`:576`）：
1. 用 `create_tcp_store`（持 listen_socket 时复用 fd）建 store；
2. 调 `torch.distributed.init_process_group(backend, timeout=..., rank=, world_size=, store=)`；
3. 返回 PG；`destroy` 时调 `stateless_destroy_torch_distributed_process_group` 把 PG 从 torch `_world.pg_map` 清除。

### get_pp_indices 调用

模型加载器（`vllm/model_executor/model_loader/loader.py` 系列）按 PP rank 调：

```python
start, end = get_pp_indices(num_hidden_layers, pp_rank, pp_size)
for i in range(start, end):
    layer = build_layer(i)
```

## 与其它模块/系统配合

- **[parallel-state](parallel-state.md)**：`StatelessProcessGroup`/`stateless_init_torch_distributed_process_group` 是 `_init_stateless_group` 与 `StatelessGroupCoordinator` 的底层；`get_pp_indices` 被 model loader 在 PP 切层时调用。
- **[stateless-coordinator](stateless-coordinator.md)**：直接消费 `StatelessProcessGroup.create` + `stateless_init_*`。
- **[kv-transfer](kv-transfer/README.md)**：NIXL/Mooncake/MoRIIO 等 connector 在握手阶段用 `StatelessProcessGroup` 交换 NIXL agent 元数据/NCCL unique id/端口。
- **[device-communicators/pynccl](device-communicators/pynccl.md)**：`PyNcclCommunicator.__init__` 接受 `StatelessProcessGroup`，调 `broadcast_obj(unique_id, src=0)` 把 rank0 的 NCCL id 播给全员。
- **[elastic-ep](elastic-ep.md)**：standby 组构造、port 协调全靠 `StatelessProcessGroup` 协调 store + `stateless_init_*` 重建 PG。
- **[eplb](eplb.md)**：`StatelessGroupCoordinator` 在 EPLB 重排通信路径上复用本文件工具。
- **[03-model-execution](../03-model-execution/README.md)**：`get_pp_indices` 决定每个 PP rank 装哪些 transformer 层。

## 历史版本演进

- **早期（v0.3）**：`StatelessProcessGroup` 为支持 PD 分离引入，初版仅有 `send_obj`/`recv_obj`/`broadcast_obj`。
- **v0.5**：补 `broadcast(tensor)`/`send`/`recv`/`all_reduce` 多用以减少 torch dist 依赖；`get_pp_indices` 引入 `VLLM_PP_LAYER_PARTITION` 覆盖。
- **v0.6**：`create_tcp_store` 增加 `listen_socket` 复用 fd 形态，服务 `_allocate_group_ports`（stateless_coordinator）端口管理。
- **v0.7（v1）**：`stateless_init_torch_distributed_process_group` 与 `stateless_destroy_torch_distributed_process_group` 公开，被 Elastic EP 与 KV connector 大量使用。
- **v0.8**：`barrier` 三段式实现（arrival/departure）落地，修 rank0 teardown 与 client 未离开的竞态。
- **v0.9–v0.11**：`get_cached_tcp_store_client` 缓存、`get_cpu_distributed_timeout_or_none`/`get_distributed_timeout_or_none` 抽出供建组超时复用。
- **v0.12/main**：随 Elastic EP/EPLB 持续打磨；`get_worker_rank_suffix` 等日志强化（待核实具体版本）。

[← 返回分布式首页](../README.md)

## 参见

- [parallel-state.md](parallel-state.md) — `StatelessProcessGroup` 在弹性路径上的使用。
- [stateless-coordinator.md](stateless-coordinator.md) — 动态建组的上层封装。
- [device-communicators/pynccl.md](device-communicators/pynccl.md) — `PyNcclCommunicator` 与 `StatelessProcessGroup` 的协作。
