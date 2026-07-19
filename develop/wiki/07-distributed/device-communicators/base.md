# base_device_communicator.py — 通信器抽象基类

[← Wiki 首页](../../README.md) > [分布式](../../README.md) > [device-communicators](README.md) > base

源码：`vllm/distributed/device_communicators/base_device_communicator.py`（约 383 行）。定义设备通信器的统一抽象 `DeviceCommunicatorBase`、EP all2all 管理器抽象 `All2AllManagerBase`、以及握手机器缓存 `Cache`。所有平台实现都继承它，把 `ProcessGroup` 封装成一组带 graph capture/P2P/all2all 语义的算子。

## 是什么

### Cache（`:13`）

简单的 kwargs→func 工厂缓存。`get_or_create(kwargs, func)`：把 kwargs 元组化做 key，命中则返回缓存对象，未命中调 `func(kwargs)` 建对象并缓存。DeepEP/FlashInfer 等 all2all manager 用它避免重复建昂贵的 plan/buffer。

### All2AllManagerBase（`:30`）

EP/sequence-parallel MoE 的 all2all 通信管理器抽象。构造接 `cpu_group, tcp_store_group=None`。提供：
- `get_handle(kwargs)` / `destroy()`：生命周期。
- `dispatch_router_logits(hidden_states, router_logits, ...)` / `dispatch(hidden_states, ...)` / `combine(hidden_states, ...)`：MoE 的 dispatch（前向：分发 hidden 与 router logits 到各 EP rank）与 combine（反向：把各 expert 输出聚合回 hidden）。
- `query_active_mask()`/`query_fault()`/`set_num_sms(n)`/`max_sms_used()`：DeepEP 等的状态/故障查询（默认 raise）。

默认实现大多 `raise NotImplementedError`，子类（[all2all.md](all2all.md)）按后端填实。

### DeviceCommunicatorBase（`:127`）

构造：`__init__(cpu_group, device=None, device_group=None, unique_name="", global_ranks=None, global_world_size=None)`（`:135`）。逻辑：

1. **stateless 探测**：`_world.pg_map.get(cpu_group, None) is None` 即视为 stateless（由 `StatelessGroupCoordinator` 建）。stateless 走 `cpu_group.rank()`/`.size()` + `global_ranks/global_world_size`；否则走 `dist.get_rank(cpu_group)` 系列（`:150`）。
2. `self.rank`/`world_size`/`ranks`/`global_rank`/`global_world_size`/`rank_in_group` 装定。
3. 读 `get_current_vllm_config_or_none()`，判断 `use_ep = data_parallel_size>1 or use_sequence_parallel_moe`，读 `all2all_backend`。
4. `is_ep_communicator = unique_name.split(":")[0] == "ep"`（`:185`）；`use_all2all = is_ep_communicator and use_ep`。
5. `all2all_manager: All2AllManagerBase | None = None`（子类按 `all2all_backend` 填）。

默认算子：`all_reduce`/`all_gather`（concat-style，注释 `:199`）/`all_gatherv`(NotImpl)/`reduce_scatter`/`reduce_scatterv`(NotImpl)/`gather`/`send`/`recv`/`broadcast`/`barrier`/`destroy`(pass)/`prepare_communication_buffer_for_model`(pass)/`dispatch_router_logits`/`dispatch`/`combine`(全 raise 或委派 all2all_manager)/`batch_isend_irecv(p2p_ops)`。

## 为什么

- **统一接口跨平台**：vLLM 要支持 CUDA/CPU/XPU/TPU 等，每平台有不同通信库。抽象基类让上层（`GroupCoordinator`/模型层）只依赖 `device_communicator.all_reduce(...)`，不被平台耦合。
- **stateless 兼容**：Elastic EP 路径下 PG 不在 torch `_world` 注册表，`dist.get_rank(cpu_group)` 会失败。基类先用 `_world.pg_map` 探测，stateless 走 `cpu_group.rank()`+外部传入 `global_ranks`，让一份算子代码同时服务主/副 PG。
- **EP-only all2all 隔离**：`is_ep_communicator` 用 `unique_name` 前缀（`ep:0`）识别，避免给 TP/PP/DP 组误建 all2all manager。
- **concat-style all_gather**：stack-style all-gather 与 torch.compile 不兼容（issue #138795），故强制 concat + reshape。
- **Cache 复用 plan**：DeepEP/FlashInfer 的 plan/buffer 创建极重，同组同 kwargs 应共享；`Cache` 把"建一次"语义内建。
- **batch_isend_irecv**：PP 通信一次可能要发/收多张量，`torch.distributed.batch_isend_irecv` 需 wrapper 统一异常/流。

## 怎么做

### 平台子类化模式

```python
class CudaCommunicator(DeviceCommunicatorBase):
    def __init__(self, cpu_group, device=None, device_group=None, unique_name="",
                 global_ranks=None, global_world_size=None, tcp_store_group=None):
        super().__init__(cpu_group, device, device_group, unique_name,
                         global_ranks, global_world_size)
        # 选 all-reduce 后端 + 建 all2all manager
```

### all2all manager 选择（基类侧约定）

子类按 `self.all2all_backend` 分派（见 [cuda.md](cuda.md)）。基类只提供 `self.all2all_manager = None` 默认与 `dispatch`/`combine` 的统一入口（子类决定是否委派）。

### stateless 路径字段

stateless 构造时 `StatelessGroupCoordinator` 必传 `global_ranks` 与 `global_world_size`，基类 assert 之（`:158`），否则报错。`ranks`/`global_rank`/`global_world_size` 由外部注入，`rank_in_group = self.rank`。

### MoE forward 典型调用

```python
# 模型层内（伪码）
fctx = get_forward_context()
comm = fctx.device_communicator   # 即 DeviceCommunicatorBase 子类
hidden, router = comm.dispatch_router_logits(hidden, router_logits, is_sequence_parallel=...)
# ... expert compute ...
hidden = comm.combine(expert_out, is_sequence_parallel=...)
```

## 与其它模块/系统配合

- **[cuda](cuda.md) / [cpu](cpu.md) / [xpu](xpu.md)**：三大平台实现的基类。
- **[parallel-state](../parallel-state.md)**：`GroupCoordinator.__init__` 在 `use_device_communicator and world_size>1` 时实例化；`unique_name`（如 `tp:0`/`ep:1`）决定 `is_ep_communicator`。
- **[stateless-coordinator](../stateless-coordinator.md)**：stateless 探测分支的来源；`global_ranks`/`tcp_store_group` 由它传入。
- **[all2all](all2all.md)**：`All2AllManagerBase` 的全部子类。
- **[03-model-execution](../../03-model-execution/README.md)**：fused MoE 层消费 `dispatch`/`combine`。
- **[09-compilation-ir](../../09-compilation-ir/README.md)**：`all_reduce`/`all_gather` 需注册 custom op + fake，piecewise CUDA graph 才能重放。

## 历史版本演进

- **早期**：`DeviceCommunicatorBase` 抽象引入，初版仅 `all_reduce`/`all_gather`/`send`/`recv`。
- **v0.7（v1）**：stateless 探测加入；`is_ep_communicator` + `all2all_manager` 字段引入支持 EP。
- **v0.8**：`dispatch_router_logits` 单列由 MoE 直接路由；`All2AllManagerBase` 抽象与 `Cache` 落地支持 DeepEP。
- **v0.9**：`batch_isend_irecv` 包装加入；`all_gatherv`/`reduce_scatterv` 接口稳定（部分平台 raise）。
- **v0.10/v0.11**：`query_active_mask`/`query_fault`/`set_num_sms` 等 DeepEP 状态接口抽象至基类（默认 raise）。
- **v0.12/main**：随 all2all_backend 列表扩展持续增字段（`mori_*`/`deepep_v2`/`nixl_ep`/`flashinfer_nvlink_*`）。（具体版本待核实）

[← 返回 device-communicators 首页](README.md)

## 参见

- [cuda.md](cuda.md) — 主平台实现。
- [all2all.md](all2all.md) — EP all2all manager 全集。
- [stateless-coordinator.md](../stateless-coordinator.md) — stateless 分支来源。
