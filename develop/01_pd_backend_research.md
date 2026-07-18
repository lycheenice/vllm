# PD 分离 KV Transfer 后端调研：NIXL 与 Mooncake

> 分支基准：vllm-project/vllm `v0.25.0`
> 目标读者：熟悉 vLLM 的工程师，为后续 H200 单机 4+4 PD 分离实验提供后端选择与 CPU 转发方案的理论依据。

## 目录

- [1. 概述](#1-概述)
- [2. KV Connector 抽象](#2-kv-connector-抽象)
- [3. NIXL 后端详解](#3-nixl-后端详解)
- [4. Mooncake 后端详解](#4-mooncake-后端详解)
- [5. NIXL vs Mooncake 对比](#5-nixl-vs-mooncake-对比)
- [6. 结论与建议](#6-结论与建议)

---

## 1. 概述

### 1.1 PD 分离背景

Prefill/Decode 分离（PD disaggregation）将长 prompt 的 prefill 与逐 token decode 部署到不同实例，独立扩缩容。核心难点是 KV cache 的搬移：P 实例完成 prefill 后须把对应 KV blocks 传给 D 实例，D 才能「带着上下文」继续 decode。搬移延迟直接决定 TTFT/ITAT，搬移路径是否复用决定 prefix cache 命中率。

vLLM V1 通过 **KV Connector** 抽象层屏蔽各后端差异，提供统一接口；具体传输实现由各后端自行完成。

### 1.2 vLLM 实现概览

当前代码中 PD 分离 **仅 V1 路径**：V0 connector 已废弃，`KVConnectorBase = KVConnectorBase_V1`（`vllm/distributed/kv_transfer/kv_connector/base.py:7`）。

关键事实：

- **无内置 PDController**：vLLM 不在引擎层做 P↔D 编排。哪个请求走 P、哪个走 D、P→D 的握手信息（`remote_engine_id` / `remote_host` / `remote_port` / `tp_size` 等 `kv_transfer_params`）由外部 **HTTP proxy / router** 维护并在多轮间转发。
- 参考实现：
  - `examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py`
  - `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`
- **后端启用方式**：通过 `--kv-transfer-config` JSON 指定 connector 名与角色。注意 NIXL 默认注册名是 `NixlConnector`（pull 模式别名），而非 `Nixl`。
  ```bash
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
  ```
  Mooncake 两个 connector 分别为 `MooncakeConnector`（P2P push）与 `MooncakeStoreConnector`（共享池）。

---

## 2. KV Connector 抽象

### 2.1 基类与角色

- **`KVConnectorBase_V1`**：`vllm/distributed/kv_transfer/kv_connector/v1/base.py:171`，所有 V1 connector 的抽象基类，统一描述 Scheduler 侧与 Worker 侧原语（`start_load_kv` / `wait_for_layer_load` / `save_kv_layer` / `wait_for_save` / `get_finished` / `request_finished` / `take_events` 等，类 docstring 见 `v1/base.py:3-41`）。
- **`KVConnectorRole`**：`vllm/distributed/kv_transfer/kv_connector/v1/base.py:124-129`
  ```python
  class KVConnectorRole(enum.Enum):
      SCHEDULER = 0  # 调度进程内
      WORKER  = 1    # worker 进程内
  ```
  每个 connector 实例在 `__init__` 中按 `role` 选择性构造 `connector_scheduler` 或 `connector_worker`，二者严格分离（注释见 `factory.py:67-74`）。
- **`SupportsHMA`**：`v1/base.py:85-121`，混入类声明 connector 兼容 Hybrid Memory Allocator。NIXL（`nixl/connector.py:79`）、Mooncake P2P（`mooncake_connector.py:469`）、Mooncake Store（`store/connector.py:87`）均继承之；工厂在创建时会校验 HMA 一致性（`factory.py:54-60`）。
- **`CopyBlocksOp`**：`v1/base.py:71-80`，CPU 转发方案的核心回调签名 `(src, dst, s_indices, d_indices, "h2d"|"d2h")`，由引擎注入实际 D2H/H2D 拷贝。

### 2.2 关键抽象方法（Worker 侧）

| 方法 | 作用 | 与 CPU 转发的关系 |
| - | - | - |
| `register_kv_caches` | 注册 KV tensor 为 NIXL/Mooncake 可识别的内存区 | CPU 模式注册 host buffer(DRAM) 而非 device KV(VRAM) |
| `start_load_kv` | 启动异步接收 | 触发 NIXL RDMA 读 |
| `wait_for_layer_load` | 阻塞至第 i 层接收完成 | CPU 模式接收在 host buffer |
| `save_kv_layer` / `wait_for_save` | 启动/等待发送 | CPU 模式在 `wait_for_save` 同步执行 D2H |
| `get_finished` | 返回本步完成收发的 req id | CPU 模式在此触发 H2D 同步 |

### 2.3 工厂与注册机制

- **`KVConnectorFactory`**：`vllm/distributed/kv_transfer/kv_connector/factory.py:27`。
  - `register_connector(name, module_path, class_name)` 懒加载注册（`factory.py:30-40`）。
  - `create_connector` 按 `role` 实例化（`factory.py:42-75`），并做 HMA 一致性校验。
- **内置注册表**：`factory.py:152-242`，集中注册以避免无条件加载所有后端。与本调研相关：
  - `NixlConnector` / `NixlPullConnector` / `NixlPushConnector`：`factory.py:176-192`
  - `MooncakeConnector`：`factory.py:218-222`
  - `MooncakeStoreConnector`：`factory.py:223-227`
- 也支持外部 connector：`kv_connector_module_path` 指定的模块路径优先于注册表（`factory.py:102-114`）。

### 2.4 创建入口

| 角色 | 创建位置 | 引用 |
| - | - | - |
| Scheduler 侧 | `Scheduler.__init__` | `vllm/v1/core/sched/scheduler.py:136` |
| Worker 侧 | KV transfer agent 单例 | `vllm/distributed/kv_transfer/kv_transfer_state.py:90` |

### 2.5 KVTransferConfig

`vllm/config/kv_transfer.py:22` 定义全局配置，关键字段：

| 字段 | 默认值 | 行号 | 说明 |
| - | - | - | - |
| `kv_connector` | `None` | `:26` | connector 名（如 `NixlConnector`） |
| `kv_role` | `None` | `:41` | `kv_producer`/`kv_consumer`/`kv_both` |
| `kv_buffer_device` | 平台默认 | `:33` | **CPU 转发开关**，取值 `cuda`/`cpu`/`xpu` |
| `kv_buffer_size` | `1e9` | `:37` | 缓冲区字节数 |
| `kv_connector_extra_config` | `{}` | `:59` | 后端子配置（如 `mooncake_protocol`、`bidirectional_kv_xfer`） |

---

## 3. NIXL 后端详解

### 3.1 文件结构与职责清单

NIXL connector 位于 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`，共 14 个 Python 文件（约 5800 行），是代码量最大的 KV connector 后端。

| 文件 | 行数 | 职责 |
| - | - | - |
| `base_worker.py` | 2420 | 核心：连接握手、内存注册、D2H/H2D、TP/PP 映射、收发调度 |
| `push_worker.py` | 742 | Push（WRITE）模式 worker 特化 |
| `pull_worker.py` | 382 | Pull（READ）模式 worker 特化（默认） |
| `connector.py` | 394 | 门面：`NixlPullConnector`/`NixlPushConnector`，`NixlConnector` 别名(`:386`) |
| `base_scheduler.py` | 455 | Scheduler 侧：请求收发编排、心跳、bidirectional |
| `push_scheduler.py` | 352 | Push 模式 scheduler 特化 |
| `pull_scheduler.py` | 275 | Pull 模式 scheduler 特化（默认） |
| `stats.py` | 266 | Prometheus/内部统计 |
| `metadata.py` | 225 | `NIXL_CONNECTOR_VERSION=4`(`:43`)、`compute_nixl_compatibility_hash`(`:79-126`)、握手数据结构 |
| `tp_mapping.py` | 142 | `compute_tp_mapping`(`:65-142`)：异构 TP 头切分映射 |
| `utils.py` | 68 | 常量 `_NIXL_SUPPORTED_DEVICE`(`:18-29`)、ZMQ 辅助 |
| `scheduler.py` / `worker.py` / `__init__.py` | 12/13/61 | 导出符号 |

3 个传输变体：**Pull**（D 主动 READ，`NixlConnector` 默认）、**Push**（P 主动 WRITE）、**Bidirectional KV**（D→P 反向，见 3.4）。

### 3.2 GDR 方案（`kv_buffer_device=cuda`）

**GDR（GPUDirect RDMA）直传**是 NIXL 的默认高性能路径。

- **触发**：`kv_buffer_device` 为 `cuda`/`xpu`。
- **`use_host_buffer = False`**：`nixl/base_worker.py:366-369`。
- **内存注册**：device KV cache 以 **VRAM** 类型注册到 NIXL（`base_worker.py:388-399` 决定 `nixl_memory_type`；`register_kv_caches` 直接注册 `kv_caches`，见 `:968-980` 的 `else` 分支）。
- **传输路径**：P GPU KV buffer → NIXL RDMA/IPC → D GPU KV buffer，**全程不落 CPU**。
- **handshake 设 CUDA context**：因后台握手线程需要有效 CUDA context 以启用 UCX cuda_ipc，`base_worker.py:551-552` 在 `not use_host_buffer` 时显式 `set_device`。

### 3.3 CPU 转发方案（`kv_buffer_device=cpu`，核心章节）

> 适用场景：设备内存无法直接注册到 NIXL（如 TPU），或希望在 CPU 侧做 buffer 中转（如 NHD/HND permute）。在 H200 上主要意义是绕过 GDR 的某些限制并提供缓冲弹性，代价是引入 D2H/H2D 拷贝。

#### 3.3.1 触发方式

**无独立开关**，仅靠 `kv_buffer_device` 取值：

```python
# nixl/base_worker.py:366-369
if self.device_type == "cpu":
    self.use_host_buffer = False
else:
    self.use_host_buffer = self.kv_buffer_device == "cpu"
```

Scheduler 侧镜像同一逻辑（`nixl/base_scheduler.py:77-82`）。

平台-缓冲设备组合受 `_NIXL_SUPPORTED_DEVICE` 约束（`nixl/utils.py:18-29`）：`cuda` 同时支持 `cuda`/`cpu`，`tpu` 仅支持 `cpu`，`xpu` 支持 `cpu`/`xpu`。

`nixl_memory_type` 随之确定：`cpu`→`DRAM`，`cuda`/`xpu`→`VRAM`（`base_worker.py:388-399`）。

#### 3.3.2 完整数据路径

CPU 转发为**点对点**路径，**无 broker / master**：

```
P GPU KV cache
   │ D2H (blocking, 每步同步)        save_kv_to_host  base_worker.py:1777-1801
   ▼
P CPU host xfer buffer   ← initialize_host_xfer_buffer (:652-695)
   │ NIXL RDMA/IPC (DRAM→DRAM)        引擎层 transfer
   ▼
D CPU host xfer buffer
   │ H2D (主线程同步)                 sync_recved_kv_to_device (:1754-1775)
   ▼
D GPU KV cache
```

P 侧 D2H 在 `wait_for_save` 中**每步同步执行**（`connector.py:290-294`），`# blocking` 注释见 `base_worker.py:1793`；D 侧 H2D 在 `get_finished` 检测到收发完成时触发（`base_worker.py:1931-1932`），运行在主线程。

#### 3.3.3 关键代码改动点

| 改动点 | 位置 | 作用 |
| - | - | - |
| `use_host_buffer` 标志 | `base_worker.py:366-369` / `base_scheduler.py:77-82` | 单一开关，驱动后续分支 |
| `initialize_host_xfer_buffer` | `base_worker.py:652-695` | 为每层建 `torch.empty(..., device="cpu")`，与 KV cache 1:1 同 shape（`:684-686`）；可选 NHD→HND permute（`:664-690`） |
| `set_host_xfer_buffer_ops` | `base_worker.py:697-706` | 接收引擎注入的 `CopyBlocksOp`（D2H/H2D 实现） |
| `register_kv_caches` 选择注册对象 | `base_worker.py:968-980` | `use_host_buffer` 真→注册 host buffer 为 DRAM；假→注册 device KV 为 VRAM |
| `save_kv_to_host`（P 侧） | `base_worker.py:1777-1801` | 阻塞 D2H，逐 group 调 `copy_blocks(..., "d2h")` |
| `sync_recved_kv_to_device`（D 侧） | `base_worker.py:1754-1775` | H2D，把 host buffer 同步回 device KV |
| `wait_for_save` 每步同步 | `connector.py:290-294` | 在 `use_host_buffer && copy_blocks` 时调用 `save_kv_to_host` |
| `get_finished` H2D 触发 | `base_worker.py:1931-1932` | 完成接收后调用 `sync_recved_kv_to_device` |
| handshake 不设 CUDA context | `base_worker.py:551-552` | CPU 模式不依赖 cuda_ipc，跳过 `set_device` |

#### 3.3.4 支持的功能组合

**关键结论**：CPU 转发在 TP/PP/EP/DP 维度与 GDR **完全一致**——共享同一套 pull/push/bidirectional 调度与 `compute_tp_mapping` 代码，**不存在任何 `if use_host_buffer` 分支限制上述并行度**。差异仅体现在内存注册对象与额外的 D2H/H2D 拷贝。

唯一限制：**Bidirectional KV 暂不支持 CPU 模式**（见下表及 `docs/features/nixl_connector_usage.md:315`：「Host-buffer support ... planned for future work」）。

#### 3.3.5 CPU 转发 vs GDR 功能组合差异表

| 功能/并行 | GDR (`cuda`) | CPU 转发 (`cpu`) | 备注 |
| - | - | - | - |
| TP（同构） | ✅ | ✅ | `tp_mapping.py:79-95`，共享代码 |
| TP（异构 P≠D） | ✅ | ✅ | `tp_mapping.py:65-142`，无 host buffer 限制分支 |
| PP > 1 跨 layer KV | ❌（数据结构预留） | ❌（同 GDR） | `EngineTransferInfo.remote_pp_rank/start/end_layer` 默认 0（`kv_connector/utils.py:385-391`）；创建时不设（`base_worker.py:1482-1487`） |
| EP / MoE | ✅ | ✅ | attention KV 照常传，无 expert-KV 特殊处理；兼容矩阵 `nixl_connector_compatibility.md:56` |
| DP | ✅ | ✅ | `base_scheduler.py:64-68` side_channel_port + dp_index |
| MLA | ✅ | ✅ | `_is_region_replicated`（`base_worker.py:229-237`）REPLICATE 模式 |
| Hybrid SSM / Mamba | ✅ | ✅ | 兼容矩阵 `:55` Host buffer ✅ |
| **Bidirectional KV** | ✅ | ❌ | `bidirectional_kv_xfer`（`base_scheduler.py:149`）仅 device-buffer；CPU 待支持 |
| HMA | ✅ | ✅ | `SupportsHMA`（`connector.py:79`） |
| HND/NHD permute | 可选 | 可选+可借 host buffer permute | `enable_permute_local_kv`，`base_worker.py:664-690` |

#### 3.3.6 性能特性

- **P 侧 D2H 阻塞**：`save_kv_to_host` 同步执行，每步阻塞 forward 流水（`base_worker.py:1793` 注释 `# blocking`）。
- **D 侧 H2D 在主线程**：`sync_recved_kv_to_device` 在 `get_finished` 中被调用（`base_worker.py:1931-1932`），与 decode forward 同线程串行。
- **host buffer 与 KV cache 1:1 同 shape**：（`base_worker.py:684-686`）`torch.empty(kv_shape, dtype=kv_dtype, device="cpu")`，保证 copy_blocks index 对齐，但**额外占用 1× KV cache 大小的 Host 内存**。
- 适合作为 GDR 不可用（如 TPU、或 CUX cuda_ipc 失败）时的回退；在 H200 + NVLink 场景下一般不优于 GDR。

### 3.4 GDR vs CPU 转发对比表

| 维度 | GDR（`cuda`） | CPU 转发（`cpu`） |
| - | - | - |
| 传输介质 | VRAM→VRAM RDMA/IPC | DRAM→DRAM RDMA + D2H/H2D |
| 额外拷贝 | 无 | P 侧阻塞 D2H + D 侧 H2D |
| 额外内存 | 无 | host buffer ≈ KV cache size |
| handshake CUDA context | 需设（`base_worker.py:551-552`） | 不设 |
| NVLink cuda_ipc 零拷贝 | ✅ | ❌（落 CPU） |
| 异构 TP/EP/DP/MLA/Mamba | 全支持 | 全支持 |
| Bidirectional KV | ✅ | ❌ |
| 主要用途 | 单机/多机 GPU 高性能 | TPU 等 NIXL 不可直注设备 / 回退 |

---

## 4. Mooncake 后端详解

Mooncake 后端位于 `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`，提供**两个**独立 connector，并用 `mooncake-transfer-engine` pip 包作为底层传输。

辅助组件：`mooncake_utils.py`（含 `MooncakeBootstrapServer`，`mooncake_utils.py:44`）、`rdma_utils.py`、`stats.py`，以及 `store/` 子系统（`worker.py` 1733 行、`scheduler.py`、`coordinator.py`、`data.py`、`protocol.py`、`metrics.py`）。

### 4.1 MooncakeConnector（P2P push）

- 类定义：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:469`（继承 `SupportsHMA`）；内部拆 `MooncakeConnectorScheduler`(`:613`) / `MooncakeConnectorWorker`(`:895`)。
- **传输模型**：P 侧主动 push。`_send_blocks` 调 `engine.batch_transfer_sync_write(remote_session, src_ptrs, dst_ptrs, lengths)`（`mooncake_connector.py:1622-1632`），直接 P GPU → D GPU。
- **GPU 存储 注册**：`register_kv_caches` 把 KV cache 的 `untyped_storage().data_ptr()` 批量注册（`batch_register_memory`，`mooncake_connector.py:1712-1725`）。
- **MLA 支持**：`MLAAttentionSpec`/`SlidingWindowMLASpec` 走 `page_size_bytes`（`mooncake_connector.py:1697`）。
- **GDN/Mamba**：GDN（vLLM 内表示为 `MambaSpec`）已测试，**Mamba2 未验证**（`mooncake_connector.py:641-643`）。HMA/GDN 支持见 PR #46807。

### 4.2 MooncakeStoreConnector（共享池）

- 类定义：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:87`（继承 `SupportsHMA`）。
- **传输模型**：P 把 KV `batch_put_from_multi_buffers` 写入由 `MooncakeDistributedStore` 管理的共享池（CPU DRAM/SSD），D 用 `batch_get_into_multi_buffers` 按 **hash key** 取回。支持跨实例 prefix cache 复用。
- **约束**（`store/connector.py:99-126` 校验）：
  - `CrossAttentionSpec` ❌（`:108`）
  - `MambaSpec` 仅 `align` 模式（block_size 相等，`:111-116`）
  - hybrid attention 且 `pcp * dcp > 1` ❌（`:119-122`）

### 4.3 工作原理与传输开关

Mooncake **无 GDR/CPU 模式开关**，传输协议靠 `kv_connector_extra_config.mooncake_protocol` 选择 `rdma`/`tcp`（`mooncake_connector.py:928-930`）：

```python
protocol = kv_transfer_config.kv_connector_extra_config.get("mooncake_protocol", "rdma")
self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, device_name)
```

P2P 模式注册的是 GPU storage 指针，本质为 GPU 直传（依赖 RDMA/NVLink）；Store 模式经 CPU DRAM/SSD 池，天然落 CPU。

### 4.4 支持的并行组合

| 并行 | MooncakeConnector(P2P) | MooncakeStoreConnector | 说明 |
| - | - | - | - |
| TP 同构 | ✅ | ✅ | |
| TP 异构 | ✅ | ✅ | 整除断言 `_get_tp_ratio`（`mooncake_connector.py:105-116`） |
| PP | ✅（PR #44528） | ✅ | `_align_transfer_regions` 按 layer-name 对齐（`mooncake_connector.py:305`） |
| DP | ✅ | ✅ | `mooncake_utils.py:20-26` dp engine index |
| EP(MoE) | ❌ 未实现 | ❌ 未实现 | grep 无 expert 相关处理 |
| MLA | ✅(`:1697`) | ✅ | |
| Hybrid Mamba/GDN | ✅(Mamba2 未验证 `:642-643`) | 仅 align | |

### 4.5 额外组件依赖

| 组件 | P2P | Store | 说明 |
| - | - | - | - |
| `mooncake-transfer-engine` pip (>=0.3.8) | 必需 | 必需 | 底层 TransferEngine |
| Bootstrap server | 必需 | 必需 | `MooncakeBootstrapServer`（`mooncake_utils.py:44`），默认端口 `VLLM_MOONCAKE_BOOTSTRAP_PORT=8998`（`envs.py:204`） |
| `mooncake_master` 进程 | — | 必需 | Store 协调器，需 JSON config + 固定 `PYTHONHASHSEED` |
| 环境变量 | `VLLM_MOONCAKE_*`（`envs.py:204-226`） | 同左 | bootstrap port、recv threads、abort timeout 等 |

部署复杂度明显高于 NIXL（NIXL 仅需 side-channel port）。

---

## 5. NIXL vs Mooncake 对比

### 5.1 功能对比表

| 维度 | NIXL | Mooncake(P2P) | Mooncake(Store) |
| - | - | - | - |
| 传输方式 | 点对点 RDMA/IPC（无 broker） | 点对点 push（TransferEngine） | 共享池（DRAM/SSD，hash key） |
| 协议/TLS | UCX：`cuda_ipc`/`cuda_copy`/`tcp`/GDS/libfabric | `mooncake_protocol=rdma/tcp` | 同 P2P + master 协调 |
| GPU 直传(GDR) | ✅（默认） | ✅（注册 GPU storage 指针） | ❌（经 CPU 池） |
| CPU 转发模式 | ✅（`kv_buffer_device=cpu`） | ❌（无开关） | 天然经 CPU（Store 池） |
| TP 同构/异构 | ✅ | ✅ | ✅ |
| PP | 数据结构预留，>1 未实现 | ✅（#44528） | ✅ |
| EP(MoE) | ✅ | ❌ 未实现 | ❌ 未实现 |
| DP | ✅ | ✅ | ✅ |
| MLA | ✅ | ✅ | ✅ |
| Hybrid Mamba/GDN | ✅ | ✅(Mamba2 未验证) | 仅 align |
| Bidirectional KV | ✅（GDR）/❌(CPU) | ❔ | ❔ |
| HMA | ✅ | ✅ | ✅ |
| Prefix cache 跨实例复用 | 依赖 APC + proxy 路由 | 依赖路由 | ✅（Store hash key 池化） |
| 连接拓扑发现 | side-channel（ZMQ） | bootstrap server | master + coordinator |
| 外部进程依赖 | 无 | bootstrap server | bootstrap + master |
| 部署复杂度 | 低 | 中 | 高 |

### 5.2 社区支持完善度

| 角度 | NIXL | Mooncake |
| - | - | - |
| 文件数 | 14（约 5800 行） | mooncake/ 5 + store/ 8（约 5900 行，含 1733 行 store worker） |
| 核心代码规模 | `base_worker.py` 2420 行 | `mooncake_connector.py` 2179 行 + `store/worker.py` 1733 行 |
| 版本号 / 成熟度 | `NIXL_CONNECTOR_VERSION=4`（`metadata.py:43`），含兼容性哈希校验（`:79-126`）与心跳租约(#4) | 无显式版本号；依赖 transfer-engine pip 版本 |
| 兼容矩阵文档 | `docs/features/nixl_connector_compatibility.md`（逐模型×能力表格，MoE `:56` 等） | 无对应矩阵文档 |
| 使用文档 | `docs/features/nixl_connector_usage.md`（含 bidirectional/多轮/单机 NVLink 指引） | 散见于 store/ 代码注释与示例 |
| 变体数 | 3（pull/push/bidirectional） + CPU 转发维度 | 2（P2P/Store），无变体 |
| 测试覆盖 | `tests/v1/kv_connector/nixl_integration/`（含 toy_proxy_server） | `tests/v1/kv_connector/` 下 mooncake 用例 |

综合判断：**NIXL 成熟度更高、文档更齐、变体更丰富**；Mooncake Store 子系统单文件可达 1733 行，复杂度集中于池化与协调。

### 5.3 性能差异定性分析

| 路径 | 预期性能 | 关键因素 |
| - | - | - |
| NIXL GDR（单机 NVLink） | **最佳** | `UCX_TLS=cuda_ipc,cuda_copy,tcp`，cuda_ipc 零拷贝 IPC，P GPU→D GPU 不落 CPU |
| NIXL CPU 转发 | 次之 | 额外阻塞 D2H + 主线程 H2D，但点对点无 broker；适合 GDR 不可用回退 |
| Mooncake P2P RDMA | 接近 GDR | 注册 GPU storage 后 `batch_transfer_sync_write` 直传；经 TransferEngine 抽象，可能比裸 UCX 多一层调度开销 |
| Mooncake Store 池化 | 更低但可复用 | 经 CPU DRAM/SSD，多一次拷贝；换取跨实例 prefix cache 命中与多 D 复用 |

> 文档中暂无 NIXL 与 Mooncake 的直接性能对比数据，需在目标硬件自测。

---

## 6. 结论与建议

### 6.1 单机 H200 场景后端选择

针对**单机 8×H200、4+4 PD 分离**实验：

- **首选 NIXL + GDR（`kv_buffer_device=cuda`）**：
  - 单机内 NVLink 下 `UCX_TLS=cuda_ipc,cuda_copy,tcp` 可零拷贝 IPC，P GPU→D GPU 直传；
  - 部署最轻量，无需 bootstrap/master 进程，仅 side-channel port；
  - TP/EP/DP/MLA/MoE 全面支持，`NIXL_CONNECTOR_VERSION=4` 兼容性校验完善；
  - 默认 pull 模式（`NixlConnector`）即可，需要 P 主动推流则用 `NixlPushConnector`。

- **CPU 转发方案作为备选**：仅当 GDR 因驱动/注册失败不可用时启用。在 H200 上它会引入每步阻塞 D2H 与主线程 H2D，且额外占用 ≈ KV cache size 的 host 内存，单机 NVLink 下通常不优于 GDR。其与 GDR 在 TP/PP/EP/DP/MLA/Mamba 维度**功能等价**，仅 Bidirectional KV 暂缺。

- **Mooncake**：
  - P2P 模式在单机亦可用 RDMA/NVLink 直传，但需额外 bootstrap server，部署较重，且 EP 未实现、Mamba2 未验证；
  - Store 模式经 CPU 池，单机直传场景下性能不占优，其价值在跨实例 prefix cache 复用与多 D 共享，单机 4+4 非其主战场。

### 6.2 启用示例（NIXL GDR，单机）

P 实例：
```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}'
```
D 实例：
```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}'
```
若回退 CPU 转发：把 `kv_buffer_device` 改为 `"cpu"` 即可，其余配置不变。

### 6.3 后续实验建议

1. 先以 NIXL GDR 跑通 4+4 基线，记录 TTFT/ITAT 与 KV 传输耗时（`stats.py` 指标）。
2. 同配置改 `kv_buffer_device=cpu` 对照，量化 D2H/H2D 开销。
3. 评估是否需要 Bidirectional KV（多轮对话反传 D→P）；若需要则必须用 GDR，CPU 转发暂不支持。
4. 若未来需要跨机或跨实例 prefix cache 复用，再评估 Mooncake Store。
