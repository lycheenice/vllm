# sglang PD 分离方案调研：H200 单机 GLM-5.2 W4AFP8 4+4 部署

> 目标读者：熟悉 vLLM PD 分离、要在 sglang 框架上独立探索 H200 单机 8 卡 PD 分离的工程师。
> 本文仅做方案调研与可行性分析，不含源码修改，所有依赖 GLM-5.2 架构数字的结论均标注「待 config.json 确认」或「待 root 授权」。
> 事实来源：sglang 官方 PD 分离文档 <https://docs.sglang.io/docs/advanced_features/pd_disaggregation>（已抓取，直接引用）。

## 目录

- [1. 概述](#1-概述)
  - [1.1 sglang PD 分离背景](#11-sglang-pd-分离背景)
  - [1.2 sglang 与 vllm PD 分离的定位差异](#12-sglang-与-vllm-pd-分离的定位差异)
- [2. sglang PD 分离机制](#2-sglang-pd-分离机制)
  - [2.1 disaggregation-mode / transfer-backend / router 三件套](#21-disaggregation-mode--transfer-backend--router-三件套)
  - [2.2 GPU 切分：base-gpu-id + CUDA_VISIBLE_DEVICES](#22-gpu-切分base-gpu-id--cuda_visible_devices)
  - [2.3 KV transfer 路径](#23-kv-transfer-路径)
- [3. Transfer backend 对比](#3-transfer-backend-对比)
  - [3.1 Mooncake backend](#31-mooncake-backend)
  - [3.2 NIXL backend](#32-nixl-backend)
  - [3.3 对比表](#33-对比表)
- [4. Heterogeneous TP 与 GPU Staging Buffer 详解](#4-heterogeneous-tp-与-gpu-staging-buffer-详解)
  - [4.1 适用场景](#41-适用场景)
  - [4.2 机制：gather / bulk RDMA / scatter](#42-机制gather--bulk-rdma--scatter)
  - [4.3 性能收益](#43-性能收益)
  - [4.4 non-MLA 限制：GLM-5.2 需确认](#44-non-mla-限制glm-52-需确认)
  - [4.5 环境变量](#45-环境变量)
- [5. DP attention 与 TP4DPA2 策略分析](#5-dp-attention-与-tp4dpa2-策略分析)
  - [5.1 DP attention 机制](#51-dp-attention-机制)
  - [5.2 TP4DPA2 命名解读](#52-tp4dpa2-命名解读)
  - [5.3 D 侧两种分配对比：D_TP2×DP2 vs D_TP1×DP4](#53-d-侧两种分配对比d_tp2xdp2-vs-d_tp1xdp4)
  - [5.4 GLM-5.2 是否支持 DP attention](#54-glm-52-是否支持-dp-attention)
- [6. GLM-5.2 W4AFP8 与量化](#6-glm-52-w4afp8-与量化)
- [7. 与 vLLM PD 分离对比（简要表）](#7-与-vllm-pd-分离对比简要表)
- [8. 已知阻塞与待确认项](#8-已知阻塞与待确认项)
- [9. 结论](#9-结论)

---

## 1. 概述

### 1.1 sglang PD 分离背景

Prefill/Decode 分离（PD disaggregation）将长 prompt 的 prefill 与逐 token decode 部署到不同实例，独立扩缩容。核心难点是 KV cache 搬移：P 实例完成 prefill 后须把对应 KV 传给 D 实例，D 才能「带着上下文」继续 decode。

sglang 内置 PD 分离支持，具体由三部分组成：

- **角色**：通过 `--disaggregation-mode prefill|decode` 指定本实例是 P 还是 D。
- **传输后端**：通过 `--disaggregation-transfer-backend mooncake|nixl|ascend` 选择 KV 搬移的底层实现。
- **router**：官方提供 `sglang_router`，负责把请求路由到 P 或 D，并在 P→D 之间传递 KV 握手信息。

### 1.2 sglang 与 vllm PD 分离的定位差异

vLLM V1 不在引擎层做 P↔D 编排，没有内置 PDController；哪个请求走 P、哪个走 D、P→D 的握手信息（`remote_engine_id` / `remote_host` / `remote_port` / `tp_size` 等通过 `kv_transfer_params`）由**外部 HTTP proxy / router** 维护并跨轮转发。参考实现是 `examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py` 与 `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`。

sglang 则把 router 作为一等公民提供：

```
python -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode  http://127.0.0.1:30001 \
    --host 0.0.0.0 --port 8000
```

定位差异总结：

| 维度 | sglang | vLLM V1 |
| - | - | - |
| PD 编排 | 内置 `sglang_router`，原生 `--pd-disaggregation` 模式 | 引擎无编排，靠外部 proxy（demo / toy_proxy_server） |
| 角色声明 | `--disaggregation-mode prefill\|decode` | `--kv-transfer-config` JSON 中 `kv_role` |
| 传输后端选择 | `--disaggregation-transfer-backend` 命令行参数 | `--kv-transfer-config` JSON 中 `kv_connector` 名 |
| CPU 转发开关 | 无（NIXL 走 UCX/LIBFABRIC RDMA，文档无 host buffer 概念） | `kv_buffer_device=cpu` 显式开关 |
| 多后端 | mooncake / nixl / ascend | NIXL / Mooncake(P2P) / Mooncake(Store) 三类 |
| 异构 TP 处理 | GPU Staging Buffer（仅 non-MLA） | `compute_tp_mapping` 按头切分 |

本调研关注单机 H200，因此 ascend 后端不展开，重点对比 mooncake 与 nixl。sglang 与 vllm 是不同框架，本分支独立探索，不放入 vllm 代码。

---

## 2. sglang PD 分离机制

### 2.1 disaggregation-mode / transfer-backend / router 三件套

一个完整的单机 4+4 PD 部署由三个进程/组件组成：

1. **Prefill 实例**：`--disaggregation-mode prefill`，承担长 prompt 的 prefill 计算，prefill 完成后把 KV 推/拉给 D。
2. **Decode 实例**：`--disaggregation-mode decode`，接收 KV 后继续逐 token decode。
3. **router**：`sglang_router.launch_router --pd-disaggregation`，对外暴露单端口（如 8000），按 `--prefill` / `--decode` 指向的两实例地址做路由与 P→D 的请求接力。

官方单机 Llama（NIXL）样例直接体现了这三件套：

```bash
# Prefill
python -m sglang.launch_server --model-path ... \
    --disaggregation-mode prefill --port 30000 \
    --disaggregation-transfer-backend nixl

# Decode（base-gpu-id 错开 GPU）
python -m sglang.launch_server --model-path ... \
    --disaggregation-mode decode --port 30001 --base-gpu-id 1 \
    --disaggregation-transfer-backend nixl

# Router
python -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode  http://127.0.0.1:30001 \
    --host 0.0.0.0 --port 8000
```

> 注意：上面样例里 `--base-gpu-id 1` 是单机「错开一张卡」的最小示例，并非 4+4。4+4 切分见 2.2。

### 2.2 GPU 切分：base-gpu-id + CUDA_VISIBLE_DEVICES

sglang 单机切分 GPU 的两种手段：

- `--base-gpu-id N`：把本实例的 GPU 编号整体偏移到从 N 开始。
- `CUDA_VISIBLE_DEVICES`：经典环境变量方式，限制可见 GPU 子集。

对 4+4 部署，推荐两者结合（以 `CUDA_VISIBLE_DEVICES` 为主，`--base-gpu-id` 作为实例内的逻辑起点）：

| 实例 | GPU | 典型写法 |
| - | - | - |
| Prefill | 0,1,2,3 | `CUDA_VISIBLE_DEVICES=0,1,2,3` + `--base-gpu-id 0`（默认） |
| Decode | 4,5,6,7 | `CUDA_VISIBLE_DEVICES=4,5,6,7` + `--base-gpu-id 0`（在可见集内的起点） |

也可只用 `--base-gpu-id`：P 不设（默认 0），D 设 `--base-gpu-id 4`。两种写法等价，`CUDA_VISIBLE_DEVICES` 形式更不易与同机其它进程冲突，本方案采用它。

### 2.3 KV transfer 路径

P 完成 prefill 后，KV cache 的页（pages）需要搬到 D。在 sglang 中：

- **传输发起**：由 router 在 P 返回「请求已 prefill」信号后，触发 D 去拉/拿 KV，或由 P 主动 push（具体取决于后端实现与配置）。
- **传输介质**：
  - Mooncake：TransferEngine，单机优先 NVLink（`INTRA_NODE_NVLINK`），辅助数据走 TCP。
  - NIXL：UCX 或 LIBFABRIC，单机 NVLink 走 cuda_ipc 类零拷贝 IPC。
- **落点**：D 侧 KV cache pages，按 KV head 切分映射到各 TP/DP rank。

> 与 vllm 不同，sglang NIXL 文档未提及「host buffer / CPU 转发」概念，意味着传输路径默认就是 GPU 直传（RDMA/IPC），不存在 `kv_buffer_device=cpu` 的等价物。

---

## 3. Transfer backend 对比

### 3.1 Mooncake backend

**安装**：

```bash
uv pip install mooncake-transfer-engine
```

**IB 设备配置**：通过 `--disaggregation-ib-device mlx5_0` 指定，或多卡用 JSON 映射。

**单机 NVLink 优化（H200 适用）**：

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export MC_INTRANODE_NVLINK=true
```

官方推荐 A100/H20/H100/H200 等机内 NVLink 机型启用此组合；辅助数据仍走 TCP。

> 多节点 NVL72 场景则是另一套 env：`SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK` + `MC_FORCE_MNNVL=True`，与单机方案不同，本调研不展开。

**关键环境变量**：

| 环境变量 | 默认值 | 说明 |
| - | - | - |
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | `int(0.75*cpu_count)//8`，限 4-12 | disaggregation 线程池大小 |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | 4 | 队列大小 |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | 300s | bootstrap 超时 |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | 5s（decode 侧） | 心跳间隔 |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | 2 | 心跳最大失败次数 |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | 300s（decode 侧） | 等待 KV 超时 |

### 3.2 NIXL backend

**安装**：

```bash
pip install nixl           # pip 安装
# 或源码编译（带 ucx_path）
```

**backend 选择**：

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX        # 或 LIBFABRIC
```

UCX 与 LIBFABRIC 是两种 RDMA 栈，单机 H200 NVLink 下 UCX 的 `cuda_ipc` 类传输通常是首选。

**与 vllm NIXL 的关键差异**：sglang 文档**未提及 CPU 转发 / host buffer 模式**。vLLM 的 NIXL 有 `kv_buffer_device=cpu` 开关，可在 host DRAM 上做中转；sglang NIXL 走 UCX/LIBFABRIC RDMA，无对应开关。这对 H200 单机不是问题（NVLink 直传即可），但意味着 sglang 不具备 vLLM 那种「设备无法直注时回退 CPU」的弹性。

### 3.3 对比表

| 维度 | Mooncake | NIXL |
| - | - | - |
| 安装 | `uv pip install mooncake-transfer-engine` | `pip install nixl` 或源码编译 |
| 单机优化 | `INTRA_NODE_NVLINK` + `MC_INTRANODE_NVLINK`（NVLink 内存池） | UCX `cuda_ipc`/`cuda_copy`/TCP（单机 IPC 零拷贝） |
| RDMA 栈 | TransferEngine（自有） | UCX 或 LIBFABRIC（`SGLANG_DISAGGREGATION_NIXL_BACKEND`） |
| IB 设备配置 | `--disaggregation-ib-device` / JSON 映射 | UCX_NET_DEVICES 之类（UCX 标准 env） |
| CPU 转发支持 | 无开关 | **无**（与 vllm NIXL 的 `kv_buffer_device=cpu` 不同） |
| 外部进程 | bootstrap（Mooncake 自带） | 无 |
| 异构 TP staging | 支持 | 支持 |
| NVL72 多节点 | `NVLINK` + `MC_FORCE_MNNVL` | 不在本调研范围 |
| 部署复杂度 | 中（需配 IB/NVLink 内存池 env） | 低（默认即可） |

> 单机 H200 结论：NIXL 部署最轻量、默认即走 NVLink；Mooncake 的 `INTRA_NODE_NVLINK` 是为单机 NVLink 专门优化的另一条路径，可作为对照实验。

---

## 4. Heterogeneous TP 与 GPU Staging Buffer 详解

这是本方案的核心章节：当 P 与 D 的 TP size 不一致（如 P_TP=4、D 用 DP attention 后等效 TP 变小）时，KV head 在两侧的分布不同，直接传输会触发逐 token slice 的低效路径。GPU Staging Buffer 是 sglang 为此提供的优化。

### 4.1 适用场景

- P 与 D TP size 不同，典型如 **P TP=4，D TP=1 with DP attention**。
- 同构 TP（P_TP == D_TP）时，staging 自动 bypass，无需启用。

### 4.2 机制：gather / bulk RDMA / scatter

```
P 侧 (TP=4, 4 个 KV head 切片分布在不同 rank)
   |  gather KV head slices -> contiguous staging buffer
   v
contiguous staging buffer  -- bulk RDMA -->  D 侧 ring buffer pool
                                               |
D 侧 (TP=1/DP, KV head 重排)  <-- scatter ------+
```

- **P 侧 gather**：把分散在各 worker 的 KV head slices 收集到一块连续 staging buffer。
- **bulk RDMA 传输**：一次性大块传，避免逐 token、逐 slice 的小消息。
- **D 侧 scatter**：从 ring buffer pool 把 KV 散射到正确的 KV cache pages。

### 4.3 性能收益

- 高并发下比默认 per-token slice 快 **2-5x**。
- 与同构 TP 基线（无 staging，TP 一致）的差距约 **5%**。

即：异构 TP 不再是「降一个数量级」的灾难，而是接近同构 TP 的性能。

### 4.4 non-MLA 限制：GLM-5.2 需确认

**关键约束**：GPU Staging Buffer **仅适用于 non-MLA 模型（GQA/MHA）**。MLA 模型（DeepSeek-V2/V3）**不可启用**。

GLM-5.2 的注意力架构「待 config.json 确认」：

- 若 `config.json` 显示为 GQA/MHA（GLM 系列历史多为 GQA），则可启用 staging，TP4DPA2 异构路径可用。
- 若为 MLA（吸收式注意力，与 DeepSeek-V3 同类），则 staging 不可用，TP4DPA2 的 D_TP1×DP4 异构路径会退回 per-token slice，性能接近未优化基线。

> 这是 TP4DPA2 方案能否拿到 staging 红利的决定性条件，须在拿到 root 授权、读取 config.json 后第一时间确认。

### 4.5 环境变量

| 环境变量 | 默认 | 作用 |
| - | - | - |
| `SGLANG_DISAGG_STAGING_BUFFER` | `0`（False） | 总开关，`1` 启用 |
| `SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB` | 64 | prefill 侧每 worker 的 staging buffer 大小 |
| `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` | 4096 | decode 侧 ring buffer pool 大小 |

启用示例：

```bash
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=128   # 视并发上调
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=8192    # 视 D 侧并发上调
```

> 容量规划：prefill staging 与 decode pool 大小取决于「在途未消费的 KV 量」，随并发请求数 × 单请求 KV 页数增长。初始可用默认值，观察到不足（pool 满、装不下）后再上调。

---

## 5. DP attention 与 TP4DPA2 策略分析

### 5.1 DP attention 机制

sglang 的 DP attention（数据并行注意力）把 attention 部分按 `dp_size` 复制多组，每组在更小的 TP 上跑 attention；MoE 部分仍跨所有 rank 做专家 all2all。这是 DeepSeek 系 MoE 模型的典型并行：attention 的 KV 并不需要跨全 TP 共享，按 DP 切分可减小 attention 通信。

官方 DeepSeek 多节点样例：

```
--tp-size 16 --dp-size 8 --enable-dp-attention --moe-a2a-backend deepep
```

总 GPU 数 = `tp_size × dp_size`（上例 16×8=128）。MoE 模型需配合 `--moe-a2a-backend deepep`（DeepEP）做专家 all2all。

### 5.2 TP4DPA2 命名解读

用户口称「TP4DPA2」含义未严格定义，最合理解读：

- **P 侧**：`TP=4`，4 卡纯 TP prefill，无 DP。
- **D 侧**：4 卡使用 DP attention，「A2」指 attention 侧 2 路 DP，于是 D 侧 4 卡 = `TP2 × DP2`。

另一种解读是 D 侧 `TP1 × DP4`（attention 完全数据并行，4 路独立 KV）。两者都符合「4 卡 DP attention」的字面，需在实测里分别验证。

### 5.3 D 侧两种分配对比：D_TP2×DP2 vs D_TP1×DP4

| 维度 | D_TP2×DP2 | D_TP1×DP4 |
| - | - | - |
| attention TP | 2 | 1（单卡自含全部 KV head） |
| DP 路 | 2 | 4 |
| 单 D 实例并发（attention 侧） | 中 | 高（4 路并行） |
| KV head 分布 | 跨 2 卡切分 | 每卡完整 |
| 与 P_TP=4 的 TP 比 | 4:2（异构 2 倍） | 4:1（异构 4 倍） |
| staging 收益 | 中（异构 2x，仍显著） | 高（异构 4x，staging 收益最大） |
| MoE a2a 范围 | 全 4 卡 | 全 4 卡（不变） |

> 两种都需 `--enable-dp-attention`；都需 `--moe-a2a-backend deepep`（**前提是 GLM-5.2 是 MoE，待 config 确认**）。
> 若 GLM-5.2 不是 MoE，则 `--moe-a2a-backend` 不适用，DP attention 是否仍可用也需确认 sglang 对 dense 模型的支持范围。

### 5.4 GLM-5.2 是否支持 DP attention

DP attention 在 sglang 主要面向 MoE 模型（DeepSeek 系列）。GLM-5.2 是否适用取决于：

1. **是否 MoE**：`config.json` 是否含 `num_experts` / 专家路由字段。待确认。
2. **是否 GQA**：DP attention 切分 attention KV，需要 attention head 可切；GQA 满足，MLA 不满足。
3. **sglang 模型支持**：GLM-5.2 是否在 sglang 已注册并支持 `--enable-dp-attention`。待安装后实测。

> 若 GLM-5.2 是 dense GQA 模型，可考虑 D 侧仍跑 TP（同构 TP4+4），DP attention 不启用，回退到对照实验 C（见实验设计文档）。若确认是 MoE GQA，则 TP4DPA2 + staging 是主路径。

---

## 6. GLM-5.2 W4AFP8 与量化

### 6.1 W4AFP8 含义推测

目录名 `W4AFP8`（用户口称 W4A8，实际命名如此）。最合理解读：

- **W4**：weight 4-bit 量化（权重大概率是 group-wise int4 / W4A 系列）。
- **A_FP8**：activation FP8（激活以 FP8 E4M3/E5M2 表示），而非 int8 激活。

这与「W4A8」的区别在于：激活侧用 FP8 而非 int8。FP8 在 Hopper（H200）上有原生 Tensor Core 支持，吞吐优于 int8，是 H200 友好的量化选择。

> 上述为命名推测，确切量化方案（per-channel/per-group、是否带 scale/zero point、FP8 格式 E4M3 还是 E5M2）「待 root 授权后读取 config.json 与 README 确认」。

### 6.2 H200 Hopper 对 FP8 的支持

- H200 SXM 141GB HBM3e，Hopper 架构，原生 FP8（E4M3 / E5M2）Tensor Core。
- W4AFP8 在 Hopper 上可跑：weight int4 dequant + activation FP8 计算，常见于 sglang/vllm 的 W4A 系列量化路径。
- 具体到 sglang 是否已支持该量化格式，需在安装后用 `--quantization` 参数实测（待确认 sglang 支持的 quantization 列表）。

### 6.3 config.json 待确认字段

拿到 root 授权（`chmod -R a+rX /data1/GLM-5.2-W4AFP8`）后，须第一时间确认以下字段，以决定 staging / DP attention 可用性：

| 关注字段 | 用途 |
| - | - |
| `architectures` / `model_type` | 判断是否 GLM 系列，是否 sglang 已注册 |
| `hidden_size` / `num_attention_heads` / `num_kv_heads` / `num_hidden_layers` | KV 显存估算、TP 切分 |
| `num_experts` / 专家路由字段 | 是否 MoE -> 决定 `--moe-a2a-backend` 与 DP attention |
| `kv_lora_rank` / `q_lora_rank`（如有） | 是否 MLA -> 决定 staging 是否可用 |
| `quantization_config` | W4AFP8 确切方案 -> 决定 `--quantization` 参数 |
| `torch_dtype` | 默认 dtype |

---

## 7. 与 vLLM PD 分离对比（简要表）

| 维度 | sglang | vLLM V1 |
| - | - | - |
| PD 编排 | 内置 `sglang_router` | 外部 proxy（demo / toy_proxy） |
| 角色声明 | `--disaggregation-mode` | `--kv-transfer-config` JSON |
| 后端选择 | `--disaggregation-transfer-backend` | `--kv-transfer-config` connector 名 |
| CPU 转发 | 无 | `kv_buffer_device=cpu` |
| 异构 TP 优化 | GPU Staging Buffer（仅 non-MLA） | `compute_tp_mapping` 按头切分（含 MLA） |
| 单机 NVLink | NIXL UCX cuda_ipc / Mooncake INTRA_NODE_NVLINK | NIXL UCX cuda_ipc / Mooncake P2P RDMA |
| DP attention | `--enable-dp-attention --moe-a2a-backend deepep` | vLLM 侧另行调研 |
| 基准工具 | `python -m sglang.bench.serving` | vLLM bench_serve 等 |
| Profiling 限制 | P 与 D 必须分开 profile（torch profiler） | 同理 |

> 关键差异：sglang 把 router 内置、把后端选择做成命令行参数，部署更「开箱即用」；vLLM 把编排留给外部、把后端做成 KV Connector 插件，灵活但需要 proxy。两者在异构 TP 处理上路线不同：sglang 用 staging buffer（non-MLA only），vLLM 用 tp_mapping（含 MLA）。

---

## 8. 已知阻塞与待确认项

**阻塞（不解决无法实验）**：

1. **GLM-5.2 读权限**：`/data1/GLM-5.2-W4AFP8` 当前 `640 root:root`，用户 `lychee` 无法读。需 `root` 执行 `chmod -R a+rX /data1/GLM-5.2-W4AFP8`。
2. **sglang 未安装**：系统 `python3.12` 无 pip、无 venv。需在 `develop/sglang/` 下建 venv 并 `uv pip install sglang[all] nixl mooncake-transfer-engine`（需网络）。

**待确认（影响方案分支）**：

1. **GLM-5.2 架构数字**：`hidden_size` / `num_kv_heads` / `num_hidden_layers` / `num_experts` / `kv_lora_rank` —— 待 root 授权后从 `config.json` 确认。
2. **是否 MLA**：决定 GPU Staging Buffer 是否可用（MLA 不可用）。
3. **是否 MoE**：决定 `--moe-a2a-backend deepep` 与 `--enable-dp-attention` 是否适用。
4. **量化方案细节**：W4AFP8 确切格式与 sglang `--quantization` 参数取值 —— 待 config / README / sglang 支持列表确认。
5. **TP4DPA2 命名**：与用户确认是 D_TP2×DP2 还是 D_TP1×DP4，两者均设为实验分支。

---

## 9. 结论

1. **框架定位**：sglang PD 分离内置 router、命令行式后端选择，单机部署比 vLLM「外部 proxy + KV Connector JSON」更轻量；本分支独立探索，不放 vLLM 代码。
2. **单机 H200 后端首选 NIXL**：默认 UCX cuda_ipc 即走 NVLink 零拷贝，部署最轻；Mooncake `INTRA_NODE_NVLINK` 作为对照实验。
3. **异构 TP 看 staging**：TP4DPA2 的红利来自 GPU Staging Buffer，但其仅 non-MLA 可用 —— GLM-5.2 是否 MLA 是方案成败的关键，须 root 授权后第一时间确认。
4. **DP attention 看 MoE**：`--enable-dp-attention` 面向 MoE，需配合 `--moe-a2a-backend deepep`；GLM-5.2 若非 MoE 则回退同构 TP4+4 对照。
5. **当前阻塞**：GLM 读权限与 sglang 安装未解决，架构数字未知 —— 实验需在两项阻塞解除后启动，期间先用本文与实验设计文档锁定参数空间，scripts 设计文档先把可复制的启动命令准备好。
