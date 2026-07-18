# sglang PD 分离方案调研：H200 单机 GLM-5.2 W4AFP8 4+4 部署

> 目标读者：熟悉 vLLM PD 分离、要在 sglang 框架上独立探索 H200 单机 8 卡 PD 分离的工程师。
> 本文基于 h200-2（10-118-89-32）实测：GLM-5.2-W4AFP8 的 `config.json` 已读取（已 `chmod a+rX` 可读），sglang 容器与启动脚本取自 `/opt/sglang-glm/`。所有架构数字均为实测值，引用 `config.json` 字段。
> 策略已收敛为 **TP4+4 同构 PD 分离 + 同配置基线对比**：不碰 GPU Staging Buffer（GLM-5.2 是 MLA，staging 不可用），不做异构 TP。
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
- [4. GPU Staging Buffer 详解（背景知识，GLM-5.2 不可用）](#4-gpu-staging-buffer-详解背景知识glm-52-不可用)
  - [4.1 适用场景](#41-适用场景)
  - [4.2 机制：gather / bulk RDMA / scatter](#42-机制gather--bulk-rdma--scatter)
  - [4.3 性能收益](#43-性能收益)
  - [4.4 non-MLA 限制：GLM-5.2 已实测确认不可用](#44-non-mla-限制glm-52-已实测确认不可用)
  - [4.5 环境变量](#45-环境变量)
- [5. 为什么放弃异构 TP：GLM-5.2 MLA 实测结论](#5-为什么放弃异构-tpglm-52-mla-实测结论)
  - [5.1 GLM-5.2 是 MLA 的实测证据](#51-glm-52-是-mla-的实测证据)
  - [5.2 staging 不可用 → 异构 TP 退回 per-token slice](#52-staging-不可用--异构-tp-退回-per-token-slice)
  - [5.3 MLA 异构 TP 的低效与 sglang 未优化](#53-mla-异构-tp-的低效与-sglang-未优化)
  - [5.4 与 vllm 异构 TP 路径对比](#54-与-vllm-异构-tp-路径对比)
  - [5.5 结论：采用 TP4+4 同构](#55-结论采用-tp44-同构)
- [6. GLM-5.2 W4AFP8 实测架构](#6-glm-52-w4afp8-实测架构)
  - [6.1 config.json 实测字段表](#61-configjson-实测字段表)
  - [6.2 W4AFP8 量化实测](#62-w4afp8-量化实测)
  - [6.3 H200 Hopper 对 FP8 的支持](#63-h200-hopper-对-fp8-的支持)
  - [6.4 权重文件实测](#64-权重文件实测)
  - [6.5 MTP 与推理相关字段](#65-mtp-与推理相关字段)
- [7. 现有 h200-2 sglang 部署现状（基线对照来源）](#7-现有-h200-2-sglang-部署现状基线对照来源)
  - [7.1 容器与镜像](#71-容器与镜像)
  - [7.2 现有 start.sh：DP4×TP2 PD 不分离单实例](#72-现有-startshdp4tp2-pd-不分离单实例)
  - [7.3 router（start-smg.sh）：非 PD 模式](#73-routerstart-smgsh非-pd-模式)
  - [7.4 docker-compose 拓扑](#74-docker-compose-拓扑)
  - [7.5 作为基线对照的口径](#75-作为基线对照的口径)
- [8. 与 vLLM PD 分离对比（简要表）](#8-与-vllm-pd-分离对比简要表)
- [9. 实测已知项](#9-实测已知项)
- [10. 结论](#10-结论)

---

## 1. 概述

### 1.1 sglang PD 分离背景

Prefill/Decode 分离（PD disaggregation）将长 prompt 的 prefill 与逐 token decode 部署到不同实例，独立扩缩容。核心难点是 KV cache 搬移：P 实例完成 prefill 后须把对应 KV 传给 D 实例，D 才能「带着上下文」继续 decode。

sglang 内置 PD 分离支持，具体由三部分组成：

- **角色**：通过 `--disaggregation-mode prefill|decode` 指定本实例是 P 还是 D。
- **传输后端**：通过 `--disaggregation-transfer-backend mooncake|nixl|ascend` 选择 KV 搬移的底层实现。
- **router**：官方提供 `sglang_router`，负责把请求路由到 P 或 D，并在 P→D 之间传递 KV 握手信息。

### 1.2 sglang 与 vllm PD 分离的定位差异

vLLM V1 不在引擎层做 P↔D 编排，没有内置 PDController；哪个请求走 P、哪个走 D、P→D 的握手信息（`remote_engine_id` / `remote_host` / `remote_port` / `tp_size` 等通过 `kv_transfer_params`）由**外部 HTTP proxy / router** 维护并跨轮转发。参考实现是 `examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py` 与 `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`（握手字段见 `toy_proxy_server.py:162-168`，回传见 `:235-237`）。

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
| CPU 转发开关 | 无（NIXL 走 UCX/LIBFABRIC RDMA，文档无 host buffer 概念） | `kv_buffer_device=cpu` 显式开关（`vllm/config/kv_transfer.py:33`，`base_worker.py:369`） |
| 多后端 | mooncake / nixl / ascend | NIXL / Mooncake(P2P) / Mooncake(Store) 三类 |
| 异构 TP 处理 | GPU Staging Buffer（仅 non-MLA） | `compute_tp_mapping` 按头切分，含 MLA 专门分支（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:65`，MLA 分支 `:79-84`） |

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
- **落点**：D 侧 KV cache pages，按 KV head 切分映射到各 TP rank。**同构 TP（P_TP == D_TP）时两侧 KV head 分布一一对应，无需重排**，这是本方案选同构 TP4+4 的直接收益（见第 5 章）。

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

**与 vllm NIXL 的关键差异**：sglang 文档**未提及 CPU 转发 / host buffer 模式**。vLLM 的 NIXL 有 `kv_buffer_device=cpu` 开关（`vllm/config/kv_transfer.py:33`，`base_worker.py:369` `use_host_buffer = kv_buffer_device == "cpu"`），可在 host DRAM 上做中转；sglang NIXL 走 UCX/LIBFABRIC RDMA，无对应开关。这对 H200 单机不是问题（NVLink 直传即可），但意味着 sglang 不具备 vLLM 那种「设备无法直注时回退 CPU」的弹性。

### 3.3 对比表

| 维度 | Mooncake | NIXL |
| - | - | - |
| 安装 | `uv pip install mooncake-transfer-engine` | `pip install nixl` 或源码编译 |
| 单机优化 | `INTRA_NODE_NVLINK` + `MC_INTRANODE_NVLINK`（NVLink 内存池） | UCX `cuda_ipc`/`cuda_copy`/TCP（单机 IPC 零拷贝） |
| RDMA 栈 | TransferEngine（自有） | UCX 或 LIBFABRIC（`SGLANG_DISAGGREGATION_NIXL_BACKEND`） |
| IB 设备配置 | `--disaggregation-ib-device` / JSON 映射 | UCX_NET_DEVICES 之类（UCX 标准 env） |
| CPU 转发支持 | 无开关 | **无**（与 vllm NIXL 的 `kv_buffer_device=cpu` 不同） |
| 外部进程 | bootstrap（Mooncake 自带） | 无 |
| 异构 TP staging | 后端支持，但 staging 仅 non-MLA（GLM-5.2 不可用，见第 4 章） | 同左 |
| NVL72 多节点 | `NVLINK` + `MC_FORCE_MNNVL` | 不在本调研范围 |
| 部署复杂度 | 中（需配 IB/NVLink 内存池 env） | 低（默认即可） |

> 单机 H200 结论：NIXL 部署最轻量、默认即走 NVLink；Mooncake 的 `INTRA_NODE_NVLINK` 是为单机 NVLink 专门优化的另一条路径，作为对照实验。两者均用于本方案的**同构 TP4+4** 路径，无需 staging。

---

## 4. GPU Staging Buffer 详解（背景知识，GLM-5.2 不可用）

> **本章仅作背景知识保留。** GPU Staging Buffer 是 sglang 为**异构 TP**（P 与 D 的 TP size 不同）提供的 KV 传输优化。但该优化**仅适用于 non-MLA 模型（GQA/MHA）**。GLM-5.2 已实测确认为 MLA（见 6.1：`model_type=deepseek_v3`、`kv_lora_rank=512`、`q_lora_rank=2048`），**staging 不可用**。本方案已采用同构 TP4+4（P_TP == D_TP），staging 本就 bypass，故本章内容不影响实验，仅说明「为什么不碰它」。

### 4.1 适用场景

- P 与 D TP size 不同，典型如 P TP=4、D TP=1 with DP attention。
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

- 高并发下比默认 per-token slice 快 2-5x。
- 与同构 TP 基线（无 staging，TP 一致）的差距约 5%。

即：异构 TP 不再是「降一个数量级」的灾难，而是接近同构 TP 的性能。**但前提是 non-MLA**。

### 4.4 non-MLA 限制：GLM-5.2 已实测确认不可用

**关键约束**：GPU Staging Buffer **仅适用于 non-MLA 模型（GQA/MHA）**。MLA 模型（DeepSeek-V2/V3、GLM-5.2）**不可启用**。

GLM-5.2 的注意力架构**已实测确认**为 MLA，`config.json` 证据（完整字段表见 6.1）：

- `model_type` = `deepseek_v3`（与 DeepSeek-V3 同型，MLA 的标志）。
- `architectures` = `GlmMoeDsaForCausalLM`。
- `kv_lora_rank` = 512、`q_lora_rank` = 2048（MLA 的低秩 KV/Q 压缩维度，GQA/MHA 无此字段）。
- `qk_head_dim` = 256、`qk_nope_head_dim` = 192、`qk_rope_head_dim` = 64、`v_head_dim` = 256（MLA 解耦 RoPE 结构）。

因此 staging 对 GLM-5.2 不可用，任何异构 TP 策略都会退回 per-token slice 的低效路径。结论见第 5 章。

### 4.5 环境变量

> 以下环境变量在本方案中**一律不启用**（`SGLANG_DISAGG_STAGING_BUFFER=0` 固定），列此仅为完整性。MLA 下即使设为 1 也不生效。

| 环境变量 | 默认 | 作用 |
| - | - | - |
| `SGLANG_DISAGG_STAGING_BUFFER` | `0`（False） | 总开关，`1` 启用（non-MLA only） |
| `SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB` | 64 | prefill 侧每 worker 的 staging buffer 大小 |
| `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` | 4096 | decode 侧 ring buffer pool 大小 |

---

## 5. 为什么放弃异构 TP：GLM-5.2 MLA 实测结论

本章取代原「TP4DPA2 策略分析」。原方案设想 P_TP=4、D 侧用 DP attention 做异构 TP（D_TP2×DP2 或 D_TP1×DP4），靠 GPU Staging Buffer 弥合异构低效。实测 `config.json` 后，该路径被否决，理由如下。

### 5.1 GLM-5.2 是 MLA 的实测证据

`config.json` 关键字段（完整表见 6.1）：

| `config.json` 字段 | 实测值 | 含义 |
| - | - | - |
| `model_type` | `deepseek_v3` | 与 DeepSeek-V3 同型，MLA 架构 |
| `kv_lora_rank` | 512 | MLA 的 KV 低秩压缩维度（GQA 无此字段） |
| `q_lora_rank` | 2048 | MLA 的 Q 低秩压缩维度 |
| `qk_nope_head_dim` / `qk_rope_head_dim` | 192 / 64 | MLA 解耦 RoPE（nope+rope），`qk_head_dim=256` |
| `v_head_dim` | 256 | MLA 的 V 头维度 |
| `num_attention_heads` / `num_key_value_heads` | 64 / 64 | MLA 下 KV 头与 Q 头同数，但缓存走 latent |

存在 `kv_lora_rank` / `q_lora_rank` 即判定为 MLA，无需更多佐证。GLM-5.2 与 DeepSeek-V3 同属 MLA 家族。

### 5.2 staging 不可用 → 异构 TP 退回 per-token slice

异构 TP 的性能红利**全部**来自 GPU Staging Buffer（第 4 章）。而 staging 仅 non-MLA 可用（4.4）。GLM-5.2 是 MLA，因此：

- staging 无法启用。
- 异构 TP 下 P（TP4）与 D（TP1/TP2 + DP）的 KV head 分布不同，KV 传输只能走默认的 **per-token slice** 路径：逐 token、逐 slice 小消息搬移，高并发下比同构 TP 慢一个数量级。
- sglang 当前未对 MLA 异构 TP 做专门优化（无 gather/scatter、无 bulk RDMA 等价路径）。

### 5.3 MLA 异构 TP 的低效与 sglang 未优化

MLA 的 KV cache 是吸收后的 latent（`kv_lora_rank` 512 + `qk_rope_head_dim` 64 = 576 维/层，见 6.1），不是按 GQA 的 num_kv_heads × head_dim 切分。这意味着：

- MLA 的 latent KV 在 TP rank 间是**复制**而非按头切分（吸收后各 rank 共享同一 latent 做投影）。
- 异构 TP 下，P 侧 4 rank 各持一份 latent，D 侧 1/2 rank 也要各持一份；slice 路径无法像 GQA 那样按 KV head 一次性 gather，只能逐 token 对齐 latent 切片，通信碎片化。
- sglang 的 staging gather/scatter 是面向 GQA head 切分设计的，对 MLA latent 没有等价实现。强行异构 TP 会得到接近「未优化基线」的传输性能，违背 PD 分离的初衷。

### 5.4 与 vllm 异构 TP 路径对比

vLLM NIXL 对异构 TP 有专门处理，且**包含 MLA 分支**：

- `compute_tp_mapping`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:65`）按 KV head 切分构建 local→remote 映射；MLA 分支在 `:79-84`（`if transfer_topology.is_mla ...`，"For MLA, we only need one remote since cache is duplicated"）。
- NIXL worker 注册 KV cache 时显式区分 MLA（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:983`，`use_mla` 日志）。

即 vLLM 侧的异构 TP + MLA 是被显式支持的；sglang 侧则无对应优化（staging non-MLA only）。这是两框架的能力差异，也是本方案在 sglang 上放弃异构 TP、改走同构的直接技术依据。

### 5.5 结论：采用 TP4+4 同构

基于上述实测：

1. **P 与 D 均为 TP=4**，`--tp-size 4` 两侧一致，不开 `--enable-dp-attention`，不开 `--moe-a2a-backend`，不设 staging env。
2. 同构 TP 下两侧 KV head 分布一一对应，KV 传输走 bulk 路径，无 gather/scatter 开销。
3. 单机 8 卡切 P(0-3) / D(4-7)，靠 NVLink 搬 KV，NIXL 与 Mooncake 各作一组对照。
4. 基线对照：同配置 TP4 不分离（或复用现有 DP4×TP2 8 卡实例，见第 7 章），口径在实验设计文档中明确。

异构 TP（tp4dp2 / tp4dp4）与 staging 相关代码在脚本设计中保留字段但固定为不可用（`STAGING=0`、`DP_ATTN=0`），见 `03_scripts_design.md`。

---

## 6. GLM-5.2 W4AFP8 实测架构

> 本节所有数字来自 h200-2 上 `/data1/GLM-5.2-W4AFP8/config.json`（已 `chmod a+rX` 可读）。容器内挂载路径为 `/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8`。

### 6.1 config.json 实测字段表

| 分类 | `config.json` 字段 | 实测值 |
| - | - | - |
| 标识 | `architectures` | `GlmMoeDsaForCausalLM` |
| 标识 | `model_type` | `deepseek_v3` |
| 标识 | `torch_dtype` | `bfloat16` |
| 体量 | `num_hidden_layers` | 78 |
| 体量 | `hidden_size` | 6144 |
| 体量 | `intermediate_size` | 12288（dense FFN） |
| 体量 | `moe_intermediate_size` | 2048（每专家） |
| 注意力 | `num_attention_heads` | 64 |
| 注意力 | `num_key_value_heads` | 64 |
| 注意力 | `head_dim` | 192 |
| MLA | `kv_lora_rank` | 512 |
| MLA | `q_lora_rank` | 2048 |
| MLA | `qk_head_dim` | 256 |
| MLA | `qk_nope_head_dim` | 192 |
| MLA | `qk_rope_head_dim` | 64 |
| MLA | `v_head_dim` | 256 |
| MoE | `n_routed_experts` | 256 |
| MoE | `num_experts_per_tok` | 8 |
| MoE | `n_shared_experts` | 1 |
| MoE | `ep_size` | 1 |
| MoE | `topk_method` | `noaux_tc` |
| MoE | `scoring_func` | `sigmoid` |
| MoE | `routed_scaling_factor` | 2.5 |
| MoE | `first_k_dense_replace` | 3（前 3 层 dense，后续 sparse） |
| 上下文 | `max_position_embeddings` | 1048576 |
| 词表 | `vocab_size` | 154880 |
| 量化 | `quantization_config.quant_method` | `w4afp8` |
| MTP | `num_nextn_predict_layers` | 1 |

**架构判定**：

- **MLA**：存在 `kv_lora_rank` / `q_lora_rank` → 多头潜在注意力（吸收式），与 DeepSeek-V3 同型。缓存的是吸收后 latent（512 + 64 = 576 维/层，远小于 GQA 的 2×64×192/层）。
- **MoE**：`n_routed_experts=256`、`num_experts_per_tok=8`、`first_k_dense_replace=3` → 前 3 层 dense、后 75 层 sparse MoE，1 个 shared expert。
- **MTP**：`num_nextn_predict_layers=1` → 支持下一代预测（与基线 `--speculative-algorithm EAGLE ... --speculative-num-steps 1` 对应）。

### 6.2 W4AFP8 量化实测

`config.json` 的 `quantization_config`：

```json
{"quant_method": "w4afp8"}
```

含义（结合命名与 sglang/vllm W4A 系列量化路径）：

- **W4**：weight 4-bit（group-wise int4，含 scale/zero point）。
- **A_FP8**：activation FP8（H200 Hopper 原生 E4M3 Tensor Core）。

权重总量约 400GB（见 6.4），与「W4 + bf16 scale」量级一致。基线 `start.sh` 未显式传 `--quantization`，由 `config.json` 的 `quant_method=w4afp8` 自动识别；KV cache 走 `--kv-cache-dtype fp8_e4m3`（与激活 FP8 同型）。

### 6.3 H200 Hopper 对 FP8 的支持

- H200 SXM 141GB HBM3e，Hopper 架构，原生 FP8（E4M3 / E5M2）Tensor Core。
- W4AFP8 在 Hopper 上：weight int4 dequant + activation FP8 计算，吞吐优于 int8，是 H200 友好的量化选择。
- sglang 镜像（见 7.1）已在该模型上跑通 W4AFP8，无需额外 `--quantization` 参数。

### 6.4 权重文件实测

- 路径（宿主）：`/data1/GLM-5.2-W4AFP8`（h200-2，已 `chmod a+rX` 可读）。
- 路径（容器）：`/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8`（docker-compose 挂载映射）。
- 内容：`config.json` + `README.md` + `chat_template.jinja` + 40 个 safetensors 分片（`model-00001-of-00040.safetensors` … `model-00040-of-00040.safetensors`）。
- 分片体积：每个约 10GB，**总权重约 400GB**。
- 量化对应：W4 使 400GB 量级合理（同等参数 bf16 会更大）。

### 6.5 MTP 与推理相关字段

- `num_nextn_predict_layers=1`：模型自带 1 层 NextN（MTP）预测头，配合 sglang 的 `--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2` 做投机解码（基线已开，见 7.2）。
- `chat_template.jinja`：GLM 对话模板，配合 `--tool-call-parser glm47 --reasoning-parser glm45`。
- `max_position_embeddings=1048576`：模型支持 1M 上下文；基线 `--context-len 300000` 取其子集。

---

## 7. 现有 h200-2 sglang 部署现状（基线对照来源）

> 本节基于 h200-2 上 `/opt/sglang-glm/` 目录实测（容器与启动脚本可读）。现有部署是 **PD 不分离的 DP4×TP2 单实例**，用作本方案基线对照来源。

### 7.1 容器与镜像

- 镜像：`br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129`（也用过 `v0.5.14-cu129`）。
- sglang build commit：`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`。
- 容器名：`sglang-glm-smg-1`（router 在跑）、`sglang-glm-sglang-1`（已退出）。
- 模型挂载：宿主 `/data1/GLM-5.2-W4AFP8` → 容器 `/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8`。

### 7.2 现有 start.sh：DP4×TP2 PD 不分离单实例

`/opt/sglang-glm/start.sh` 实测内容（**单实例、PD 不分离**，8 卡 DP attention + MLA）：

```bash
python3 -m sglang.launch_server \
    --model /mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8 \
    --served-model-name glm --trust-remote-code \
    --port 8001 --host 0.0.0.0 \
    --context-len 300000 \
    --tool-call-parser glm47 --reasoning-parser glm45 \
    --schedule-policy fcfs \
    --enable-metrics --enable-cache-report \
    --tp-size 8 --dp-size 4 --enable-dp-attention \
    --enable-dp-attention-local-control-broadcast --enable-dp-lm-head \
    --chunked-prefill-size 32768 \
    --max-running-requests 64 --max-queued-requests 512 \
    --mem-fraction-static 0.85 --watchdog-timeout 1800 \
    --speculative-algorithm EAGLE --speculative-num-steps 1 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
    --enable-dynamic-chunking --enable-hierarchical-cache --hicache-size 195 \
    --cuda-graph-max-bs 128 --kv-cache-dtype fp8_e4m3
```

要点：

- 并行：`--tp-size 8 --dp-size 4 --enable-dp-attention`，8 卡 = TP2×DP4（attention 数据并行 4 路、每路 TP2），配合 `--enable-dp-attention-local-control-broadcast --enable-dp-lm-head`。**这是单实例 PD 不分离**，不是 PD 分离。
- 量化：未显式 `--quantization`，由 `config.json` 的 `w4afp8` 自动识别；`--kv-cache-dtype fp8_e4m3`。
- 投机：EAGLE，`num-steps 1 / eagle-topk 1 / num-draft-tokens 2`，对应 MTP（`num_nextn_predict_layers=1`）。
- 缓存：`--enable-hierarchical-cache --hicache-size 195`（分层 KV 缓存，约 195GB 层级池）。
- 显存：`--mem-fraction-static 0.85`。

本方案的 PD 启动参数由此 `start.sh` 派生（去 DP attention、加 `--disaggregation-mode`，见实验设计文档第 6 节）。

### 7.3 router（start-smg.sh）：非 PD 模式

`/opt/sglang-glm/start-smg.sh` 跑的是 `sglang_router.launch_router`，但**不是 PD 模式**（无 `--pd-disaggregation`），而是 DP-aware `cache_aware` 策略，对外端口 18080（业务）+ 29000（prometheus 指标）。它把请求在 DP4 路间做缓存感知路由，与 PD 分离的 `--pd-disaggregation --prefill --decode` 模式不同。

本方案的 router 改用 PD 模式（见 `03_scripts_design.md` launch_router.sh）。

### 7.4 docker-compose 拓扑

`/opt/sglang-glm/docker-compose.yaml` 实测：

- service `sglang`（port 8001）+ service `smg`（port 18080 / 29000）。
- 挂载：`/data1/GLM-5.2-W4AFP8` → 容器 `/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8`。
- `shm_size: 32gb`，capabilities：`SYS_NICE` + `IPC_LOCK`，`--gpus all`。

PD 分离需起两个 sglang 实例（P/D 各 4 卡）+ 一个 router，容器执行方式见 `03_scripts_design.md` 第 8 节。

### 7.5 作为基线对照的口径

现有部署可作为基线，但需明确对照口径：

- **口径一（同配置 TP4 不分离）**：新建一个 TP4 单实例（无 PD、无 DP attention），与本方案 PD 的 P/D 单实例配置完全一致，是最公平的「PD vs 不 PD」对照。推荐为主基线（实验 C）。
- **口径二（复用现有 DP4×TP2 8 卡）**：直接用 7.2 的现网实例做参考量级，但它开了 DP attention + EAGLE + hicache，配置复杂、非同口径，仅作吞吐量级的上限参考（实验 D 可选）。
- 两口径的取舍在 `02_h200_glm_pd_experiment_design.md` 第 4 节实验矩阵中明确。

---

## 8. 与 vLLM PD 分离对比（简要表）

| 维度 | sglang | vLLM V1 |
| - | - | - |
| PD 编排 | 内置 `sglang_router` | 外部 proxy（demo / toy_proxy） |
| 角色声明 | `--disaggregation-mode` | `--kv-transfer-config` JSON |
| 后端选择 | `--disaggregation-transfer-backend` | `--kv-transfer-config` connector 名 |
| CPU 转发 | 无 | `kv_buffer_device=cpu`（`vllm/config/kv_transfer.py:33`，`base_worker.py:369`） |
| 异构 TP 优化 | GPU Staging Buffer（**仅 non-MLA**，GLM-5.2 不可用） | `compute_tp_mapping` 按头切分，**含 MLA 分支**（`tp_mapping.py:65`，MLA `:79-84`） |
| MLA 感知 | staging 不支持；同构 TP 可用 | NIXL 显式 `use_mla`（`base_worker.py:983`） |
| 单机 NVLink | NIXL UCX cuda_ipc / Mooncake INTRA_NODE_NVLINK | NIXL UCX cuda_ipc / Mooncake P2P RDMA |
| DP attention | `--enable-dp-attention --moe-a2a-backend deepep`（本方案不用） | vLLM 侧另行调研 |
| 基准工具 | `python -m sglang.bench.serving` | vLLM bench_serve 等 |
| Profiling 限制 | P 与 D 必须分开 profile（torch profiler） | 同理 |

> 关键差异：sglang 把 router 内置、后端做成命令行参数，部署更「开箱即用」；vLLM 把编排留给外部、后端做成 KV Connector 插件，灵活但需 proxy。两者在异构 TP 处理上路线不同：sglang 用 staging buffer（non-MLA only），vLLM 用 `compute_tp_mapping`（含 MLA 专门分支）。**对 GLM-5.2（MLA），sglang 只能走同构 TP，本方案 accordingly 选 TP4+4。**

---

## 9. 实测已知项

原「已知阻塞与待确认项」中的阻塞已全部解除（权限已 `chmod a+rX`、sglang 已在容器内可用、架构数字已读）。以下为实测后的已知项：

1. **MLA 限制（决定性）**：GLM-5.2 是 MLA（`model_type=deepseek_v3`、`kv_lora_rank=512`），GPU Staging Buffer 不可用，异构 TP 路径被否决。本方案固定同构 TP4+4。
2. **MoE 但本方案不开 DP attention**：模型是 MoE（256 专家、8/tok），但同构 TP4 不开 `--enable-dp-attention`，故也不需 `--moe-a2a-backend`。MoE 在纯 TP4 下走 sglang 默认 MoE-TP 路径。
3. **权重 400GB / 40 分片**：总权重约 400GB，W4 量化；TP4 每卡约 100GB，H200 141GB 剩约 41GB（显存估算见实验设计文档第 3 节）。
4. **容器路径映射**：宿主 `/data1/GLM-5.2-W4AFP8` ↔ 容器 `/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8`。sglang 跑在镜像内，脚本须在容器内执行或 `docker run` 新容器（见 `03_scripts_design.md` 第 8 节）。
5. **sglang 已在容器内可用**：镜像 `v0.5.15.post1-cu129`（commit `0b3bb0c`）已含 sglang，`install_sglang.sh` 仅用于容器内或新 venv 补装 nixl/mooncake，不需在宿主装。
6. **EAGLE 在 PD 下可用性待验证**（唯一保留的待实测项）：基线开了 EAGLE 投机；PD 分离下 D 侧通常可保留 EAGLE、P 侧一般不开，但 sglang 是否完全支持 PD + EAGLE 组合需实测确认（见实验设计文档第 8 节风险点）。

---

## 10. 结论

1. **策略定案**：TP4+4 同构 PD 分离。P/D 均 `--tp-size 4`，不开 DP attention、不开 staging、不开 moe-a2a-backend，靠 NVLink 搬 MLA latent KV。
2. **理由**：GLM-5.2 实测为 MLA（`model_type=deepseek_v3`、`kv_lora_rank=512`），GPU Staging Buffer 不可用；异构 TP 会退回 per-token slice 低效路径且 sglang 未优化（vLLM 侧有 `compute_tp_mapping` MLA 分支，sglang 无）。同构 TP 两侧 KV head 一一对应，无 gather/scatter 开销。
3. **后端对照**：NIXL（UCX cuda_ipc，默认走 NVLink，部署最轻）为主，Mooncake（`INTRA_NODE_NVLINK`）为对照。
4. **基线对照**：同配置 TP4 不分离（主基线）+ 可选复用现有 DP4×TP2 8 卡实例（量级参考）。口径在实验设计文档明确。
5. **参数派生**：PD 启动命令由现网 `start.sh` 派生，保留 `--context-len 300000 --tool-call-parser glm47 --reasoning-parser glm45 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85` 等业务参数，加 `--disaggregation-mode/--disaggregation-transfer-backend`。
6. **待验证**：EAGLE 在 PD 下的可用性是唯一保留的实测项；MLA 下 sglang PD 分离功能是否完全支持需首轮实测验证（vLLM NIXL 对 MLA 有专门处理，sglang 需实测）。
