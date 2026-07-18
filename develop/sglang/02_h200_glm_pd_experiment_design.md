# H200 单机 GLM-5.2 W4AFP8 sglang 4+4 PD 分离实验设计

> 配套文档：`01_sglang_pd_research.md`（机制调研）、`03_scripts_design.md`（可复制脚本）。
> 本文给出 H200 单机 8 卡、GLM-5.2 W4AFP8、sglang 4+4 PD 分离的实验目标、矩阵、部署拓扑、启动参数、验证步骤与风险回退。
> 凡依赖 GLM-5.2 架构数字的估算均标注「待 config.json 确认」或「待 root 授权」，未编造具体数值。

## 目录

- [1. 实验目标](#1-实验目标)
- [2. 环境前提与阻塞](#2-环境前提与阻塞)
- [3. 显存估算](#3-显存估算)
- [4. 实验矩阵](#4-实验矩阵)
- [5. 部署拓扑](#5-部署拓扑)
- [6. 启动参数清单](#6-启动参数清单)
- [7. 验证步骤](#7-验证步骤)
- [8. 关键注意点](#8-关键注意点)
- [9. 风险与回退](#9-风险与回退)

---

## 1. 实验目标

在单机 8×H200 SXM（每卡 141GB HBM3e，NVLink/NVSwitch 互联）上，用 sglang 框架对 GLM-5.2 W4AFP8 模型做 4+4 PD 分离，验证以下命题：

1. **TP4DPA2 异构 TP 可行性**：P 侧 TP=4、D 侧 DP attention（D_TP2×DP2 或 D_TP1×DP4），配合 GPU Staging Buffer，能否在异构 TP 下逼近同构 TP4+4 的性能（差距 ~5% 量级）。
2. **transfer backend 对照**：NIXL（UCX cuda_ipc 单机 NVLink）vs Mooncake（`INTRA_NODE_NVLINK`）在单机的 KV 搬移延迟 / 吞吐差异。
3. **staging buffer 增益**：异构 TP 下，启用 vs 不启用 `SGLANG_DISAGG_STAGING_BUFFER` 的 KV 传输耗时对比（预期 2-5x）。
4. **整体 PD 收益**：相对于单实例 TP8 无 PD 基线，4+4 PD 在 TTFT/ITAT/吞吐上的变化。

> 上述命题的成立前提包含「GLM-5.2 为 non-MLA」「GLM-5.2 为 MoE 或可启用 DP attention」等条件，均「待 config.json 确认」。若条件不满足，按第 9 节回退。

---

## 2. 环境前提与阻塞

### 2.1 硬件与系统

| 项 | 取值 |
| - | - |
| 节点 | h200-2，单机 8×H200 SXM |
| 单卡显存 | 141GB HBM3e |
| 互联 | NVLink / NVSwitch（机内全互联） |
| 用户 | lychee |

### 2.2 模型

| 项 | 取值 | 状态 |
| - | - | - |
| 路径 | `/data1/GLM-5.2-W4AFP8` | 存在 |
| 权限 | `640 root:root` | **lychee 当前不可读（阻塞）** |
| 内容 | 40 个 safetensors 分片 + config.json + README.md + chat_template.jinja | 待授权后核验 |
| 量化 | W4AFP8（weight 4-bit + activation FP8，推测） | 待 config/README 确认 |

可写区：`/data1/models`（777），可用于放置后续下载的对照模型或缓存，但 GLM-5.2 本身位于不可读的 `/data1/GLM-5.2-W4AFP8`。

### 2.3 软件

| 项 | 状态 |
| - | - |
| 系统 python | python3.12，无 pip，无 venv |
| sglang | **未安装（阻塞）** |
| venv 位置 | 计划建在 `develop/sglang/`（见 `03_scripts_design.md` install_sglang.sh） |

### 2.4 阻塞清单（须先解决）

1. **GLM-5.2 读权限**：需 root 执行
   ```bash
   sudo chmod -R a+rX /data1/GLM-5.2-W4AFP8
   ```
   授权后立即读取 `config.json`，回填第 3 节显存估算与第 6 节待确认参数。
2. **sglang 安装**：在 `develop/sglang/` 建 venv，`uv pip install sglang[all] nixl mooncake-transfer-engine`（需网络）。
3. **架构数字确认**：从 `config.json` 取 `hidden_size` / `num_kv_heads` / `num_hidden_layers` / `num_experts` / `kv_lora_rank` / `quantization_config`。

> 上述三项未解决前不启动实验，但可先用本设计与 `03_scripts_design.md` 把所有命令模板准备到位。

---

## 3. 显存估算

> 公式给出，数值「待 config.json 确认」后回填。所有估算以 W4AFP8（weight int4、activation FP8）为前提。

### 3.1 权重显存

- 单分片体积 ≈ 总权重 / 40。40 个 safetensors 分片，待读取后用 `du` / 分片大小求和。
- W4 量化下：权重参数量 × 0.5 byte（int4）+ scale/zero point 开销（per-group，约 +5-10%）。
- TP=4 时每卡权重 ≈ 总权重显存 / 4。

| 项 | 公式 | 数值 |
| - | - | - |
| 总权重量 | 待 config 确认参数量后计算 | 待回填 |
| W4 权重显存 | 参数量 × ~0.55 byte | 待回填 |
| TP=4 每卡权重 | 总权重显存 / 4 | 待回填 |

### 3.2 KV cache 显存

KV cache 单 token 显存（per layer，单 KV head，FP8 = 1 byte）：

```
kv_per_token = 2 (K+V) × num_hidden_layers × num_kv_heads × head_dim × dtype_bytes
```

- W4AFP8 模型 KV 通常以 FP8 存（待 config 确认 `kv_cache_dtype`），dtype_bytes=1。
- TP=4 下每卡承担 `num_kv_heads / 4` 个 KV head（GQA 假设，待确认）。
- 单卡可分配 KV = (141GB × utilization − 权重 − 激活) ，utilization 取 0.90。

| 项 | 公式 | 数值 |
| - | - | - |
| 单 token KV（全模型） | `2 × L × H_kv × D × 1` | 待回填 |
| TP=4 每卡单 token KV | 上式 / 4 | 待回填 |
| 每卡可用 KV 显存 | `141GB × 0.9 − 权重/4 − 激活余量` | 待回填 |
| 最大并发 token（TP=4 每卡） | 可用 KV / 单 token KV | 待回填 |

### 3.3 staging buffer / pool 占用

- prefill 侧：`SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB`（默认 64，每 worker）。
- decode 侧：`SGLANG_DISAGG_STAGING_POOL_SIZE_MB`（默认 4096，4GB ring buffer pool）。
- decode 侧 4GB pool 需从 KV 显存预算中扣除，规划 D 侧并发时计入。

> 结论：精确显存预算须待 config.json 回填。粗略上 H200 141GB/卡、TP=4 分担权重，KV 预算充裕；主要约束在 decode 侧 staging pool 与并发 KV 的权衡。

---

## 4. 实验矩阵

以策略（strategy）与传输后端（transport）为两维，对照同构与基线：

| 编号 | strategy | transport | P 侧 | D 侧 | staging | 说明 |
| - | - | - | - | - | - | - |
| A | tp4dp2 | nixl | TP4 | TP2×DP2（enable-dp-attention） | 1 | TP4DPA2 主路径，NIXL |
| B | tp4dp2 | mooncake | TP4 | TP2×DP2（enable-dp-attention） | 1 | 同 A，换 Mooncake + INTRA_NODE_NVLINK |
| C | tp4tp4 | nixl | TP4 | TP4（同构） | 0（auto bypass） | 同构对照，staging 不启用 |
| D | tp4dp4 | nixl | TP4 | TP1×DP4（enable-dp-attention） | 1 | 异构 4x，staging 收益最大 |
| 基线 | tp8 | - | TP8 单实例无 PD | - | - | 性能 / 正确性基线 |

策略命名（与 `03_scripts_design.md` 一致）：

- `tp4tp4`：P_TP4 / D_TP4，同构，staging bypass，D 不开 DP attention。
- `tp4dp2`：P_TP4 / D_TP2×DP2，异构 2x，开 DP attention，开 staging。
- `tp4dp4`：P_TP4 / D_TP1×DP4，异构 4x，开 DP attention，开 staging。
- `tp8`：单实例 TP8，无 PD，作基线。

> 前置条件：A/B/D 需 GLM-5.2 为 non-MLA（staging 可用）且支持 DP attention（MoE 或 sglang 对该模型支持）。若任一不满足，按第 9 节回退；C 与基线不依赖 DP attention / staging，是最稳的对照。

实验执行顺序建议：基线 → C（同构，验证 PD 链路通） → A（异构 2x + staging） → B（换后端） → D（异构 4x）。

---

## 5. 部署拓扑

### 5.1 GPU 划分

```
+-------------------+   +-------------------+
|  Prefill (P)      |   |  Decode (D)       |
|  GPU 0,1,2,3      |   |  GPU 4,5,6,7      |
|  TP=4             |   |  TP2×DP2 / TP1×DP4|
|  port 30000       |   |  port 30001       |
+---------+---------+   +---------+---------+
          |                       |
          +------- NVLink --------+
                   (KV transfer)
          |                       |
          +---------+-------------+
                   |
            +------+------+
            |   router    |   port 8000 (对外)
            | sglang_router|
            +-------------+
```

### 5.2 端口分配

| 组件 | 端口 | 备注 |
| - | - | - |
| Prefill `launch_server` | 30000 | OpenAI 兼容 API |
| Decode `launch_server` | 30001 | OpenAI 兼容 API |
| router | 8000 | 对外统一入口 |
| 基线 TP8 | 30002 | 仅基线实验时启动 |

> 与 vllm 那套（8100/8200/8000/5600-5601 side channel）不同：sglang 无独立 side channel 端口概念，NIXL/Mooncake 的辅助连接由后端自管；router 占 8000。

### 5.3 目录与日志

| 路径 | 用途 |
| - | - |
| `develop/sglang/scripts/` | 启动/停止/bench 脚本 |
| `develop/sglang/logs/` | prefill.log / decode.log / router.log / bench.log |
| `develop/sglang/run/pids/` | 各进程 PID 文件 |
| `develop/sglang/.venv/` | sglang venv（install_sglang.sh 创建） |

---

## 6. 启动参数清单

> 以下参数模板「待 config.json 确认」后微调（`--quantization`、`--moe-a2a-backend`、`--max-model-len` 等）。

### 6.1 通用参数（P 与 D 共用）

| 参数 | 取值 | 说明 |
| - | - | - |
| `--model-path` | `/data1/GLM-5.2-W4AFP8` | 待 root 授权可读 |
| `--trust-remote-code` | 开 | GLM 系列通常需要 |
| `--quantization` | 待确认 | W4AFP8 在 sglang 的 quantization 名（待 config / README / sglang 支持列表） |
| `--kv-cache-dtype` | fp8（待确认） | 与 W4AFP8 激活量化匹配，待 sglang 支持 |
| `--max-model-len` | 32768（可调） | 视显存预算 |
| `--host` | 127.0.0.1 | 单机回环，router 同机 |
| `CUDA_VISIBLE_DEVICES` | P=0,1,2,3 / D=4,5,6,7 | GPU 切分 |

### 6.2 Prefill 参数（按 strategy）

公共：

```
--disaggregation-mode prefill
--port 30000
--tp-size 4
--disaggregation-transfer-backend <nixl|mooncake>
```

所有 strategy 中 P 侧均为 TP4，不启 DP attention。

### 6.3 Decode 参数（按 strategy）

公共：

```
--disaggregation-mode decode
--port 30001
--base-gpu-id 0          # 配合 CUDA_VISIBLE_DEVICES=4,5,6,7
--disaggregation-transfer-backend <nixl|mooncake>
```

strategy 差异：

| strategy | `--tp-size` | `--dp-size` | `--enable-dp-attention` | `--moe-a2a-backend` | staging env |
| - | - | - | - | - | - |
| tp4tp4 | 4 | - | 否 | -（待 MoE 确认） | 0（bypass） |
| tp4dp2 | 2 | 2 | 是 | deepep（待 MoE 确认） | 1 |
| tp4dp4 | 1 | 4 | 是 | deepep（待 MoE 确认） | 1 |

> `--moe-a2a-backend deepep` 仅当 GLM-5.2 为 MoE 时加入；非 MoE 则去掉该参数，并需确认 sglang 是否允许 dense 模型开 `--enable-dp-attention`（待实测）。

### 6.4 router 参数

```
python -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode  http://127.0.0.1:30001 \
    --host 0.0.0.0 --port 8000
```

### 6.5 transport 相关环境变量

NIXL：

```
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
```

Mooncake（单机 NVLink）：

```
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export MC_INTRANODE_NVLINK=true
```

staging（仅异构 strategy）：

```
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=128      # 视并发调
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=8192       # 视 D 侧并发调
```

---

## 7. 验证步骤

### 7.1 健康检查

- P：`curl http://127.0.0.1:30000/health` 返回 200。
- D：`curl http://127.0.0.1:30001/health` 返回 200。
- router：`curl http://127.0.0.1:8000/health` 返回 200（或 router 自有 health 端点，待 sglang 版本确认）。
- 日志关键字（见 `03_scripts_design.md` 第 6 节）核实 P↔D 握手成功。

### 7.2 正确性验证

- 单条请求：经 router 8000 发一条 chat completion，断言返回非空且无 KV 搬移错误。
- 对比基线：同一 prompt 在基线 TP8（30002）与 PD（8000）下输出一致性比对（容许采样温度为 0 取 greedy）。
- 长 context：构造接近 `max-model-len` 的 prompt，验证 KV 跨 P→D 搬移后 decode 上下文连贯（无截断、无重复）。

### 7.3 性能 bench

- sglang 自带：`python -m sglang.bench.serving --url http://127.0.0.1:8000 ...`（参数待 sglang 版本确认）。
- 兼容 vLLM bench_serve 指标口径（TTFT / ITAT / 吞吐），便于与 vllm 那套对照。
- 逐实验记录：TTFT、ITAT、KV 传输耗时（若 sglang 暴露指标）、吞吐、显存峰值。

### 7.4 PD 模式 profiling 注意

sglang 官方明确：PD 模式下 prefill 与 decode worker **必须分开 profile**（torch profiler 限制）。profiling 时各自单独启用，不要同时对 P 与 D 开 profiler。

---

## 8. 关键注意点

1. **staging 仅 non-MLA**：A/B/D 实验的 staging 红利仅在 GLM-5.2 为 GQA/MHA 时成立；若 MLA，staging 不可启，异构 strategy 退回 per-token slice 低效路径（此时优先跑 C 同构）。
2. **单机 NVLink 用 INTRA_NODE_NVLINK**：Mooncake backend 在单机 H200 必须设 `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK` + `MC_INTRANODE_NVLINK=true`，否则不享受机内 NVLink 优化；辅助数据仍走 TCP。
3. **DP attention 需要 MoE a2a backend**：`--enable-dp-attention` 在 sglang 主要面向 MoE，需配 `--moe-a2a-backend deepep`；若 GLM-5.2 非 MoE，DP attention 路径可能不可用，回退同构。
4. **同构 TP staging 自动 bypass**：C 实验无需也不应手动启 staging（设了也会 bypass，但避免干扰）。
5. **base-gpu-id 与 CUDA_VISIBLE_DEVICES**：本方案用 `CUDA_VISIBLE_DEVICES` 切 P/D，`--base-gpu-id` 在各自可见集内取 0；不要 D 侧再设 `--base-gpu-id 4`（在已裁剪的可见集内会越界或错位）。
6. **router 单点**：router 是 P↔D 编排与对外入口，崩了即全链路断；实验时先确认 router 健康再 bench。
7. **profiling 分开**：见 7.4。
8. **量化参数待确认**：`--quantization` 与 `--kv-cache-dtype` 的确切取值须从 config/README + sglang 支持列表对齐，错误取值会导致加载失败或退回精度。

---

## 9. 风险与回退

### 9.1 风险矩阵

| 风险 | 触发条件 | 影响 | 回退 |
| - | - | - | - |
| GLM-5.2 为 MLA | config.json 含 `kv_lora_rank` | staging 不可用 | 放弃 A/B/D，仅跑 C + 基线 |
| GLM-5.2 非 MoE | config.json 无 `num_experts` | DP attention 可能不支持 | 放弃 A/B/D 的 DP attention，改 C 同构 |
| sglang 不识别该量化 | `--quantization` 取值不在支持列表 | 加载失败 | 查 README / sglang 支持列表，换 quantization 名或退回 fp8/fp16 |
| nixl 安装失败 | 源码编译缺 ucx | NIXL backend 不可用 | 仅跑 Mooncake（B 改 transport=mooncake 主跑） |
| Mooncake bootstrap 超时 | `BOOTSTRAP_TIMEOUT` 不足 | D 拉不到 KV | 调大 `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` |
| staging pool 满 | 并发过高、`POOL_SIZE_MB` 不足 | KV 丢/卡 | 上调 `SGLANG_DISAGG_STAGING_POOL_SIZE_MB`，或降并发 |
| DP attention 显存爆 | D_TP1×DP4 单卡 KV head 全量 | OOM | 回退 D_TP2×DP2（A），或降 `--max-model-len` |

### 9.2 回退优先级

1. **最稳**：基线 TP8（无 PD）+ C 同构 TP4+4（NIXL，无 staging、无 DP attention）。即使 GLM-5.2 是 MLA 或非 MoE，这组仍可跑，用于验证 sglang PD 链路本身。
2. **次稳**：A（TP4DPA2 / D_TP2×DP2 / NIXL + staging），需 non-MLA + DP attention 可用。
3. **进阶**：D（异构 4x）、B（Mooncake），在前述稳定后再跑。
4. **不可跑**：若 GLM-5.2 为 MLA，A/B/D 的 staging 路径全部放弃，仅保留 1 的结论。

### 9.3 失败判定

- 健康检查任一不通过：停所有进程，查日志关键字，修正参数后重试。
- 正确性比对与基线不一致（greedy 下）：疑为 KV 搬移丢页或量化精度问题，停 PD，单独验 P、D 各自正确性。
- bench 异常（TTFT 远超基线）：疑为 staging 未生效或 transport 走 TCP 回退，核对 env 与日志传输路径。
