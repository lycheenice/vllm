# H200 单机 GLM-5.2 W4AFP8 sglang 4+4 PD 分离实验设计

> 配套文档：`01_sglang_pd_research.md`（机制调研）、`03_scripts_design.md`（可复制脚本）。
> 本文基于 h200-2 实测：GLM-5.2-W4AFP8 `config.json` 已读、`/opt/sglang-glm/` 启动脚本已读。给出 H200 单机 8 卡、GLM-5.2 W4AFP8、sglang **TP4+4 同构 PD 分离**的实验目标、矩阵、部署拓扑、启动参数、验证步骤与风险回退。
> 策略已收敛为同构 TP4+4 + 同配置基线对比：不碰 GPU Staging Buffer（GLM-5.2 是 MLA，staging 不可用），不做异构 TP（理由见调研文档第 5 章）。

## 目录

- [1. 实验目标](#1-实验目标)
- [2. 环境](#2-环境)
- [3. 显存估算（实测数字）](#3-显存估算实测数字)
- [4. 实验矩阵](#4-实验矩阵)
- [5. 部署拓扑](#5-部署拓扑)
- [6. 启动参数清单（由 start.sh 派生）](#6-启动参数清单由-startsh-派生)
- [7. 验证步骤](#7-验证步骤)
- [8. 关键注意点与风险](#8-关键注意点与风险)
- [9. 风险与回退](#9-风险与回退)

---

## 1. 实验目标

在单机 8×H200 SXM（每卡 141GB HBM3e，NVLink/NVSwitch 互联）上，用 sglang 框架对 GLM-5.2 W4AFP8（MLA MoE，实测架构见调研文档第 6 章）做 **TP4+4 同构 PD 分离**，验证以下命题：

1. **同构 PD 链路可用性**：P_TP=4 / D_TP=4，NIXL 与 Mooncake 分别能否跑通 MLA 的 latent KV 跨实例搬移（P→D），正确性一致（greedy 对齐基线）。
2. **transfer backend 对照**：NIXL（UCX cuda_ipc 单机 NVLink）vs Mooncake（`INTRA_NODE_NVLINK`）在单机的 KV 搬移延迟 / TTFT / ITAT / 吞吐差异。
3. **PD vs 不 PD 收益**：相对于同配置 TP4 不分离基线（实验 C），PD 在长 prompt（输入 7500）+ 短输出（200）负载下 TTFT/吞吐的变化；可选对照现网 DP4×TP2 8 卡实例（实验 D）作量级参考。
4. **MLA 下 sglang PD 支持度**：sglang PD 分离对 MLA 的支持是否完整（vLLM NIXL 对 MLA 有专门处理，sglang 需实测，见 8.2）。

> 命题不含「异构 TP」「staging 增益」——GLM-5.2 是 MLA，staging 不可用，异构 TP 已否决（调研文档 5.5）。bench 负载统一：输入 7500 / 输出 200，num_prompts 50 与 100 各跑一轮。

---

## 2. 环境

### 2.1 硬件与系统

| 项 | 取值 |
| - | - |
| 节点 | h200-2（10-118-89-32），单机 8×H200 SXM |
| 单卡显存 | 141GB HBM3e |
| 互联 | NVLink / NVSwitch（机内全互联） |
| 用户 | lychee |

### 2.2 模型（实测）

| 项 | 取值 | 状态 |
| - | - | - |
| 宿主路径 | `/data1/GLM-5.2-W4AFP8` | 已 `chmod a+rX`，可读 |
| 容器路径 | `/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8` | docker-compose 挂载映射 |
| 内容 | 40 个 safetensors 分片 + `config.json` + `README.md` + `chat_template.jinja` | 已核验 |
| 总权重 | 约 400GB（每分片约 10GB） | 实测 |
| 量化 | W4AFP8（`quantization_config.quant_method=w4afp8`） | 实测 |
| 架构 | `GlmMoeDsaForCausalLM` / `model_type=deepseek_v3`，MLA（`kv_lora_rank=512`）+ MoE（256 专家） | 实测 |

### 2.3 软件（实测）

| 项 | 状态 |
| - | - |
| sglang 镜像 | `lmsysorg/sglang:v0.5.15.post1-cu129`（commit `0b3bb0c`），已在 h200-2 容器内可用 |
| 现网部署 | `/opt/sglang-glm/`：单实例 DP4×TP2 PD 不分离 + EAGLE + hicache（见调研文档 7.2） |
| sglang 安装 | 容器内已含；`install_sglang.sh` 仅用于容器内或新 venv 补装 nixl/mooncake |

> 原阻塞已解除：读权限已 `chmod a+rX`，sglang 已在容器内可用，架构数字已读。本设计不含「待 config 确认」项（EAGLE 在 PD 下可用性为唯一待实测项，见 8.2）。

---

## 3. 显存估算（实测数字）

> 所有数字基于 6.1 实测字段。GLM-5.2 是 MLA，KV cache 是吸收后 latent，远小于普通 GQA。

### 3.1 权重显存

- 总权重实测约 400GB（W4AFP8：weight int4 + bf16 scale/zero，见调研文档 6.4）。
- TP=4 每卡权重 ≈ 400GB / 4 = **100GB**。
- H200 单卡 141GB，扣权重后 **剩约 41GB**。

| 项 | 公式 | 数值 |
| - | - | - |
| 总权重 | 实测（40 分片 × ~10GB） | ~400GB |
| TP=4 每卡权重 | 400GB / 4 | ~100GB |
| 每卡扣权重后剩余 | 141GB − 100GB | ~41GB |

### 3.2 KV cache 显存（MLA latent）

MLA 吸收后，每 token 每层缓存 latent：

```
latent_dim_per_layer = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576 维
```

- latent 在 TP rank 间**复制**（各 rank 共享同一 latent 做头投影），故每卡承担完整 576 维/层/token。
- KV cache dtype = `fp8_e4m3`（与基线 `start.sh` 一致），1 byte/维。
- 单 token 单层 = 576 bytes。
- 单 token 全模型（78 层）= 576 × 78 = 44,928 bytes ≈ **43.9 KiB**。

对比普通 GQA（假设 2×64×192×78）：约 1.83 MiB/token，**MLA 约为其 1/43**。

| 项 | 公式 | 数值 |
| - | - | - |
| latent 维/层/token | `kv_lora_rank + qk_rope_head_dim` | 576 |
| 单 token KV（78 层，FP8） | `576 × 78 × 1` | 44,928 B ≈ 43.9 KiB |
| 每卡 KV pool（`mem-fraction-static=0.85`） | `141GB × 0.85 − 100GB` | ~20GB |
| 每卡最大并发 token | `20GB / 43.9 KiB` | ~47.8 万 token |
| 单请求 token（输入 7500+输出 200） | - | 7700 |
| 每卡最大并发请求（KV 估算） | `47.8 万 / 7700` | ~62 路 |

> P 侧 `--mem-fraction-static 0.85` 与基线一致；D 侧可适当下调（如 0.80）给 decode 激活/输出留余量，但 MLA KV 占比极小，0.85 通常也可用。`--max-running-requests` 建议沿用基线 64（与 KV 估算的 ~62 路吻合）。

### 3.3 不再有 staging 占用

同构 TP4+4 不启用 staging（MLA 不可用，且同构本就 bypass）。原 staging buffer / pool 的显存预算（`SGLANG_DISAGG_STAGING_POOL_SIZE_MB`）**不再计入**，KV 预算全部给 D 侧正常 KV cache。

> 结论：H200 141GB/卡、TP=4 分担 100GB 权重，KV pool 约 20GB；MLA latent 极小，context-len 300000 下并发余量充足。主要约束不在显存，而在 KV 搬移延迟与 sglang PD+MLA 的功能支持度（见 8.2）。

---

## 4. 实验矩阵

以传输后端（transport）为变量，同构 TP4+4 为唯一 PD 策略，对照同配置基线：

| 编号 | strategy | transport | P 侧 | D 侧 | staging | 说明 |
| - | - | - | - | - | - | - |
| A | tp4tp4 | nixl | TP4 | TP4（同构） | 0（固定） | 同构 PD 主路径，NIXL |
| B | tp4tp4 | mooncake | TP4 | TP4（同构） | 0（固定） | 同 A，换 Mooncake + INTRA_NODE_NVLINK |
| C | tp4（基线） | - | TP4 单实例无 PD | - | 0 | 同配置不分离，主基线 |
| D（可选） | tp8 或复用现网 | - | TP8 单实例无 PD / 现网 DP4×TP2 8 卡 | - | - | 量级参考基线 |

策略命名（与 `03_scripts_design.md` 一致）：

- `tp4tp4`：P_TP4 / D_TP4，同构，不开 DP attention，不开 staging。
- `tp4`（基线）：单实例 TP4，无 PD。
- `tp8`（可选基线）：单实例 TP8，无 PD。
- 异构策略 `tp4dp2` / `tp4dp4` 已移除（MLA 不可用，见调研文档 5.5）。

> 前置条件已满足：GLM-5.2 架构已实测（MLA），无需再等 config 确认。A/B/C/D 均不依赖 staging / DP attention。

bench 负载（每组两个轮次）：

| 轮次 | 输入 token | 输出 token | num_prompts |
| - | - | - | - |
| 1 | 7500 | 200 | 50 |
| 2 | 7500 | 200 | 100 |

执行顺序建议：C（基线，验证单实例加载与 bench 链路） → A（同构 PD + NIXL，验证 PD 链路通） → B（换 Mooncake） → D（可选量级对照）。

---

## 5. 部署拓扑

### 5.1 GPU 划分

```
+-------------------+   +-------------------+
|  Prefill (P)      |   |  Decode (D)       |
|  GPU 0,1,2,3      |   |  GPU 4,5,6,7      |
|  TP=4             |   |  TP=4（同构）     |
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

基线 C/D：单实例占 GPU 0-7（TP8）或 0-3（TP4），port 30002
```

### 5.2 端口分配

| 组件 | 端口 | 备注 |
| - | - | - |
| Prefill `launch_server` | 30000 | OpenAI 兼容 API |
| Decode `launch_server` | 30001 | OpenAI 兼容 API |
| router | 8000 | 对外统一入口（`--pd-disaggregation`） |
| 基线单实例 | 30002 | 仅基线实验时启动 |

> sglang 无独立 side channel 端口概念，NIXL/Mooncake 辅助连接由后端自管；router 占 8000。与 vllm 那套（8100/8200/8000/5600-5601 side channel）不同。

### 5.3 目录与日志

| 路径 | 用途 |
| - | - |
| `develop/sglang/scripts/` | 启动/停止/bench/基线脚本 |
| `develop/sglang/logs/` | prefill.log / decode.log / router.log / baseline.log / bench.log |
| `develop/sglang/run/pids/` | 各进程 PID 文件 |

---

## 6. 启动参数清单（由 start.sh 派生）

> 以下 PD 参数由现网 `/opt/sglang-glm/start.sh`（调研文档 7.2）派生：保留业务参数，去掉 DP attention 相关，加 `--disaggregation-mode` / `--disaggregation-transfer-backend`。

### 6.1 通用参数（P 与 D 共用，由 start.sh 保留）

| 参数 | 取值 | 来源 |
| - | - | - |
| `--model` | `/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8`（容器内） | start.sh |
| `--served-model-name` | `glm` | start.sh |
| `--trust-remote-code` | 开 | start.sh |
| `--context-len` | `300000`（P/D 一致） | start.sh |
| `--tool-call-parser` | `glm47` | start.sh |
| `--reasoning-parser` | `glm45` | start.sh |
| `--schedule-policy` | `fcfs` | start.sh |
| `--enable-metrics` | 开 | start.sh |
| `--enable-cache-report` | 开 | start.sh |
| `--chunked-prefill-size` | `32768`（P 侧重用） | start.sh |
| `--max-running-requests` | `64` | start.sh |
| `--max-queued-requests` | `512` | start.sh |
| `--mem-fraction-static` | `0.85`（P）；D 可降为 `0.80` | start.sh |
| `--watchdog-timeout` | `1800` | start.sh |
| `--cuda-graph-max-bs` | `128` | start.sh |
| `--kv-cache-dtype` | `fp8_e4m3` | start.sh |
| `--host` | `0.0.0.0`（容器内对外） / `127.0.0.1`（单机回环） | 视容器模式 |

> 量化：不显式传 `--quantization`，由 `config.json` 的 `quant_method=w4afp8` 自动识别（与 start.sh 一致）。

### 6.2 Prefill 参数

公共（6.1）基础上加：

```
--disaggregation-mode prefill
--disaggregation-transfer-backend <nixl|mooncake>
--tp-size 4
--port 30000
```

P 侧不开 DP attention、不开 staging、不开 `--moe-a2a-backend`（同构 TP4 走 sglang 默认 MoE-TP 路径）。P 侧通常不开 EAGLE 投机（见 8.2）。

### 6.3 Decode 参数

公共（6.1）基础上加：

```
--disaggregation-mode decode
--disaggregation-transfer-backend <nixl|mooncake>
--tp-size 4
--port 30001
```

D 侧与 P 同构 TP4，KV head 一一对应。D 侧可保留 EAGLE 投机（基线已开，对应模型 MTP `num_nextn_predict_layers=1`），但 PD 下 EAGLE 可用性待实测（见 8.2）：先默认关，验证 PD 链路后再开 D 侧 EAGLE 对照。

### 6.4 router 参数（PD 模式）

```
python -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode  http://127.0.0.1:30001 \
    --host 0.0.0.0 --port 8000
```

> 与现网 `start-smg.sh`（非 PD 的 DP-aware cache_aware router，port 18080/29000）不同：本方案 router 用 `--pd-disaggregation` 显式 PD 模式。

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

staging：**固定不启用**（`SGLANG_DISAGG_STAGING_BUFFER=0`，MLA 不可用，同构本就 bypass）。

### 6.6 基线参数（C/D）

基线 C（TP4 不分离，主基线）：与 6.1 公共参数一致，加 `--tp-size 4 --port 30002`，无 `--disaggregation-mode`。

基线 D（可选 TP8）：加 `--tp-size 8 --port 30002`，无 PD。可复用现网 DP4×TP2 8 卡实例（调研文档 7.2）作量级参考，但口径不同（开了 DP attention + EAGLE + hicache），仅参考吞吐量级。

---

## 7. 验证步骤

### 7.1 健康检查

- P：`curl http://127.0.0.1:30000/health` 返回 200。
- D：`curl http://127.0.0.1:30001/health` 返回 200。
- router：`curl http://127.0.0.1:8000/health` 返回 200（端点名以 sglang 版本为准，可能为 `/health_ready`）。
- 日志关键字（见 `03_scripts_design.md` 第 6 节）核实 P↔D 握手成功。

### 7.2 正确性验证

- 单条请求：经 router 8000 发一条 chat completion，断言返回非空且无 KV 搬移错误。
- 对比基线：同一 prompt（temperature=0 greedy）在基线 C（30002）与 PD（8000）下输出一致性比对。
- 长 context：构造接近 `context-len`（如 7500+ 长 prompt）的请求，验证 KV 跨 P→D 搬移后 decode 上下文连贯（无截断、无重复）。

### 7.3 性能 bench

- sglang 自带：`python -m sglang.bench.serving --url http://127.0.0.1:8000 ...`，负载输入 7500 / 输出 200 / num_prompts 50 与 100。
- 兼容 vLLM bench_serve 指标口径（TTFT / ITAT / 吞吐），便于横向对比。
- 逐实验记录：TTFT、ITAT、KV 传输耗时（若 sglang 暴露指标）、吞吐、显存峰值。
- 基线 C/D 用 `--baseline` 走 30002 端口 bench，与 PD 同负载对照。

### 7.4 PD 模式 profiling 注意

sglang 官方明确：PD 模式下 prefill 与 decode worker **必须分开 profile**（torch profiler 限制）。profiling 时各自单独启用，不要同时对 P 与 D 开 profiler。

---

## 8. 关键注意点与风险

1. **MLA 下 sglang PD 支持度需实测（核心风险点）**：GLM-5.2 是 MLA（`kv_lora_rank=512`）。vLLM NIXL 对 MLA 有专门处理（`compute_tp_mapping` MLA 分支 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:79-84`、`base_worker.py:983` `use_mla`），sglang 侧 PD 分离对 MLA 的支持是否完整需首轮实测：先跑 A（NIXL）验证 KV 搬移是否成功、正确性是否一致；若 sglang PD 对 MLA 有未覆盖路径，回退到基线 C 并记录现象。
2. **EAGLE 在 PD 下可用性待验证**：基线开了 EAGLE 投机（`--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`，对应 MTP `num_nextn_predict_layers=1`）。PD 分离下 D 侧通常可保留 EAGLE、P 侧一般不开。本方案先默认关闭 EAGLE 验证 PD 链路，再在 D 侧单独开 EAGLE 做对照。这是本设计唯一保留的待实测项。
3. **单机 NVLink 用 INTRA_NODE_NVLINK**：Mooncake backend 在单机 H200 必须设 `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK` + `MC_INTRANODE_NVLINK=true`，否则不享受机内 NVLink 优化；辅助数据仍走 TCP。
4. **同构 TP 不开 DP attention / moe-a2a**：本方案 P/D 均 TP4，不开 `--enable-dp-attention`、不开 `--moe-a2a-backend`。MoE 在纯 TP4 下走 sglang 默认 MoE-TP 路径。现网 start.sh 的 `--enable-dp-attention-local-control-broadcast --enable-dp-lm-head` 等 DP 专用参数也不加。
5. **base-gpu-id 与 CUDA_VISIBLE_DEVICES**：用 `CUDA_VISIBLE_DEVICES` 切 P/D（P=0,1,2,3 / D=4,5,6,7），`--base-gpu-id` 在各自可见集内取 0；不要 D 侧再设 `--base-gpu-id 4`（在已裁剪可见集内会越界或错位）。
6. **router 单点**：router 是 P↔D 编排与对外入口，崩了即全链路断；实验时先确认 router 健康再 bench。
7. **profiling 分开**：见 7.4。
8. **容器路径映射**：sglang 跑在镜像内，`--model` 用容器路径 `/mnt/file/default-gpfs-official-2/GLM-5.2-W4AFP8`；脚本执行方式见 `03_scripts_design.md` 第 8 节。

---

## 9. 风险与回退

### 9.1 风险矩阵

| 风险 | 触发条件 | 影响 | 回退 |
| - | - | - | - |
| sglang PD 对 MLA 支持不完整 | A 实验中 KV 搬移失败 / 正确性不一致 | PD 链路不可用 | 回退基线 C；记录 sglang 日志现象，作为后续上游反馈 |
| EAGLE + PD 不可用 | D 侧开 EAGLE 后报错或结果异常 | 投机收益拿不到 | 关闭 EAGLE，PD 走非投机路径（不影响 PD 本身） |
| nixl backend 不可用 | nixl 安装/加载失败 | NIXL 组不可用 | 仅跑 Mooncake（B 为主） |
| Mooncake bootstrap 超时 | `BOOTSTRAP_TIMEOUT` 不足 | D 拉不到 KV | 调大 `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` |
| D 侧 OOM | `mem-fraction-static` 过高 + 激活 | OOM | D 侧降 `--mem-fraction-static`（0.85→0.80）或降 `--max-running-requests` |
| KV 搬移走 TCP 回退 | NVLink env 未生效 | 延迟飙升 | 核对 `INTRA_NODE_NVLINK` / UCX_TLS，确认日志走 NVLink/cuda_ipc |

### 9.2 回退优先级

1. **最稳**：基线 C（TP4 不分离）+ A（同构 TP4+4 NIXL）。这组不依赖 staging / DP attention / EAGLE，用于验证 sglang PD 链路本身（含 MLA 支持度）。
2. **次稳**：B（同构 TP4+4 Mooncake `INTRA_NODE_NVLINK`）。
3. **可选**：D（TP8 或现网 DP4×TP2）量级对照。
4. **不可跑**：若 sglang PD 对 MLA 完全不支持（A 验证失败），则 PD 路径整体搁置，仅保留基线结论并记录阻塞。

### 9.3 失败判定

- 健康检查任一不通过：停所有进程，查日志关键字，修正参数后重试。
- 正确性比对与基线不一致（greedy 下）：疑为 KV 搬移丢页或 MLA latent 映射问题，停 PD，单独验 P、D 各自正确性。
- bench 异常（TTFT 远超基线）：疑为 transport 走 TCP 回退或 MLA KV 路径低效，核对 env 与日志传输路径。
