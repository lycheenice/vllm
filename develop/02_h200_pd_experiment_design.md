# H200 单机 8 卡 NIXL PD 分离实验设计（MiniMax-M2.5）

> 适用机型：h200-2（单机 8×H200 SXM）。目标读者：执行本实验的工程师。
> 文档目标：可直接照做。所有命令假定 vLLM 已在 `.venv` 中以可编辑模式安装（`uv pip install -e .`），
> `vllm` / `python` 取自 `.venv/bin`。代码引用格式为 `file_path:line_number`，路径相对仓库根
> `/home/lychee/mycode/vllm`。

## 目录

- [1. 实验目标](#1-实验目标)
- [2. 环境前提](#2-环境前提)
- [3. 显存估算](#3-显存估算)
- [4. 实验矩阵](#4-实验矩阵)
- [5. 部署拓扑](#5-部署拓扑)
- [6. 启动参数清单](#6-启动参数清单)
- [7. 验证步骤](#7-验证步骤)
- [8. 关键注意点清单](#8-关键注意点清单)
- [9. 风险与回退](#9-风险与回退)

---

## 1. 实验目标

在单机 8×H200 上验证 NIXL（NixlConnector）Prefill/Decode 分离的 4+4 部署，对比两种 KV 传输方案：

- **方案 A（GDR / cuda_ipc）**：`kv_buffer_device=cuda`，KV 直接在 GPU 显存间通过 UCX `cuda_ipc` 转移，零拷贝。
- **方案 B（CPU 转发）**：`kv_buffer_device=cpu`，KV 经 host DRAM 中转（`base_worker.py:353-399`）。

验证维度：功能正确性（PD 输出与基线一致）、KV transfer 链路建立（指标日志可见）、吞吐/延迟对比、以及单机内 GDR 与 CPU 转发的差异。同时给出无 PD 分离的 TP=8 基线作为对照。

---

## 2. 环境前提

| 项目 | 内容 |
| --- | --- |
| 机器 | h200-2，单机 8×H200 SXM，每卡 141GB HBM3e，NVLink/NVSwitch 全互联 |
| 用户 | lychee（无 root；`/data1/models` 可写 777；docker socket 无权限） |
| 模型 | MiniMax-M2.5，FP8（dynamic activation, per-128×128 weight block, `float8_e4m3fn`），已下载至 `/data1/models/MiniMax-M2.5`（约 230GB，125 个 safetensors 分片） |
| vLLM | 分支 `v0.25.0`；研发目录 `/home/lychee/mycode/vllm/develop`，仓库根 `/home/lychee/mycode/vllm` |
| 依赖 | `uv pip install nixl`（见 `docs/features/nixl_connector_usage.md:9-14`）；UCX 运行时随 nixl 提供 |

### 模型架构数字（config.json，已从 hf-mirror 核对）

| 字段 | 值 |
| --- | --- |
| `num_hidden_layers` | 62 |
| `hidden_size` | 3072 |
| `num_attention_heads` | 48 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `num_experts` / top-k | 256 / 8（MoE） |
| `vocab_size` | 200064 |
| `max_position_embeddings` | 196608 |
| 量化 | `quant_method=fp8`，`activation_scheme=dynamic`，`weight_block_size=[128,128]`，`fmt=float8_e4m3fn` |

架构分类：**MoE**，非 MLA，非 hybrid SSM/Mamba。在 NIXL 兼容矩阵中 MoE 行对 Basic PD / Hetero TP / Host buffer 均为支持（`docs/features/nixl_connector_compatibility.md:56`）。vLLM 对 `MiniMaxM2ForCausalLM` 有原生实现映射（`vllm/model_executor/models/registry.py:153`，`vllm/model_executor/models/minimax_m2.py:432`，带 `SupportsPP` / `SupportsEagle3`），但因仓库附带 `configuration_minimax_m2.py` / `modeling_minimax_m2.py`，加载自定义 Config 仍需 `--trust-remote-code`。启动用例参考 `tests/compile/h100/test_startup.py:171`（`MiniMaxAI/MiniMax-M2.5`，带 `hf_overrides`）。

---

## 3. 显存估算

### 3.1 权重

FP8 权重约 230GB，TP=4 每卡约 `230 / 4 ≈ 57.5GB`。H200 单卡 141GB，扣除权重后剩余约 `141 − 57.5 ≈ 83GB` 给 KV cache + activations + 临时开销（下文预算取约 80–83GB）。

### 3.2 KV cache（每 token 每卡）

公式：

```
bytes_per_token_per_rank =
    2 (K+V) × num_hidden_layers × num_kv_heads × head_dim × dtype_bytes / TP
```

代入 bf16（dtype_bytes=2）：

```
2 × 62 × 8 × 128 × 2 / 4 = 63,488  B/token/rank   (bf16)
```

FP8 KV cache（dtype_bytes=1）：

```
2 × 62 × 8 × 128 × 1 / 4 = 31,744  B/token/rank   (fp8)
```

> 说明：每 token 字节数 FP8 是 bf16 的一半，因此同等显存下 FP8 KV 可容纳 token 数约为 bf16 的 2 倍。

### 3.3 容量估算（每卡，预算 ~83GB）

| KV dtype | 每 token 每卡 | 单卡可容纳 token 数 |
| --- | --- | --- |
| bf16 | 63,488 B | `83 × 1024^3 / 63,488 ≈ 1.40M` |
| fp8 | 31,744 B | `83 × 1024^3 / 31,744 ≈ 2.80M` |

绑定约束是**同时在途**序列的 KV 总量，而非 `num_prompts` 总量。本实验单请求 7500+200=7700 token，即使数十路并发，在途 KV 远小于 1.4M，bf16 KV 充裕。

### 3.4 CPU 转发（方案 B）额外开销

`kv_buffer_device=cpu` 时，转移侧在 host DRAM 上分配与待传 KV block 同 shape 的 `host_xfer_buffers`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:363-369`），NIXL 内存类型为 `DRAM`（`base_worker.py:388-394`）。该 buffer 占用的是 **CPU DRAM 而非 GPU 显存**，大小与在途转移窗口成正比；需确认单机 DRAM 足够容纳一侧同时在途的 KV（本实验量级轻松满足）。

### 3.5 结论

- 4+4（TP=4 + TP=4）可行；bf16 KV 足够支撑 7500/200 负载。
- 若显存吃紧可切 FP8 KV cache，容量约翻倍；但 **FP8 KV 必须用静态 scale**（从 checkpoint 加载），`dynamic`（运行时 per-block scale）当前不支持，因为 per-block scale 不会随 KV 数据一起转移（`docs/features/nixl_connector_compatibility.md:98-104`）。MiniMax-M2.5 权重是 FP8 dynamic，但权重量化方案与 KV cache 量化方案相互独立——若要对 KV 用 FP8，需自行把静态 scale 烘焙进 checkpoint 后再加载。

---

## 4. 实验矩阵

| 编号 | 方案 | `kv_buffer_device` | UCX_TLS | P_TP / D_TP | 状态 |
| --- | --- | --- | --- | --- | --- |
| A | NIXL GDR（cuda_ipc） | `cuda` | `cuda_ipc,cuda_copy,tcp` | 4 / 4 | 主实验 |
| B | NIXL CPU 转发 | `cpu` | `cuda_ipc,cuda_copy,tcp` | 4 / 4 | 主实验 |
| C | 异构 TP（可选） | `cuda` | `cuda_ipc,cuda_copy,tcp` | 4 / 2 或 2 / 4 | 可选 |
| 基线 | 无 PD 分离 | — | — | 8（单实例） | 对照 |

- 实验 A/B/C 均通过 `toy_proxy_server.py` 路由（端口 8000）。
- 基线为单实例 TP=8，客户端直连，不经 proxy。
- 异构 TP 在 MoE 架构下支持（`docs/features/nixl_connector_compatibility.md:56`），`tensor-parallel-size` 不进入兼容性哈希（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py:94-96,111-126`）。

### 每组基准参数

固定 `--random-input-len 7500 --random-output-len 200`，对每组扫描：

| `--num-prompts` | `--burstiness` |
| --- | --- |
| 50 / 100 / 200 | 1.0 / 100 |

`--request-rate` 默认 2；burstiness 100 模拟突发，burstiness 1.0 模拟泊松到达（参数风格参考 `examples/disaggregated/mooncake_connector/run_mooncake_connector.sh:213-216`）。

---

## 5. 部署拓扑

```
                ┌───────────────┐
   client ────► │  proxy :8000  │  (toy_proxy_server.py)
                └───────┬───────┘
            round-robin │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 ┌──────────────┐               ┌──────────────┐
 │ Prefill :8100│               │ Decode :8200 │
 │ GPU 0,1,2,3  │  NIXL xfer    │ GPU 4,5,6,7  │
 │ TP=4         │ ◄───────────► │ TP=4         │
 │ side_ch 5600 │               │ side_ch 5601 │
 └──────────────┘               └──────────────┘
```

| 角色 | `CUDA_VISIBLE_DEVICES` | TP | serve 端口 | side_channel |
| --- | --- | --- | --- | --- |
| Prefill (P) | 0,1,2,3 | 4 | 8100 | 5600 |
| Decode (D) | 4,5,6,7 | 4 | 8200 | 5601 |
| Proxy | — | — | 8000 | — |
| 基线 | 0..7 | 8 | 8300 | — |

- side_channel 每实例唯一：TP-only 部署下实际端口 = `VLLM_NIXL_SIDE_CHANNEL_PORT + data_parallel_index`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:64-68`），DP=1 即等于 base。P=5600、D=5601 不冲突。
- 单机 UCX 传输：`UCX_TLS=cuda_ipc,cuda_copy,tcp`（参考 `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_vllm_launcher.sh:31`）。`cuda_ipc` 提供 GPU 显存零拷贝，`cuda_copy` / `tcp` 作为回退。
- NCCL 环境变量对 NIXL 无效，传输只由 UCX 控制（`docs/features/nixl_connector_usage.md:38-39`）。

---

## 6. 启动参数清单

### 6.1 实验 A — GDR（cuda_ipc）

**Prefill：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
UCX_TLS=cuda_ipc,cuda_copy,tcp \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
VLLM_KV_CACHE_LAYOUT=HND \
vllm serve /data1/models/MiniMax-M2.5 \
  --port 8100 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
```

**Decode：**

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
UCX_TLS=cuda_ipc,cuda_copy,tcp \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
VLLM_KV_CACHE_LAYOUT=HND \
vllm serve /data1/models/MiniMax-M2.5 \
  --port 8200 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
```

**Proxy：**

```bash
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8000 \
  --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts  localhost --decoder-ports  8200
```

### 6.2 实验 B — CPU 转发

在 A 的基础上，P 与 D 的 `--kv-transfer-config` 各加一个 `"kv_buffer_device":"cpu"` 字段，其余不变（`tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh:53-58` 即按此拼装）：

```bash
# Prefill
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_buffer_device":"cpu"}'

# Decode
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_buffer_device":"cpu"}'
```

> `cuda_ipc` 仅对 GPU 显存生效；CPU 转发时数据在 host DRAM，实际走 `sm`/`tcp` 路径，UCX_TLS 保持不变无害。

### 6.3 基线（无 PD 分离，TP=8）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve /data1/models/MiniMax-M2.5 \
  --port 8300 \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

### 6.4 通用参数说明

| 参数 | 说明 |
| --- | --- |
| `--max-model-len 8192` | 覆盖 7500+200=7700；非 handshake 硬性，但 P/D 实践对齐 |
| `--gpu-memory-utilization` | P/D 可不同（KV blocks 数由各自可用显存决定，`docs/features/nixl_connector_compatibility.md:86-90`） |
| `--enforce-eager` | 非必须；CUDA graph 与 NIXL 兼容（`docs/features/nixl_connector_compatibility.md:22`）。首跑建议加 `--enforce-eager` 排障，确认链路后再去掉测性能 |
| `--trust-remote-code` | MiniMax-M2.5 附带自定义 Config/Model 文件，必需 |
| `VLLM_KV_CACHE_LAYOUT=HND` | NixlConnector 默认 HND 以获得最优转移性能（`docs/features/nixl_connector_compatibility.md:92-96`） |
| `--kv-transfer-config` | 连接器、角色、失败策略；`kv_load_failure_policy=fail`（默认）失败即报错，避免 D 端隐式重算（`docs/features/nixl_connector_usage.md:393-401`） |

### 6.5 handshake 必须一致项

兼容性哈希在握手期校验（`metadata.py:79`，因子见 `metadata.py:111-126`；文字说明见 `docs/features/nixl_connector_compatibility.md:76-81`）。P/D 必须一致：

- vLLM 版本 + NIXL connector 版本
- 模型（architecture、dtype、`num_kv_heads`、`head_size`、`num_hidden_layers`）
- `attn_backend_name`（P/D 用同一 `--attention-backend`）
- `cache_dtype`（KV cache 数据类型）
- `cross_layers_blocks`、`is_hma_enabled`

可安全不同：`tensor-parallel-size`、`block-size`、KV blocks 数（`docs/features/nixl_connector_compatibility.md:86-90`）。

---

## 7. 验证步骤

### 7.1 健康检查

```bash
# 各 vllm 实例就绪（P/D）
curl -s localhost:8100/v1/completions >/dev/null && echo "P up"
curl -s localhost:8200/v1/completions >/dev/null && echo "D up"
# proxy 状态
curl -s localhost:8000/healthcheck
```

`/healthcheck` 返回 prefill/decode 实例数（`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:274-281`）。

### 7.2 单请求正确性（temperature=0）

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/data1/models/MiniMax-M2.5","prompt":"vLLM 的核心思想是","temperature":0,"max_tokens":64}' \
  | jq -r '.choices[0].text'
```

### 7.3 KV transfer 链路确认

在 P 与 D 的日志中确认周期性指标行（格式见 `docs/features/nixl_connector_usage.md:431-435`）：

```
KV Transfer metrics: Num successful transfers=N, Avg xfer time (ms)=..., P90 xfer time (ms)=..., Throughput (MB/s)=...
```

重点看 `Num successful transfers > 0`、`Throughput (MB/s)` 非 0。CPU 转发（实验 B）的吞吐应明显低于 GDR（实验 A）。

### 7.4 benchmark

```bash
vllm bench serve \
  --port 8000 --backend vllm --model /data1/models/MiniMax-M2.5 \
  --dataset-name random \
  --random-input-len 7500 --random-output-len 200 \
  --num-prompts 200 --burstiness 100 --request-rate 2
```

`num-prompts` 与 `burstiness` 按 [第 4 节](#4-实验矩阵) 扫描。基线组把 `--port` 改为 8300 并直连。

### 7.5 正确性对比（PD 分离 vs 基线）

同一 prompt、`temperature=0`、固定 `seed`，分别打 proxy（:8000）与基线（:8300），逐 token 比对输出文本应一致（PD 分离不改变数值结果，仅改变调度）。可复用 `tests/v1/kv_connector/nixl_integration/test_accuracy.py` 的对比思路（脚本入口见 `run_accuracy_test.sh:278-280`）。

---

## 8. 关键注意点清单

1. **proxy 不能省**：客户端必须打 `:8000`，由 `toy_proxy_server.py` 先发 P（`max_tokens=1`，`do_remote_decode=true`）取回 `kv_transfer_params`，再转发给 D 流式返回（`toy_proxy_server.py:155-251`）。直接打 P 或 D 不会触发 PD 协作。
2. **side_channel_port 每实例唯一**：P=5600、D=5601；多实例时再加 DP rank 偏移（`base_scheduler.py:64-68`）。
3. **CPU 转发占 DRAM**：`kv_buffer_device=cpu` 的 `host_xfer_buffers` 在 host 内存（`base_worker.py:363-369`），不占 GPU 显存；确保单机 DRAM 足够。
4. **双向 KV 仅 GDR 支持**：bidirectional（D→P 拉取）目前仅支持 device-buffer（CUDA），host-buffer（CPU 转发）暂不支持（`docs/features/nixl_connector_usage.md:314-315`）。本实验为单轮，不开 bidirectional。
5. **FP8 KV cache 必须静态 scale**：`dynamic` / per-block scale 不随 KV 转移，握手会失败/结果错（`docs/features/nixl_connector_compatibility.md:98-104`）。权重 FP8 与 KV FP8 是两件事，不要混用 dynamic。
6. **trust-remote-code**：MiniMax-M2.5 自带 `configuration_minimax_m2.py` / `modeling_minimax_m2.py`，必加 `--trust-remote-code`；vLLM 对该架构有原生实现（`registry.py:153`，`minimax_m2.py:432`）。
7. **UCX 而非 NCCL**：NIXL 传输受 UCX_TLS / UCX_NET_DEVICES 控制，NCCL_× 变量无效（`docs/features/nixl_connector_usage.md:38-39`）。
8. **attn_backend / cache_dtype 对齐**：P/D 必须相同，否则兼容哈希不一致（`metadata.py:122-124`）。

---

## 9. 风险与回退

| 风险 | 判断 | 回退 |
| --- | --- | --- |
| `cuda_ipc` 建链失败 | 日志无 `KV Transfer metrics`，或 UCX 报 `cuda_ipc` 错 | UCX_TLS 退化为 `cuda_copy,tcp` 或 `sm,tcp`；功能可跑，性能下降 |
| GPU OOM | vllm 启动期或运行期 OOM | 降 `--gpu-memory-utilization`；或切 FP8 KV（须静态 scale，见 [8.5](#8-关键注意点清单)）；或缩短 `--max-model-len` |
| CPU 转发过慢（实验 B） | `Throughput (MB/s)` 远低于 A，TTFT 劣化 | 仅作功能验证，性能结论以实验 A 为准 |
| 握手失败 | 启动期 hash 不匹配报错 | 核对 [6.5](#65-handshake-必须一致项) 各项；确无特殊需求不要关闭 `enforce_handshake_compat`（`docs/features/nixl_connector_compatibility.md:83-84`） |
| 异构 TP（实验 C）异常 | D 端 KV 头分配报错 | 先确保 A（同构 4+4）通过，再试 C；MoE 异构 TP 理论支持（`docs/features/nixl_connector_compatibility.md:56`） |
