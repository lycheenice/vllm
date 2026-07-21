# vLLM NIXL PD 分离性能分析与优化方案

**日期**: 2026-07-21  
**作者**: lychee  
**模型**: MiniMax-M2.5 (230GB, fp8 dynamic w8a8, 256 experts top-8, 62 layers, GQA 48 heads / 8 kv_heads)  
**硬件**: h200-2 (8×H200 SXM 141GB, NVLink/NVSwitch)  
**vLLM 版本**: v0.25.0  
**基准测试**: kvcache-benchmarks `ramp_test.py`, `codex_swebenchpro.json` (610 records, ~30 turns/trial)  

---

## 一、性能对比：PD Basic vs Baseline

### 1.1 核心数据

| C | 指标 | Baseline (TP8) | PD Basic (P4+D4) | 差异 |
|---|------|----------------|-------------------|------|
| 1 | TTFT P50 | 237ms | 561ms | **+137%** |
| | TPOT P50 | 7.7ms | 8.5ms | +10% |
| | 吞吐 (tok/s) | 112 | 90 | -20% |
| | 延迟 P50 | 5,843ms | 4,448ms | -24% (唯一优势) |
| 4 | TTFT P50 | 205ms | 437ms | **+113%** |
| | TPOT P50 | 9.2ms | 11.1ms | +21% |
| | 吞吐 | 321 | 276 | -14% |
| 16 | TTFT P50 | 233ms | 523ms | **+124%** |
| | TPOT P50 | 16.0ms | 28.3ms | **+77%** |
| | 吞吐 | 803 | 488 | **-39%** |
| 32 | TTFT P50 | 255ms | 578ms | **+127%** |
| | TPOT P50 | 29.4ms | 50.3ms | **+71%** |
| | 吞吐 | 883 | 615 | -30% |
| | 错误率 | 0.8% | **64.8%** | 灾难性 |
| 64 | 吞吐 | 1,085 | 0 | **100% 崩溃** |
| | 错误率 | 0.9% | **100%** | 完全失败 |

### 1.2 结论

PD Basic 在所有并发级别下均为净负收益——峰值吞吐下降 43%，TTFT 恶化 2-3 倍，C≥32 时灾难性崩溃。唯一优势是 C=1 单用户时延迟略低（-24%），因为 decode 独立节点消除了单实例内 prefill/decode 资源争用。

---

## 二、性能问题根因分析

### 2.1 根因 1：无 D→P KV 回传（最核心）

`disagg_proxy_demo.py` 的工作流：

```
Turn 1: Client → Proxy → P(prefill) → NIXL KV transfer → D(decode) → response
Turn 2: Client → Proxy → D(decode) directly  ← P 被绕过
Turn 3: Client → Proxy → D(decode) directly
...
```

问题链条：
1. Turn 1 后 KV cache 在 D 节点
2. Turn 2 的 prompt = Turn 1 prompt + assistant response + new user input
3. D 节点虽然有前一轮的 KV，但新生成的 assistant response tokens + new user input tokens 没有 KV cache——D 必须做 prefill 计算
4. D 节点为 decode 优化（`max_num_seqs` 高，`max_num_batched_tokens` 低），做 prefill 效率极差
5. P 节点完全空闲，却因拿不到 D 的 KV，无法复用前缀

对比 Baseline：单实例 TP8 下多轮对话 Turn 2+ 的 prefix cache hit rate 达 91-96%。

量化影响：每条 trial ~30 轮，PD Basic 只有第 1 轮用 P 做 prefill，后 29 轮全在 D 上做 prefill+decode 混合。

### 2.2 根因 2：TP4 vs TP8 算力减半

| 节点 | GPU 数 | prefill 算力 | decode 算力 |
|------|--------|-------------|-------------|
| Baseline | 8 (TP8) | 8× H200 | 8× H200 |
| PD P | 4 (TP4) | 4× H200 | — |
| PD D | 4 (TP4) | — | 4× H200 |

- Prefill：TP4 只有 4 卡并行，prefill 时间约翻倍
- Decode：TP4 的 batch 吞吐约 TP8 一半，C=16 时 TPOT 暴涨 77%
- C=64 崩溃：D 节点 4 卡 KV cache 显存不足以容纳 64 并发请求

### 2.3 根因 3：NIXL KV 传输开销

每个首轮请求的 TTFT 包含：
1. P 计算 prefill（比 baseline 慢，因 TP4）
2. NIXL 传输 KV cache P→D（通过 NVLink/RDMA）
3. D 加载 KV cache 到本地显存
4. D 开始 decode

MiniMax-M2.5: 62 层 × 8 kv_heads × 128 head_dim × seq_len × fp8 = 可观数据量。C=1 时 TTFT 从 237ms→561ms（+324ms），相当部分是 NIXL 传输。但此开销仅 Turn 1 发生，如实现 D→P 回传则被摊薄到 1/30。

### 2.4 根因 4：Proxy 序列化开销

FastAPI 外部代理引入额外 HTTP 跳转，C=1 约 20-50ms，高并发时代理本身成为瓶颈。

### 2.5 根因 5：kv_role 配置可能不正确

当前 `run_all.sh` 使用旧式 `kv_producer`/`kv_consumer`，而 v0.25.0 官方集成测试统一使用 `kv_both`。这可能导致 NIXL 双向 KV 传输未正确启用。

---

## 三、当前缓存层级状态

| 层级 | 机制 | 状态 | 说明 |
|------|------|------|------|
| L0 (GPU prefix cache) | `--enable-prefix-caching` | ✅ 已开 | GPU 显存内 KV 复用，baseline hit rate 92-96% |
| L1 (跨实例 NIXL) | `NixlConnector` | ✅ 已开 | P→D KV 传输（单向） |
| **L2 (CPU offload)** | `OffloadingConnector` | ❌ 未开 | KV 卸载到 CPU 内存/磁盘 |

当前 `kv-transfer-config` 仅配了单个 NixlConnector，无 L2。

---

## 四、优化方案

### 4.1 方案 1：开启 L2 缓存（MultiConnector: NIXL + OffloadingConnector）

**优先级**: P0  
**框架能力**: vLLM `MultiConnector`（`vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`）支持组合多个子 connector。

MultiConnector 语义（源码 `multi_connector.py:128`）：
- **Load**：按子 connector 顺序查询 `get_num_new_matched_tokens`，取命中最大值（NIXL 给 0 tokens → CPU offload 给 N tokens → 取 N）
- **Save**：`save_kv_layer` 对所有子 connector 调用（GPU KV → NIXL 传输 + CPU 内存留存）
- `wait_for_layer_load` 所有子完成才算完成（与语义）
- scheduler 看到的是单个 connector，不感知 Multi

配置改法（参考 `tests/v1/kv_connector/nixl_integration/run_multi_connector_accuracy_test.sh:49`）：

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
      {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
       "kv_connector_extra_config": {"cpu_bytes_to_use": 17179869184}}
    ]
  }
}
```

关键参数（参考 `docs/features/kv_offloading_usage.md`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `cpu_bytes_to_use` | 17179869184 (16GB) ~ 128GB | CPU 内存总量（所有 worker 共享），h200-2 内存充裕建议 64-128GB |
| `block_size` | 默认=GPU block size | offloaded block token 数，需为 GPU block size 倍数 |
| `eviction_policy` | `lru`（默认） | CPU tier 淘汰策略 |
| `spec_name` | `CPUOffloadingSpec`（默认） | 单 CPU tier；`TieringOffloadingSpec` 为多层（CPU+FS） |

多层 L2（CPU + 文件系统）配置：

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 10737418240,
    "block_size": 16,
    "secondary_tiers": [
      {"type": "fs", "root_dir": "/mnt/kv_cache", "n_read_threads": 32, "n_write_threads": 16}
    ]
  }
}
```

L2 对多轮 agent 场景的价值：
1. D 节点 GPU KV eviction 后旧轮次 KV 落到 CPU RAM，新轮次仍可命中
2. 多轮间隔较长（agent 思考），GPU prefix cache 过期后 CPU L2 容量大 10-20 倍仍可命中
3. 命中 L2 时只需 GPU←CPU DMA 拷贝（`cudaMemcpyAsync` 异步），远快于重新计算

**同时修正 kv_role**：从旧式 `kv_producer`/`kv_consumer` 改为 `kv_both`，与 v0.25.0 官方测试一致。

### 4.2 方案 2：实现双向 PD（D→P KV 回传）

**优先级**: P0  
**框架能力**: vLLM NixlConnector 支持双向 KV 传输。`disagg_proxy_multiturn_autoid.py` 已为此设计。

工作流：
```
Turn 1: P(prefill) → KV P→D → D(decode) → response
Turn 1 结束后: D → KV D→P → P 接收新 KV（含 assistant response 的 KV）
Turn 2: P(prefill 新增 tokens，复用已有 KV) → KV P→D → D(decode)
...循环
```

预期改善：
- P 节点复用 prefix cache，turn 2+ 只 prefill 新增 tokens（几十~几百个），而非整个对话历史
- TTFT 大幅下降（prefill 计算量减少 90%+）
- D 节点不再做 prefill，专注 decode，TPOT 回到正常水平

实现要点：
- D 节点每轮 decode 结束后，通过 NIXL 将新增 KV 回传给 P
- P 节点接收 KV 后更新本地 prefix cache
- `conversation_id` 路由：同一对话始终 P→D→P→D 交替
- 需修正 `kv_role` 为 `kv_both`

### 4.3 方案 3：非对称 GPU 分配

**优先级**: P1  
**框架能力**: vLLM PD 支持任意 TP 配置，只要 P 和 D 的 `kv_cache_shape` 兼容（layers/kv_heads/head_dim 一致，TP 不影响 KV shape）。

当前 P4+D4 对多轮 agent 工作负载不合理：

| 工作负载 | Prefill 负载 | Decode 负载 | 推荐配置 |
|----------|-------------|-------------|----------|
| 单轮短 prompt | 高 | 低 | P6+D2 或 P8 inline |
| 多轮长对话 | 中高（每轮新增少） | 高（持续生成） | P4+D4 或 P3+D5 |
| 我们的 case（~30轮，长context） | 每轮新增~500 token | 每轮~1000 decode token | **P2+D6** 或 **P3+D5** |

理由：多轮对话中随轮次增加 KV cache 积累，D 节点显存压力远大于 P。且每轮新增 prompt tokens 少（~500），P 不需太多算力。D 需更多 GPU 来：
1. 容纳更大总 KV cache（更多并发）
2. 提高 decode batch 吞吐

若已实现方案 2（双向 KV），P 的 prefill 计算量极小（仅新增 tokens），P2 甚至足够。

### 4.4 方案 4：调优 `max_num_batched_tokens`（P 节点）

**优先级**: P2  
**框架能力**: vLLM 支持 prefill/decode batch token 数独立调优。

当前 P 节点 `--max-num-batched-tokens 8192`，对 MiniMax-M2.5 长 prompt（~30k token）需 4 个 chunk。

建议 P 节点增大到 `32768` 或 `65536`：
- 减少 chunked prefill 步数，降低 prefill 延迟
- 允许同时 prefill 多请求的 chunk，提高 P 吞吐
- 代价：更高瞬时显存使用

### 4.5 方案 5：调优 `max_num_seqs`（D 节点）

**优先级**: P2  
**框架能力**: `max_num_seqs` 控制 decode batch 大小上限。

当前 D 节点 `--max-num-seqs 128`。4×H200 (564GB) 减模型权重(~115GB)，约 450GB 可用于 KV cache。每序列 ~30k token × 62 层 × 8 kv_heads × 128 dim × 1 byte(fp8) ≈ 1.9GB/序列。450GB / 1.9GB ≈ 237 序列。

建议：
- 如保持 P4+D4：降到 `192` 避免 OOM
- 如改 P2+D6：可设 `256` 或更高
- 根本解法是方案 3（增加 D 的 GPU 数）

### 4.6 方案 6：KV Lease 调优

**优先级**: P2  
**框架能力**: NixlConnector 支持 `kv_lease_duration` 参数，控制 KV cache 在对端的存活时间。

当前 `kv_lease_duration: 60`（秒）。多轮对话间隔可能较长（agent 思考），lease 太短 P 上 prefix cache 可能在 turn 2 前被驱逐。建议 `300`（5 分钟）。

### 4.7 方案 7：消除 Proxy 中间层（长期）

**优先级**: P3  
**框架能力**: vLLM v0.25.0 发展中内置 PD 路由能力。

当前 proxy 开销：HTTP 序列化 ~5-10ms/请求，streaming 每 chunk ~1ms，C=64 时 FastAPI 单进程成瓶颈。

短期：用更高性能代理（Rust/Go，或 nginx 级反代）。  
长期：使用 vLLM 内置路由。

### 4.8 优化优先级矩阵

| 方案 | 预期收益 | 实现难度 | 优先级 |
|------|---------|---------|--------|
| 1. 开启 L2 (MultiConnector) | D 节点 eviction 后仍可命中, C≥32 可用 | 低（改配置） | **P0** |
| 2. 双向 KV 回传 | TTFT -60%, TPOT -40% | 中（已有脚本框架） | **P0** |
| 3. 非对称 P2+D6 | C=64 可用, 吞吐 +50% | 低（改配置） | **P1** |
| 4. P 增大 batched tokens | TTFT -20% | 低（改参数） | P2 |
| 5. D 调 max_num_seqs | 减少 OOM | 低（改参数） | P2 |
| 6. KV lease 调优 | 减少缓存驱逐 | 低（改参数） | P2 |
| 7. 去除 proxy | 延迟 -10ms | 高 | P3 |

---

## 五、建议的下一轮实验配置

### 5.1 Phase 3A：PD Bidir + L2

```bash
# P 和 D 共用配置
KV_CFG='{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
      {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
       "kv_connector_extra_config": {"cpu_bytes_to_use": 68719476736}}
    ]
  }
}'

docker run ... \
  --tensor-parallel-size 4 \
  --port 8100 \
  --kv-transfer-config "$KV_CFG" \
  $COMMON_ARGS

docker run ... \
  --tensor-parallel-size 4 \
  --port 8200 \
  --kv-transfer-config "$KV_CFG" \
  $COMMON_ARGS

# Proxy: disagg_proxy_multiturn_autoid.py
```

### 5.2 Phase 3B：PD Bidir + L2 + 非对称 P2+D6

```bash
# P 节点: 2 GPU
-e CUDA_VISIBLE_DEVICES=0,1 \
--tensor-parallel-size 2 \
--max-num-batched-tokens 65536

# D 节点: 6 GPU
-e CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
--tensor-parallel-size 6 \
--max-num-seqs 256
```

### 5.3 Phase 4：参数调优

在 Phase 3 最优配置基础上调优：
- `kv_lease_duration`: 60 → 300
- `max_num_batched_tokens`: 8192 → 32768/65536（P 节点）
- OffloadingConnector `eviction_policy`: lru → arc

---

## 六、相关文件

- `develop/vllm/deploy/minimax/run_all.sh` — 部署脚本（需修改）
- `develop/vllm/deploy/minimax/disagg_proxy_multiturn_autoid.py` — 双向 PD 代理
- `develop/vllm_pd_baseline_vs_pdbasic_report.md` — Phase 1/2 基准测试报告
- `develop/vllm/minimax_vllm_pd_plan.md` — 4 阶段实验计划
- `develop/01_pd_backend_research.md` — vLLM NIXL/Mooncake 研究
- vLLM 源码:
  - `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py` — MultiConnector
  - `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` — CPU offload
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` — NIXL connector
  - `vllm/distributed/kv_transfer/kv_connector/factory.py` — connector 注册
- vLLM 文档:
  - `docs/features/disagg_prefill.md` — PD 文档
  - `docs/features/kv_offloading_usage.md` — OffloadingConnector 用法
- 官方集成测试:
  - `tests/v1/kv_connector/nixl_integration/run_multi_connector_accuracy_test.sh` — MultiConnector 配置参考
- Wiki:
  - `develop/wiki/07-distributed/kv-transfer/multi.md` — MultiConnector 中文解析
