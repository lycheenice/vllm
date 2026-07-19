# MiniMax-M2.5 PD 分离对比实验设计

> 模型: MiniMax-M2.5 (FP8 dynamic, w8a8, `MiniMaxM2ForCausalLM`)
> 硬件: h200-2, 8×H200 SXM 141GB, 主机 RAM 2015 GB, NVLink/NVSwitch
> 推理框架: sglang v0.5.15.post1 (sglang 有 `srt/models/minimax_m2.py` 原生支持)
> 发压工具: kvcache-benchmarks (codex_swebenchpro_traces, 610 records, p50=30 turns, p90=57 turns)

---

## 1. 模型架构与关键差异（vs GLM-5.2-W4AFP8）

| 维度 | GLM-5.2-W4AFP8 | MiniMax-M2.5 | 对 PD 的影响 |
|---|---|---|---|
| 注意力机制 | MLA (kv_lora_rank=512) | **标准 GQA** (48 heads, 8 kv_heads, head_dim 128) | hicache 兼容! 无 element_size=656 问题 |
| 量化 | w4afp8 | **FP8 dynamic (w8a8)** | 无 DeepGEMM JIT, 启动秒级 |
| 层数 | 78 | **62** | KV/token 更小 |
| hidden | 6144 | 3072 | 权重更轻 |
| 专家 | 256 top-8 | 256 top-8 | 相同 |
| vocab | — | 200064 | — |
| max_position | 1048576 | 196608 | 实际 context 受限 |
| 总大小 | ~400GB (40 shards) | ~215GB (125 shards) | TP8 权重仅 ~27GB/GPU |
| KV/token (TP8, fp8) | MLA latent ~80KB | **~16KB** (2×1×128×62) | L1 池容量 ~6× 于 GLM |
| L1 池预估 (TP8) | 441K tokens | ~5-6M tokens | **cache 驱逐不再是瓶颈** |

### 关键结论

1. **hicache 可用**: MiniMax-M2.5 标准 GQA，无 MLA kernel 兼容问题（GLM 的 P1.1 失败不适用）
2. **L1 池极大**: KV/token 仅 ~16KB（TP8），L1 池可容纳 5-6M tokens → c64 (64×25k=1.6M) 远在容量内
3. **无冷启动偏差**: 无 DeepGEMM JIT 编译，启动秒级，冷栈==热栈
4. **L1 驱逐不应是问题**: 5-6M tokens 池容量 vs codex traces 单 session ~25k token → 理论 c128 无驱逐

---

## 2. 实验矩阵

### 2.1 基线: TP8 非分离 (单服务器混合 prefill+decode)

| 参数 | 值 |
|---|---|
| tp_size | 8 (GPU 0-7) |
| 模式 | 非 PD (sglang 标准 serve) |
| chunked_prefill_size | 8192 (P1.3 验证有效) |
| mem_fraction_static | 0.90 (MiniMax 权重轻, 可给更多 KV) |
| kv_cache_dtype | fp8_e4m3 |
| context_len | 100000 (max_position=196608, 取实用值) |
| max_running_requests | 128 (单服务器, 无 PD 开销) |
| cuda_graph | decode full, max_bs 256 |

### 2.2 PD Config A: 基础 PD (对照 GLM exp1)

| 参数 | 值 |
|---|---|
| 拓扑 | P=TP4 (GPU 0-3) + D=TP4 (GPU 4-7) |
| NIXL | UCX cuda_ipc |
| chunked_prefill_size | 32768 (GLM exp1 同值) |
| SGLANG_DISAGG_QUEUE | 8, THREAD 12, TIMEOUT 600 |
| decode radix cache | 启用 |
| router | 标准 (默认熔断) |
| hicache | 关闭 |

目的: 与 GLM exp1 对照, 验证 MiniMax 是否也有 c16 cache 雪崩。

### 2.3 PD Config B: P1.3 验证配置 (chunked 8192 + nocb)

| 参数 | 值 |
|---|---|
| 拓扑 | P=TP4 + D=TP4 |
| chunked_prefill_size | **8192** |
| router | **--disable-circuit-breaker** |
| 其余 | 同 Config A |

目的: 验证 GLM P1.3 的优化是否对 MiniMax 同样有效。

### 2.4 PD Config C: P1.3 + hicache (MiniMax 优势配置)

| 参数 | 值 |
|---|---|
| 拓扑 | P=TP4 + D=TP4 |
| chunked_prefill_size | 8192 |
| router | --disable-circuit-breaker |
| **hicache** | **--enable-hierarchical-cache --hicache-ratio 3 --hicache-io-backend direct** |
| 其余 | 同 Config B |

目的: MiniMax 标准 MHA 下 hicache 可用, 验证 L2 缓存对 PD 跨 turn cache 复用的增益。

---

## 3. 发压方案

### 参数

| 参数 | 值 |
|---|---|
| 工具 | kvcache-benchmarks/scripts/ramp_test.py |
| 数据集 | codex_swebenchpro_traces (610 records) |
| 并发级别 | **1, 4, 16, 32, 64** |
| trials_per_user | 5 |
| max_tokens | 4096 |
| turns | **all** (不限轮次, 全量) |
| streaming | 开启 |

### 每组测试顺序

1. 停旧栈 → 启新栈 → 等 health → 冒烟测试
2. 按 1→4→16→32→64 顺序逐级发压
3. 每级记录: summary.json, result.json, run.log
4. 每级采集运行时 metrics (prefill/decode /metrics)
5. 组间重启栈清退化态

### 总组数

| 组 | 配置 | 并发级数 | 预计时长 |
|---|---|---|---|
| 1 | TP8 baseline | 5 (1/4/16/32/64) | ~3-5h |
| 2 | PD Config A | 5 | ~5-8h |
| 3 | PD Config B | 5 | ~5-8h |
| 4 | PD Config C | 5 | ~5-8h |
| **总计** | 4 组 | 20 级 | ~18-29h |

---

## 4. 对比指标

| 维度 | 指标 | 来源 |
|---|---|---|
| 成功率 | succeeded / total | bench summary |
| 延迟 | ttft p50/p90, tpot p50/p90 | bench summary |
| 缓存 | cache_rate mean/p50 | bench summary |
| 稳定性 | router 503 数, prefill/decode restart 次数 | docker logs / metrics |
| L1 利用 | kv_used_tokens, evicted_tokens | /metrics |
| L2 利用 | hicache hit rate (Config C) | /metrics |
| 并行 | num_prefill_inflight, num_decode_running | /metrics |
| 吞吐 | gen_throughput | /metrics |

### 核心对比问题

1. **TP8 baseline vs PD**: 同 8 卡, PD 分离是否比混合更快？在什么并发级别 PD 开始胜出？
2. **Config A vs B**: chunked 8192 对 MiniMax 是否同样关键？（MiniMax L1 池大, cache 驱逐不严重, 可能 chunked 影响小）
3. **Config B vs C**: hicache 在 PD 模式下是否有增益？（L1 已 5-6M tokens, L2 增量可能有限）
4. **MiniMax vs GLM 交叉**: 同配置下两个模型的 PD 行为差异

---

## 5. 前置条件

### 模型下载

MiniMax-M2.5 当前**不在 h200-2** (`/data1/models/` 为空)。需下载:

```bash
# h200-2 有 DNS 问题, 可能需从其他机器下载后 scp
# HuggingFace repo: Eco-Tech/MiniMax-M2.5-w8a8-QuaRot (w8a8 FP8 量化版)
# 原始 repo: MiniMaxAI/MiniMax-M2 (BF16, 更大)
# 目标路径: /data1/models/MiniMax-M2.5/
# 大小: ~215GB, 125 shards
# 磁盘: /data1 剩 1.1TB, 充足
```

### 启动速度预期

| 模型 | JIT 编译 | 权重加载 | 总启动 |
|---|---|---|---|
| GLM-5.2-W4AFP8 | 10-20min (DeepGEMM) | 44s | ~15-25min |
| MiniMax-M2.5 | **0** (无 w4afp8) | ~20s | **<1min** |

---

## 6. 风险与注意事项

1. **模型下载**: h200-2 DNS 问题可能阻断 HF 下载, 需从 88-236 或其他机器中转
2. **sglang MiniMax-M2 支持**: sglang 有 `srt/models/minimax_m2.py` 和 `MiniMaxM2ForCausalLM` 类, 但实际加载需验证 chat template / tokenizer
3. **FP8 精度**: MiniMax-M2.5 w8a8 可能与 vLLM 的 FP8 实现有差异, sglang 对 w8a8 的支持需验证
4. **PD transfer 兼容**: MiniMax 标准 MHA 的 KV layout 应与 NIXL 兼容, 但需实际验证
5. **hicache ratio**: MiniMax 标准 MHA, element_size 应为标准值, 但仍需监控 "Unsupported element_size" 警告
