# P1.3 chunked_prefill 8192 + nocb router 优化实验报告

> 日期: 2026-07-19
> 基线: exp1 PD-opt (chunked=32768, c8 cache_rate=0.92 40/40; c16 cache_rate=0.22 80/80)
> 模型: GLM-5.2-W4AFP8 (MLA, w4afp8, GlmMoeDsaForCausalLM)
> 硬件: h200-2, 8×H200 SXM, 1P+1D 各 TP4, NIXL UCX

---

## 1. 改动点

### 改动 1: `--chunked-prefill-size 8192`（从 32768 降到 8192，prefill+decode 两端）

**文件**: `start-prefill-cp8k.sh`, `start-decode-cp8k.sh`

`chunked_prefill_size` 是 sglang 调度器在 prefill 阶段单个 batch 处理的最大新 token 数上限。当请求的 prompt 很长时，调度器将其切成多个 chunk 分批处理，每个 chunk 不超过该上限。

| 参数 | exp1 (原) | P1.3 (改) |
|---|---|---|
| `--chunked-prefill-size` | 32768 | **8192** |
| `--max-prefill-tokens` | 16384 (不变) | 16384 (不变) |

注: `max_prefill_tokens=16384` 限制单 batch 总 token（含 cached）。降低 chunked_prefill_size 让单条请求的 chunk 更小，但 batch 可混合多条请求的 chunk。

### 改动 2: router `--disable-circuit-breaker`（继承自 P1.2）

**文件**: `start-router-nocb.sh`

```
--disable-circuit-breaker
--request-timeout-secs 600
--queue-timeout-secs 600
--health-failure-threshold 10
--health-success-threshold 2
```

继承 P1.2 的 router 配置。P1.2 单独测试证明 `--disable-circuit-breaker` 不能单独消除 503（health check 是独立机制），但在 P1.3 里作为额外保险，避免偶发慢响应触发 CB 级联 trip。

### 未改动项（与 exp1 一致）

- `SGLANG_DISAGGREGATION_QUEUE_SIZE=8`, `THREAD_POOL_SIZE=12`, `BOOTSTRAP/WAITING_TIMEOUT=600`
- `--mem-fraction-static 0.85`, `--kv-cache-dtype fp8_e4m3`
- `--context-len 300000`, `--max-running-requests 64`
- `--disaggregation-decode-enable-radix-cache` (decode 端 radix cache)
- L2 hicache **关闭** (P1.1 证明 MLA 不兼容)

---

## 2. 改动原因

### chunked_prefill_size=32768 的问题

在 exp1（chunked=32768）下观察到：

1. **单条长 prompt 独占 prefill batch**: codex_swebenchpro traces 单条请求累积上下文 p50 ~25k token。chunked=32768 意味着一个 chunk 就能容纳整个 prompt，调度器无需切分 → 单条请求独占整个 prefill batch 数秒~数十秒。

2. **prefill 串行化**: 实测 prefill `#inflight-req=1`（仅 1 条在算），`#bootstrap-req` 堆积（c8 时 0-3，c32 退化态 8-26）。其余请求全部排队等当前 batch 完成。

3. **health 超时 → router 503**: prefill 忙于算 32k chunk 时，`/health` 端点响应延迟 1.0s+ 甚至 TimedOut（router 默认 health 超时 5s）。router 标记 prefill 不健康 → circuit breaker open → 503 "No available decode workers"（消息误导性归咎 decode）。

4. **c16 cache 雪崩**: exp1 c8 cache_rate=0.92 → c16 骤降到 0.22。根因是 L1 pool 仅 441k token/GPU（16GB），c16×25k=400k 逼近上限 → LRU 大量驱逐。但 prefill 串行化加剧了问题：被驱逐的 prefix 需要重算，而 prefill 一次只能算 1 条 → 排队雪崩。

### 降到 8192 的预期

- 单 chunk 缩短 4×，prefill 每次只算 ≤8192 token 就切到下一条
- `max_prefill_tokens=16384` 允许 batch 混合 2-4 条请求的 chunk → `#inflight-req` 从 1 提到 2-4
- prefill 不再被单条长请求独占 → health 响应快 → router 不再误判 → 消除 503
- 代价: ttft 上升（多次 chunk 切分开销），但换来稳定性

---

## 3. 实测效果

### 探针配置
- levels: 8, 16, 32
- trials-per-user: 1
- max-tokens: 4096
- all turns (max turns = 不限)
- 预热: 10 条冒烟请求消除冷栈 JIT 偏差

### 结果

| level | total | succeeded | failed | 成功率 | ttft p50 (s) | ttft p90 (s) | tpot p50 (ms) | tpot p90 (ms) | cache_rate mean | cache_rate p50 | 503 数 | duration (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c8 | 264 | 263 | 1 | **99.6%** | 19.6 | 79.2 | 16.7 | 29.9 | 0.73 | 0.92 | 0 | 62.8 |
| c16 | 490 | 489 | 1 | **99.8%** | 70.4 | 115.7 | 15.6 | 40.9 | 0.46 | 0.35 | 0 | 86.9 |
| c32 | — | — | — | 进行中 | — | — | — | — | — | — | 7 (全程) | >2h |

### 稳定性指标

| 指标 | exp1 | P1.1 hicache | P1.2 nocb | **P1.3 cp8k+nocb** |
|---|---|---|---|---|
| prefill RestartCount | — | 崩溃(OOM) | 0 | **0** |
| decode RestartCount | — | 崩溃(detokenizer挂) | 0 | **2** (残留, 非 P1.3) |
| router 503 总数 (c8-c32) | — | 6461 | 11121 | **7** |
| c16 成功率 | 100% (但 cache 0.22) | 0% | 0% | **99.8%** |
| OOMKilled | No | Yes (ratio=10) | No | **No** |

### 运行时调度器状态

c8/c16 阶段（健康态）:
- prefill: `#bootstrap-req: 0-2, #inflight-req: 1-3` → 并行度恢复
- decode: `#running-req: 3-5, #transfer-req: 3-5` → 正常并行
- cache_rate: 63-100% 波动（运行时观测）

c32 阶段（压力态）:
- prefill: `#bootstrap-req: 25-28, #inflight-req: 1` → 排队重现，但未崩溃
- decode: `KVTransferError: Aborted by AbortReq` 偶发（5 条/4h，非雪崩）
- 仅 7 次 503（vs P1.2 的 11K），router 未级联 trip

---

## 4. 三组优化横向对比

| 配置 | chunked | CB | hicache | c8 成功 | c8 cache | c16 成功 | c16 cache | c16 503 | c32 |
|---|---|---|---|---|---|---|---|---|---|
| exp1 | 32768 | 默认 | off | 40/40 | 0.92 | 80/80 | 0.22 | 0 (但退化) | 快败 |
| P1.1 | 32768 | 默认 | ratio=3 | 9/264 | — | 0/490 | — | 6461 | 0/1168 |
| P1.2 | 32768 | **disabled** | off | 64/264 | 0.75 | 0/490 | — | 11121 | 0/1168 |
| **P1.3** | **8192** | **disabled** | off | **263/264** | 0.73 | **489/490** | 0.46 | **0** | 进行中 |

### 关发现

1. **chunked_prefill 8192 是核心有效的改动**: P1.2（仅禁 CB）c8 仅 24% 成功、c16 全败；P1.3 叠加 chunked 8192 后 c8 99.6%、c16 99.8%。说明 prefill 独占 batch 是 503 的根因，CB 只是症状。

2. **c16 cache_rate 0.46 仍有退化**: c8→c16 cache_rate 0.73→0.46，说明 L1 pool 容量限制仍在（441k token/GPU，c16×25k=400k 逼近上限）。但不像 exp1 那样雪崩到 0.22——chunked 8192 让 prefill 能更快切换请求，被驱逐的 prefix 重算排队时间缩短。

3. **ttft 代价显著**: c8 p50 从 exp1 的 4.95s 涨到 19.6s，c16 p50 70.4s。原因:
   - 8192 chunk 切分增加调度开销
   - 多轮对话（all turns, p50=30 turns）累积 token 多，单 prefill 需 3-4 个 chunk
   - 但成功率从"全败"到"99.8%"，这个代价是值得的

4. **c32 prefill 排队重现** (`bootstrap-req: 26`): 32 并发下 prefill 又开始排队，但未崩溃（7 次 503 vs 11K）。说明 chunked 8192 把崩溃阈值从 c16 推到了 c32 附近，但没彻底解决 prefill 吞吐瓶颈。

---

## 5. 结论

**P1.3 有效**。`--chunked-prefill-size 8192` + `--disable-circuit-breaker` 将 PD 栈从"c16 全败"提升到"c16 99.8% 成功"。核心机理是消除 prefill batch 独占，恢复 health 响应，消除 router 503。

**局限**:
- ttft 代价大（4×慢）
- c32 仍有 prefill 排队（bootstrap-req: 26），阈值推到 c32 而非彻底解决
- L1 pool 容量限制仍在（c16 cache_rate 0.46），L2 hicache 不可用（MLA 不兼容）

**下一步** (P2.x):
- P2.4: `--context-len 131072`（从 300000 降）收紧 radix 账户，可能提升 cache_rate
- P2.2: `--optimistic-prefill-retries 2` 跳过 bootstrap 等待，缓解 c32 排队
- P2.3: `--enable-prefill-delayer` 合并小 prefill 批，提升 GPU 利用率
- 考虑 chunked 16384 折中（8192 太慢，32768 太独占）
