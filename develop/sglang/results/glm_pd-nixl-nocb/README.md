# P1.2 router --disable-circuit-breaker 优化实验报告

> 日期: 2026-07-19
> 基线: exp1 PD-opt (c8 cache_rate=0.92, 40/40 成功; c16 cache_rate=0.22, 80/80 成功)
> 改动: router 加 `--disable-circuit-breaker --request-timeout-secs 600 --queue-timeout-secs 600 --health-failure-threshold 10 --health-success-threshold 2`
> 注: prefill/decode 仍用 exp1 原配置 (无 hicache)

## 结果

| level | total | succeeded | failed | cache_rate | ttft p50 (s) | 状态 |
|---|---|---|---|---|---|---|
| c8 | 264 | 64 (24%) | 200 | **0.75** | 11.70 | 部分成功 |
| c16 | 490 | 0 | 490 | n/a | n/a | 全 503 |
| c32 | 1168 | 0 | 1168 | n/a | n/a | 全 503 |

router 503 错误总计: **11,121 条** (c16+c32 阶段)

## 分析

### 503 未消除

`--disable-circuit-breaker` 仅关闭了 circuit breaker 机制，但 router 的 **health check 是独立机制**。当 decode 的 /health 端点超时（5s）连续 10 次（`--health-failure-threshold 10`），router 将 decode 标记为不健康，仍返回 503。

错误消息仍为 `No available decode workers (all circuits open or unhealthy)`，说明消息模板未区分 CB 开/health 失败。

### c8 部分改善 (vs hicache 崩溃态)

- hicache 崩溃后 c8: 9/264 成功
- P1.2 c8: 64/264 成功, cache_rate=0.75
- 但远低于 exp1 clean 态 c8: 40/40 成功, cache_rate=0.92

### 冷栈 vs 热栈偏差

P1.2 在刚启动 15min 的冷栈上测试，exp1 baseline 在运行 1h+ 的热栈上测试。冷栈第一波请求触发 DeepGEMM JIT 编译，导致前几十个请求极慢（ttft p50=11.7s vs 热态 4.95s），诱发 prefill health 超时 → 级联 503。

### 根因不变

decode detokenizer 在 c16 并发下挂起（与 exp1 + hicache 失败同模式），health check 超时是症状而非根因。P1.2 无法解决 PD 管道在并发下的结构性不稳定。

## 结论

**P1.2 单独无效**。CB 关闭不能消除 503，因为 health check 是独立机制。PD 管道在 c16+ 并发下的 detokenizer 挂起是根因，不是 router 配置。

保留 P1.2 的 nocb router 作为安全措施（避免假阴性 CB trip），P1.3 叠加验证 chunked_prefill 效果。

## 下一步

P1.3: `--chunked-prefill-size 8192`（从 32768 降），+ P1.2 nocb router，+ 预热请求消除冷栈偏差。
