# sglang NIXL PD 优化实验记录

机器: h200-2 (10-118-89-32), 8×H200, GLM-5.2-W4AFP8 (MLA, 78层, 256专家, w4afp8)
工具: kvcache-benchmarks (codex_swebenchpro, 610 traces)
拓扑: P=GPU0-3 TP4, D=GPU4-7 TP4, NIXL UCX, host 网络, bootstrap 8998

## baseline (无优化, c8) — 2026-07-19 03:21
参数: 默认 SGLANG_DISAGG_*, decode disable_radix_cache=True
现象:
- prefill #bootstrap-req: 9, #inflight-req: 1 (串行)
- decode KVTransferError: Aborted by AbortReq (8 并发全 abort)
- 所有 GPU 0%, 请求 25s+ 无响应
根因: NIXL KV transfer 串行化, decode 等不到 KV 超 300s abort, 链式雪崩

## exp1: 加 SGLANG_DISAGG 并行 + decode radix — 2026-07-19
改动:
- SGLANG_DISAGGREGATION_QUEUE_SIZE=8 (默认4)
- SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12 (默认4-12动态)
- SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 (默认300)
- SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 (默认300)
- decode 加 --disaggregation-decode-enable-radix-cache
结果: (待跑)

## exp1 实测结果 — 2026-07-19 04:24 (c8 启动后 90s 采样)
对比 baseline:
| 指标 | baseline | exp1 |
|---|---|---|
| prefill #inflight-req | 1 (串行) | 3 (并行) |
| prefill #bootstrap-req 堆积 | 9 | 0 |
| prefill input throughput | 1 tok/s | 10487 tok/s |
| decode #running-req | 0 (全 abort) | 5 |
| decode #transfer-req | 0 | 3 (并行 KV transfer) |
| decode gen throughput | 70 tok/s (单req假忙) | 242 tok/s (5 req 真并发) |
| decode token usage | - | 0.43 |
| KVTransferError abort | 8 全 abort | 0 |
| GPU prefill util | 0% | 0% (短脉冲) |
| GPU decode util | 100% (假忙) | 100% (真忙) |

结论: 优化生效。根因被对症: SGLANG_DISAGGREGATION_QUEUE_SIZE=8+THREAD_POOL_SIZE=12 让 NIXL KV transfer 并行, decode --disaggregation-decode-enable-radix-cache 减少重复拉取。无 abort。
c8 ramp 跑通, 等 c8 result.json 完整指标。

## exp1 全量 ramp 结果 — 2026-07-19 05:15-06:33 (已保存 commit 09c539101)
| level | trials | cache_rate | ttft p50 | 状态 |
|---|---|---|---|---|
| c8 | 40/40 | 0.92 (p50 0.99) | 4.95s | OK |
| c16 | 80/80 | 0.22 (p50 0.03) | 94.80s | cache 雪崩 |
| c32 | 160 | n/a (133s 快败, 无 lat 字段) | — | 异常 |
| c64 | — | 0% | 301s 超时 | 2h44m 手动停 |

## 退化态诊断 — 2026-07-19 09:57 (c64 风暴后未重启)
prefill: bootstrap_queue=8, inflight=1 (串行化重现), 152 bootstrap 失败, 169 abort, 35.8M evicted
decode: 163 abort, 10 bootstrap 失败, 5.8M evicted, KVTransferError Aborted by AbortReq
结论: 过载后 PD 栈不可自恢复, 需重启
根因: L1 仅 16GB/441k token (权重占 94.5GB), L2 hicache 关闭, chunked_prefill=32768 独占 batch, router 熔断误判 prefill

## 待办
- 优化方案已写入 04_optimization_analysis.md (Priorities P1.1-P4.2)
- Phase A 待跑: P1.1 (开 hicache ratio=10) + P1.2 (禁熔断) + P1.3 (chunked 8192) → c8/c16/c32 验
- Mooncake backend 对照 (P4.1)
- TP4 不分离基线 (P4.2)

