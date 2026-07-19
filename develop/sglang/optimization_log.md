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

## 待办
- 等 c8/c16/c32/c64/c128 各级 result.json
- 若 prefill GPU 仍 0%, 考虑 P2+D6 或 P6+D2 重配比
- Mooncake backend 对照 (需另起栈)
- TP4 不分离基线

