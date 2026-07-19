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

## P1.1 L2 hicache — 失败 (2026-07-19)
改动: --enable-hierarchical-cache --hicache-ratio N
结果: ratio=10 OOM (4×271GB>NUMA); ratio=3+direct MLA element_size=656 不兼容 → detokenizer 挂起, 9/264 成功
结论: sglang v0.5.15.post1 hicache kernel 不支持 MLA 压缩 KV layout, 代码级不兼容

## P1.2 router 禁熔断 — 单独无效 (2026-07-19)
改动: --disable-circuit-breaker --request-timeout-secs 600 --health-failure-threshold 10
结果: c8 64/264 (24%), c16 0/490, c32 0/1168, router 503=11K
结论: CB 关闭不够, health check 独立机制仍标记不健康, decode detokenizer 挂起是根因

## P1.3 chunked_prefill 8192 + nocb — 有效! (2026-07-19)
改动: --chunked-prefill-size 8192 (从 32768) + P1.2 nocb router
结果: c8 263/264 (99.6%, cache 0.73), c16 489/490 (99.8%, cache 0.46), c32 进行中 (>2h, 503=7)
核心机理: chunked 8192 消除 prefill batch 独占, health 响应快, router 不误判, 503 消除
代价: ttft 4×慢 (c8 p50 4.95s→19.6s), c32 仍有 prefill 排队 (bootstrap-req: 26)
文档: results/glm_pd-nixl-cp8k/README.md

## 当前运行状态快照 (2026-07-20 00:50)

### 1. h200-2 sglang GLM PD 分离 (运行中)

**栈**: P1.3 配置 (chunked=8192 + nocb router), 容器名 pd-prefill/pd-decode/pd-router
- prefill: GPU0-3 TP4, port 8001, start-prefill-cp8k.sh
- decode: GPU4-7 TP4, port 8002, start-decode-cp8k.sh
- router: port 8000, start-router-nocb.sh (--disable-circuit-breaker)
- 模型: GLM-5.2-W4AFP8 (/data1/GLM-5.2-W4AFP8 → /mnt/file/...)
- 部署脚本: h200-2:/opt/sglang-glm-pd/ + develop/sglang/deploy/
- c8/c16 已完成 (99.6%/99.8% 成功), c32 探针可能仍在跑 (pd-probe 容器)

**环境**: ssh root@h200-2; sglang 镜像 br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129; kvcache-benchmarks 在 /home/lychee/mycode/kvcache-benchmarks (h200-2 本地也有)

**git**: fork=git@github.com:lycheenice/vllm.git, branch=v0.25.0, 最新 commit=34881cde3; push 用 `git push fork refs/heads/v0.25.0:refs/heads/v0.25.0`; develop 目录=/home/lychee/mycode/vllm/develop

**P1.3 结果** (results/glm_pd-nixl-cp8k/):
- c8: 263/264 success, cache 0.73, ttft p50 19.6s
- c16: 489/490 success, cache 0.46, ttft p50 70.4s
- c32: 进行中 (503=7, prefill bootstrap-req: 26 排队但未崩溃)

**已知限制**: GLM MLA → hicache 不可用 (element_size=656); L1 pool 仅 441k tok/GPU; DeepGEMM JIT 10-20min 每次冷启

### 2. h200-2 MiniMax-M2.5 PD 分离 (即将测试)

**模型**: MiniMax-M2.5 (w8a8 FP8, MiniMaxM2ForCausalLM, 62层, 标准 GQA 非 MLA, 125 shards 230GB)
- 已拷贝到 h200-2:/data1/models/MiniMax-M2.5/ (完整性验证通过: 96103 权重, 62 层全覆盖)
- sglang 有 srt/models/minimax_m2.py 原生支持
- **优势**: 标准 MHA → hicache 可用; 无 DeepGEMM JIT → 秒级启动; KV/token ~16KB → L1 pool ~5-6M tokens

**实验设计**: develop/sglang/minimax_experiment_design.md
**脚本**: develop/sglang/deploy/minimax/ (13 个脚本, 未 scp 到 h200-2)

**4 组配置**:
1. TP8 baseline (非 PD, 8 卡混合): start-baseline-tp8.sh, run_baseline_tp8.sh
2. PD Config A (基础, chunked=32768, 默认熔断): start-prefill-A.sh + start-decode-A.sh
3. PD Config B (P1.3 验证: chunked=8192 + nocb): start-prefill-B.sh + start-decode-B.sh
4. PD Config C (B + hicache ratio=3): start-prefill-C.sh + start-decode-C.sh
- PD 启停: run_pd.sh <A|B|C> [start|stop]
- 发压: run_bench.sh <case> [levels] [trials], levels=1,4,16,32,64, trials=5, turns=all
- 停止: stop_all.sh

**发压**: kvcache-benchmarks codex_swebenchpro (610 traces, p50=30 turns), all turns, max_tokens=4096

**执行顺序**: TP8 baseline → PD-A → PD-B → PD-C, 每组 1/4/16/32/64

**待办 (GLM PD)**:
- c32 完成后补充 P1.3 完整数据
- P2.4: context-len 131072; P2.2: optimistic-prefill-retries 2; P2.3: prefill-delayer
- 折中: chunked 16384; Mooncake 对照; TP4 基线

