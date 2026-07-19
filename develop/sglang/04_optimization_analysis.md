# sglang NIXL PD 优化分析报告 — GLM-5.2-W4AFP8 on h200-2

> 基于 2026-07-19 实测数据 + sglang v0.5.15.post1 源码/CLI 调研
> 模型: GLM-5.2-W4AFP8 (MLA, 78层, 256专家, w4afp8, `model_type=deepseek_v3`, `GlmMoeDsaForCausalLM`)
> 硬件: h200-2, 8×H200 SXM 141GB, 主机 RAM 2015 GB (可用 1881 GB), NVLink/NVSwitch
> 拓扑: 1P+1D, 各 TP4, NIXL UCX (cuda_ipc), host 网络, bootstrap 8998

---

## 1. 执行摘要

当前 PD 栈在 c8 并发下可工作 (cache_rate 0.92)，但 c16 起 cache 命中雪崩 (0.22)、c32 快速失败、c64 集体超时。根因集中三点：**(1) L1 KV 池过小 + L2 hicache 关闭导致缓存驱逐无兜底；(2) prefill 在过载后串行化且不可自恢复；(3) router 熔断器对 prefill 慢响应误判触发 503**。优化优先级最高、难度最低、收益最大的是开启 L2 hicache（主机 1.8TB RAM 几乎免费把缓存容量放大 10-50×），配合 router 熔断调优 + chunked_prefill 降低，预计可将可用并发从 c8 推到 c64+。

---

## 2. 当前配置（ServerArgs 实测 dump）

| 类别 | 参数 | 值 | 备注 |
|---|---|---|---|
| 模型 | model_path | GLM-5.2-W4AFP8 | `max_position_embeddings=1048576` (1M) |
| 并行 | tp_size | 4 (P: GPU0-3, D: GPU4-7) | MLA → staging buffer 不可用, 无 DP attention |
| 上下文 | context_length | 300000 | 未超模型上限, 主要影响单请求上限 |
| 调度 | chunked_prefill_size | 32768 | **过大**: 单大 prefill 独占 batch |
| 调度 | max_prefill_tokens | 16384 | |
| 调度 | max_running_requests | 64 | |
| 调度 | schedule_policy | fcfs | |
| 内存 | mem_fraction_static | 0.85 | 权重占 94.5GB/GPU, 池分配后剩 ~16GB |
| KV | kv_cache_dtype | fp8_e4m3 | |
| KV | page_size | 64 | |
| **L1** | disable_radix_cache | False (开启) | radix LRU |
| **L2** | enable_hierarchical_cache | **False (关闭)** | **根因之一** |
| L2 | hicache_ratio | 2.0 (默认, 未激活) | |
| L2 | hicache_storage_backend | None | |
| PD | disaggregation_transfer_backend | nixl | |
| PD | disaggregation_decode_enable_radix_cache | True (decode 端) | exp1 已开 |
| PD | disaggregation_decode_enable_offload_kvcache | **False** | 需配 hicache_storage_backend |
| PD | num_reserved_decode_tokens | 512 | |
| PD | optimistic_prefill_retries | 0 | |
| PD | enable_prefill_delayer | False | |
| CUDA graph | prefill disable_cuda_graph | True | prefill 无图 |
| CUDA graph | decode backend | full, max_bs 128 | |
| PD 环境变量 | SGLANG_DISAGGREGATION_QUEUE_SIZE | 8 | exp1, 默认4 |
| PD 环境变量 | SGLANG_DISAGGREGATION_THREAD_POOL_SIZE | 12 | exp1 |
| PD 环境变量 | BOOTSTRAP/WAITING_TIMEOUT | 600 | exp1, 默认300 |
| router | launch_router --pd-disaggregation | 默认熔断参数 | **误判 prefill** |

---

## 3. 运行时数据采集

### 3.1 L1 KV 池容量 (metrics 实测)

| 指标 | prefill (GPU0-3) | decode (GPU4-7) |
|---|---|---|
| max_total_num_tokens / GPU | 441,792 | ~441,792 |
| num_pages / GPU | 6,903 | 6,903 |
| startup_available_gpu_memory_gb | **16.24 GB** | ~17.9 GB |
| 权重占用 / GPU | 94.5 GB | 94.5 GB |
| GPU 显存 (nvidia-smi) | 142.4/143.8 GB (99%) | 126.8/143.8 GB (88%) |

**L1 池仅 ~16GB/GPU → 441k tokens/GPU**。对 codex_swebenchpro traces (p50=30 turns, 累积上下文 ~20-30k token/session)：c8×25k=200k 容得下；c16×25k=400k 逼近上限；c32×25k=800k **远超 441k → 驱逐**。

### 3.2 已保存基准结果 (05:15 clean run, trials_per_user=5)

| level | trials | ret | ttft p50/p90 (s) | tpot p50/p90 (ms) | cache_rate mean | 时长 | 状态 |
|---|---|---|---|---|---|---|---|
| c8 | 40 | 0 | 4.95 / 13.73 | 13.48 / 14.84 | **0.92** (p50 0.99) | 32min | OK |
| c16 | 80 | 0 | 94.80 / 184.08 | 18.52 / 54.13 | **0.22** (p50 0.03) | 45min | cache 雪崩 |
| c32 | 160 | 0 | n/a (无 lat 字段) | n/a | n/a | 133s | 异常快败 |
| c64 | — | — | 卡 301s 超时 | — | 0% | >2h44m | 手动停 |

c8→c16 的 cache_rate 0.92→0.22 与 L1 容量极限精确吻合：并发翻倍即驱逐。

### 3.3 退化态 metrics (c64 风暴后 09:57, 栈未重启)

| 指标 | prefill | decode |
|---|---|---|
| num_prefill_bootstrap_queue_reqs | **8** (全卡住) | 0 |
| num_prefill_inflight_queue_reqs | **1** (串行化重现) | 0 |
| num_bootstrap_failed_reqs_total | **152** | 10 |
| num_aborted_requests_total | **169** | **163** |
| evicted_tokens_total | **35.87M** | 5.83M |
| kv_used_tokens | 64 (空) | 0 (空) |
| kv_available_tokens | 87,296 | 440,896 |
| kv_transfer_latency_ms_count (生命周期) | 283 | — |
| cache_hit_rate | 0.0 | 0.0 |
| gen_throughput | 0.0 | 0.0 |

**关键发现：过载后 PD 栈进入不可自恢复的退化态**。L1 池虽已腾空 (kv_used≈0)，但 prefill 仍维持 inflight=1 串行 (queue=8 全排队)，decode 持续 `KVTransferError: Aborted by AbortReq`。生命周期 169 abort / 152 bootstrap 失败 / 35.8M 驱逐 —— 即便用户不再施压，栈也无法回到 exp1 的健康并行态 (inflight=3, throughput=10487 tok/s)。需重启恢复。

### 3.4 早期 6× c08 全失败 (02:43–05:08)

全部 1343/0 失败，HTTP 503 `No available decode workers (all circuits open or unhealthy)`。实测根因不是 decode 不健康，而是 **prefill /health 超时** (router 日志: `HTTP health check failed for http://127.0.0.1:8001/health: TimedOut`)，router 熔断器打开后错误信息误导性地归咎 decode。

---

## 4. 根因分析

### R1. L1 KV 池过小 + L2 hicache 关闭（首要根因）
- 权重 94.5GB/GPU 吃掉大部分显存，L1 仅 16GB/441k tokens。
- `enable_hierarchical_cache=False` → 驱逐的 KV 直接丢失，无 host 兜底。
- c16 并发 × ~25k token/session ≈ 400k 逼近 L1 上限 → LRU 大量驱逐 → cache_rate 0.92→0.22。
- 生命周期 35.8M token 被驱逐 = 反复重算，prefill 被迫重做已驱逐的 prefix。
- **主机 1881 GB 可用 RAM 完全闲置**，L2 几乎可以"无限大"。

### R2. Prefill 串行化且不可自恢复
- `chunked_prefill_size=32768` 让单条大 prefill 独占 batch 数秒~数十秒，其余请求堆在 `bootstrap_queue`。
- exp1 的 QUEUE_SIZE=8/THREAD=12 在**冷启 clean 态**下确实把 inflight 提到 3 (04:24 实测)；但**经 c64 2h44m 风暴后**，prefill 沉降到 inflight=1，152 次 bootstrap 失败留下残留状态，scheduler 不再并行填报 inflight。
- 即便负载撤离，栈不自愈 → 必须重启。

### R3. Router 熔断器误判 prefill
- router 默认对 :8001/:8002 做健康探测，prefill 忙时 /health 响应 1.0s+ 甚至超时。
- 熔断器打开后返回 503，消息模板把任何 PD 失败都说成 "No available decode workers"，掩盖真实是 prefill 慢。
- 单 P+单 D 拓扑下熔断**无故障转移价值**，只会制造假阴性。

### R4. Decode 无异步 KV 卸载
- `disaggregation_decode_enable_offload_kvcache=False`。该选项需配 `--hicache-storage-backend`，即依赖 L2 栈。
- 开启后 decode 可把非活跃 prefix 异步卸到 host/NIXL 存储，腾出 L1 给活跃 decode 批次 → 直接提高 decode 并发容量。
- 当前关闭 = decode L1 与 prefill L1 同样受限。

### R5. 上下文/调度参数联动
- `context_length=300000` 虽未超模型 1M 上限，但与 `chunked_prefill_size=32768`+`max_prefill_tokens=16384` 联动，使单条长上下文请求长时间独占 prefill batch 槽。
- `optimistic_prefill_retries=0`、`enable_prefill_delayer=False` —— 两个本可缓解 prefill 阻塞的机制均未启用。

---

## 5. 可用优化机制（源码 + CLI 调研）

### 5.1 L2 hicache 完整栈
```
--enable-hierarchical-cache
--hicache-ratio <float>          # host 池 = ratio × device 池, 默认 2.0
--hicache-size <int>             # 直接指定 host 池 GB, 覆盖 ratio
--hicache-write-policy           # write_back | write_through | write_through_selective
--hicache-io-backend             # direct | kernel | kernel_ascend
--hicache-mem-layout             # layer_first | page_first | page_first_direct | ...
--hicache-storage-backend        # file | mooncake | hf3fs | nixl | aibrix | dynamic | ...
--hicache-storage-prefetch-policy  # best_effort | wait_complete | timeout
```
特殊：`--disaggregation-decode-enable-offload-kvcache` **仅当 hicache_storage_backend 提供时生效**（源码 `server_args.py:5785` 校验），二者是配套的。

### 5.2 PD prefill 缓解机制
```
--optimistic-prefill-retries <int>   # 跳过 bootstrap 等待的重试次数 (默认0)
--enable-prefill-delayer             # 延迟 prefill 以批合并到达中的 transfer
  --prefill-delayer-max-delay-passes
  --prefill-delayer-token-usage-low-watermark
  --prefill-delayer-queue-min-ratio
  --prefill-delayer-max-delay-ms
```

### 5.3 Router 熔断/健康控制
```
--disable-circuit-breaker
--cb-failure-threshold / --cb-success-threshold / --cb-timeout-duration-secs
--health-failure-threshold / --health-success-threshold
--retry-max-retries / --retry-initial-backoff-ms / --retry-max-backoff-ms
--request-timeout-secs / --connect-timeout-secs / --queue-timeout-secs
```

### 5.4 Decode 卸载
```
--disaggregation-decode-enable-offload-kvcache   # 需 hicache_storage_backend
--num-reserved-decode-tokens <int>               # 默认 512
--disaggregation-decode-extra-slots <int>        # in-transfer 预留槽位, 默认 0/2x running
--disaggregation-decode-polling-interval <int>   # 默认 1
```

---

## 6. 优化方案与优先级

难度: ★(改1-2行参数) / ★★(多参数联动需验证) / ★★★(改栈/重配/重跑验证)
收益: 按"能否把可用并发从 c8 推高到 c64+"评估

### Priority 1 — 难度低、收益大（先做）

| ID | 改动 | 难度 | 收益 | 说明 |
|---|---|---|---|---|
| **P1.1** | 两端开 L2 hicache: `--enable-hierarchical-cache --hicache-ratio 10` (或 `--hicache-size 320` 直配 320GB host 池) | ★ | **极高** | 主机 1.8TB RAM 闲置, ratio=10 → host L2 = 640GB (≈1.6M tokens), cache 容量 ~10×, 直接解 R1, 预期 c16/c32 cache_rate 回到 0.9+ |
| **P1.2** | router `--disable-circuit-breaker` (单 P+D 无转移价值) 或 `--health-failure-threshold 10 --health-success-threshold 3 --cb-timeout-duration-secs 60` | ★ | **高** | 解 R3, 消除 prefill 慢响应误判 503 |
| **P1.3** | `--chunked-prefill-size 8192` (两端, 从 32768 降) | ★ | **高** | 解 R2 的一部分: 单 prefill 不再独占 batch, inflight 可恢复 3-4 |

### Priority 2 — 难度中、收益大

| ID | 改动 | 难度 | 收益 | 说明 |
|---|---|---|---|---|
| **P2.1** | decode 开异步卸载: `--disaggregation-decode-enable-offload-kvcache --hicache-storage-backend nixl` (与 P1.1 配套) | ★★ | **高** | 解 R4: decode L1 活跃池翻倍, decode 并发容量 +50-100% |
| **P2.2** | `--optimistic-prefill-retries 2` | ★ | **中** | 跳过 bootstrap 等待重试, 减少 prefill 串行 stall |
| **P2.3** | `--enable-prefill-delayer` + `--prefill-delayer-queue-min-ratio 0.3 --prefill-delayer-max-delay-passes 5` | ★★ | **中** | 合并小 prefill 批, 提升 GPU 利用率 (当前 prefill util 0%) |
| **P2.4** | `--context-length 131072` (从 300000 降) | ★ | **中低** | 收紧单请求上限与 radix 账户, 腾出有效槽位; 对 codex traces (p90=57 turns 但单请求 max_tokens=4096) 影响小 |

### Priority 3 — 难度中、收益中/不确定

| ID | 改动 | 难度 | 收益 | 说明 |
|---|---|---|---|---|
| **P3.1** | decode `--disaggregation-decode-extra-slots 16` (从 0/默认) | ★ | **中** | 为 in-transfer 请求预留更多槽位, 减少 decode 排队 |
| **P3.2** | `--hicache-write-policy write_back` (默认 write_through) | ★★ | **中** | 写回策略提升 L2 写入吞吐, 但需验证一致性 |
| **P3.3** | PD 栈健康自愈: docker healthcheck + `--restart unless-stopped` + 周期 curl /health 失败即重启 | ★★ | **中** | 解 R2 不可自恢复: 过载后自动重建而非沉死 |
| **P3.4** | GPU 切分 P6+D2 / P2+D6 重配比 | ★★★ | **不确定** | GLM w4afp8 prefill 重, 但 TP 改变需重分片; MLA + 8卡单节点建议保持 TP4+4, 暂不动 |

### Priority 4 — 对照/扩展（已排队）

| ID | 改动 | 难度 | 收益 | 说明 |
|---|---|---|---|---|
| **P4.1** | Mooncake P2P/Store backend 对照 (run_pd_mooncake.sh) | ★★ | **数据点** | NIXL vs Mooncake GDR/Store 横向对比 |
| **P4.2** | TP4 非分离基线 (start-baseline-tp4.sh) | ★★ | **数据点** | 量化 PD 分离相对收益 |

---

## 7. 推荐实施顺序

**Phase A（最小改动验证 cache 假设）** — 改 3 个参数, 重跑 c8/c16/c32:
1. P1.1 (两端 `--enable-hierarchical-cache --hicache-ratio 10`)
2. P1.2 (router `--disable-circuit-breaker`)
3. P1.3 (两端 `--chunked-prefill-size 8192`)
4. 重启 PD 栈 (清退化态) → c8/c16/c32 探针 (trials_per_user=1 快验)
5. **预期**: c16 cache_rate 从 0.22 回到 ≥0.85, c32 不再快败

**Phase B（decode 容量提升）** — 若 Phase A 达标:
1. P2.1 (decode `--disaggregation-decode-enable-offload-kvcache --hicache-storage-backend nixl`)
2. P2.4 (`--context-length 131072`)
3. c64/c128 验证

**Phase C（prefill 效率 + 自愈）** — 若 c64 仍 prefill 瓶颈:
1. P2.2 + P2.3 (optimistic + delayer)
2. P3.3 (健康自愈)

**Phase D（对照）** — Mooncake + 基线

---

## 8. 风险与注意事项

- **L2 hicache + MLA**: MLA 的压缩 KV latent (kv_lora_rank=512) 在 host 池的 layout 需 `page_first` (默认) 兼容; `write_through` 默认策略最稳, 先用默认验证再试 `write_back`。
- **DeepGEMM JIT**: 重启容器会触发 10-20min W4AFP8 编译, 安排进等待时间。
- **hicache_ratio 上限**: ratio=10 → host 池 640GB; 若欲更大建议用 `--hicache-size` 直配, 避免按 ratio 从 16GB device 放大时的浮点误差。host RAM 1.8TB 充足。
- **不可自恢复是独立问题**: 即便优化后, 持续过载仍可能沉死; Phase C 的健康自愈 (P3.3) 是长期必需。
- **熔断关闭的代价**: `--disable-circuit-breaker` 后若 decode 真的 crash, router 会持续把请求打到死掉的 decode; 需配合 decode 容器 `--restart unless-stopped` (已有)。

---

## 9. 验证用 metrics（每级探针采样）

重启 + 每 Phase 后跑 c8/c16/c32/c64 探针 (trials_per_user=1), 采样下列指标做前后对比:

| 维度 | metrics | 期望 (优化后) |
|---|---|---|
| L1 命中 | prefill/decode `cache_hit_rate` | ≥0.85 @ c16 |
| L1 驱逐 | `evicted_tokens_total` 增量 | c16 增量 < 1M (现 35.8M) |
| L2 命中 | hicache 相关计数器 (启用后出现) | >0, 随并发上升 |
| prefill 并行 | `num_prefill_inflight_queue_reqs` | ≥3 @ c8 (现退化态=1) |
| bootstrap 堆积 | `num_prefill_bootstrap_queue_reqs` | ≤2 @ c8 (现=8) |
| abort | `num_aborted_requests_total` 增量 | 0 @ c8/c16 |
| 传输完成 | `kv_transfer_latency_ms_count` 增量 | ≈ 并发×turns |
| ttft | bench summary p50/p90 | c16 p50 < 10s (现 94.8s) |
| 端到端 | bench cache_rate mean | ≥0.85 @ c16 |
