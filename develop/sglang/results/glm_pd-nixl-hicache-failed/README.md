# P1.1 L2 hicache 优化实验报告 — 失败

> 日期: 2026-07-19
> 基线: exp1 PD-opt (无 hicache), c8 cache_rate=0.92, c16 cache_rate=0.22
> 改动: 两端加 `--enable-hierarchical-cache --hicache-ratio N --hicache-io-backend direct`

## 尝试 1: hicache_ratio=10, io_backend=kernel (默认)

**结果: OOM Killed**

- 每 TP rank 分配 226.06 GB host (ratio=10 × 25.29 GB device pool) + 45.49 GB DSA indexer = 271 GB/rank
- 4 ranks × 271 = 1086 GB total host memory
- NUMA constraint (`mems_allowed=0-1`) 限制可用内存约 900 GB → OOM
- `shmem-rss: 250396952 kB ≈ 250 GB` 时被 kernel oom-killer 杀死
- dmesg: `Out of memory: Killed process 1635676 (sglang::schedul) total-vm:1049023412kB, shmem-rss:250396952kB`

**关键日志**:
```
[TP2] Allocating 226.06 GB host memory for hierarchical KV cache.
[TP0] Unsupported element_size = 656 for JIT HiCache kernel
[TP0] Allocating 45.49 GB host memory for DSA indexer (layout=page_first).
→ OOM kill
```

## 尝试 2: hicache_ratio=3, io_backend=direct, mem_layout=page_first_direct

**结果: 功能性失败（9/264 成功后崩溃）**

- 每 rank 67.82 GB host + 13.65 GB DSA indexer = 81.5 GB/rank × 4 = 326 GB → 无 OOM
- 容器正常启动，prefill + decode 健康
- 冒烟测试通过（PD 管道通）
- **c8 探针启动 1.5 分钟后 decode detokenizer 心跳停止**
- prefill 随后崩溃重启（DeepGEMM JIT 重新编译）

### 失败时间线

| 时间 | 事件 |
|---|---|
| 11:03 | prefill 健康 (10min JIT 编译完成) |
| 11:11 | decode 健康 |
| 11:12:31 | c8 探针启动 (8 并发, trials_per_user=1) |
| 11:13:47 | decode: `#running-req:3, #transfer-req:5, gen 168 tok/s` — 正在工作 |
| 11:13:47 | decode: `KVTransferError: Aborted by AbortReq` (1 请求 abort) |
| **11:13:54** | **decode detokenizer 最后心跳** — 进程挂起 |
| 11:15:09 | decode health check 开始失败 (detokenizer 20s 无响应) |
| 11:18:04 | prefill 重启 (DeepGEMM warmup 从头开始) |
| 11:18:44 | router: `No available prefill workers (all circuits open or unhealthy)` |
| 全程 | c08: 9/264 成功, c16: 0/490, c32: 0/1168 |

### 根因: MLA KV layout 与 hicache 不兼容

```
[TP0] Unsupported element_size = 656 for JIT HiCache kernel
```

- GLM-5.2 使用 MLA (Multi-Latent Attention), `kv_lora_rank=512`
- 每 page 的 KV element_size = 656 字节 (512 latent + metadata)
- sglang v0.5.15.post1 的 hicache kernel 不支持此 element_size
- `--hicache-io-backend direct` 绕过了 JIT kernel compile，但底层 D2H/H2D 传输路径仍依赖此 kernel
- 运行时实际触发 hicache D2H transfer 时，内核不兼容导致 detokenizer 挂起

### 排除其他原因

- **非 OOM**: ratio=3 总 host 用量 326 GB, 主机可用 1881 GB, dmesg 无新 OOM 条目
- **非显存 OOM**: GPU mem 126 GB/143 GB, L1 pool 不变
- **非 PD 管道问题**: 冒烟测试成功, 前 9 个请求正常
- **非 router 熔断**: prefill/decode 崩溃在前, router 503 在后

## 结论

**P1.1 (L2 hicache) 对 GLM-5.2-W4AFP8 (MLA) 无效**。sglang v0.5.15.post1 的 hicache 实现不支持 MLA 的压缩 KV layout (element_size=656)。这是代码级不兼容，非参数可调。

### 需要的修复（超出本实验范围）

- sglang hicache kernel 需支持 MLA element_size=656
- 或提供 MLA-aware 的 hicache 直接传输路径
- 或使用 `--hicache-storage-backend nixl` 绕过内置 kernel（需 P2.1 验证）

## 下一步

P1.1 标记为失败。转 P1.2 (router `--disable-circuit-breaker`)，回到 exp1 原始配置（无 hicache），单独验证熔断关闭对 503 的改善。
