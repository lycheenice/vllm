# MiniMax-M2.5 + vLLM PD 分离实验计划

> 机器: h200-2 (10-118-89-32), 8×H200 SXM 141GB, NVLink/NVSwitch
> 模型: MiniMax-M2.5 (w8a8 FP8, `MiniMaxM2ForCausalLM`, 62层, 标准 GQA)
> 框架: vLLM v0.25.0 (docker 镜像 `docker.1ms.run/vllm/vllm-openai:v0.25.0`, 已含 nixl)
> 发压: kvcache-benchmarks (codex_swebenchpro, 610 traces, p50=30 turns)
> 日期: 2026-07-20

---

## 1. 环境确认

### 1.1 已就绪

| 项目 | 状态 | 详情 |
|---|---|---|
| vLLM 镜像 | ✅ | `docker.1ms.run/vllm/vllm-openai:v0.25.0` (18.7GB), 含 nixl |
| MiniMax-M2.5 模型 | ✅ | h200-2:/data1/models/MiniMax-M2.5/ (230GB, 125 shards) |
| vLLM 模型支持 | ✅ | `MiniMaxM2ForCausalLM` 在 registry, `vllm/model_executor/models/minimax_m2.py` |
| NIXL 库 | ✅ | 镜像内 `import nixl` 通过 |
| GLM PD 容器 | ✅ 已清理 | pd-prefill/pd-decode/pd-router/pd-probe 已 stop+rm |
| GPU | ✅ 全空闲 | 8×H200, 之前的容器已清理 |

### 1.2 模型架构关键参数

```
model_type:         minimax_m2
architectures:      [MiniMaxM2ForCausalLM]
quantization:       fp8 (dynamic, w8a8, block 128×128)
num_hidden_layers:  62
hidden_size:        3072
num_attention_heads: 48
num_key_value_heads: 8
head_dim:           128
num_local_experts:  256
num_experts_per_tok: 8
max_position_embeddings: 196608
use_mtp:            true (num_mtp_modules: 3)
vocab_size:         200064
```

### 1.3 KV/token 估算 (TP8, fp8)

- 每层每 token: 2 (k+v) × 8 (kv_heads) × 128 (head_dim) × 1 byte (fp8) = 2048 bytes
- 62 层: 2048 × 62 = 127KB/token (TP8, 每 GPU 仅存 1/8 → ~16KB/token/GPU)
- L1 池容量 (141GB - 27GB 权重 = ~114GB, mem_fraction 0.90): ~103GB → ~6.4M tokens/GPU
- 对比 GLM-5.2 (441K tokens): **14× 更大**, L1 驱逐不应是瓶颈

### 1.4 Docker 镜像检查命令

```bash
# 验证 vLLM 版本和 nixl
docker run --rm --gpus all --entrypoint python3 \
  docker.1ms.run/vllm/vllm-openai:v0.25.0 \
  -c "import vllm; print(vllm.__version__); import nixl; print('nixl OK')"
# 输出: 0.25.0 / nixl OK  (已验证)
```

---

## 2. vLLM PD 分离机制 (与 sglang 的关键差异)

### 2.1 架构对比

| 维度 | sglang PD | vLLM PD |
|---|---|---|
| 路由器 | 内置 router (Rust) | 外部 Python proxy (FastAPI) |
| KV 传输 | NIXL (sglang 封装) | NIXL connector (vllm 原生) |
| P→D KV | P push → D pull | NixlConnector (默认 pull: D 主动读 P) |
| D→P 反传 | 不支持 | **bidirectional_kv_xfer** (P 从 D 拉 KV, 多轮复用) |
| 跨轮 cache | 依赖 L1 radix | APC (P 侧) + bidirectional (D→P) |
| 启动参数 | SGLANG_DISAGG_* 环境变量 | `--kv-transfer-config` JSON |
| 侧信道 | bootstrap port 8998 | VLLM_NIXL_SIDE_CHANNEL_PORT (默认 5600) |
| 熔断 | --disable-circuit-breaker | 无内置熔断 (proxy 层处理) |

### 2.2 vLLM PD 三组件

1. **Prefill 实例** (kv_producer): `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'`
2. **Decode 实例** (kv_consumer): `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'`
3. **Proxy**: `disagg_proxy_multiturn.py` (支持 bidirectional 多轮) 或 `disagg_proxy_demo.py` (基础轮询)

### 2.3 关键配置项

| 参数 | 默认 | 说明 |
|---|---|---|
| `kv_buffer_device` | cuda | GDR 直传 (H200 NVLink 零拷贝) |
| `kv_load_failure_policy` | fail | KV 加载失败行为 (fail/recompute) |
| `kv_lease_duration` | 30s | P 侧 KV 块租约, 等 D 来读 |
| `bidirectional_kv_xfer` | false | D→P 反向 KV (多轮复用) |
| `kv_recompute_threshold` | 64 | bidirectional 最小 token 阈值 |
| `decoder_kv_blocks_ttl` | 480s | D 侧 bidirectional KV 块 TTL |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | 5600 | NIXL 握手端口 |
| `UCX_TLS` | all | 传输层 (单机: cuda_ipc,cuda_copy,tcp) |

### 2.4 conversation_id 问题

vLLM 的 `disagg_proxy_multiturn.py` 需要请求体包含 `conversation_id` 字段来关联多轮对话, 实现 bidirectional KV 复用。kvcache-benchmarks 的 `load_test.py` 当前不发送此字段。

**解决方案**: 修改 `disagg_proxy_multiturn.py`, 当请求无 `conversation_id` 时, 基于首条 user message 内容哈希自动生成。这样无需修改 benchmark 工具。

---

## 3. 实验方案

### Phase 1: TP8 基线 (非 PD, 混合 prefill+decode)

**目的**: 建立 MiniMax-M2.5 在 vLLM 上的性能基线, 作为 PD 对照。

| 参数 | 值 |
|---|---|
| tp_size | 8 (GPU 0-7) |
| 模式 | 标准 vllm serve |
| port | 8000 |
| kv_cache_dtype | fp8 (模型本身 w8a8) |
| max_model_len | 100000 (max_position=196608, 取实用值) |
| gpu_memory_utilization | 0.90 |
| max_num_seqs | 128 |
| enable_prefix_caching | true (默认) |
| chunked_prefill_size | 8192 (sglang P1.3 验证有效) |
| speculative_config | 待验证: use_mtp=true 是否自动启用 |

**启动命令**:
```bash
docker run -d --name minimax-baseline --gpus all --network host \
  --shm-size 16g \
  -v /data1/models/MiniMax-M2.5:/models/MiniMax-M2.5 \
  -v /home/lychee/mycode/kvcache-benchmarks:/bench \
  docker.1ms.run/vllm/vllm-openai:v0.25.0 \
  --model /models/MiniMax-M2.5 \
  --served-model-name MiniMax-M2.5 \
  --tensor-parallel-size 8 \
  --port 8000 \
  --max-model-len 100000 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --chunked-prefill-size 8192 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

**发压**: 1/4/16/32/64, turns=all, max_tokens=4096
**预计**: 3-5h

### Phase 2: PD 基础 (无 bidirectional)

**目的**: 验证 vLLM PD 基本功能, P→D 单向 KV 传输。P 侧靠 APC 实现跨轮 prefix 复用。

| 组件 | 参数 |
|---|---|
| Prefill | TP4 (GPU 0-3), port 8100, kv_producer, side_channel 5600 |
| Decode | TP4 (GPU 4-7), port 8200, kv_consumer, side_channel 5601 |
| Proxy | disagg_proxy_demo.py, port 8000 |
| UCX_TLS | cuda_ipc,cuda_copy,tcp (单机 NVLink) |
| kv_lease_duration | 60 (默认 30 太短) |
| chunked_prefill_size | 8192 |
| APC | P 侧开启 (默认) |
| bidirectional | 关闭 |

**启动**:
```bash
# Prefill (producer)
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_NIXL_SIDE_CHANNEL_PORT=5600
vllm serve ... --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"kv_lease_duration":60}}'

# Decode (consumer)
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_NIXL_SIDE_CHANNEL_PORT=5601
vllm serve ... --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

# Proxy
python disagg_proxy_demo.py --model MiniMax-M2.5 \
  --prefill localhost:8100 --decode localhost:8200 --port 8000
```

**发压**: 1/4/16/32/64
**关键观察**: TTFT (P prefill + KV transfer + D start), cache_rate (P APC), 503/超时

### Phase 3: PD + Bidirectional KV (多轮复用)

**目的**: 启用 D→P 反向 KV 传输, 多轮对话中 P 只 prefill 新增 token。

| 组件 | 参数 |
|---|---|
| Prefill | TP4, kv_producer, bidirectional_kv_xfer=true |
| Decode | TP4, kv_consumer, bidirectional_kv_xfer=true |
| Proxy | disagg_proxy_multiturn.py (修改: 自动生成 conversation_id) |
| kv_recompute_threshold | 64 (默认) |
| decoder_kv_blocks_ttl | 480 (默认) |

**启动**:
```bash
# Prefill
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"bidirectional_kv_xfer":true,"kv_lease_duration":60}}'

# Decode
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"bidirectional_kv_xfer":true}}'

# Proxy (multiturn, 支持跨轮 KV 参数缓存)
python disagg_proxy_multiturn.py --host 0.0.0.0 --port 8000 \
  --prefiller-host localhost --prefiller-port 8100 \
  --decoder-host localhost --decoder-port 8200
```

**发压**: 1/4/16/32/64
**关键观察**: TTFT 对比 Phase 2 (bidirectional 应降低多轮 TTFT), D→P 传输耗时

### Phase 4: PD 自主优化 (迭代)

基于 Phase 2/3 的指标, 按优先级尝试:

| 策略 | 参数 | 预期效果 | 优先级 |
|---|---|---|---|
| 4.1 chunked_prefill 调优 | 8192 vs 16384 vs 32768 | 平衡 prefill 独占与 TTFT | P0 |
| 4.2 kv_lease_duration 调优 | 60→120→300 | 避免高并发下 P 块过期 | P1 |
| 4.3 max_num_seqs 调优 | P: 32 vs 64 vs 128 | P 侧并发 prefill 数 | P1 |
| 4.4 gpu_memory_utilization | 0.85 vs 0.90 vs 0.92 | KV 池大小 vs OOM 边界 | P2 |
| 4.5 kv_recompute_threshold | 64 vs 128 vs 256 | bidirectional 触发阈值 | P2 |
| 4.6 UCX_TLS 调优 | cuda_ipc,cuda_copy vs all | 传输效率 | P3 |
| 4.7 Push vs Pull 模式 | NixlConnector vs NixlPushConnector | P push 可能减少 D 等待 | P3 |

**迭代原则**: 每次只改一个参数, 对比前一组结果, 记录 metrics, 提交报告。

---

## 4. 指标采集

### 4.1 benchmark 指标 (kvcache-benchmarks summary.json)

| 指标 | 来源 |
|---|---|
| 成功率 | succeeded / total |
| TTFT p50/p90 | summary |
| TPOT p50/p90 | summary |
| cache_rate | cached_tokens / prompt_tokens |
| prompt/completion tokens | usage |

### 4.2 vLLM 运行时指标 (/metrics)

| 指标 | 端点 |
|---|---|
| vllm:num_requests_running | P:8001/metrics, D:8002/metrics |
| vllm:num_requests_waiting | 同上 |
| vllm:gpu_cache_usage_perc | 同上 |
| vllm:cache_eviction | 同上 |
| vllm:nixl_xfer_time_seconds | P+D /metrics |
| vllm:nixl_bytes_transferred | P+D /metrics |
| vllm:nixl_num_kv_expired_reqs | P /metrics |

### 4.3 Proxy 日志

- disagg_proxy_multiturn.py: 打印 prefill 耗时, cache HIT/MISS, KV blocks 数
- disagg_proxy_demo.py: 基本请求转发日志

### 4.4 GPU 利用率

```bash
# 每 5s 采样
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 5
```

---

## 5. 部署脚本结构

```
develop/vllm/
├── minimax_vllm_pd_plan.md          ← 本文档
├── deploy/
│   ├── minimax/
│   │   ├── start-baseline-tp8.sh     Phase 1: TP8 基线
│   │   ├── start-prefill-pd.sh       Phase 2: PD prefill (无 bidir)
│   │   ├── start-decode-pd.sh        Phase 2: PD decode (无 bidir)
│   │   ├── start-proxy-demo.sh       Phase 2: disagg_proxy_demo.py
│   │   ├── start-prefill-bidir.sh    Phase 3: PD prefill (bidir)
│   │   ├── start-decode-bidir.sh     Phase 3: PD decode (bidir)
│   │   ├── start-proxy-multiturn.sh  Phase 3: disagg_proxy_multiturn.py (修改版)
│   │   ├── run_bench.sh              发压入口 (调用 kvcache-benchmarks)
│   │   └── stop_all.sh               停止所有容器
│   └── proxy/
│       └── disagg_proxy_multiturn_autoid.py  修改版 (自动 conversation_id)
└── results/                          实验结果
```

---

## 6. 执行顺序

1. **Phase 1**: 启动 TP8 baseline → 等 health → 冒烟测试 → 发压 1/4/16/32/64 → 收集结果
2. **Phase 2**: 停 baseline → 启 PD (无 bidir) → 等 health → 冒烟 → 发压 → 收集
3. **Phase 3**: 停 PD → 启 PD (bidir) → 等 health → 冒烟 → 发压 → 收集
4. **Phase 4**: 基于 Phase 2/3 结果, 选最优 PD 配置作为基线, 逐个试优化策略
5. 每组完成后写 README + 提交 + push

---

## 7. 风险与注意事项

1. **MTP (Multi-Token Prediction)**: config 有 `use_mtp: true`, vLLM 可能需要 `--speculative-config` 启用, 或自动处理。Phase 1 冒烟测试时验证。
2. **conversation_id**: Phase 3 需要修改 proxy 或 benchmark 工具注入 conversation_id, 否则 bidirectional 不生效。
3. **UCX_TLS 单机**: 单机 8 卡需 `cuda_ipc,cuda_copy,tcp`, 而非默认 `all` (避免 IB 探测延迟)。
4. **无联网**: h200-2 无 DNS, 所有文件 (proxy 脚本, bench 工具) 须从本机 SCP 过去或挂载。
5. **Docker 共享内存**: PD 模式 NIXL 侧信道需要大 SHM, `--shm-size 16g`。
6. **Docker 网络**: `--network host` 简化端口映射 (P:8100, D:8200, proxy:8000, side_channel:5600/5601)。
