[← Wiki 首页](../../README.md) > [可观测](../../README.md) > v1/metrics/perf

# Perf（PerfStats / ModelMetrics / MFU 估算）

> 源码：`vllm/v1/metrics/perf.py`（1634 行）

## 是什么

`perf.py` 实现 v1 的 **Model FLOPs Utilization（MFU）与内存带宽估算**——不依赖硬件计数器，纯解析估算：根据 `VllmConfig`（模型架构 + 并行 + 量化）计算每步理论 flops / read bytes / write bytes，喂给 [loggers.py](loggers.md) 输出 `vllm:estimated_flops_per_gpu_total` 等 Counter，让用户用 PromQL `rate(...)/1e12` 算出 TF/s 与厂商标称峰值对比。

顶层结构：

| 类/函数 | 行号 | 角色 |
|---|---|---|
| `PerfStats` | `perf.py:94` | dataclass：`num_flops_per_gpu`/`num_read_bytes_per_gpu`/`num_write_bytes_per_gpu` + 可选 `DebugPerfStats`；嵌入 `SchedulerStats.perf_stats` |
| `ExecutionContext` | `perf.py:102` | 一个 batch 的 prefill/decode token × context 累积量；`add(num_tokens, context_len, is_prefill)` |
| `Parser` Protocol + `ParserChain` | `perf.py:194`/`perf.py:203` | 把 `VllmConfig` 解析为 `ParsedArgs` 的责任链 |
| `ComponentMetrics` (ABC + `__init_subclass__` 注册) | `perf.py:227` | pydantic BaseModel，每子类对应一个模型组件（attn/mla_attn/ffn/unembed），实现 `get_num_flops_breakdown`/`get_read_bytes_breakdown`/`get_write_bytes_breakdown` |
| `AttentionMetrics` | `perf.py:412` | 标准 MHA/GQA 注意力 |
| `MLAAttentionMetrics` | `perf.py:583` | DeepSeek MLA（kv_lora_rank 压缩 KV cache） |
| `FfnMetrics` | `perf.py:945` | Dense FFN + MoE routed + shared expert，支持 EP/TP |
| `UnembedMetrics` | `perf.py:1202` | lm_head unembedding |
| `ModelMetrics` | `perf.py:1268` | 聚合所有可实例化的 `ComponentMetrics`；`get_step_perf_stats_per_gpu(scheduler_output)` 是热路径入口 |
| `PerfMetricsLogging` | `perf.py:1480` | 周期 `log()` 打 "MFU: X.X TF/s/GPU Y.Y GB/s/GPU" |
| `PerfMetricsProm` | `perf.py:1548` | 注册三个 Prom Counter：`vllm:estimated_flops_per_gpu_total`、`vllm:estimated_read_bytes_per_gpu_total`、`vllm:estimated_write_bytes_per_gpu_total` |
| `PerfMetricsDebugLogging` | `perf.py:1405` | `VLLM_DEBUG_MFU_METRICS` 时按 component breakdown 聚合，`.log()` dump 为 JSON |
| `_QUANT_WEIGHT_BYTE_SIZE` | `perf.py:52` | 量化方法→权重字节映射（fp8=1B，int4/awq/gptq=0.5B 等） |

热路径 `ModelMetrics.get_step_perf_stats_per_gpu()`（`perf.py:1339`）：

1. 从 `SchedulerOutput` 遍历 `scheduled_new_reqs` + `scheduled_cached_reqs`，把每个请求的 `num_scheduled_tokens` 与 `context_len` 喂给 `ExecutionContext.add()`（cached req 若 `num_tokens>1` 算 prefill，否则 decode）。
2. 对每个 `ComponentMetrics` 调 `get_*_breakdown(ctx, per_gpu=True)`，按 TP/PP/EP 切分。
3. 求和构造 `PerfStats`；若 `VLLM_DEBUG_MFU_METRICS` 则附加 `DebugPerfStats`（含 calc_duration、各 component 明细）。

```mermaid
flowchart LR
    SO["SchedulerOutput<br/>scheduled_new_reqs / scheduled_cached_reqs"]
    SO --> EC["ExecutionContext<br/>prefill/decode aggregates"]
    MM["ModelMetrics<br/>(per ComponentMetrics)"]
    EC --> MM
    MM -->|"get_*_breakdown(ctx, per_gpu=True)|" PS["PerfStats<br/>num_flops/read_bytes/write_bytes_per_gpu"]
    PS --> SCHEDSTATS["SchedulerStats.perf_stats"]
    SCHEDSTATS --> LOG["PerfMetricsLogging.observe()"]
    SCHEDSTATS --> PROM["PerfMetricsProm.observe()"]

    style MM fill:#fde,stroke:#c30
    style PS fill:#dfd,stroke:#393
```

## 为什么

- **不依赖 CUPTI/硬件计数器的吞吐量**：跨厂商（NVIDIA/AMD/TPU/Intel）统一口径；用户只需 PromQL `rate(vllm:estimated_flops_per_gpu_total[1m])/1e12` 即可得 TF/s，再除以 GPU 标称峰值即 MFU。
- **per-gpu 视角**：所有计算都按 `tp_size`/`pp_size`/`ep_size` 切分后返回单卡量，避免多卡部署时数字膨胀——直接和单卡峰值比。
- **ComponentMetrics 注册表**：用 `__init_subclass__` 自动注册，新增组件（如未来 Mamba）只需写子类即被 `ModelMetrics` 收纳。
- **Parser Chain 解耦**：从 `VllmConfig` 解析字段（hidden_size/num_heads/...）拆成独立 Parser，可复用、可覆盖（如 `AttentionQuantizationConfigParser` 覆盖 weight_byte_size）。
- **量化感知**：`_QUANT_WEIGHT_BYTE_SIZE` 表让 FP8/INT4/AWQ/GPTQ 等量化的权重字节正确反映在 read_bytes 估算里。
- **MLA 单列**：DeepSeek MLA 的 KV cache 压缩到 `kv_lora_rank + qk_rope_head_dim`，与标准 MHA 计算公式完全不同，故单列 `MLAAttentionMetrics`，由 `MLADetectionParser` 仅在 `is_deepseek_mla` 时实例化。
- **MoE load balance 假设**：`FfnMetrics` 假设完美负载均衡（`num_activated_experts = min(num_activated_tokens, num_experts)`），并用 `ffn_ep_size` 切——文档显式标 `FIXME: Assume perfect load balancing for now`，已知偏差。
- **debug breakdown**：`VLLM_DEBUG_MFU_METRICS` 把每 component 的 flops/bytes breakdown 累积并以 JSON 形式 `logger.debug` 出，调 MFU 异常时定位哪个组件估算偏差。
- **aggregate logger 禁用 perf**：`AggregatedLoggingStatLogger._enable_perf_stats()` 直接 `return False`，避免多 engine 数字相加产生误导。

## 怎么做

**用户开启**：`--enable-mfu-metrics` 触发 `ObservabilityConfig.enable_mfu_metrics=True`，`LoggingStatLogger._enable_perf_stats() -> True`，构造 `PerfMetricsLogging`；`PrometheusStatLogger` 总是构造 `PerfMetricsProm`（observe 时若 0 则跳过 inc）。

**调试明细**：`VLLM_DEBUG_MFU_METRICS=1` 让 `PerfStats.debug_stats` 不为 `None`，进 `PerfMetricsDebugLogging` 聚合，每周期 `logger.debug` 出 JSON（含 calc_duration 占比 `mfu_calc_overhead`）。

**PromQL**：

```promql
# TF/s per GPU
rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12

# GB/s per GPU
(rate(vllm:estimated_read_bytes_per_gpu_total[1m])
 + rate(vllm:estimated_write_bytes_per_gpu_total[1m])) / 1e9
```

**自定义 ComponentMetrics**：写 `ComponentMetrics` 子类，`__init_subclass__` 自动注册；实现 `component_type()`/`get_parser()`/`get_*_breakdown()`；`get_parser()` 必须提供所有声明字段的 ParsedArgs，否则 `from_vllm_config` 抛 `InvalidComponent`（被 `ModelMetrics` 静默跳过）。

## 与其它模块/系统配合

- **[stats.py](stats.md)**：`PerfStats` 字段挂在 `SchedulerStats.perf_stats`。
- **[loggers.py](loggers.md)**：`PrometheusStatLogger` 持有 `PerfMetricsProm`；`LoggingStatLogger` 持有 `PerfMetricsLogging` 并在 `log()` 末尾调 `perf_metrics_logging.log()`。
- **SchedulerOutput**（待核实）：`ModelMetrics.get_step_perf_stats_per_gpu(scheduler_output)` 是热路径入口，由 EngineCore 在每步调度后调用。
- **`vllm/utils/torch_utils.py`**：`get_dtype_size`/`get_kv_cache_torch_dtype`/`STR_DTYPE_TO_TORCH_DTYPE` 决定权重与 KV cache 字节大小。
- **QuantizationConfig**：`cfg.get_name()` 决定走 `_QUANT_WEIGHT_BYTE_SIZE` 哪一项；不支持的量化方法抛 `InvalidComponent` 整个组件被跳过。
- **spec decode**：spec decode 的 draft 模型 forward 不计入 `PerfStats`（仅 target model），故 MFU 估算不含 spec decode 开销（待核实）。
- **ObservabilityConfig**：`enable_mfu_metrics` 主开关。

## 历史版本演进

- **v0.5/v0.6（v0）**：无 MFU 估算（v0 不含此模块）。
- **v0.7（v1 落地）**：无 perf.py（待核实）。
- **v0.8**：引入 `perf.py` 初版，仅 `AttentionMetrics` + `UnembedMetrics`；`PerfStats` 单字段 `num_flops_per_gpu`。
- **v0.9**：`FfnMetrics`（Dense + MoE）；`MLAAttentionMetrics` 单列；`read_bytes`/`write_bytes` 三联；`PerfMetricsProm` 三个 Counter；`VLLM_DEBUG_MFU_METRICS` debug breakdown。
- **v0.10–v0.12/main**：`FfnParallelParser` 区分 EP/TP；`InterleaveMoeLayerStepParser`（Llama4）+ `MoeLayerFreqParser`（Deepseek）正确推导 `num_moe_layers`；`PerfMetricsDebugLogging` JSON dump。具体版本归属（待核实）。

[← 返回可观测首页](../../README.md)

## 参见

- [stats.md](stats.md) — `PerfStats` 嵌入 `SchedulerStats.perf_stats`。
- [loggers.md](loggers.md) — `PerfMetricsLogging`/`PerfMetricsProm` 由 logger 持有。
- `../../10-config/observability-config.md` — `enable_mfu_metrics` 开关。
- `../../10-config/quantization-config.md` — 量化方法 → weight_byte_size 映射。
