# SchedulerConfig（scheduler.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > SchedulerConfig

源码：`vllm/config/scheduler.py`（约 321 行）。`SchedulerConfig` 描述调度器的批次与策略参数：最大 token/seq 数、chunked prefill、async scheduling、policy、DP prefill 节拍、watermark 等。它是 `VllmConfig.scheduler_config`，被 `Scheduler`/`AsyncScheduler` 与 `EngineCore` 消费。

## 是什么

`@config` 装饰（`scheduler.py:25`）。用 `InitVar` 传 `max_model_len`/`is_encoder_decoder` 给 `__post_init__`（这类"从其它配置借来供校验/默认推导"的值用 `InitVar` 不持久化为字段）。`DEFAULT_MAX_NUM_BATCHED_TOKENS=2048`、`DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP=256`、`DEFAULT_MAX_NUM_SEQS=128`。

| 字段 | 默认 | 含义 |
|---|---|---|
| `runner_type` | `"generate"` | `generate`/`pooling`/`draft` |
| `max_num_batched_tokens` | `2048` | 单步最多处理 token 数 |
| `max_num_scheduled_tokens` | `None` | 单步最多下发 token 数（投机解码时小于 `max_num_batched_tokens` 以留 drafter 槽） |
| `max_num_seqs` | `128` | 单步最多 seq 数 |
| `max_num_partial_prefills` | `1` | chunked prefill 并发部分填充 seq 上限 |
| `max_long_partial_prefills` | `1` | 长 prompt（>`long_prefill_token_threshold`）并发 prefill 上限（小于此则短 prompt 可插队） |
| `long_prefill_token_threshold` | `0` | 长 prompt 阈值，0=禁用 |
| `enable_chunked_prefill` | `True` | 是否允许 prefill 分块 |
| `is_multimodal_model` | `False` | 是否多模态 |
| `max_num_encoder_input_tokens` | init=False |=`max_num_batched_tokens` | 多模态编码器计算预算 |
| `encoder_cache_size` | init=False |=`max_num_batched_tokens` | 多模态编码器缓存 |
| `policy` | `"fcfs"` | `fcfs`/`priority` |
| `disable_chunked_mm_input` | `False` | 禁止分割多模态项（保证一个图像/item 整体调度） |
| `scheduler_cls` | `None` | 调度器类或 `"mod.custom_class"` 路径 |
| `disable_hybrid_kv_cache_manager` | `None`(三态) | 关 HMA：`True` 强制关、`None` 自动、`False` 强制开 |
| `scheduler_reserve_full_isl` | `True` | admit 前检查完整 ISL 是否能容下（防 chunked prefill 过度 admit） |
| `watermark` | `0.0` | KV cache 空闲块水位（[0,1)，0=禁用） |
| `prefill_schedule_interval` | `1` | DP 下每 N 步才 admit 新 prefill（跨 DP rank 对齐） |
| `async_scheduling` | `None`(三态) | 异步调度（GPU 利用率提升） |
| `stream_interval` | `1` | 流式 token 缓冲间隔（1=逐 token，大值批处理减开销） |

`InitVar`：`max_model_len`（仅用于 `verify_max_model_len` 与 `long_prefill_token_threshold` 默认推导）、`is_encoder_decoder`（强制关 chunked prefill + prefix caching）。

### 关键方法

- `default_factory(**kwargs)`（`scheduler.py:169`）：给 `InitVar` 填默认（`max_model_len=8192`/`is_encoder_decoder=False`）的工厂，供 `VllmConfig` `Field(default_factory=SchedulerConfig.default_factory)` 用。
- `get_scheduler_cls()`（`scheduler.py:180`）：按 `async_scheduling` 选 `AsyncScheduler`/`Scheduler`，支持自定义路径（带 warning）。
- `compute_hash()`（`scheduler.py:203`）：仅纳入 `max_num_batched_tokens`（因 LoRA 静态 buffer + Inductor 32/64-bit 索引依赖，见 [issue #29585](https://github.com/vllm-project/vllm/issues/29585)）。
- `verify_max_model_len(max_model_len)`（`scheduler.py:272`）：校验 `max_num_batched_tokens` 与 `max_model_len`/`max_num_seqs` 关系，`max_num_partial_prefills>1` 时校验 chunked prefill 与 `long_prefill_token_threshold`。
- `__post_init__(max_model_len, is_encoder_decoder)`：encoder-decoder 强制关 chunked prefill + 长 prefill 阈值；`max_num_partial_prefills>1` 时默认 `long_prefill_token_threshold = max_model_len*0.04`；调 `verify_max_model_len`。

## 为什么

- **统一调度模型**：v1 取消显式 prefill/decode phase，所有请求按 `num_new_tokens` 切分。`max_num_batched_tokens` 是总预算，`max_num_scheduled_tokens` 是扣除 drafter 预留后的可下发量（`VllmConfig._set_max_num_scheduled_tokens` 推导）。
- **`InitVar` 借值**：`max_model_len`/`is_encoder_decoder` 已在 `ModelConfig`，`SchedulerConfig` 只在校验/默认推导时需要，用 `InitVar` 避免重复存储与不一致。
- **三态开关**：`disable_hybrid_kv_cache_manager`/`async_scheduling` 用 `None` 让 `VllmConfig` 按平台/特性/connector 自动决定，显式 True/False 走硬校验（防止用户在不支持场景强开导致静默错误）。
- **`compute_hash` 精简**：仅 `max_num_batched_tokens` 影响 LoRA 静态 buffer 形状与 Inductor 索引位宽，其余策略字段不改变图形状。
- **DP prefill 节拍**：`prefill_schedule_interval` 让各 DP rank 对齐 admit 时机，平衡每步 forward 时长。

## 怎么做

- **批次上限**：`--max-num-batched-tokens 8192 --max-num-seqs 256`。
- **chunked prefill**：默认开；`--max-num-partial-prefills 4 --max-long-partial-prefills 1` 让短 prompt 插队。
- **async scheduling**：`--async-scheduling`（显式开，硬校验）或留 `--no-async-scheduling`；不开则 `VllmConfig` 按 spec/executor/pooling 自动决定。
- **priority**：`--scheduler-policy priority`，请求带 `priority` 字段（值小优先）。
- **DP 节拍**：`--prefill-schedule-interval 4` 每 4 步 admit 一次 prefill。
- **HMA**：`--no-disable-hybrid-kv-cache-manager` 显式开（与不支持 connector 配置时 raise）；默认自动。
- **watermark**：`--scheduler-watermark 0.1` 留 10% 空闲块防抖动。

## 与其它模块/系统配合

- **调度器（[`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)）**：`max_num_*`/`policy`/`async_scheduling` 直接驱动 `Scheduler`/`AsyncScheduler`；`disable_hybrid_kv_cache_manager` 决定 `KVCacheManager` 是否走 `HybridKVCacheCoordinator`。
- **EngineCore（[`01-engine-core/engine-core-process.md`](../01-engine-core/engine-core-process.md)）**：`async_scheduling` 与 `max_concurrent_batches`(VllmConfig 派生) 决定 inflight batch 数；`stream_interval` 影响输出回流。
- **多模态（[multimodal-config.md](multimodal-config.md) 与 [`11-multimodal/`](../11-multimodal/README.md)）**：`is_multimodal_model`/`encoder_cache_size` 驱动 `EncoderCacheManager`；`disable_chunked_mm_input` 保证图像整体调度。
- **LoRA（[lora-config.md](lora-config.md)）**：`max_num_batched_tokens` 决定 LoRA 静态 buffer 形状（故进哈希）。
- **投机解码（[speculative-config.md](speculative-config.md)）**：`max_num_scheduled_tokens` 由 `VllmConfig._set_max_num_scheduled_tokens` 按 `max_num_new_slots_for_drafting*max_num_seqs` 扣减；`async_scheduling` 限制 spec method 白名单。
- **编译（[compilation-config.md](compilation-config.md)）**：`max_num_batched_tokens` 是 `compile_range_end` 与 cudagraph 尺寸推导上界。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`async_scheduling` 三态决策、`disable_hybrid_kv_cache_manager` 三态与 connector `SupportsHMA` 判定、pooling 模型默认关 async。

## 历史版本演进

- **v0.5/v0.6（v0）**：`SchedulerConfig` 含 `max_num_batched_tokens`/`max_num_seqs`/`enable_chunked_prefill`（默认 `False`）/`policy`/`max_num_batched_tokens` 等；显式 prefill/decode phase。
- **v0.7（v1 落地）**：v1 `Scheduler` 重写，统一 `num_computed_tokens` 模型；`enable_chunked_prefill` 默认 `True`；`async_scheduling` 字段引入；`InitVar` 模式用于 `max_model_len`/`is_encoder_decoder`；`default_factory` 支持默认 `InitVar`。
- **v0.8**：`async_scheduling` 三态（`bool | None`）正式化；`AsyncScheduler` 落地；`max_concurrent_batches` 派生；`disable_hybrid_kv_cache_manager` 三态引入。
- **v0.9**：`scheduler_reserve_full_isl` 准入门控；`prefill_schedule_interval` DP prefill 节拍；`disable_chunked_mm_input`；动态投机 `dynamic_sd_lookup`（在 `VllmConfig` 侧联动 `cudagraph_mode`）。
- **v0.10**：`DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP`（batched DP 256）；`watermark` 字段加入防抖动；`stream_interval` 暴露为可调。
- **v0.11 / v0.12 / main**：`scheduler_cls`/`async_scheduling` 用 `_skip_none_validation` wrap validator 支持延迟初始化；MRv2（`v1/worker/gpu/`）与 `async_scheduling`+PP 的 `pp_size+1` 并发；`long_prefill_token_threshold` 与 Mamba align 模式联动校验。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [vllm-config.md](vllm-config.md) — `async_scheduling`/HMA 三态决策与 `max_num_scheduled_tokens` 推导。
- [cache-config.md](cache-config.md) — `kv_cache_size_tokens` 与 `scheduler_reserve_full_isl` 配合。
- [speculative-config.md](speculative-config.md) — `max_num_new_slots_for_drafting` 影响 `max_num_scheduled_tokens`。
- [../01-engine-core/scheduler/scheduler.md](../01-engine-core/scheduler/scheduler.md) — 主消费方。
