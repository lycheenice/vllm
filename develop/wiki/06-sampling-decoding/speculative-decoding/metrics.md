[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > SpecDecoding Metrics

# SpecDecoding Metrics（指标采集）

> 源码：`vllm/v1/spec_decode/metrics.py`

---

## 是什么

`metrics.py` 提供 spec decode 的三类统计组件：

1. **`SpecDecodingStats`**（行 17）：dataclass，per-step 累积计数器——`num_drafts`、`num_draft_tokens`、`num_accepted_tokens`、`num_accepted_tokens_per_pos`、`num_draft_tokens_per_pos`。scheduler 每步调 `observe_draft(num_draft_tokens, num_accepted_tokens)` 累积。
2. **`SpecDecodingLogging`**（行 52）：周期性 log 输出 mean acceptance length / draft throughput / accepted throughput / per-position acceptance rate；也支持 diffusion LLM 模式（`is_diffusion=True`）把同套计数器改语义输出 denoising steps / canvas tokens / committed tokens。
3. **`SpecDecodingProm`**（行 177）：Prometheus 指标采集——`vllm:spec_decode_num_drafts` / `num_draft_tokens` / `num_accepted_tokens` / `num_accepted_tokens_per_pos` 三个 Counter（前两个 base 按 engine_idx 分 label），可用于 Grafana / Datadog 看板。

## 为什么

- **可观测性必要**：spec decode 加速比强依赖 drafter 质量；用户需观察 acceptance rate 与 mean acceptance length 判断 spec decode 是否物有所值（如果接受率 < 30% 通常不该开 spec decode）。
- **per-position acceptance rate**：drafter 在第 1 个 draft 位置通常高接受率，越后越低；per-position 数据让用户决定最佳 K（持续 K 个位置但接受率太低反而有害）。
- **diffusion LLM 复用**：diffusion LLM (dLLM) 模型复用 spec-decode 数据流但语义不同——每个 draft = 一次去噪 step，draft_tokens = canvas 位置数，accepted = 已确定 token。`SpecDecodingLogging._log_diffusion` 复用同一套 counter 改语义输出。
- **Prometheus label per engine**：多 engine 部署（DBO / DP）时按 `engine_idx` 分 label 让看板能区分。
- **time-window reset**：LoggingStatLogger 周期性 reset 而非累积——`last_log_time = time.monotonic()`，每次 `log()` 后 reset 计数器，让输出反映最近 window 而非全 session 累积。

## 怎么做

### SpecDecodingStats

```python
@dataclass
class SpecDecodingStats:
    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_draft_tokens_per_pos: list[int] = field(default_factory=list)

    def observe_draft(self, num_draft_tokens, num_accepted_tokens):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
        for i in range(num_draft_tokens):
            self.num_draft_tokens_per_pos[i] += 1
```

`scheduler.new()` 创建一个 `SpecDecodingStats.new(num_spec_tokens)`；每 step 调 `observe_draft` 累积；step 结束时把 stats 放入 `EngineCoreOutputs.SchedulerStats`。

### SpecDecodingLogging

```python
class SpecDecodingLogging:
    def __init__(self, is_diffusion=False): ...
    def observe(self, spec_decoding_stats): ...
    def log(self, log_fn=logger.info): ...
```

`observe` 把每次 stats append 到内部 list；`log` 在 `last_log_time` 至少经过预设间隔时触发：

- 计算 window 内总 drafts / draft_tokens / accepted_tokens。
- 计算吞吐：`draft_throughput = num_draft_tokens / elapsed_time`、`accepted_throughput = num_accepted_tokens / elapsed_time`。
- `mean_acceptance_length = 1 + num_accepted_tokens / num_drafts`（含 bonus）。
- `acceptance_rates[pos] = sum(num_accepted_tokens_per_pos) / num_drafts`。
- `draft_acceptance_rate = accepted / drafted * 100%`。
- 调 `log_fn("SpecDecoding metrics: Mean acceptance length: %.2f, ..." % (...))`。

`is_diffusion=True` 时调 `_log_diffusion`：输出改为 "Committed token throughput / Mean denoising steps per canvas / Mean tokens committed per denoising step / Committed tokens / Denoising steps / Canvas positions evaluated"。

### SpecDecodingProm

```python
class SpecDecodingProm:
    def __init__(self, speculative_config, labelnames, per_engine_labelvalues, is_diffusion=False):
        if not self.spec_decoding_enabled: return
        if is_diffusion:
            counter_specs = [
                ("vllm:diffusion_num_denoising_steps", ...),
                ("vllm:diffusion_num_canvas_positions", ...),
                ("vllm:diffusion_num_committed_tokens", ...),
            ]
        else:
            counter_specs = [
                ("vllm:spec_decode_num_drafts", ...),
                ("vllm:spec_decode_num_draft_tokens", ...),
                ("vllm:spec_decode_num_accepted_tokens", ...),
            ]
        counters = [create_metric_per_engine(Counter(...), per_engine_labelvalues) for ...]
        self.counter_spec_decode_num_drafts = counters[0]
        ...
        # per-position counter
        if not is_diffusion:
            pos_labelnames = labelnames + ["position"]
            base_counter = Counter("vllm:spec_decode_num_accepted_tokens_per_pos", ...)
            self.counter_spec_decode_num_accepted_tokens_per_pos = {
                idx: [base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)]
                for idx, lv in per_engine_labelvalues.items()
            }
    
    def observe(self, spec_decoding_stats, engine_idx=0):
        self.counter_spec_decode_num_drafts[engine_idx].inc(spec_decoding_stats.num_drafts)
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(...)
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(...)
        for pos, counter in enumerate(self.counter_spec_decode_num_accepted_tokens_per_pos.get(engine_idx, [])):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
```

PromQL 计算示例（见 docstring 行 178–195）：

```promql
rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
rate(vllm:spec_decode_num_draft_tokens_total[$interval])
```

### scheduler 集成

scheduler 在每 step consolidate spec decoding stats：

- 遍历所有 requests 的 `num_acceptance_tokens`、`num_draft_tokens`。
- 对每个 request 调 `SpecDecodingStats.observe_draft(num_draft_tokens, num_accepted_tokens)`。
- 把 stats 通过 `EngineCoreOutputs.SchedulerStats` 发到 frontend；frontend 的 metrics aggregator 调 `SpecDecodingLogging.observe` 与 `SpecDecodingProm.observe`。

## 与其它模块/系统配合

- [../rejection-sampler.md](../rejection-sampler.md)：rejection sampler 输出的 `[batch, K+1]` token 矩阵中 `PLACEHOLDER == -1` 个数 = num_rejected； scheduler 据此算 num_accepted。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：scheduler 是 stats 采集发起者。
- [可观测性-metrics](../../16-observability/README.md)（待补充）：`vllm:v1/metrics/utils.py:create_metric_per_engine` 是 metric 注册 helper；Prometheus Counter 全部走此 helper。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：ModelRunner 在 rejection 后 parse output 得到每请求的 accepted tokens count，回传给 scheduler。
- [diffusion LLM 模型库](../../04-model-zoo/README.md)（待补充）：dLLM 模型走 `is_diffusion=True` 复用同一 metric 框架。

## 历史版本演进

- **v0.7.0**：V1 `SpecDecodingStats` / `SpecDecodingLogging` 与 spec decode 同期 landfall；只支持 spec decode 语义。
- **v0.8.0**：`SpecDecodingProm` Prometheus counters 加入；per-position counter 引入。
- **v0.9.0**：`mean_acceptance_length` 公式确定（含 bonus token：`1 + accepted/drafts`）。
- **v0.10.0**：`create_metric_per_engine` 让多 engine counter 共享 base 不同 label；`_log_diffusion` 引入服务 dLLM。
- **v0.10.5+**：周期 reset 而非累积；`reset()` 在 `log()` 末尾统一调用；num_accepted_tokens_per_pos 的索引边界检查完善。
- **v0.12 / main**：差异 coverage——`is_diffusion` 跳过 per-position counter（语义不适用）。

[← 返回投机解码](../README.md)

## 参见

- [../rejection-sampler.md](../rejection-sampler.md)
- [可观测性-metrics](../../16-observability/README.md)
- [dynamic.md](dynamic.md)：dynamic K 下 metric 的 par-position 解读
