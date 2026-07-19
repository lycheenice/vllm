[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > Dynamic SD

# Dynamic Speculative Decoding（动态 K 调度）

> 源码：`vllm/v1/spec_decode/dynamic/utils.py` + scheduler 端集成

---

## 是什么

Dynamic SD 让 spec decode 的 K（每步 draft token 数）不再是固定常量，而是根据当前 batch size 动态调整——batch 越大 K 越小，避免大 batch + 高 K 时 drafter forward 反而拖累 target verify 的吞吐。用户通过 `num_speculative_tokens_per_batch_size` 配置项指定一组"batch_size 区间 → K"的映射，scheduler 在每步查表得到当前 K。

调度数据结构（`vllm/v1/spec_decode/dynamic/utils.py:4`）：

```python
DynamicSDSchedule = list[tuple[int, int, int]]
# 每项: (range_start, range_end, num_speculative_tokens)
```

## 为什么

- **大 batch 时 K 太大反而慢**：target verify K+1 个位置成本随 K 线性增加；batch 越大每步 base forward 越贵，再加 K 个 draft verify 在显存/算力上得不偿失。Dynamic SD 让用户根据 workload 自适应。
- **drafter 自身瓶颈**：EAGLE 类 drafter 在 K=4 以上 multi-pass 开销显著；当 batch 大时 drafter forward 也变贵，最好在每个 request scale down。
- **配置驱动而非运行时启发式**：vLLM 选择把决策权交给用户配置而非运行时预测。配置是显式的"区间 → K"列表，避免运行时学习的不确定性。
- **dense lookup table**：`build_dynamic_sd_schedule_lookup` 在初始化时把用户配置展开成 `[0..max_batch_size+1]` 的 dense list，让 scheduler 每 step O(1) 查表，而非 O(M) 线性扫描配置 ranges。
- **区间非覆盖约束**：`validate_and_normalize_dynamic_sd_schedule` 强制 ranges 不重叠、必须从 1 开始、按起始排序；缺省的"间隙"用上一个 K 自动填充（carry-forward），让用户写配置时不必列全所有 batch size。

## 怎么做

### validate_and_normalize_dynamic_sd_schedule（行 7）

输入 `num_speculative_tokens_per_batch_size`（list of 3-tuples），输出 `DynamicSDSchedule`（normalized list）。

校验规则：

- 必须是 list，非空；每项 3 元素。
- `range_start <= range_end`、均正数。
- `num_speculative_tokens >= 0`（0 表示本区间不 spec decode）。
- 排序后 ranges 不可重叠（`range_start > previous_end`）。
- **第一个 range 必须从 1 开始**——保证任何 batch_size ≥ 1 都有 K 值可查。
- gaps 自动在 `build_dynamic_sd_schedule_lookup` 中用 carry-forward 填充，不在 validator 中拒绝。

举例：

```
用户配置: [(1, 16, 4), (32, 128, 2)]
展开后 dense lookup:
  batch_size ∈ [1, 16]   → K=4
  batch_size ∈ [17, 31]  → K=4（carry-forward）
  batch_size ∈ [32, 128] → K=2
  batch_size ∈ [129, max_batch_size] → K=2（carry-forward tail）
```

### build_dynamic_sd_schedule_lookup（行 77）

```python
def build_dynamic_sd_schedule_lookup(
    num_speculative_tokens_per_batch_size, vllm_max_batch_size, vllm_num_speculative_tokens,
) -> list[int]:
    ...
```

构造 `[0] * (vllm_max_batch_size + 1)`，按 ranges 与 carry-forward 填充：

- 进每个 range 前：若有 gap 且 `last_num_speculative_tokens is not None`，用前一个 K 填充 gap。
- 进 range 内：`dense_schedule[bs] = min(vllm_num_speculative_tokens, K_in_range)`——vLLM 全局 `num_speculative_tokens` 是上限。
- 退 range 后的 tail：用最后 K carry-forward 到 max_batch_size。

返回的 list 是 1-indexed：`dense_schedule[batch_size] = K`。

### scheduler 端使用

`vllm/v1/core/sched/scheduler.py:253` 处（待核实行号与具体函数）：scheduler 每步查 `dense_schedule[current_batch_size]` 得当前 K；若 K=0 走非 spec decode 路径；若 K>0 走 spec decode 路径，并把 `num_speculative_tokens` 传给 drafter.propose。

drafter 接收 `num_speculative_tokens` 参数（基类 propose 第一参数），按其值生成 K 个 draft token——与固定 K 路径完全相同。

###drafter 端的"K=0 早返回"

`SpecDecodeBaseProposer.propose` 行 610：

```python
if self.num_speculative_tokens == 0:
    return torch.empty(batch_size, 0, device=sample_hidden_states.device, dtype=torch.int64)
```

Dynamic SD 让 drafter 在 K=0 时仍跑 first pass forward（保持 KV cache 同步），但返回空 draft tensor；scheduler 走非 spec 路径，rejection sampler 不被触发。

## 与其它模块/系统配合

- [llm-base-proposer.md](llm-base-proposer.md)：drafter 层的 K=0 早返回。
- [../rejection-sampler.md](../rejection-sampler.md)：K=0 时不调 rejection sampler；K>0 时按常规。
- [eagle.md](eagle.md) / [draft-model.md](draft-model.md) / [ngram.md](ngram.md)：所有 drafter 都支持 dynamic K；ngram 走 `assert num_speculative_tokens <= self.k`（`ngram_proposer.py:145`）。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：scheduler 是 dense_schedule 查表的发起者。
- [配置体系-SpeculativeConfig](../../10-config/README.md)（待补充）：`num_speculative_tokens_per_batch_size` 字段。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：每步 receive scheduler 的 `num_speculative_tokens` 并传给 drafter。

## 历史版本演进

- **v0.10.0**：Dynamic SD landfall；`num_speculative_tokens_per_batch_size` 字段加入 SpeculativeConfig。
- **v0.10.5**：`validate_and_normalize_dynamic_sd_schedule` 完善；first range must start at 1 强制。
- **v0.11.0**：与 padded drafter batch + dynamic SD 协调——当 K 动态变化时 padded buffer 大小仍按 `vllm_num_speculative_tokens` 分配，K<max 时部分 slot padding。
- **v0.12 / main**：dense lookup 在 ModelRunner 初始化时一次性建立，避免每步重新解析配置。

[← 返回投机解码](../README.md)

## 参见

- [llm-base-proposer.md](llm-base-proposer.md)
- [metrics.md](metrics.md)：dynamic K 下 metric 的可观察性
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)
