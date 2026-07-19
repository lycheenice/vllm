[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > SuffixDecodingProposer

# SuffixDecodingProposer（后缀树解码）

> 源码：`vllm/v1/spec_decode/suffix_decoding.py`
> 论文：[Suffix Decoding](https://arxiv.org/pdf/2411.04975)（Snowflake Arctic Inference）

---

## 是什么

`SuffixDecodingProposer` 是基于后缀树的 spec decode drafter——它在 prompt 与已生成 response 上构建后缀树（suffix tree），每步用最近 `max_tree_depth` 个 token 作 pattern 查询，按概率采 K 个 draft token。本质是 n-gram 的推广：n-gram 只用单个 n 长度，suffix decoding 用全局共享后缀树覆盖任意长度匹配。

类签名（`vllm/v1/spec_decode/suffix_decoding.py:9`）：

```python
class SuffixDecodingProposer:
    def __init__(self, vllm_config: VllmConfig): ...
    def propose(self, num_speculative_tokens, input_batch, sampled_token_ids,
                slot_mappings=None) -> list[list[int]]: ...
```

实际算法委托给第三方库 `arctic_inference.suffix_decoding.SuffixDecodingCache`，vLLM 这层只做：

1. Lazy import `arctic_inference`（若未安装则该 method 不可用）。
2. 维护 per-request 生命周期：start_request / add_active_response / stop_request。
3. 把 `input_batch.token_ids_cpu` 与 `num_tokens_no_spec` 转交 arctic 接口。
4. 收集 `draft.token_ids` 返回 list[list[int]]。

## 为什么

- **动态长度 draft**：suffix decoding 每步每请求可提议不同数量的 draft token（小于等于 K），比固定 K 的 n-gram 灵活。当 suffix tree 命中短分支时少提议、长分支时多提议，提高接受率与吞吐。
- **跨请求共享 cache**：`SuffixDecodingCache` 在多请求间共享后缀树（max_cached_requests 控制），适合 batch 内有相似 prompt 的场景（如 RAG 同源文档）。
- **复用代码产出 token**：output 中重复的 code 段、JSON 模板、markdown 表格等会形成强 suffix tree 分支；suffix decode 能直接"复用"既有输出，加速极显著。
- **概率阈值控制**：`min_token_prob` 让低概率分支不进入 draft——避免无价值的 low-confidence draft 浪费 target forward。
- **max_spec_factor**：限制 raw draft 长度不超过 K * factor，防止意外长分支撑爆 mask 张量。

## 怎么做

### __init__（行 16）

```python
def __init__(self, vllm_config):
    config = vllm_config.speculative_config
    self.num_speculative_tokens = config.num_speculative_tokens
    self.max_tree_depth = config.suffix_decoding_max_tree_depth
    self.max_spec_factor = config.suffix_decoding_max_spec_factor
    self.min_token_prob = config.suffix_decoding_min_token_prob
    self.max_model_len = vllm_config.model_config.max_model_len
    # Lazy import
    from arctic_inference.suffix_decoding import SuffixDecodingCache
    self.suffix_cache = SuffixDecodingCache(
        max_tree_depth=config.suffix_decoding_max_tree_depth,
        max_cached_requests=config.suffix_decoding_max_cached_requests,
    )
```

### propose 主流程（行 35）

每请求独立处理：

1. 跳过 `sampled_ids` 为空的请求（partial prefill）。
2. 跳过 `num_tokens >= max_model_len` 的请求。
3. 若该请求未在 cache 中：调 `start_request(req_id, prompt_token_ids)` 构建该请求的 suffix tree。
   - 如果该请求之前在 `cached_requests` 中（已结束的请求被回收）：先 `evict_cached_response` 重置。
4. `add_active_response(req_id, sampled_ids)` 把本步采样添加到 cache。
5. 提取 pattern：`pattern = token_ids_cpu[i, start:num_tokens]`，其中 `start = max(0, num_tokens - max_tree_depth)`。
6. 调 `self.suffix_cache.speculate(req_id, pattern, max_spec_tokens, max_spec_factor, min_token_prob)` 得 `draft.token_ids`。
7. batch 结束后：对未在本步出现的 `active_requests` 调 `stop_request`。

### 与 uite batch 的差异

每请求 draft 长度可能不同（甚至为 0），`SpecDecodeMetadata.make_dummy` 接受 `list[list[int]]` 异构长度的 draft_token_ids 并自动计算 `num_draft_tokens` 与 `cu_num_draft_tokens`。这要求 `RejectionSampler` 内部 `max_spec_len = max(num_draft_tokens)`，对短 draft 请求补 placeholder。

### load_model

```python
def load_model(self, *args, **kwargs):
    pass  # No model to load.
```

无 drafter 模型权重。

## 与其它模块/系统配合

- [ngram.md](ngram.md)：概念上的"父算法"；suffix decoding = n-gram + suffix tree + 跨请求共享。
- [../rejection-sampler.md](../rejection-sampler.md)：`NO_DRAFT_PROBS=True` 路径；rejection kernel 按请求实际 draft 长度切分。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：`speculative_config.method == "suffix"` 触发实例化；与其他 drafter 一样由 ModelRunner 在每步驱动。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：K = num_speculative_tokens 仍由 scheduler 决定，但实际 draft 长度可能小于 K；scheduler 端 dynamic SD 机制不直接感知此差异。
- 第三方依赖：`arctic_inference` 包；未安装时使用 suffix method 会 ImportError。

## 历史版本演进

- **v0.9.0**：SuffixDecodingProposer landfall；与 ngram GPU、custom_class 同期引入。
- **v0.10.0**：`suffix_decoding_max_tree_depth` / `suffix_decoding_max_spec_factor` / `suffix_decoding_min_token_prob` / `suffix_decoding_max_cached_requests` 字段加入 `SpeculativeConfig`。
- **v0.10.5+**：稳定维护；与 padded drafter batch 兼容（动态长度走 placeholder 填充路径）。
- **v0.12 / main**：`stop_request` 在批次结束时统一触发，避免 cache 泄漏。

[← 返回投机解码](../README.md)

## 参见

- [ngram.md](ngram.md)
- [ngram-gpu.md](ngram-gpu.md)
- [custom-class.md](custom-class.md)：用户可注册自定义 drafter，suffix decoding 是其中典型用例
