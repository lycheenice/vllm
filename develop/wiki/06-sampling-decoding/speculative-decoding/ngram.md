[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > NgramProposer

# NgramProposer（CPU N-gram drafter）

> 源码：`vllm/v1/spec_decode/ngram_proposer.py`

---

## 是什么

`NgramProposer` 是不依赖任何 drafter 模型的 prompt-lookup 风格 spec decoder。它在 CPU 上用 Numba JIT 加速的 LPS（longest prefix suffix）算法在 prompt + 已生成 token 序列中找出与当前末尾 suffix 匹配的最长 n-gram（长度在 `[min_n, max_n]` 内），把匹配处之后的 K 个 token 当作 draft 提议。

类签名（`vllm/v1/spec_decode/ngram_proposer.py:12`）：

```python
class NgramProposer:
    def __init__(self, vllm_config: VllmConfig): ...
    def propose(self, num_speculative_tokens, sampled_token_ids,
                num_tokens_no_spec, token_ids_cpu, slot_mappings=None) -> list[list[int]]:
        ...
```

`propose` 的输入是 CPU numpy 张量，输出是 `list[list[int]]`——这与 EAGLE/MTP 等 GPU drafter 的 `[B, K]` torch.Tensor 输出不同；前者在 scheduler/model_runner 端做进一步拼装（转入 GPU、构造 `SpecDecodeMetadata`）。

## 为什么

- **零额外模型成本**：不需要任何 drafter 权重，适合"模型无法选 drafter 时也想要 spec decode 加速"的场景（如自研模型、闭源 API 模型）。
- **prompt_lookup 应用场景**：代码补全、RAG、文档问答等输出大量重复 prompt 内容的场景。Suffix match 命中率高时接受率极高。
- **LPS 算法 O(N)**：用类似 KMP 的 LPS 数组 `[0..max_n-1]` 在反转序列上找最长前缀=后缀；O(N) 时间复杂度避免 brute force 的 O(N²)。
- **batch 加速**：`batch_propose_numba` 用 `@njit(parallel=True)` + `prange` 并行处理 batch 内多请求；总 token 数 ≥ 8192 时启用多线程（`num_tokens_threshold`，行 36），否则单线程减少开销。
- **TP-aware 线程数**：`num_numba_thread_available = min(1, cpu_count // 2) // tp_size`——TP > 1 时 shrink 防止抢占；当前限制为 1（注释说明 TP 并行化尚未实现，待提升 8）。
- **JIT precompile**：构造时主动跑一次 `propose` with 1024 dummy 请求触发 Numba JIT（行 57–62），避免首次请求时编译延迟。

## 怎么做

### propose 主入口（行 135）

```python
def propose(self, num_speculative_tokens, sampled_token_ids, num_tokens_no_spec,
            token_ids_cpu, slot_mappings=None):
    assert num_speculative_tokens <= self.k
    valid_ngram_requests = []
    for i, sampled_ids in enumerate(sampled_token_ids):
        if not sampled_ids: continue           # 跳过 partial prefill
        if num_tokens_no_spec[i] >= self.max_model_len: continue
        valid_ngram_requests.append(i)
    return self.batch_propose(len(sampled_token_ids), valid_ngram_requests,
                              num_tokens_no_spec, token_ids_cpu, num_speculative_tokens)
```

- 跳过还没采出 token 的请求（partial prefill）。
- 跳过已达 max_model_len 的请求。
- 其余走 batch_propose。

### batch_propose（行 64）

```python
def batch_propose(self, num_requests, valid_ngram_requests, num_tokens_no_spec,
                  token_ids_cpu, k):
    # 自适应线程数
    if num_ngram_requests := len(valid_ngram_requests):
        original = get_num_threads()
        total_tokens = np.sum(num_tokens_no_spec)
        if total_tokens >= self.num_tokens_threshold:
            set_num_threads(max(1, min(self.num_numba_thread_available, num_ngram_requests)))
        else:
            set_num_threads(1)
        batch_propose_numba(...)  # @njit(parallel=True) kernel
        set_num_threads(original)
    # 拼装输出：valid_ngram_requests 之外为空 list
    for i in range(num_requests):
        if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
            draft_token_ids.append(self.valid_ngram_draft[i, :n].tolist())
        else:
            draft_token_ids.append([])
    return draft_token_ids
```

`valid_ngram_draft` 与 `valid_ngram_num_drafts` 是预分配的 `(max_num_seqs, K)` numpy buffer，避免每次分配。

### _find_longest_matched_ngram_and_propose_tokens（行 206）

核心算法（`@jit(nopython=True)`）：

1. 反转 token 序列：`tokens = origin_tokens[::-1]`。
2. 维护 LPS 数组 `lps[0..max_n-1]`：`lps[i]` = 反转后位置 i 的最长"前缀=后缀"长度。
3. 标准 KMP-style 双指针扫描（行 249–281）：
   - match 时 `prev_lps += 1`，若 `prev_lps >= longest_ngram` 更新 `longest_ngram` 与 `position`。
   - mismatch 时 `prev_lps = lps[prev_lps-1]` fallback 到次长前缀。
   - 当 `prev_lps == max_n` 时 cap，避免 ngram 超过 `max_ngram`。
4. 若 `longest_ngram < min_ngram`：返回空数组（无有效匹配）。
5. 否则取 `origin_tokens[start_position : start_position + k]` 作为 draft。

### 配置项

`SpeculativeConfig` 中相关字段：

- `prompt_lookup_min`：min_ngram（默认 1，待核实）。
- `prompt_lookup_max`：max_ngram（默认 4 或类似，待核实当前默认值）。
- `num_speculative_tokens`：k。
- `max_model_len`：防止 OOB。

### 与 ModelRunner 协作

`GPUModelRunner` 在 `speculative_config.method == "ngram"` 时实例化 `NgramProposer`（`gpu_model_runner.py:587–590`）。每步：

1. ModelRunner 收集 `sampled_token_ids`（每请求的平均采样 token）、`num_tokens_no_spec`、`token_ids_cpu`（numpy `(B, max_model_len)` 视图）。
2. 调 `drafter.propose(...)` 得 `list[list[int]]`。
3. 通过 `SpecDecodeMetadata.make_dummy(draft_token_ids, device)` 拼成 metadata。
4. RejectionSampler 走 `NO_DRAFT_PROBS=True` 路径（因为 ngram 无 draft 概率）。

## 与其它模块/系统配合

- [ngram-gpu.md](ngram-gpu.md)：GPU 加速版，对大 batch / 长 prompt 更快。
- [../rejection-sampler.md](../rejection-sampler.md)：ngram 无 draft_probs，rejection kernel 走 `NO_DRAFT_PROBS`。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：scheduler 端不感知 drafter 类型；`SpecDecodeMetadata` 统一接口。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：CPU 调 ngram → GPU 跑 target verify；CPU/GPU 间存在 token_ids 拷贝（`token_ids_cpu` 由 GPU tensor `.cpu()` 而来）。
- [suffix.md](suffix.md)：suffix decoding 是 ngram 的"全局 + per-request 后缀树"扩展。

## 历史版本演进

- **v0.5–v0.6（V0）**：V0 已有 prompt_lookup_nominal_ngrams 在 `vllm/spec_decode/`，纯 Python，无批量加速。
- **v0.7.0**：V1 NgramProposer landfall；Numba JIT 引入；`batch_propose_numba` + `prange` 并行。
- **v0.7.5**：JIT precompile 在构造时触发，避免首请求延迟。
- **v0.8.0**：`num_tokens_threshold = 8192` 引入——总 token 低于阈值时单线程，避免 numba thread spawn 开销。
- **v0.9.0**：TP-aware 线程数；与 NgramGPUProposer 并存。
- **v0.10.5+**：稳定维护；与 thinking_budget、spec_token_ids 拼接等新机制兼容（待核实：ngram + thinking_budget 是否完全支持）。

[← 返回投机解码](../README.md)

## 参见

- [ngram-gpu.md](ngram-gpu.md)
- [suffix.md](suffix.md)
- [../rejection-sampler.md](../rejection-sampler.md)
