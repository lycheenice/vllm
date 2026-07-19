[← Wiki 首页](../README.md) > [采样与解码](../README.md) > Sampling Ops

# Sampling Ops（采样算子集）

> 源码：`vllm/v1/sample/ops/`

---

## 是什么

`ops/` 子目录把 sampler 与 rejection sampler 共用的低层算子拆成独立模块，目前包含四个文件：

| 文件 | 暴露函数/类 | 用途 |
|---|---|---|
| `topk_topp_sampler.py` | `TopKTopPSampler`（`nn.Module`） + `apply_top_k_top_p` / `random_sample` / `flashinfer_sample` / `aiter_sample` | top-k / top-p 联合过滤 + 指数噪声采样；多后端 dispatch（CUDA/FlashInfer/ROCm/aiter/CPU/XPU） |
| `topk_topp_triton.py` | `apply_top_k_top_p_triton` | 基于 Qrita 算法的合并 Triton kernel，大 batch 优路径 |
| `penalties.py` | `apply_all_penalties` | repetition / frequency / presence penalties 的批处理包装 |
| `bad_words.py` | `apply_bad_words` / `apply_bad_words_with_drafts` | 多 token 禁词前缀屏蔽（含 spec decode 版） |
| `logprobs.py` | `batched_count_greater_than` | 编译过的"统计每行 ≥ 阈值的元素个数"，用于算 sampled token 的 rank |

## 为什么

- **解耦**：sampler 与 rejection sampler 都需要 top-k/top-p、penalties、bad_words，把它们抽出来避免循环依赖；`rejection_sampler.py` 直接 import `apply_top_k_top_p` 这个函数级 API（`vllm/v1/sample/rejection_sampler.py:20`）。
- **平台特化**：top-k/top-p 的实现差异巨大——FlashInfer 用 rejection sampling 避免全 vocab 排序（`flashinfer_sample` 注释 `topk_topp_sampler.py:477`），aiter 是 ROCm 专用 C++ 实现，XPU 走自定义 `torch.ops.vllm.xpu_topk_topp_sampler`，CPU 走 Triton 或 sort。`TopKTopPSampler.__init__` 在构造时根据平台 + logprobs_mode 一次性 dispatch `forward` 到 `forward_cuda`/`forward_native`/`forward_cpu`/`forward_hip`/`forward_xpu`。
- **GPU-friendly**：`random_sample` 用 `q = torch.empty_like(probs); q.exponential_(); probs.div_(q).argmax(-1)` 实现 Gumbel-max，等价于 multinomial 但避免 CPU-GPU 同步（注释见 `topk_topp_sampler.py:451`）。
- **bitmask 一致性**：top-k/top-p 的输出语义需要与 FlashInfer/aiter 等后端统计等价但不要求 bit-wise 一致（FlashInfer 内部用不同 RNG）。docstring 明确"outputs do not necessarily match ... statistically equivalent"（行 483）。

## 怎么做

### TopKTopPSampler（`topk_topp_sampler.py:70`）

构造时根据：

```text
platform == CUDA + logprobs_mode not in (processed_*) + FlashInfer 可用 + VLLM_USE_FLASHINFER_SAMPLER!=0
  → forward = forward_cuda
platform == CUDA 其他情况
  → forward = forward_native
platform == CPU + arch in (RISCV, POWERPC)
  → forward = forward_native
platform == CPU 其他
  → forward = forward_cpu
platform == XPU + VLLM_XPU_USE_SAMPLER_KERNEL
  → forward = forward_xpu
ROCm + aiter 启用
  → forward = forward_hip
其他
  → forward = forward_native
```

`flashinfer_sampler_supported()`（行 21）负责检查环境、计算能力、env var，必要时给出 `RuntimeError`（用户显式 opt-in 但不支持）或 `warning_once`（自动降级）。

### forward_native / forward_cpu / forward_cuda / forward_hip / forward_xpu

- **forward_native**（行 123）：`apply_top_k_top_p(logits, k, p)` → 可能返回处理后的 logits/log_softmax → `softmax` → `random_sample(probs, generators, use_fp64_gumbel)`。
- **forward_audio_cuda**（行 147，`forward_cuda`）：当 `k is None and p is None` 或有 per-request generators 时回退到 native；否则直接 `flashinfer_sample(logits.contiguous(), k, p, generators)`。logits 必须连续，flex_attn/triton_attn fp32 路径产出的 logits 可能不连续，故显式 `.contiguous()`。
- **forward_cpu**（行 176）：当 batch size 与 generators 数量匹配且 `use_fp64_gumbel=False` 时走 `compiled_random_sample`（`torch.compile(dynamic=True)`）；否则 fall-through 到 native 风格但用 `empty_exponential_noise_like` + 手动 per-generator exponential。
- **forward_hip**（行 222）：ROCm aiter 路径。`aiter_sample`（行 247）分 top-k-only / top-p-only / joint 三条，分别调 `aiter.ops.top_p_sampling_from_probs`、`top_k_sampling_from_probs`、`top_k_top_p_sampling_from_probs`，且 `deterministic=True`。
- **forward_xpu**（行 285）：调 `torch.ops.vllm.xpu_topk_topp_sampler` 自定义 kernel；手动同步 default generator 的 offset（行 327–331）以保证 RNG 可重现。

### apply_top_k_top_p（行 345）

```python
def apply_top_k_top_p(logits, k, p) -> torch.Tensor:
    if p is None and k is None: return logits
    if current_platform.is_cpu():
        if HAS_TRITON: return apply_top_k_top_p_triton(logits, k, p)
        return apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)
    return apply_top_k_top_p_pytorch(logits, k, p)
```

策略：CPU 优先 Triton；GPU 在 batch≥8 时用 Triton，小 batch 用 PyTorch（sort 实现更省 kernel 启动开销）。

### apply_top_k_top_p_pytorch（行 363）

经典实现：`logits.sort(descending=False).values` → 按 k 取阈值 `masked_fill_(-inf)` → softmax → cumsum → 按 p 阈值 mask → `scatter_` 还原顺序。当只有 k 无 p 时调 `apply_top_k_only`（行 407）避免全 vocab 排序——它只取 `topk(max_k)` 然后按 k 阈值 mask，但需要 GPU→CPU 同步拿 `max_top_k`。

### apply_top_k_top_p_triton（`topk_topp_triton.py`）

基于 Park 等人的 Qrita 算法（论文 `https://arxiv.org/abs/2602.01518`，待核实：链接可能打不开，arXiv 编号格式异常），用 pivot-based truncation + selection 把 top-k 与 top-p 融合到一个 kernel。预计算两张 LUT（`_NORMAL_CDF_TO_SIGMA_TABLE`、`_PERCENTILE_TO_STD_TABLE`，行 22–66）用于把 top-p 阈值映射到 logit 空间近似阈值，再做 selective scan。`_TRITON_TABLE_CACHE` / `_TRITON_BUFFER_CACHE` 缓存 device-bound 张量。

### random_sample（行 446）

```python
def random_sample(probs, generators, use_fp64_gumbel=False):
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return sample_with_exponential_noise(probs, q)
```

`sample_with_exponential_noise`：`scores = probs.div_(q).argmax(-1)`（fp32）或 `q.reciprocal_().mul_(probs).argmax(-1)`（fp64 Gumbel）。

### apply_all_penalties（`penalties.py:10`）

包装 `vllm.model_executor.layers.utils.apply_penalties`（C++ extension），负责把请求级 list-of-list 转 padded tensor：`make_tensor_with_pad(output_token_ids, pad=vocab_size, ...)`。`vocab_size` 本身被当作 pad 值，避免与有效 token id 冲突；再用 `masked_fill_(output_tokens_t == -1, vocab_size)` 处理异步调度中未跑 penalty 的 placeholder 行（行 29）。

### apply_bad_words / apply_bad_words_with_drafts（`bad_words.py`）

多 token 禁词：给定 `bad_words_token_ids: list[list[int]]`，检查 `past_tokens[-len(prefix):] == bad_word_ids[:-1]` 即在最后一 token 位 mask `-inf`。`apply_bad_words_with_drafts` 是 spec decode 版——按 `num_draft_tokens` 切片，对每个 draft 位置独立调用 `_apply_bad_words_single_batch`（含 spec 历史前缀匹配）。

### batched_count_greater_than（`logprobs.py`）

```python
@torch.compile(backend=current_platform.simple_compile_backend)
def batched_count_greater_than(x, values):
    torch._check(x.shape[0] >= 1)
    torch._check(x.shape[0] == values.shape[0])
    return (x >= values).sum(-1)
```

`sampler.gather_logprobs` 用它算 sampled token 的 rank（行 347）；显式 shape check 防止 Inductor 在 batch=0 时崩。

## 与其它模块/系统配合

- [sampler.md](sampler.md)：`TopKTopPSampler` 是 sampler 的成员；`apply_top_k_top_p` 与 `apply_all_penalties`、`apply_bad_words` 直接被 sampler 调用。
- [rejection-sampler.md](rejection-sampler.md)：`apply_sampling_constraints` 调 `apply_top_k_top_p`、`apply_all_penalties`、`apply_bad_words_with_drafts`、`MinTokens.apply_with_spec_decode` 都来自 ops/ 包。
- [结构化输出](structured-output/README.md)：`apply_grammar_bitmask` 在 sampler 调用 top-k/top-p 之前 apply；clean 后的 logits 经 top-k/top-p 仍是合法 token。
- [平台子系统](../08-platforms/README.md)：`current_platform.is_cuda/is_cpu/is_xpu/is_rocm` 是 op 选择的关键开关；`get_device_capability` 决定 FlashInfer 是否可用。
- [编译与 IR](../09-compilation-ir/README.md)：`compiled_random_sample` 用 `torch.compile(dynamic=True)`；`batched_count_greater_than` 走 `simple_compile_backend`；Triton kernel 通过 `triton_utils` 包装以支持禁用 Triton 的环境（如 TPU）。
- aiter/FlashInfer 是外部依赖：`requirements/cuda.txt` 默认安装 FlashInfer；ROCm aiter 由 `vllm._aiter_ops.rocm_aiter_ops.is_enabled()` 判定。

## 历史版本演进

- **v0.5–v0.6（V0）**：top-k/top-p 实现在 `vllm/model_executor/layers/sampler.py`，纯 PyTorch，对大 vocab 慢；penalties 在另一个文件 `vllm/model_executor/layers/rejection_sampler.py`（V0 路径）。
- **v0.7.0**：V1 `TopKTopPSampler` 引入，融合 FlashInfer 后端；`apply_top_k_top_p_pytorch` 抽出独立函数；`random_sample` 改用 Gumbel-max。
- **v0.7.5**：`apply_top_k_only` 优化路径加入，当 batch 全是 top-k-only 时避免全 vocab sort。
- **v0.8.0**：`TopKTopPSampler` 拆 `nn.Module` 出 `ops/`；引入 `apply_bad_words_with_drafts` 服务 spec decode。
- **v0.8.5**：CPU 路径加 Triton kernel `apply_top_k_top_p_triton`（HAS_TRITON 判定）。
- **v0.9.0**：`use_fp64_gumbel` 配置项落地，给 Xeon/iGPU 等数值敏感场景提供 float64 Gumbel；XPU 自定义 kernel 路径加入（`VLLM_XPU_USE_SAMPLER_KERNEL`）。
- **v0.10.0**：ROCm aiter sampler 路径加入（`forward_hip`），与 FlashInfer 路径对称；`flashinfer_sampler_supported` 函数化，支持降级 warning。
- **v0.11.0**：PowerPC/RISCV CPU 路径显式不走 `forward_cpu`（`torch.compile` argmax 在 PowerPC 上有 bug，PR #26987）。
- **v0.12 / main**：Qrita 合并 Triton kernel 落地（`topk_topp_triton.py` 964 行），大 batch + 同时 top-k/top-p 场景显著加速；`_TRITON_TABLE_CACHE` / `_TRITON_BUFFER_CACHE` 引入避免重复分配。

[← 返回采样与解码](../README.md)

## 参见

- [sampler.md](sampler.md)
- [rejection-sampler.md](rejection-sampler.md)
- [logits-processor.md](logits-processor.md)：`MinTokens.apply_with_spec_decode` 在 ops/ 之外但语义上是 spec decode ops 的延伸
- [平台子系统](../08-platforms/README.md)
