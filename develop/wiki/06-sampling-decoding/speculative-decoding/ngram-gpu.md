[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > NgramProposerGPU

# NgramProposerGPU（GPU N-gram drafter）

> 源码：`vllm/v1/spec_decode/ngram_proposer_gpu.py`

---

## 是什么

`NgramProposerGPU` 是 `NgramProposer` 的 GPU 加速版本——把 n-gram 匹配从 CPU Numba 迁移到纯 PyTorch tensor 操作，全程 GPU 上完成 unfold + 比较 + argmax + gather，避免每步 CPU→GPU token_ids 拷贝与 CPU 端 kernel 启动。它由 `@support_torch_compile()` 装饰，可走 `torch.compile` 加速。

类签名（`vllm/v1/spec_decode/ngram_proposer_gpu.py:28`）：

```python
@support_torch_compile()
class NgramGPUKernel(nn.Module):
    def __init__(self, vllm_config, prefix="", device="cuda"): ...
    def forward(self, ...): ...
```

整个 proposer 类（包含 `NgramGPUKernel` 实例 + 外层 `propose` 包装）在 `gpu_model_runner.py:597` 实例化（`speculative_config.use_ngram_gpu()` 触发）。

## 为什么

- **避免 D2H 同步**：CPU ngram 每步需要 `token_ids_gpu.cpu()` 把全 batch 的 token id 拷贝到 host；GPU 版直接在 device 上算，省 H2D/D2H 一次大拷贝。
- **大 batch / 长 prompt 性能**：unfold() 是 O(1) view，但匹配矩阵是 `(B, num_windows)`，对 B × windows 大小 GPU 友好；CPU Numba 在大场景下会成为瓶颈。
- **`torch.compile` 可选**：`@support_torch_compile()` 让 vLLM 的 compile dispatcher 能在合适场景编译 `NgramGPUKernel.forward`；相较 Numba JIT 更深度集成进 vLLM 编译体系。
- **必须 GPU 持有 token_ids**：为此 GPUModelRunner 在实例化时创建 `num_tokens_no_spec_gpu` 与 `token_ids_gpu_tensor (max_num_reqs, max_model_len)`，每步 spec decode 时这两个张量被同步推进（行 599–613 in `gpu_model_runner.py`）。
- **支持 spec_token_ids 历史**：GPU 路径与 spec_token_ids 推进逻辑配合——拒绝的 spec token 会被下一 step 重新计算，因此 token_ids_gpu_tensor 的更新需要精确。

## 怎么做

### 核心算法 _find_first_and_extract_all_n_parallel（行 47）

```python
def _find_first_and_extract_all_n_parallel(
    self, token_ids, seq_lengths, min_ngram_len, max_ngram_len, num_draft_tokens,
) -> torch.Tensor:
    ...
```

对每个 n-gram 长度 `n` 从 `min_n` 到 `max_n` 并行算：

1. **suffix** 提取：每序列末 `n` 个 token——`suffix_indices[i, :] = token_ids[i, seq_len[i]-n : seq_len[i]]`。
2. **windows**：`token_ids.unfold(1, n, 1)` 给所有长度为 n 的窗口（O(1) view）。
3. **matches**：`(windows == suffix.unsqueeze(1)).all(dim=-1)` 得到 `(B, num_windows)` match matrix。
4. **valid_mask**：window 起始位置必须满足"匹配后还能再跟至少一个 draft token"——`window_pos <= seq_len - n - 1`。
5. **first_match_idx = argmax(final_matches.int())`**（注：argmax 在全 False 时返回 0，需 `has_match` 验证）。
6. 存入 `first_match_positions[:, i]`。

之后跨 n-gram 长度选最长匹配（行 113）：

```python
best_ngram_idx = (first_match_positions >= 0).int().flip(dims=[1]).argmax(dim=1)
best_ngram_idx = num_ngram_sizes - 1 - best_ngram_idx  # 翻转回
```

`has_any_match = best_match_pos >= 0`；`draft_start = best_match_pos + best_ngram_len`；`draft_indices = draft_start.unsqueeze(1) + arange(K)` clamp ∈ [0, max_seq_len-1]；`draft_tokens = torch.gather(token_ids, 1, draft_indices)`。最后 `valid_positions = arange(K) < tokens_available` 对越界位置填 -1。

### 数据流

`GPUModelRunner` 内部：

1. 每步 forward 后更新 `token_ids_gpu_tensor[i, num_tokens_no_spec[i]] = sampled_token_id`（仅有 single sampled token 时直接写）。
2. spec decode 路径：`num_tokens_no_spec_gpu` 与 `token_ids_gpu_tensor` 都在 GPU 上维护；与 padded drafter batch 模式配合，padding 部分填零。
3. `drafter.propose(...)` 接收 CUDA tensor，直接跑 `_find_first_and_extract_all_n_parallel`，输出 `[B, K]` torch.Tensor（CPU 版是 `list[list[int]]`）。
4. 拒绝采样后，CPU 端用 `_ngram_pinned_idx_buf` / `_ngram_pinned_val_buf` 处理 index/val 拷贝（行 608–613 in `gpu_model_runner.py`）。

### @support_torch_compile 装饰

`vllm/compilation/decorators.py:support_torch_compile` 让该类的 `forward` 可被 vLLM 的 compile dispatcher 选择性编译。当 `compilation_config` 启用时该 kernel 被编译进 inductor；不启用时仍跑 eager。

## 与其它模块/系统配合

- [ngram.md](ngram.md)：CPU 版，算法等价但执行路径不同。
- [../rejection-sampler.md](../rejection-sampler.md)：GPU ngram 仍无 draft_probs，rejection kernel 走 `NO_DRAFT_PROBS`。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：实例化时分配 `num_tokens_no_spec_gpu` / `token_ids_gpu_tensor` / `_ngram_pinned_*_buf`；GPU 端状态维护与 CPU 版本完全不同。
- [编译与 IR](../../09-compilation-ir/README.md)：`@support_torch_compile` 装饰器；`CompilationMode` 选择是否编译。
- [平台子系统-CUDA](../../08-platforms/README.md)：仅 CUDA 路径支持；CPU/TPU/XPU 不支持。

## 历史版本演进

- **v0.9.0**：NgramProposerGPU 引入（`use_ngram_gpu()` 开启），用 unfold+argmax+gather 全 GPU 流水。
- **v0.9.5**：`@support_torch_compile` 装饰加入，让 vLLM 编译 dispatcher 统一管理 kernel 编译。
- **v0.10.0**：与 padded drafter batch 兼容——`is_masked_token_mask` 路径支持。
- **v0.10.5**：max_seq_len clamp 与越界 -1 填充完善，避免 OOB。
- **v0.11.0**：`_ngram_pinned_idx_buf` / `_ngram_pinned_val_buf` 引入减少 Pinned memory allocation 次数。
- **v0.12 / main**：优化 max_ngram_len > 8 的 unfolded tensor 内存占用（待核实：具体优化 commit）。

[← 返回投机解码](../README.md)

## 参见

- [ngram.md](ngram.md)
- [suffix.md](suffix.md)
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)
