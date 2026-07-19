[← Wiki 首页](../README.md) > [采样与解码](../README.md) > Sampler

# Sampler（采样器）

> 源码：`vllm/v1/sample/sampler.py`

---

## 是什么

`Sampler` 是一个 `nn.Module`，是 vLLM V1 把模型前向输出的 logits 转换成实际"被采出 token"的唯一入口。它在 `GPUModelRunner` 每步前向之后被调用，输入一个 `[num_tokens, vocab_size]` 的 logits 张量与一份 `SamplingMetadata`，输出一个 `SamplerOutput`（含 `sampled_token_ids` 和可选的 `logprobs_tensors`）。

类签名与其文档字符串里固定了 9 步处理流水线（见 `vllm/v1/sample/sampler.py:20`）：

1. 若需要 logprobs，先备份原始 logits 或计算原始 logprobs（用于"raw_logprobs"/"raw_logits"模式）。
2. logits 转 float32。
3. 应用 `allowed_token_ids_mask`（白名单 mask 为 `-inf`）。
4. 应用 `bad_words` 多 token 禁词（前缀匹配）。
5. 应用 **非 argmax 不变量** logits processors：`MinTokensLogitsProcessor`、`LogitBiasLogitsProcessor`（影响 greedy）。
6. 应用 penalties（repetition / frequency / presence）。
7. 在 `sample()` 内：argmax（greedy）或 temperature→argmax-invariant processors（默认 `MinPLogitsProcessor`）→ top-k/top-p → 指数噪声采样。
8. 若需要 logprobs，gather top-k 与被采样 token 的 logprob/rank。
9. 组装 `SamplerOutput`。

## 为什么

- **正确性优先**：vLLM 采样必须与 HF `transformers` 的 `temperature/top_k/top_p/repetition_penalty` 统计等价，但又要避免 `torch.multinomial` 的 CPU-GPU 同步（见 `random_sample` 注释 `vllm/v1/sample/ops/topk_topp_sampler.py:451`）。
- **批异构**：一个 batch 里可能同时存在 greedy（temp=0）与 random（temp>0）请求、有/无 penalties、有/无 logprobs 请求；`SamplingMetadata.all_greedy` / `all_random` 两个极端标志位让短路路径避免无谓 kernel 启动。
- **argmax 不变量分类**：能改 greedy 结果的 processor（min_tokens 把 stop token 屏蔽为 -inf，logit_bias 调整某 token 偏置）必须**在** greedy 决定之前跑；不能改 greedy 结果的 processor（min_p：把概率低于阈值的全 mask）放在 greedy 之后。这条二分由 `LogitsProcessors.argmax_invariant` / `non_argmax_invariant` 两个分组承载（见 `vllm/v1/sample/logits_processor/state.py:148`）。
- **logprobs 模式多样化**：vLLM 配置项 `LogprobsMode` 提供 `raw_logprobs` / `raw_logits` / `processed_logits` / `processed_logprobs` 四种语义，分别对应"用未处理 logits 算 log_softmax" / "返回未处理 logits" / "返回采样前 logits" / "返回采样前 log_softmax"。`Sampler.forward` 的 `logprobs_mode_override` 参数专门为 rejection sampler 的内部调用留出后门（`vllm/v1/sample/sampler.py:77`）。

## 怎么做

### forward 主路径

```python
def forward(
    self,
    logits: torch.Tensor,                # [num_tokens, vocab_size]
    sampling_metadata: SamplingMetadata,
    predict_bonus_token: bool = False,
    logprobs_mode_override: LogprobsMode | None = None,
) -> SamplerOutput:
    ...
```

调用顺序见 `vllm/v1/sample/sampler.py:72`。关键点：

- 行 86–93：先决定 `raw_logprobs` 用哪种语义。`raw_logits` 模式仅 `clone()`（fp32）或 `to(float32)`，不算 log_softmax。
- 行 96：logits **强制** float32；下游所有算子假定 fp32 入参。
- 行 98：`apply_logits_processors` 内部依次跑 allowed-token mask / bad_words / non-argmax-invariant processors / penalties / thinking budget（详见下文）。
- 行 102：`sample()` 内对 `all_greedy` 短路直接 argmax；否则 temperature→argmax-invariant→top-k/top-p→指数噪声采样。
- 行 109：`sampled.long()` 是因为 FlashInfer 采样返回 int32，而后续 `gather_logprobs` 要求 int64 index。
- 行 114–136：`logprob_token_ids` 是 `generative_scoring` API 的特殊路径——只 gather 指定 token ids 的 logprobs，而非 top-k。`num_logprobs == -1` 表示"返回全 vocab"。
- 行 139：最终 `sampled.to(int32)` 以减小张量体积。

### sample 子方法

```python
def sample(self, logits, sampling_metadata, logprobs_mode_override=None):
    ...
```

- `all_greedy` 且需要 logprobs：用 `processed_logits` 或 `processed_logprobs` 模式返回处理后的 logits/log_softmax（行 263–271）。
- 非 `all_random`：先算 `greedy_sampled = logits.argmax(-1)` 当作 fall-back；`all_random` 跳过此步以省一次 argmax。
- 行 276：`apply_temperature` 用 `div_` 原地除，避免新建张量；`all_random=False` 时把 temp < 1e-5 的请求置为 1.0 防止除零（因为那些请求会走 `greedy_sampled`）。
- 行 282：argmax-invariant processors 在 temperature 后跑；当前唯一注册的是 `MinPLogitsProcessor`。
- 行 286：`TopKTopPSampler` 是个 dispatch 模块，会根据平台/编译目标切换到 FlashInfer / aiter / Triton / native 实现，详见 [sampling-ops.md](sampling-ops.md)。
- 行 296：`torch.where(temp < EPS, greedy, random)` 在混合 batch 中复用 `greedy_sampled` 张量作为输出缓冲（`out=greedy_sampled`）。

### gather_logprobs 与 gather_specific_token_logprobs

- `gather_logprobs`（行 308）：标准 top-k logprobs。用 `torch.topk` 求 top-k，再 `gather` 出 sampled token 的 logprob，并用 `batched_count_greater_than`（`ops/logprobs.py`）计算 sampled token 在 vocab 内的 rank。`mark_unbacked` 用来防止 dynamo 在 batch=1 → 2 时重编译。
- `gather_specific_token_logprobs`（行 151）：用于 `generative_scoring` API。把每个请求的 `logprob_token_ids` 列表 pad 到统一长度后 `gather`，无效位 mask 成 `-inf`；首列固定为实际 sampled token，rank 同样由 `batched_count_greater_than` 算出。

### apply_logits_processors

```python
def apply_logits_processors(self, logits, sampling_metadata, predict_bonus_token):
    ...
```

- 行 384：如果 spec decode 开启且当前在跑 `predict_bonus_token`，需要把 `output_token_ids` 与 `spec_token_ids` 拼接作为"用于 penalty 计算的历史"——否则 penalties 会漏掉 spec 部分。
- 行 396：`allowed_token_ids_mask` 直接 `masked_fill_(-inf)`。
- 行 400：`apply_bad_words` 多 token 前缀匹配屏蔽。
- 行 404：非 argmax-invariant processors。
- 行 408：`apply_all_penalties`（repetition/frequency/presence）。
- 行 409–419：若 `ThinkingBudgetStateHolder` 跟踪了请求，先 `update_state`（基于 output/spec 推进 thinking 计数），再 `apply_to_logits`（强制把 end-of-thinking token logits 拉到 1e9）。详见 [thinking-budget.md](thinking-budget.md)。

### 与 spec decode 的协作

`predict_bonus_token=True` 路径只在 `RejectionSampler.forward` 中调用（`vllm/v1/sample/rejection_sampler.py:130`）。此时 sampler 把 bonus token 视作"独立一次采样"，但对 penalties/bad_words 的历史看到的 token 序列会被 spec_token_ids 扩展——这保证 bonus 的语义等价于"在已经接受所有 draft 之后走一步正常采样"。

## 与其它模块/系统配合

- [引擎核心-调度](../01-engine-core/scheduler/README.md)：调度器组装 `SamplingMetadata`，包括 temperature/top_k/top_p/repetition 等批级张量、`generators`（per-request RNG seed）、`no_penalties` 快速路径标志。
- [执行层-GPUModelRunner](../02-execution/worker/README.md)：`GPUModelRunner.execute` 在模型前向后调用 `self.sampler(logits, sampling_metadata)`；spec decode 路径则调 `self.rejection_sampler(...)`。
- [logits-processor.md](logits-processor.md)：`LogitsProcessors` 容器把所有 processor 分到 argmax/non-argmax 两组，sampler 按序调用。
- [sampling-ops.md](sampling-ops.md)：top-k/top-p、penalties、bad_words、logprobs 的实际算子都在 `ops/` 下，sampler 只负责编排。
- [rejection-sampler.md](rejection-sampler.md)：spec decode 的多 token 采样是 sampler 的"复合扩展版"。
- [thinking-budget.md](thinking-budget.md)：在 penalties 之后强制收尾思考段。
- [结构化输出-manager](structured-output/manager.md)：`apply_grammar_bitmask` 在 sampler 调用之前修改 logits；sampler 本身并不感知 grammar。
- LoRA：当 LoRA 启用时 `compute_logits` 已经包含 LoRA 路径；sampler 不需改动（待核实：logit_bias + LoRA 同时使用时的优先级顺序）。

## 历史版本演进

- **v0.5–v0.6（V0）**：采样逻辑在 `vllm/model_executor/layers/sampler.py`，支持 `_apply_penalties` / `_apply_top_k_top_p` 但与 V1 完全不同——V0 用 `torch.multinomial`，对大 vocab 有 CPU 同步问题；V0 还不支持 argmax-invariant 二分。
- **v0.7.0**：V1 `Sampler` 落地。引入 `_SAMPLING_EPS = 1e-5` 区分 greedy 与 random；引入 `LogitsProcessors` 容器；`all_greedy`/`all_random` 短路；用 `empty_exponential_noise_like` + `exponential_` 来生成 Gumbel 噪声近似，避免 multinomial 同步。
- **v0.7.5**：`logprobs_mode` 引入，初始只有 `raw_logprobs` 与 `processed_logits`；为 logprobs 计算分离 `raw_logprobs` 与 `processed_logprobs` 路径。
- **v0.8.0**：`TopKTopPSampler` 拆出独立 `nn.Module`（原内联在 sampler 内）；CUDA 路径接入 FlashInfer（`VLLM_USE_FLASHINFER_SAMPLER=1`）。
- **v0.8.5**：为 spec decode 加入 `predict_bonus_token` 参数，让 sampler 能在 rejection sampling 上下文中被复用。
- **v0.9.0**：`logprob_token_ids` API（generative scoring）落地，`gather_specific_token_logprobs` 引入；`num_logprobs == -1` 表示返回全 vocab。
- **v0.10.0**：四种 logprobs 模式齐备（`raw_logprobs`/`raw_logits`/`processed_logits`/`processed_logprobs`）；rejection sampler 通过 `logprobs_mode_override` 在内部触发 `processed_logits`。
- **v0.10.5**：`ThinkingBudgetStateHolder` 钩入 `apply_logits_processors` 末尾。
- **v0.11.0**：`batched_count_greater_than` 通过 `torch.compile(simple_compile_backend)` + `mark_unbacked` 修复 batch 从 1 到 2 的重编译。
- **v0.12 / main**：`TopKTopPSampler` 增加对 ROCm aiter、XPU kernel 的多后端 dispatch；Qrita Triton kernel 进入 `ops/topk_topp_triton.py` 用于合并 top-k+top-p（pivot-based truncation）。

[← 返回采样与解码](../README.md)

## 参见

- [rejection-sampler.md](rejection-sampler.md)：spec decode 验证器
- [logits-processor.md](logits-processor.md)：argmax/非 argmax 不变量分类
- [sampling-ops.md](sampling-ops.md)：top-k/top-p/penalties/bad_words/logprobs 实算子
- [thinking-budget.md](thinking-budget.md)：思考预算强制收尾
- [引擎核心-调度](../01-engine-core/scheduler/README.md)
- [执行层-GPUModelRunner](../02-execution/worker/README.md)
