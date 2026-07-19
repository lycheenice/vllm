[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > MedusaProposer

# MedusaProposer（Medusa 多头）

> 源码：`vllm/v1/spec_decode/medusa.py`

---

## 是什么

`MedusaProposer` 实现 [Medusa 算法](https://arxiv.org/abs/2401.10774)（待核实链接）：在 target 模型末层 hidden states 后接 K 个并行的"Medusa head"（每个 1 层 transformer + lm_head），一次性预测 K 个未来 token。它与 EAGLE 的关键差异是 **不需要 multi-pass drafter forward**——一次 forward 算完所有 K 个 head，但每个 head 都基于同一 hidden state 独立预测，无法使用上一步 head 输出作为下一步输入（无 autoregressive draft chain）。

类签名（`vllm/v1/spec_decode/medusa.py:18`）：

```python
class MedusaProposer:
    def __init__(self, vllm_config, device): ...
    def propose(self, num_speculative_tokens, target_hidden_states,
                sampling_metadata, slot_mappings=None) -> torch.Tensor:
        blocks = self.model(target_hidden_states)
        logits = self.model.compute_logits(blocks)
        draft_tokens = torch.stack([logit.argmax(-1) for logit in logits], dim=1)
        return draft_tokens
```

注意它**不继承 `SpecDecodeBaseProposer`**，因为 Medusa 没有自身的 forward 链；它直接消耗 target 末层 hidden states 而不需要 positions / attention metadata。

## 为什么

- **架构简单**：Medusa head 是 target 模型的旁支，参数极少（每 head ~1 层），训练成本低；适合作为快速 spec decode 加速器。
- **并行度高**：K 个 head 完全并行，GPU 上单次 forward 即出 K 个 draft；相比 EAGLE 的 multi-pass 在 K 较大时延迟更低。
- **接受率较低**：因为每个 head 独立预测无 chain，第 K 个 head 的准确率随 K 衰减明显；通常 K=4~5。
- **拒绝采样兼容**：draft_probs 可由 head logits 算 softmax 得到，rejection sampler 仍能使用；当前 vLLM 实现默认走 argmax，draft_probs=None（与 EAGLE 同）。
- **不支持 EPLB + MoE**：`is_mixture_of_experts(self.model) and enable_eplb` 时显式 assert 拒绝（`medusa.py:69`）。

## 怎么做

### load_model（行 60）

```python
def load_model(self, target_model: nn.Module) -> None:
    with set_model_tag("medusa_head"):
        self.model = get_model(
            vllm_config=self.vllm_config,
            model_config=self.spec_config.draft_model_config,
        )
```

`set_model_tag("medusa_head")` 让编译后端识别这是 drafter 模型，避免与 target 的 compile cache 冲突。`is_mixture_of_experts + enable_eplb` 在此处被 assert。

### propose（行 40）

```python
def propose(self, num_speculative_tokens, target_hidden_states,
            sampling_metadata, slot_mappings=None) -> torch.Tensor:
    blocks = self.model(target_hidden_states)
    logits = self.model.compute_logits(blocks)
    draft_tokens = torch.stack([logit.argmax(-1) for logit in logits], dim=1)
    return draft_tokens
```

- `self.model(target_hidden_states)` 返回 K 个"blocks"（每个 head 的 transformer 输出）。
- `compute_logits(blocks)` 对每 head 算 lm_head → `[K, B, V]`。
- argmax 后 stack 成 `[B, K]`。

`sampling_metadata` 与 `slot_mappings` 参数仅为接口兼容（unused），与 EAGLE 风格 drafter 保持 propose 签名一致。

### dummy_run（行 73）

`@torch.inference_mode()` 装饰，构造零张量 hidden_states 跑一次 model forward，用于：

- 内存 profiling（决定 KV cache 大小）。
- CUDA graph 捕获前的 warmup。
- torch.compile 触发。

### 与 EAGLE 路径的差异

| 维度 | EAGLE | Medusa |
|---|---|---|
| forward 次数 | K 次 multi-pass | 1 次 |
| head 关系 | autoregressive chain | K 个独立 head |
| 需要 positions/attn_metadata | 是 | 否 |
| 继承 SpecDecodeBaseProposer | 是 | 否 |
| draft_probs 可用 | （未启用） | （未启用） |
| draft + multimodal | 警告但允许 | 待核实 |
| EPLB + MoE | 支持 | assert 拒绝 |

## 与其它模块/系统配合

- [../rejection-sampler.md](../rejection-sampler.md)：`draft_probs=None`，rejection kernel 走 NO_DRAFT_PROBS 分支。
- [eagle.md](eagle.md)：vLLM 推荐 EAGLE3 替代 Medusa（接受率更高），但 Medusa 仍保留用于老 checkpoint。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：`self.drafter = MedusaProposer(...)` 在 `gpu_model_runner.py:630`。drafter forward 与 target forward 共用 hidden_states（target computed → medusa head forward 一步完成）。
- [模型库-Medusa heads](../../04-model-zoo/README.md)（待补充：Medusa 模型实现位置）。
- [编译与 IR](../../09-compilation-ir/README.md)：`set_model_tag("medusa_head")` 让 compile backend 选择合适的 cpp 后端。

## 历史版本演进

- **v0.5–v0.6（V0）**：Medusa 已有 V0 实现，在 `vllm/spec_decode/medusa.py`；走 `MultiStepWorker`。
- **v0.7.0**：V1 MedusaProposer landfall，签名极简：`propose(target_hidden_states, ...)`。无 multi-pass 需求。
- **v0.8.0**：`is_mixture_of_experts + enable_eplb` 的 assert 加入。
- **v0.8.5–v0.10**：基本无变化——EAGLE3 逐渐成为主流，Medusa 维护性更新。
- **v0.11.0+**：仍维护，但社区新模型多用 EAGLE3 / MTP（待核实：vLLM 是否仍推荐 Medusa 用于新部署）。

[← 返回投机解码](../README.md)

## 参见

- [eagle.md](eagle.md)：更现代的替代
- [mtp.md](mtp.md)：另一种单 forward drafter
- [../rejection-sampler.md](../rejection-sampler.md)
