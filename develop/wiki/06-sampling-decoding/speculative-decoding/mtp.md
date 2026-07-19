[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > MTP

# MTP（DeepSeek Multi-Token Prediction）

> 源码：基类流程见 `vllm/v1/spec_decode/llm_base_proposer.py`（`SpecDecodeBaseProposer`）；DeepSeek MTP 模型实现见 `vllm/model_executor/models/deepseek_mtp.py`（待核实具体路径）；Step3.5 / Gemma4 / DFlash 等 MTP 变体见对应页。

---

## 是什么

MTP（Multi-Token Prediction）是 DeepSeek 系（DeepSeek-V2/V3/V4、Qwen3 MTP、Gemma4 MTP、Step3.5 MTP 等）共用的"用 target 末层 hidden 状态 + 自回归 drafter 多步预测 K 个 token"的 drafter 风格。在 vLLM V1 中 MTP 是 `SpeculativeConfig.method == "mtp"` 的统称，由 `SpecDecodeBaseProposer` 直接承载（无独立 `MTPProposer` 类）。

`SpecDecodeBaseProposer.model_returns_tuple`（行 1007）区分 MTP 家族：

```python
def model_returns_tuple(self) -> bool:
    if self.method == "mtp":
        # DeepSeek-family MTP (deepseek_mtp.py) recycles the post-final-
        # norm hidden, so its forward returns (logit_hidden,
        # recycle_hidden). Other MTP families return a single tensor.
        return "DeepSeekMTPModel" in (
            self.draft_model_config.hf_config.architectures or []
        )
    return self.method not in ("mtp", "draft_model", "dflash")
```

仅当 drafter 是 `DeepSeekMTPModel` 时返回 tuple——`logit_hidden` 用于算 logits，`recycle_hidden` 作为下一步 hidden_states 输入（recycle 机制）。

## 为什么

- **DeepSeek 原生架构**：DeepSeek 训练时就有 MTP head，可作为 drafter 直接复用其权重——无需训练 EAGLE 风格 draft 网络。这是 MTP 相对 EAGLE 的优势：权重已存在，零额外训练成本。
- **MTP layer 通常单层或浅**：DeepSeek MTP layer 仅 1 层 transformer，forward 极轻；与 EAGLE 类似但少一层。
- ** recycling hidden states**：DeepSeek MTP 末层 hidden 不直接进入 lm_head，而是经过一层 norm 后分两路：一路算 logits、一路作为下一步 hidden 输入。这种"recycle"机制需要 `model_returns_tuple=True` 让基类正确拆分。
- **与 EAGLE3 的差异**：EAGLE3 用 aux hidden states（多个中间层）+ `combine_hidden_states`；MTP 只用末层 hidden 但加 recycle 路径。两者在基类 `propose` 内部按 `method` 字符串分流。
- **MTP 家族成员**：
  - **DeepSeek-V2/V3 MTP**：`DeepSeekMTPModel`，tuple 返回。
  - **DeepSeek V4 MTP**：`hc_mult` 字段扩展 hidden_size（行 100–104 in `llm_base_proposer.py`）。
  - **Qwen3 MTP**：单 tensor 返回，复用基类默认路径。
  - **Gemma4 / Step3.5 / DFlash**：各自有专门子类，见 [gemma4.md](gemma4.md) / [step3p5.md](step3p5.md) / [dflash.md](dflash.md)。

## 怎么做

###drafter 实例化

`gpu_model_runner.py:614–617`（Gemma4 / Step3.5 / DFlash 各自分支）之外：

- 当 `speculative_config.method == "mtp"` 且非 Gemma4/Step3.5/DFlash：drafter 走基类 `SpecDecodeBaseProposer`（即 EAGLEProposer 的父类）。`gpu_model_runner.py:624` 的 `use_eagle()` 分支是 EAGLE/EAGLE3，MTP 必须先走 Gemma4/Step3.5/DFlash 的 method 检测。
- 待核实：DeepSeek V2/V3 MTP 是否在 `use_eagle()` 分支前被识别，还是有独立 `elif method == "mtp"` 分支。

### model_returns_tuple 的 implications

基类 `propose` 中对 tuple 的处理（行 591–595）：

```python
ret_hidden_states = self.model(**model_kwargs)
if not self.model_returns_tuple():
    last_hidden_states = ret_hidden_states
    hidden_states = last_hidden_states
else:
    last_hidden_states, hidden_states = ret_hidden_states
```

之后 `sample_hidden_states = last_hidden_states[token_indices_to_sample]` 与 `hidden_states = hidden_states[token_indices_to_sample]` 分别取走：

- `last_hidden_states`：用于 `compute_logits` + 采样本 token。
- `hidden_states`：作为下一步 multi-pass 的 hidden 输入（DeepSeek recycle）。

### DeepSeek V4 MTP 的 hidden_size 扩展

`llm_base_proposer.py:97–104`：

```python
draft_hf_config = self.draft_model_config.hf_config
if hasattr(draft_hf_config, "compress_ratios") and hasattr(draft_hf_config, "hc_mult"):
    self.hidden_size = self.hidden_size * draft_hf_config.hc_mult
```

DSV4 MTP drafter 消耗 target 的 pre-hc_head residual stream，shape `(T, hc_mult * hidden_size)`。基类据此扩展 `self.hidden_size` 与 `hidden_states` buffer 大小。

### MTP index share 优化

`_share_mtp_indices` 字段（基类行 82、571–603）：

- Step 0 时 `set_skip_topk(False)`，让 MTP layer 自己计算 topk indices。
- Step 1+ 时 `set_skip_topk(True)` + `compact_topk_indices(token_indices_to_sample)`，复用 step 0 算出的 indices，跳过重复 topk 计算。
- 这是 DeepSeek MLA 系（含 sparse attention）MTP 的优化：topk indices 在 K 个 draft step 间是稳定的，可缓存复用。

### 并行 drafting 与 MTP

并行 drafting（DFlash/DSpark 走 mask token 一次算 K 位置）通常不适用于 DeepSeek MTP（必须 autoregressive chain）。但 DeepSeek V4 系可能引入并行方案（待核实）；`num_speculative_tokens > 1` 路径下 MTP 走多步。

## 与其它模块/系统配合

- [llm-base-proposer.md](llm-base-proposer.md)：MTP 完全走基类流程。
- [eagle.md](eagle.md)：MTP 与 EAGLE 都用 target hidden 输入；MTP 不需要 aux hidden states。
- [gemma4.md](gemma4.md) / [step3p5.md](step3p5.md) / [dflash.md](dflash.md)：MTP 家族的其他成员，子类化。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：scheduler 通过 `num_speculative_tokens` 推进请求，与 MTP 关系直接。
- [模型库-DeepSeek 架构](../../04-model-zoo/README.md)（待补充）：DeepSeek V2/V3/V4 主模型与 drafter。
- [注意力后端-MLA](../../05-attention/README.md)：MTP layer 与 target 共享 MLA sparse SWA group。

## 历史版本演进

- **v0.6.x（V0）**：DeepSeek MTP 已有 V0 实现，作为 `vllm/spec_decode/` 下 draft_model 风格 drafter。
- **v0.10.5**：V1 DeepSeek MTP landfall，`model_returns_tuple` 区分 DeepSeekMTPModel 与其他 MTP 家族；`_share_mtp_indices` 优化引入。
- **v0.10.5+**：DeepSeek V4 支持——`hc_mult` 字段扩展 hidden_size；与 sparse SWA attention 集成。
- **v0.11.0**：Gemma4 / Step3.5 / DFlash 等 MTP 家族子类化，统一 `method == "mtp"` 之外的特化路径。
- **v0.12 / main**：`compact_topk_indices` 接口稳定；MTP + thinking_budget 联动路径完善。

[← 返回投机解码](../README.md)

## 参见

- [llm-base-proposer.md](llm-base-proposer.md)
- [eagle.md](eagle.md)
- [gemma4.md](gemma4.md)
- [step3p5.md](step3p5.md)
- [dflash.md](dflash.md)
