# 旋转位置编码（rotary_embedding/）

[← Wiki 首页](../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > 旋转位置编码

`vllm/model_executor/layers/rotary_embedding/` 实现 RoPE（Rotary Positional Embedding）及其全部变体。它是几乎所有现代 LLM（LLaMA / Mistral / Qwen / DeepSeek / Gemma / Phi / MiniMax …）位置编码的真正实现，被注意力后端在 `q`/`k` 投影之后、SDPA 之前调用。

## 是什么

模块对外暴露统一工厂函数 `get_rope(...)`（`rotary_embedding/__init__.py:33`），根据 `rope_parameters["rope_type"]` 分派到下列具体实现：

| 类 | 文件 | 含义 |
|---|---|---|
| `RotaryEmbeddingBase` | `base.py:15` | 抽象基类，注册为 `CustomOp` 命名 `rotary_embedding`；持有 `cos_sin_cache`、`head_size`、`rotary_dim`、`is_neox_style`、`apply_rotary_emb` |
| `RotaryEmbedding` | `base.py:139` | "原版" RoPE：`_compute_inv_freq` 用 `base ** (arange/rotary_dim)`；前向 in-place 修改 q/k |
| `LinearScalingRotaryEmbedding` | `linear_scaling_rope.py` | `t = t / scaling_factor` 的线性缩放 |
| `NTKScalingRotaryEmbedding` | `ntk_scaling_rope.py` | NTK-aware 缩放，改写 `inv_freq` |
| `DynamicNTKScalingRotaryEmbedding` | `dynamic_ntk_scaling_rope.py` | 动态 NTK（按当前 seq len 自适应） |
| `DynamicNTKAlphaRotaryEmbedding` | `dynamic_ntk_alpha_rope.py` | 带 `alpha` 的 Dynamic NTK 变体 |
| `YaRNScalingRotaryEmbedding` | `yarn_scaling_rope.py:10` | YaRN 缩放，含 `beta_fast/beta_slow`、`mscale` |
| `Llama3RotaryEmbedding` | `llama3_rope.py` | Llama3 频率混合（low/high freq factor） |
| `Phi3LongRoPEScaledRotaryEmbedding` | `phi3_long_rope_scaled_rope.py:16` | Phi-3 LongRoPE：`short_factor` / `long_factor` 二态 + mscale |
| `DeepseekScalingRotaryEmbedding` / `DeepseekV4ScalingRotaryEmbedding` | `deepseek_scaling_rope.py` | DeepSeek-V2/V3/V4 的 YaRN 变体带 `mscale/mscale_all_dim` |
| `MRotaryEmbedding` / `MRotaryEmbeddingInterleaved` | `mrope.py` / `mrope_interleaved.py` | Qwen2-VL 多模态 mRoPE：时间/高/宽三段 `mrope_section`，可选 interleave |
| `Gemma4RotaryEmbedding` | `gemma4_rope.py` | Gemma4 global attention 的 sparse/fractional RoPE |
| `Llama4VisionRotaryEmbedding` | `llama4_vision_rope.py` | Llama4 视觉编码器 RoPE |
| `DualChunkRotaryEmbedding` | `dual_chunk_rope.py` | Qwen long-context dual-chunk RoPE |
| `FourierRotaryEmbedding` | `fope.py` | Fourier RoPE（部分模型 `use_fope` 开启） |
| `XDRotaryEmbedding` | `xdrope.py` | XD-RoPE（带 `xdrope_section`） |
| `TeleChat3RoPEScaledRotaryEmbedding` | `telechat3_scaling_rope.py` | TeleChat3 YaRN 变体 |
| `Ernie45VLRotaryEmbedding` | `ernie45_vl_rope.py` | ERNIE 4.5-VL |
| `ApplyRotaryEmb` | `common.py` | 共享的 in-place / 静态 apply 实现 |

## 为什么

把 RoPE 独立成一个层库子模块的动机：

1. **变体太多**：仅 `rope_type` 就有 `default / linear / ntk / dynamic / yarn / llama3 / longrope / mllama4 / deepseek_yarn / xdrope / telechat3-yarn / proportional / openpangu` 等十余种，每一种都改写 `_compute_inv_freq` 或 `_compute_cos_sin_cache`，统一到 `RotaryEmbeddingBase` 后只需通过 `get_rope(...)` 工厂按配置分派，模型代码无需关心。
2. **与注意力后端解耦**：vLLM 的注意力后端（FlashAttn、FlashInfer、Triton、ROCm AITER…）的 RoPE 应用方式不同——有些在后端内部完成（如 FlashAttn 把 `cos_sin_cache` 作为参数传入），有些需要在外部 in-place 修改 q/k。`RotaryEmbeddingBase` 同时提供 `forward_cuda`（调 `_custom_ops.rotary_embedding`）、`forward_hip`（AITER 路径）、`forward_native`（torch 路径）、`forward_xpu`/`forward_cpu`，由 `CustomOp` 在 `__init__` 期决定。
3. **缓存复用**：`cos_sin_cache` 是按 `max_position_embeddings × rotary_dim` 预算的，`_ROPE_DICT` 全局缓存（`__init__.py:30`）按 `(head_size, rotary_dim, max_position, is_neox_style, rope_parameters, dual_chunk_attention_args, dtype)` 复用，避免每层都建一份。
4. **支持运行期长度切换**：`Phi3LongRoPEScaledRotaryEmbedding` 会读取 `get_current_vllm_config().model_config.max_model_len` 来决定是否 `use_long_rope`（`phi3_long_rope_scaled_rope.py:54-55`），允许训练长度以上的运行期延展。

## 怎么做

### 基类 `RotaryEmbeddingBase.__init__`

`base.py:20-78` 的关键工作：

1. 计算 `inv_freq = 1.0 / (base ** (arange(0, rotary_dim, 2) / rotary_dim))`（CPU/GPU 数值差异由注释说明，`base.py:82-92`）。
2. `cos_sin_cache = einsum(t, inv_freq) → cos|sin`，按 `dtype` 转换并 `register_buffer(..., persistent=False)`。
3. ROCm 路径会额外预存一份 `bf16` 缓存 `cos_sin_cache_bf16`（`base.py:66-74`），用于 AITER + `torch.compile` 路径。
4. 实例化 `ApplyRotaryEmb(is_neox_style=...)` 复用 apply 逻辑。
5. `use_flashinfer`（默认 False，FlashInfer 受 head_size 限制）与 `use_aiter`（ROCm 受 env 控制）决定 `forward_*` 走专用内核还是通用 `_custom_ops.rotary_embedding`。

### 工厂 `get_rope(...)`

`__init__.py:33-384` 是一大块 `if scaling_type == ...` 分派：

- 先把 `rope_parameters` 中所有 `list` 字段 `tuple` 化，便于作为缓存 key。
- `rotary_dim` 优先取 `rope_parameters["rope_dim"]`；否则 `int(head_size * partial_rotary_factor)`。
- `dual_chunk_attention_config` 单独走 `DualChunkRotaryEmbedding` 分支（`__init__.py:86-100`）。
- `default` 分支再细分：含 `mrope_section` → `MRotaryEmbedding`；`use_fope=True` → `FourierRotaryEmbedding`；否则 `RotaryEmbedding`。
- `yarn` 分支再细分：含 `mrope_section` → `MRotaryEmbedding`（用 YaRN 的 mscale 系数）；否则 `YaRNScalingRotaryEmbedding`。
- `deepseek_yarn` / `deepseek_llama_scaling` 同构，按 `is_deepseek_v4` 切换 `DeepseekV4ScalingRotaryEmbedding` vs `DeepseekScalingRotaryEmbedding`，传入 `mscale` / `mscale_all_dim`。
- `openpangu` 分支必须含 `mrope_section` 且 `mrope_interleaved=True`，使用 `MRotaryEmbeddingInterleaved`。

### 前向：`is_neox_style` 与 in-place

`is_neox_style=True`（GPT-NeoX 风格）：把 head 的前一半与后一半旋转交错；`False` 则是"原始 GPT-J"风格相邻配对。两种布局都由 `ApplyRotaryEmb` 处理（`common.py`）。

`RotaryEmbedding.forward_cuda`（`base.py:221-252`）调用 `_custom_ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style)`，是 **in-place** 算子：query/key 张量被原地修改，因此调用方传入的 `q`/`k` 必须是被 `QKVParallelLinear` 直接产出的张量，不能是 view。

### mRoPE 与多模态

`MRotaryEmbedding`（`mrope.py`）针对 Qwen2-VL 等多模态模型：把 `head_dim` 按 `mrope_section = [t, h, w]` 切三段，分别按时间/高/宽位置计算 cos/sin。`cos_sin_cache` 形状为 `(3, num_tokens, head_dim // 2)`（`mrope.py:14-70` 的 Triton kernel 注释）。`MRotaryEmbeddingInterleaved` 是 OpenPangu 风格的 interleave 变体。

### LongRoPE 的运行期自适应

`Phi3LongRoPEScaledRotaryEmbedding.__init__` 在初始化时根据 `vllm_config.model_config.max_model_len > original_max_position_embeddings` 决定 `use_long_rope`（`phi3_long_rope_scaled_rope.py:54-60`）。若超出原训练长度则强制用 `long_factor` 并发出 warning，避免 KV cache 失配。

### DeepSeek mscale

`DeepseekScalingRotaryEmbedding` 在 YaRN 基础上叠加 `mscale = yarn_get_mscale(scaling_factor) * mscale_all_dim`（具体在 `deepseek_scaling_rope.py`，`(待核实)` 详见源码），与 attention 中的 `qkv_normalization` 联合工作以稳住长上下文数值。

## 与其它模块/系统配合

- [attention #05](../../05-attention/README.md)：`RotaryEmbedding.forward_*` 在 `QKVParallelLinear` 之后被调用；FlashAttn 后端会把 `cos_sin_cache` 作为内部参数；MLA 后端（DeepSeek）通常不走 RoPE 而走自己的 decoupled RoPE（细节 `(待核实)`）。
- [linear.md](linear.md)：`QKVParallelLinear` 的 head 切分必须与 `RotaryEmbedding.head_size` / `rotary_dim` 对齐，否则旋转错位。
- [config #10](../../10-config/README.md)：`rope_parameters` 来自 `transformers_utils` 解析的 HF `config.json` 中 `rope_scaling` 字段；`get_rope` 把 list 转 tuple 以便 hash。
- [compilation-ir #09](../../09-compilation-ir/README.md)：`CustomOp.enabled()` 走 `compilation_config.custom_ops`；`_match_cos_sin_cache_dtype` 在 `torch.compiler.is_compiling()` 时返回新张量而非写回 `self.cos_sin_cache` buffer，避免 cudagraph tracing 时改 buffer 触发图重建（`base.py:127-131`）。
- [multimodal #11](../../11-multimodal/README.md)：mRoPE 的 `mrope_section` 来自模型多模态预处理产生的 position_ids（时间/高/宽三维）。
- [platforms #08](../../08-platforms/README.md)：`use_flashinfer`、`use_aiter` 由 `current_platform` 与 `rocm_aiter_ops.is_triton_rotary_embed_enabled()` 决定。

## 历史版本演进

- **v0.5–v0.6**：只有 `RotaryEmbedding` + `LinearScalingRotaryEmbedding` + `DynamicNTKScalingRotaryEmbedding`，仍按 v0 引擎组织。
- **v0.6**：`YaRNScalingRotaryEmbedding` 引入，支持 Phi-2 / Mistral 等长上下文模型。
- **v0.6–v0.7**：`get_rope` 工厂落地，把 rope_type 分派逻辑从模型代码上提；`Phi3LongRoPEScaledRotaryEmbedding` 引入支持 Phi-3 128k。
- **v0.7–v0.8**：`Llama3RotaryEmbedding` 引入支持 Llama-3.1 128k（`low_freq_factor/high_freq_factor/original_max_position_embeddings`）。
- **v0.8–v0.9**：`DeepseekScalingRotaryEmbedding` 引入服务 DeepSeek-V2，带 `mscale/mscale_all_dim`。
- **v0.9–v0.10**：`MRotaryEmbedding` 引入支持 Qwen2-VL；后续补 `MRotaryEmbeddingInterleaved` 支持 OpenPangu；mRoPE 的 cos/sin cache 形状改为 `(3, num_tokens, head_dim // 2)`。
- **v0.10–v0.11**：`CustomOp.register("rotary_embedding")` 统一注册；ROCm AITER 的 Triton 路径接入；`cos_sin_cache_bf16` 备份缓存引入以规避 cudagraph tracing 期 dtype 同步问题。
- **v0.11–v0.12**：`DualChunkRotaryEmbedding` 引入支持 Qwen long-context dual-chunk attention；`FourierRotaryEmbedding`、`XDRotaryEmbedding`、`TeleChat3RoPEScaledRotaryEmbedding`、`Ernie45VLRotaryEmbedding`、`Gemma4RotaryEmbedding`、`Llama4VisionRotaryEmbedding` 陆续加入。
- **v0.12 / main**：`DeepseekV4ScalingRotaryEmbedding` 引入服务 DeepSeek-V4；`use_long_rope` 运行期判断成熟；`partial_rotary_factor` 校验从 0 到 1 区间严格化（`__init__.py:70-72`）。

[← 返回层库首页](../README.md)

## 参见

- [linear.md](linear.md)：`QKVParallelLinear` 与 RoPE 的几何对齐。
- [attention #05](../../05-attention/README.md)：注意力后端如何消费 `cos_sin_cache` 或 in-place q/k。
- [custom-op.md](custom-op.md)：`CustomOp.register` 与 `dispatch_forward` 在 RoPE 上的应用。
