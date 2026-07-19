# 激活函数（activation.py）

[← Wiki 首页](../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > 激活函数

`vllm/model_executor/layers/activation.py`（~836 行）实现 vLLM 所有"门控激活"`*AndMul` 类、`GELU` 家族变体、`xIELU`，以及一个带 learned scale 的 `ScaledActivation`。它们都是 `CustomOp` 子类，注册到 `op_registry` 后可由 `CompilationConfig.custom_ops` 全局开关，也支持 `forward_native` 走 `torch.compile` 融合路径。

## 是什么

文件公开的算子（全部继承 `CustomOp`，注册名见 `@CustomOp.register("...")`）：

| 类 | 注册名 | 含义 |
|---|---|---|
| `SiluAndMul` | `silu_and_mul` (`activation.py:117`) | SwiGLU：`silu(x[:d]) * x[d:]`，MLP 默认 |
| `SiluAndMulWithClamp` | `silu_and_mul_with_clamp` (`activation.py:154`) | SwiGLU + clamp + alpha/beta，MoE shared experts 用 |
| `MulAndSilu` | `mul_and_silu` (`activation.py:214`) | `x[:d] * silu(x[d:])`，顺序倒置的变体 |
| `SwigluOAIAndMul` | `swigluoai_and_mul` (`activation.py:433`) | GPT-OSS SwiGLU（alpha=1.702, limit=7.0），输入按奇偶切片 |
| `SwigluStepAndMul` | `swiglustep_and_mul` (`activation.py:465`) | 带 clamp 的 SwiGLU step（Triton kernel） |
| `GeluAndMul` | `gelu_and_mul` (`activation.py:370`) | GeGLU：`gelu(x[:d]) * x[d:]`，带 `approximate="none"|"tanh"` |
| `GeluAndMulSparse` | `gelu_and_mul_sparse` (`activation.py:251`) | Gemma3n 稀疏 GeGLU：先高斯 top-k 稀疏化再 GELU |
| `FatreluAndMul` | `fatrelu_and_mul` (`activation.py:78`) | MiniCPM-S FATReLU：`F.threshold(x[:d])*x[d:]` |
| `GELU` / `GELUTanh` / `NewGELU` / `FastGELU` / `QuickGELU` | `gelu` / `gelu_tanh` / `gelu_new` / `gelu_fast` / `quick_gelu` | 非 mul 版 GELU 变体 |
| `ReLUSquaredActivation` | `relu2` (`activation.py:586`) | `relu(x)**2` |
| `XIELU` | `xielu` (`activation.py:604`) | 实验 xIELU，带 `alpha_p/alpha_n/beta` 参数与外部 CUDA 包 |
| `ScaledActivation` | — | 含 `scales` 参数的后缩放激活（AWQ 用） |

外加两个公开函数：

- `get_act_fn(act_fn_name) -> nn.Module`（`activation.py:796`）：返回普通激活（无 mul），按 `_ACTIVATION_REGISTRY: LazyDict` 懒构造。
- `get_act_and_mul_fn(act_fn_name, *, compile_native=True)`（`activation.py:824`）：返回"激活 + mul"算子（如 `SiluAndMul`），按 `_ACTIVATION_AND_MUL_REGISTRY` 取，`silu`/`swish` 支持 `compile_native=False` 来跳过 torch.compile。

`_ACTIVATION_REGISTRY` 内置 `gelu / gelu_fast / gelu_new / gelu_pytorch_tanh / relu / relu2 / silu / swish / quick_gelu / tanh / sigmoid / xielu`；`_ACTIVATION_AND_MUL_REGISTRY` 内置 `gelu / gelu_pytorch_tanh / silu / swish / geglu / swigluoai`。

## 为什么

将所有激活集中到一个文件并以 `CustomOp` 包装，是为了：

1. **GEMM 融合做不掉的 mul 由专用 kernel 完成**：`MergedColumnParallelLinear` 产出的 `gate_up` 张量是 `[num_tokens, 2*d]`，下游必须做 `silu(gate)*up` 这种"按通道切片乘"。CUDA 上用 `torch.ops._C.silu_and_mul` 单 kernel 完成切片+激活+乘，避免三次访存。
2. **平台差异集中化**：例如 ROCm 上 `torch.compile + GELU(tanh)` 数值不稳（`activation.py:273-279`、`397-402`），统一在算子层 fallback；ARM CPU 上有 `activation_lut_bf16` 与 `gelu_tanh` 专用内核（`activation.py:319-322`）。
3. **`torch.compile` 协同**：`SiluAndMul.__init__` 默认 `compile_native=True`，让 `forward_native` 在不透明 custom op 内部（如 fused_moe）也能被简单后端编译（参见 [custom-op.md](custom-op.md)），避免在 opaque op 内出现 raw torch ops。
4. **支持 MoE shared experts 的特殊 SwiGLU**：`SiluAndMulWithClamp`、`SwigluStepAndMul`、`SwigluOAIAndMul` 都带 clamp/alpha/beta，对应 GPT-OSS、DeepSeek-V3.5、MiniMax 等模型在 MoE shared 专家路径上对激活值范围有约束的设定。

## 怎么做

### 标准前向（以 `SiluAndMul` 为例）

`activation.py:130-151`：

```python
def __init__(self, *, compile_native: bool = True):
    super().__init__(compile_native=compile_native)
    if current_platform.is_cuda_alike() or current_platform.is_xpu():
        self.op = torch.ops._C.silu_and_mul
    elif current_platform.is_cpu():
        self._forward_method = self.forward_native

@staticmethod
def forward_native(x):  # 走 torch.compile 的路径
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

def forward_cuda(self, x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    self.op(out, x)
    return out
```

- `CustomOp.__init__` 通过 `dispatch_forward(compile_native=...)` 在 `__init__` 期把 `self._forward_method` 绑定到 `forward_cuda`/`forward_native`/`forward_hip`/…，运行期 `CustomOp.forward` 直接转调（`custom_op.py:135-136`）。
- 若 `enabled()=False`（被 `compilation_config.custom_ops` 显式 `-silu_and_mul`），`dispatch_forward` 返回 `maybe_compile(self.forward_native, ...)`，即把 native 路径 torch.compile 一遍。

### 注册表与工厂

`_ACTIVATION_REGISTRY` 与 `_ACTIVATION_AND_MUL_REGISTRY` 都是 `LazyDict`（`vllm/utils/collection_utils.py`），首次按名字访问才实例化。`get_act_fn` 还处理 `torch.nn.modules.Identity` 这种"模块路径字符串"（`activation.py:800-804`）。`get_act_and_mul_fn` 对 `silu`/`swish` 且 `compile_native=False` 的情况显式构造 `SiluAndMul(compile_native=False)`，避免缓存命中造成 compile 行为不一致。

### 稀疏激活 GeGLU

`GeluAndMulSparse`（`activation.py:251`）用于 Gemma3n：在 GELU 之前先用 `_gaussian_topk` 按 Gaussian 分位数稀疏化（`activation.py:288-296`）。`std_multiplier = Normal(0,1).icdf(activation_sparsity)` 在 `__init__` 期一次算好。前向只走 `forward_native`（没有CUDA 专用 kernel）。

### `SwigluOAIAndMul` 的奇偶切片

`activation.py:443-451`：标准 SwiGLU 把张量切成连续 `[gate | up]`，而 GPT-OSS 的权重布局是 `[gate0, up0, gate1, up1, ...]`，因此用 `x[..., ::2]` 与 `x[..., 1::2]` 取出 gate/up。`alpha=1.702` 与 `limit=7.0` 是 GPT-OSS 默认常数。

### `ScaledActivation`（AWQ 路径）

`activation.py:718-757`：包装一个普通 activation 并按通道除以 `scales` 参数；`input_is_parallel=True` 时 `scales` 也按 TP 切分加载。这是 AWQ 量化路径专用——量化误差通过 per-channel scale 在激活后补偿。

## 与其它模块/系统配合

- [linear.md](linear.md)：`MergedColumnParallelLinear` 产出的 `gate_up` 直接喂给 `SiluAndMul`/`GeluAndMul`；`skip_bias_add` 让 bias 延后到激活后再加。
- [fused-moe.md](fused-moe.md)：FusedMoE 的 `MoEActivation.from_str(activation)` 决定是否 `is_gated`；`SiluAndMulWithClamp`/`SwigluStepAndMul` 用于 MoE shared experts 的受限 SwiGLU；`swiglu_limit/swiglu_alpha/swiglu_beta` 由 `FusedMoE(...)` 透传到 `FusedMoEConfig`。
- [quantization/](quantization/README.md)：AWQ 的 `ScaledActivation` 在量化 method 创建时挂上；`fusion/quant_activation.py` 把量化与激活改为融合算子（`(待核实)` 由 quantization agent 描述）。
- [compilation-ir #09](../../09-compilation-ir/README.md)：`forward_native` 是 Inductor pattern matcher 可内联的目标（参见 `compilation/passes/fusion/` 下的 rms_quant_fusion 等）；`CustomOp.enabled()` 受 `compilation_config.custom_ops` 控制。
- [platforms #08](../../08-platforms/README.md)：分支常带 `current_platform.is_cuda_alike()` / `is_cpu()` / `is_rocm()` / `is_xpu()`，决定是否调用 `torch.ops._C.*` 或专用 LUT。
- [model-zoo #04](../../04-model-zoo/README.md)：模型 `__init__` 中通常用 `get_act_fn(config.hidden_act)` 与 `get_act_and_mul_fn(...)` 而不直接构造类。

## 历史版本演进

- **早期**：`SiluAndMul` / `GeluAndMul` 以普通 `nn.Module` 实现，前向仅 `forward_cuda`。
- **v0.6**：随 `CustomOp` 抽象引入迁移，注册名 `silu_and_mul` / `gelu_and_mul` 入 `op_registry`。
- **v0.7**：`get_act_fn` 注册表统一，把模型里散落的 `act_fn` 字符串映射集中。
- **v0.8**：`QuickGELU` / `NewGELU` / `FastGELU` 引入支持 CLIP 等视觉模型；`MulAndSilu` 加入（顺序倒置）。
- **v0.9–v0.10**：`SwigluOAIAndMul` 引入支持 GPT-OSS（HuggingFace `transformers` v4.55）；ROCm 上对 `GELU(tanh)` + torch.compile 的 fallback 文案与行为多轮调整。
- **v0.10**（PR #32806 `[torch.compile] Compile CustomOp.forward_native for SiluAndMul and QuantFP8...`）：`SiluAndMul.__init__` 默认 `compile_native=True`，让 opaque op 内部的 `forward_native` 也能编译，避免在 fused_moe 内出现 raw torch ops。
- **v0.10末–v0.11**：`SiluAndMulWithClamp`（带 alpha/beta/clamp）与 `SwigluStepAndMul` 引入，服务 MoE shared experts。`GeluAndMulSparse` 引入支持 Gemma3n。
- **v0.11–v0.12**：`xIELU` 实验性接入，依赖外部 `nickjbrowning/XIELU` 包；`ScaledActivation` 与 AWQ 的关系稳定化。
- **v0.12 / main**：`get_act_and_mul_fn` 新增 `compile_native` 参数；`_ACTIVATION_AND_MUL_REGISTRY` 加入 `swigluoai`；ARM CPU LUT 路径扩展。`FatreluAndMul` 早期用于 MiniCPM-S，仍保留。

[← 返回层库首页](../README.md)

## 参见

- [custom-op.md](custom-op.md)：`CustomOp` 的 dispatch/enable 机制。
- [fused-moe.md](fused-moe.md)：`MoEActivation` 与 `swiglu_limit/alpha/beta` 的传导。
- [quantization/](quantization/README.md)：`ScaledActivation` 与 AWQ `fusion/quant_activation.py`。
