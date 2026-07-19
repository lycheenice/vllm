# CustomOp 与 PluggableLayer（custom_op.py）

[← Wiki 首页](../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > CustomOp/PluggableLayer

`vllm/model_executor/custom_op.py`（353 行）是层库最核心的"横切抽象"：定义两个基类 `CustomOp` 与 `PluggableLayer`，统一管理"按平台 dispatch"和"out-of-tree 整层替换"两套机制。所有热点算子层（RMSNorm、SiluAndMul、RotaryEmbedding、LinearBase、VocabParallelEmbedding、MambaMixer…）都从这里继承。

## 是什么

文件公开的核心组件：

| 组件 | 行 | 角色 |
|---|---|---|
| `op_registry` | `custom_op.py:21` | `dict[str, type[CustomOp | PluggableLayer]]`，按注册名索引所有 in-tree 算子层 |
| `op_registry_oot` | `:22` | `dict[str, type]`，按 in-tree 类名索引 OOT 替换类（同一名字只能有一个 OOT 替换） |
| `maybe_get_oot_by_class` | `:25` | 给定 in-tree 类，查 OOT 注册表，若不存在则返回原类 |
| `PluggableLayer` | `:32` | 整层可 OOT 替换的 `nn.Module` 基类；`__new__` 在实例化时按 `op_registry_oot[layer_class_name]` 决定真正实例化的类 |
| `PluggableLayer.register(name)` | `:69` | 装饰器：把 in-tree 层注册到 `op_registry`，赋予 `.name` |
| `PluggableLayer.register_oot(...)` | `:83` | 装饰器：把 OOT 替换层注册到 `op_registry_oot[cls.__name__]` |
| `CustomOp` | `:103` | 按平台 dispatch 的 `nn.Module` 基类，提供 `forward_cuda/hip/xpu/cpu/tpu/oot/native` 分支 |
| `CustomOp.__init__` | `:130` | 调 `dispatch_forward(compile_native=...)`，把 `self._forward_method` 在实例化期绑死 |
| `CustomOp.forward` | `:135` | 直接转调 `self._forward_method(*args, **kwargs)`，运行期无 dispatch 分支 |
| `CustomOp.dispatch_forward` | `:174` | 决策：enabled → 选平台 method；disabled → `maybe_compile(self.forward_native, ...)` |
| `CustomOp.maybe_compile` | `:209` | 在不透明 custom op 内部时把 `forward_native` torch.compile（仅 simple_compile_backend） |
| `CustomOp.enabled` | `:271` | 根据 `CompilationConfig.custom_ops`（`+name`/`-name`/`all`/`none`）判断本算子是否启用 |
| `CustomOp.default_on` | `:291` | 默认行为：Inductor 后端模式下默认 `none`（全关），其他后端默认 `all`（全开） |
| `CustomOp.register(name, dynamic_arg_dims=None)` | `:307` | 装饰器：注册 in-tree op；`dynamic_arg_dims` 透传给 `maybe_compile` 标记动态维度 |
| `CustomOp.register_oot(...)` | `:331` | 装饰器：注册 OOT 替换 op |

## 为什么

把"平台 dispatch"与"OOT 替换"统一抽象的动机：

1. **运行期零分支**：`CustomOp.__init__` 在构造期就调 `dispatch_forward()` 把 `self._forward_method` 绑死为 `forward_cuda` 或 `forward_native`（编译后版本）。运行期 `forward` 只做一次属性查找（`:135-136`），避免大批量 decode 时每步走 if/else 路径。
2. **平台抽象**：vLLM 支持 CUDA / ROCm (HIP) / XPU / CPU / TPU / OOT 六类后端。`dispatch_forward`（`:174-207`）按 `current_platform.is_rocm/is_cpu/is_tpu/is_xpu/is_out_of_tree()` 选择 `forward_*`，HIP/TPU/CPU 默认转发到 CUDA/native 路径，子类按需 override。
3. **与 torch.compile 协同的两种姿态**：
   - 启用（`enabled()=True`）：算子层作为不透明 custom op（如 `torch.ops._C.silu_and_mul`）出现于模型级 torch.compile 图中；编译器不内联。
   - 禁用（`enabled()=False`）：`dispatch_forward` 调 `maybe_compile(self.forward_native, ...)`，把 `forward_native` 走 simple_compile_backend 编译一遍。这是为了解决"在 `fused_moe` 等 opaque 调用内部出现 raw torch ops 会让 Inductor 把整图黑盒"的痛点（PR #32806，注释 `:191-194`）。
4. **Inductor 模式 default_off**：当 `CompilationConfig.backend == inductor` 时 `custom_ops` 默认 `none`，即全部 CustomOp 走 `forward_native + torch.compile`，让 Inductor 有机会做跨算子融合（如 RMSNorm + 量化）。其他后端（eager/cudagraphs）默认 `all`，即全部走专用 CUDA kernel。
5. **OOT 全层替换**：`PluggableLayer.__new__`（`:47-66`）在实例化期根据 `op_registry_oot[layer_class_name]` 决定真正实例化的类。这让厂商可以提供 `HPUColumnParallelLinear`、`XPUUnquantizedFusedMoEMethod` 等替代实现而不需要改 vLLM 主线代码。`register_oot` 的两种调用形态（带/不带括号）覆盖了"按名字替换"与"按被替换类替换"两种用法。
6. **`CustomOp.register_oot` 与 `PluggableLayer.register_oot` 区别**：`CustomOp.register_oot` 用 `cls.__name__` 作 key，让 OOT 类替换某 in-tree CustomOp（如 `HPUUnquantizedFusedMoEMethod` 替换 `UnquantizedFusedMoEMethod`）；`PluggableLayer.register_oot` 在新的 OOT 类上设置 `.name = cls.__name__`，让 `PluggableLayer.__new__` 在 `op_registry_oot[layer_class_name]` 查找到的 OOT 类替换整层。

## 怎么做

### 注册一个 in-tree CustomOp

```python
@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    def __init__(self, *, compile_native: bool = True):
        super().__init__(compile_native=compile_native)
        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            self.op = torch.ops._C.silu_and_mul
    @staticmethod
    def forward_native(x): ...    # torch.compile 内联路径
    def forward_cuda(self, x): ... # 专用 CUDA 路径
```

实例化时 `CustomOp.__init__` → `dispatch_forward(compile_native=True)` → 若 `enabled()` 为真则根据平台返回 `forward_cuda` 方法；若为假则 `maybe_compile(self.forward_native, enable=True)`。

### 注册一个 in-tree PluggableLayer

```python
@PluggableLayer.register("column_parallel_linear")
class ColumnParallelLinear(LinearBase):
    ...
```

`ColumnParallelLinear` 实例化时 `PluggableLayer.__new__` 先看 `op_registry_oot["ColumnParallelLinear"]` 是否存在 OOT 类；若存在则实例化 OOT 类，否则实例化自身。这让厂商通过：

```python
@PluggableLayer.register_oot(name="ColumnParallelLinear")
class HPUColumnParallelLinear(ColumnParallelLinear): ...
```

即可让所有 `ColumnParallelLinear` 实际上是 `HPUColumnParallelLinear`。

### OOT 替换的两种姿态

```python
# 形态 1：不带括号，按"被替换父类名"注册
@CustomOp.register_oot
class HPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod): ...

# 形态 2：带括号/带 name，显式指定注册名
@CustomOp.register_oot(name="UnquantizedFusedMoEMethod")
class HPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod): ...
```

两种形态最终都把 `op_registry_oot["UnquantizedFusedMoEMethod"] = HPUUnquantizedFusedMoEMethod`。

### `enabled()` 与 `default_on()`

`CustomOp.enabled()`（`:271-289`）：

```python
compilation_config = get_cached_compilation_config()
custom_ops = compilation_config.custom_ops   # 形如 ["all"] / ["none"] / ["+rms_norm", ...]
enabled = f"+{cls.name}" in custom_ops
disabled = f"-{cls.name}" in custom_ops
return (CustomOp.default_on() or enabled) and not disabled
```

`default_on()`（`:291-304`）：`custom_ops.count("none") + count("all") == 1`，若 `none` 出现则默认关。Inductor 后端自动加 `"none"`，其他后端自动加 `"all"`，由 `CompilationConfig` 在初始化时决定。

### `maybe_compile` 与 dynamic_arg_dims

`custom_op.py:209-269`：当 `compilation_config.mode == NONE`、`backend == "eager"` 或 `enable=False` 时直接返回原函数；否则按 `current_platform.simple_compile_backend` 走 `torch.compile(fn, dynamic=True, backend=...)`。若算子注册时设置了 `_dynamic_arg_dims`（如 `{ "x": [0, -1] }`），则用 `torch._dynamo.mark_dynamic(arg, real_d)` 显式标注动态维度，避免 batch/head 变化时重编译。`silu_and_mul`(早期) `/ QuantFP8` 是典型用例（PR #32806），后续越来越多算子注册时显式声明 `dynamic_arg_dims`。

### 与 `compilation_config.custom_ops` 的关系

`CompilationConfig.custom_ops: list[str]` 是 vLLM 用户可配置的"算子开关"，CLI 与 `VLLM_PLATFORM_OVERRIDE` 间接驱动。`enforce_enable=True`（`:130`）是 ViT 模型内强制开启 device-specific kernel 的特殊机制，注释 `:179-184` 说明这是过渡方案，未来会通过单独的 MM compilation_config 取代。

## 与其它模块/系统配合

- [linear.md](linear.md)：`LinearBase(PluggableLayer)`——线性层整体可被 OOT 替换；本身不做按平台 dispatch（GEMM 由 PyTorch/Inductor 处理）。
- [norm.md](norm.md)：`RMSNorm(CustomOp)` 与 `RMSNormGated(CustomOp)`——`forward_native` 走 `vllm.ir.ops.rms_norm`，让 Inductor pattern matcher 可以与下游量化/all-reduce 融合；`enabled()=False` 是触发融合的入口。
- [activation.md](activation.md)：所有 `*AndMul`/`GELU*`/`XIELU` 都是 `CustomOp`，注册名 `silu_and_mul` 等；`compile_native=True` 让 opaque op 内部也能 torch.compile。
- [rotary.md](rotary.md)：`RotaryEmbeddingBase(CustomOp)`——按平台选 `forward_cuda`/`forward_hip`（AITER）/`forward_native`。
- [embedding.md](embedding.md)：`VocabParallelEmbedding(PluggableLayer)` / `ParallelLMHead(PluggableLayer)`——支持厂商整层替换，例如 HPU 等。
- [fused-moe.md](fused-moe.md)：`FusedMoE` 内部的 `RoutedExperts`/`quant_method` 多为 `CustomOp`（如 `UnquantizedFusedMoEMethod`）；`moe_forward` 通过 `direct_register_custom_op`（`vllm/utils/torch_utils.py`）注册，不走 `CustomOp` 体系，但 `MoERunner` 注册到 `compilation_config.static_forward_context` 协同。
- [mamba-ssm.md](mamba-ssm.md)：`MambaMixer`/`MambaMixer2(PluggableLayer)`，`ShortConv/Mixer2RMSNormGated(CustomOp)`；`mamba_mixer`/`mamba_mixer2` 同样走 `direct_register_custom_op`。
- [parameter.md](parameter.md)：`BasevLLMParameter` 不直接依赖 custom_op，但 `set_weight_attrs` 等加载期逻辑与 `op_registry` 协同（如 OOT 层可能创建 OOT-specific Parameter）。
- [compilation-ir #09](../../09-compilation-ir/README.md)：`op_registry` 与 `CompilationConfig.custom_ops`/`enabled_custom_ops`/`disabled_custom_ops` 双向校准；IR ops priority 机制决定 `forward_native` 走 vLLM IR op 还是纯 aten。`static_forward_context` 把 `MoERunner`/`MambaMixer` 等"按名字反查层"的机制放在 compilation config 中。
- [platforms #08](../../08-platforms/README.md)：`current_platform` 是 dispatch 的决策源；`is_rocm/is_cpu/is_tpu/is_xpu/is_out_of_tree/is_cuda_alike()` 与 `simple_compile_backend`、`use_sync_weight_loader()` 都来自 platform。
- [LoRA #12](../../12-lora/README.md)：LoRA 通过替换 `quant_method` 或包装层外挂 LoRA，不直接走 `register_oot`；但 PR #37181 `[MM][OOT] Add ability to replace oot ops when using lora`（顶层 git log 显示）让 OOT + LoRA 能共存。

## 历史版本演进

- **早期**：vLLM 算子层无统一抽象，每个文件直接 `if is_cuda(): ... else: ...`，重复且无 OOT 概念。
- **v0.6–v0.7**：`CustomOp` 引入，把 `forward_cuda/native` 分支集中；`op_registry` 初版成形。早期 `SiluAndMul`、`RMSNorm` 迁移过来。
- **v0.7–v0.8**：`compilation_config.custom_ops` + `enabled()/default_on()` 引入，让用户能用 `+rms_norm`/`-silu_and_mul` 显式开关；Inductor 后端默认 `none` 触发 `forward_native + torch.compile`。
- **v0.8–v0.9**：`register_oot` 引入支持 HPU/XPU 等厂商整层替换；`op_registry_oot` 与 `op_registry` 分离。
- **v0.10（PR #32744 `[PluggableLayer][1/N] Define PluggableLayer`）**：`PluggableLayer` 抽象落地，把"全层 OOT 替换"与"按平台 dispatch"两条路径拆成两个基类。`LinearBase`、`VocabParallelEmbedding`、`ParallelLMHead`、`MambaMixer`、`MambaMixer2` 等改为继承 `PluggableLayer`，`CustomOp` 仍保留为按平台 dispatch 的算子层基类。
- **v0.10（PR #32806 `[torch.compile] Compile CustomOp.forward_native for SiluAndMul and QuantFP8...`）**：`maybe_compile` 引入，让 opaque custom op（fused_moe 等）内部的 `forward_native` 也能 torch.compile；`compile_native` 参数加入 `SiluAndMul`/`CustomOp.__init__`。
- **v0.10末**：`enforce_enable` 引入支持 ViT 模型在 graph mode 强制启用 device kernel（注释 `:179-184` 标记为过渡方案）。
- **v0.10末–v0.11**：`dynamic_arg_dims` 引入（PR `#34900` `[Model Bash][DSR1] Add selective dynamic shape marking for CustomOp`），让特定算子在 `maybe_compile` 路径下显式 declare dynamic dims。
- **v0.11–v0.12**：PR `#37181` `[MM][OOT] Add ability to replace oot ops when using lora` 让 OOT 替换与 LoRA 兼容；`op_registry_oot` 注册路径在 LoRA 场景下保留启用。
- **v0.12（PR `#36605`）`[MM][OOT] Support CPU seq_lens for OOT MMEncoderAttention kernels`**：OOT MM 注意力层在 CPU 上支持 `seq_lens` 处理，进一步扩展 OOT 适用范围。
- **v0.12 / main**：`CustomOp` 与 `PluggableLayer` 已经成为层库基础设施；`compilation_config.custom_ops` 维持 `+name`/`-name` 协议；`enforce_enable` 仍在等待 MM compilation_config 取代 `(待核实)`。

[← 返回层库首页](../README.md)

## 参见

- [parameter.md](parameter.md)：参数层的特殊化与 OOT 替换的耦合。
- [compilation-ir #09](../../09-compilation-ir/README.md)：`custom_ops` 配置、IR ops priority 与 `static_forward_context`。
- [platforms #08](../../08-platforms/README.md)：`current_platform` 决策源。
