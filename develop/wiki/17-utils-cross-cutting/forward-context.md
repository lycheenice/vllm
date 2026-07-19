# 步前向上下文（forward_context）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 步前向上下文

本页覆盖 `vllm/forward_context.py`（376 行），描述 vLLM 一步模型前向期间的"全局上下文"机制：`ForwardContext`、`BatchDescriptor`、`DPMetadata` 与 `set_forward_context` 上下文管理器。

## 是什么

### `ForwardContext`（`vllm/forward_context.py:132`）

模块级单例风格的上下文对象，每步前向被填入并暴露给所有自定义算子 / 模型层。核心字段：

- `no_compile_layers: dict[str, Any]`：从 `compilation_config.static_forward_context` 拷贝来的"不参与编译的层"映射。
- `attn_metadata`：`dict[str, AttentionMetadata]`（v1）或 `list[...]`（DBO 双微批），按注意力层名 → metadata；前向动态设置。
- `slot_mapping: dict[str, torch.Tensor]`：KV cache slot 映射。
- `dp_metadata: DPMetadata | None`：DP/SP-MoE 跨秩 token 分布信息。
- `cudagraph_runtime_mode: CUDAGraphMode`：`FULL`/`PIECEWISE`/`NONE`，决定 cudagraph 分派风格。
- `batch_descriptor: BatchDescriptor | None`：cudagraph 池化键。
- `ubatch_slices: UBatchSlices | None`：DBO ubatch 切片。
- `is_padding: torch.Tensor | None`：padding 行掩码，供算子跳过假 token。
- `skip_compiled: bool`：绕过编译图，直接 `.forward()`（用于 warmup/profiling）。
- `all_moe_layers: list[str] | None` + `moe_layer_index: int`：为避免 torch.compile 把字符串硬编码进图，把 MoE 层名列表存于此，自定义算子按序 pop（`vllm/forward_context.py:160` 注释，关联 issue #31985）。
- `additional_kwargs: dict[str, Any]`：由 `current_platform.set_additional_forward_context` 注入的平台专属字段。

### `BatchDescriptor`（`vllm/forward_context.py:29`，`frozen=True`）

cudagraph 池化键，字段尽量少以唯一描述 padded batch：`num_tokens`、`num_reqs`（PIECEWISE 可为 None）、`uniform`（所有 request token 数相同）、`has_lora`、`num_active_loras`（开启 `cudagraph_specialize_lora_count` 时按 LoRA 数分别捕图）。

### `DPMetadata`（`vllm/forward_context.py:73`）

封装 `num_tokens_across_dp_cpu: torch.Tensor` 与可选 `local_sizes`（SP-MoE 切片上下文）；`sp_local_sizes()` 提供序列并行切分；`cu_tokens_across_sp(sp_size)` 给出 SP+DP 维度累积 token。`DPMetadata.make()` 校验 `dp_rank` 行等于本地 batchsize。

### 上下文管理器

- `set_forward_context(...)`（`vllm/forward_context.py:260`）：ModelRunner 在 `execute_model` 前进入；内部构造 `DPMetadata`（需要时调 `coordinate_batch_across_dp`）、可选生成 `BatchDescriptor`、调 `current_platform.set_additional_forward_context(...)` 注入平台字段，再用 `override_forward_context` 把全局 `_forward_context` 替换为新建 `ForwardContext`；退出时若 `VLLM_LOG_BATCHSIZE_INTERVAL>=0` 做 `current_platform.synchronize()` + 批量耗时统计并按间隔打日志。
- `get_forward_context()`（`:199`）/`is_forward_context_available()`（`:208`）：自定义算子/层内读取当前上下文的入口；`_forward_context` 为 `None` 时 `get_forward_context` 断言失败。
- `override_forward_context(fc)`（`:245`）：临时改写上下文的内部管理器（测试/嵌套前向用）。

## 为什么

- **算子需要步级而非全局信息**：注意力 metadata、slot mapping、padding 掩码每步都变；自定义算子（`vllm.moe_forward` 等）无法从闭包拿到，故走模块级单例。
- **torch.compile 友好**：把可变副作用（层名、metadata）放进运行期 context，避免被 Dynamo 当作 graph break 或硬编码进图；`all_moe_layers` 的 pop 模式正是为此（注释 #31985）。
- **cudagraph 分派需要稳定键**：`BatchDescriptor` 让相同形状/LoRA 配置的 batch 复用同一张图。
- **DP/SP-MoE 统一记账**：`DPMetadata` 把"各 DP 秩 token 数"一次性 all_reduce 后缓存，避免逐算子重复通信。

## 怎么做

```python
from vllm.forward_context import set_forward_context, get_forward_context

with set_forward_context(attn_metadata, vllm_config, num_tokens=n,
                         cudagraph_runtime_mode=CUDAGraphMode.FULL):
    fc = get_forward_context()           # 在算子/层内
    attn_md = fc.attn_metadata[name]
    ...
```

- 平台扩展：实现 `Platform.set_additional_forward_context`，把私有字段经 `additional_kwargs` 注入，由 `ForwardContext.additional_kwargs` 暴露给算子。
- MoE 层名冷启动：`compilation_config.fast_moe_cold_start=True` 时 `create_forward_context` 填入 `all_moe_layers`，MoE 自定义算子从中按序取字符串。

## 与其它模块/系统配合

- [执行层](../02-execution/README.md)/[模型执行](../03-model-execution/README.md)：ModelRunner（V1/V2）在 `execute_model` 入口进入 `set_forward_context`。
- [编译与 IR](../09-compilation-ir/README.md)：`no_compile_layers`、`fast_moe_cold_start`、`static_all_moe_layers` 均来自 `compilation_config`；cudagraph wrapper 读 `batch_descriptor`/`cudagraph_runtime_mode` 分派（见 [cuda-graph.md](../09-compilation-ir/cuda-graph.md)、[breakable-cudagraph.md](../09-compilation-ir/breakable-cudagraph.md)）。
- [注意力](../05-attention/README.md)：各后端把构造好的 `AttentionMetadata` 放进 `attn_metadata`，层内经 `fc` 取回。
- [分布式](../07-distributed/README.md)：`DPMetadata` 服务 DP/SP-MoE；`coordinate_batch_across_dp` 在此触发跨秩协调。
- [平台](../08-platforms/README.md)：`current_platform.set_additional_forward_context` 与 `synchronize` 注入/同步钩子。
- [envs](envs.md)：`VLLM_LOG_BATCHSIZE_INTERVAL` 控制批量耗时日志。

## 历史版本演进

- **v0.7（v1 引入）**：`forward_context.py` 随 torch.compile 集成诞生，最初只承载 `attn_metadata`。
- **v0.8–v0.9**：加入 `BatchDescriptor`（cudagraph 池化键）与 `DPMetadata`；`all_moe_layers`/`moe_layer_index` 为解决 MoE 层名硬编码引入。
- **v0.10–v0.11**：引入 DBO 双微批（`attn_metadata` 变为 list 形态）、`ubatch_slices`、`is_padding` 掩码、`skip_compiled`；`BatchDescriptor` 增加 `num_active_loras` 支持 LoRA 数特化 cudagraph。
- **v0.12/main**：PIECEWISE/breakable cudagraph 与 `num_reqs=None` 兼容、SP-MoE `sp_local_sizes` 上下文就位（待核实具体版本）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [envs.md](envs.md)：`VLLM_LOG_BATCHSIZE_INTERVAL` 等。
- [编译与 IR · cuda-graph](../09-compilation-ir/cuda-graph.md)
- [分布式](../07-distributed/README.md)
