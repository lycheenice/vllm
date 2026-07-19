# CompilationConfig + CompilationMode + CUDAGraphMode + PassConfig + DynamicShapesConfig（compilation.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > CompilationConfig

源码：`vllm/config/compilation.py`（约 1521 行）。本模块是 vLLM 与 `torch.compile`/Inductor/CUDA graph 集成的全部控制面：编译模式、cudagraph 模式与捕获尺寸、Inductor pass 融合开关、动态形状策略、编译缓存、splitting ops。`CompilationConfig` 是 `VllmConfig.compilation_config`，被 `vllm/compilation/` 与 `ModelRunner` 消费。

## 是什么

### `CompilationMode`（`compilation.py:37`，`IntEnum`）

| 值 | 含义 |
|---|---|
| `NONE`(0) | 全 eager，无 torch.compile |
| `STOCK_TORCH_COMPILE`(1) | 标准 `torch.compile` 流水线 |
| `DYNAMO_TRACE_ONCE`(2) | 单次 Dynamo trace，去 guard 防重编译 |
| `VLLM_COMPILE`(3) | vLLM 自定义 Inductor 后端：缓存 + 分段编译 + 形状特化 + 自定义 pass（v1 默认） |

### `CUDAGraphMode`（`compilation.py:53`，`Enum`）

| 值 | 含义 |
|---|---|
| `NONE` | 不捕获 cudagraph |
| `PIECEWISE` | 分段（cudagraph-unsafe op 留外） |
| `FULL` | 整批全图捕获 |
| `FULL_DECODE_ONLY` | 仅 decode 批全图（P/D 的 D 实例省显存） |
| `FULL_AND_PIECEWISE` | decode 批全图 + prefill/混合批分段（v1 默认） |

值可为 tuple `(FULL, NONE)` 表示"decode 用 FULL、prefill 用 NONE"。方法：`decode_mode()`/`mixed_mode()`/`has_mode()`/`requires_piecewise_compilation()`/`has_full_cudagraphs()`/`separate_routine()`/`valid_runtime_modes()`/`__bool__`（`NONE→False`）。

### `DynamicShapesType`（`compilation.py:317`）与 `DynamicShapesConfig`（`compilation.py:337`）

`DynamicShapesType`：`BACKED`(默认)/`UNBACKED`/`BACKED_SIZE_OBLIVIOUS`（实验）。

| 字段 | 默认 | 含义 |
|---|---|---|
| `type` | `BACKED` | torch.compile 动态形状处理 |
| `evaluate_guards` | `False` | 调试：Dynamo 守卫不丢弃，重编译即报错（需 `VLLM_USE_BYTECODE_HOOK=0`） |
| `assume_32_bit_indexing` | `False` | 假设 32-bit 索引（需 PyTorch 2.10+） |

### `PassConfig`（`compilation.py:106`）

自定义 Inductor pass 的开关。多数字段默认 `None`（`_skip_none_validation` wrap validator，供 `VllmConfig` 延迟填默认）。

**融合开关**

| 字段 | 含义 |
|---|---|
| `fuse_norm_quant` | RMSNorm + quant 融合 |
| `fuse_act_quant` | SiluMul + quant 融合 |
| `fuse_attn_quant` | Attention/MLAAttention + quant 融合 |
| `fuse_allreduce_rms` | flashinfer allreduce 融合（TP>1 + Hopper/Blackwell + flashinfer） |
| `fuse_act_padding` | ROCm：RMSNorm + padding 融合 |
| `fuse_mla_dual_rms_norm` | ROCm/AITER：MLA q/kv 配对 RMS norm 融合 |
| `fuse_rope_kvcache` | ROCm：QK rope + KV cache 融合 |
| `fuse_rope_kvcache_cat_mla` | MLA KV cache update + RoPE 融合 |
| `enable_qk_norm_rope_fusion` | Q/K RMSNorm + RoPE 融合 |
| `enable_sp` | 序列并行（async TP 基础，需 TP>1） |
| `fuse_gemm_comms` | async TP（GEMM+通信融合），隐式开 `enable_sp` |
| `eliminate_noops` | 默认 `True`，消除 no-op（融合依赖） |

**阈值**

| 字段 | 默认 | 含义 |
|---|---|---|
| `rope_kvcache_fusion_max_token_num` | `256` | ROCm rope+kv 融合上限 token |
| `fi_allreduce_fusion_max_size_mb` | `None`(平台默认) | flashinfer allreduce 融合通信量上限（按 compute capability × world_size 默认） |
| `sp_min_token_num` | `None`(自动阈值) | SP 启用最小 token 数 |

方法：`flashinfer_max_size(world_size)`、`default_fi_allreduce_fusion_max_size_mb()`、`__post_init__`（平台不支持融合则关闭并 warning）、`log_enabled_passes()`、`compute_hash()`（全字段纳入）。

### `CompilationConfig`（`compilation.py:380`）

**顶层编译控制**

| 字段 | 默认 | 含义 |
|---|---|---|
| `mode` | `None`(→VLLM_COMPILE 当 O>0) | `CompilationMode` |
| `debug_dump_path` | `None` | 调试转储路径 |
| `cache_dir` | `""` | 编译图缓存目录（默认按模型信息生成） |
| `compile_cache_save_format` | `binary`/`unpacked` | 缓存保存格式（`binary`多进程安全） |
| `backend` | `""`(→inductor) | 编译后端：`""`/`eager`/`openxla`/全限定名 |
| `custom_ops` | `[]` | `all`/`none`/`+op`/`-op` 列表，控制 CustomOp 启停 |
| `ir_enable_torch_wrap` | `None` | vLLM IR torch custom op wrapping（默认 Inductor+VLLM_COMPILE 时 True） |
| `splitting_ops` | `None`(→attention ops) | 分段编译拆分点；`[]`=不拆（全图） |
| `compile_mm_encoder` | `False` | 编译多模态编码器 |
| `use_inductor_graph_partition` | `None` | Inductor codegen 时分区（支持 full+piecewise 不编译两次） |

**CUDAGraph 捕获**

| 字段 | 默认 | 含义 |
|---|---|---|
| `cudagraph_mode` | `None`(→FULL_AND_PIECEWISE) | `CUDAGraphMode` |
| `cudagraph_num_of_warmups` | `0` | 捕获前 warmup 次数 |
| `cudagraph_capture_sizes` | `None`(自动推导) | 捕获尺寸列表 |
| `max_cudagraph_capture_size` | `None` | 捕获上限 |
| `cudagraph_copy_inputs` | `False` | 是否拷贝输入（仅 PIECEWISE 有效） |
| `cudagraph_specialize_lora` | `True` | LoRA 有无各捕获一份图 |

**多模态编码器 CUDA graph**

| 字段 | 默认 | 含义 |
|---|---|---|
| `cudagraph_mm_encoder` | `False` | 捕获 ViT 为 cudagraph |
| `encoder_cudagraph_token_budgets` | `[]` | token 预算等级（自动按 2 幂推导） |
| `encoder_cudagraph_max_vision_items_per_batch` | `0`(自动) | 每批图像/视频上限 |
| `encoder_cudagraph_max_frames_per_batch` | `None`(自动) | 每批视频帧上限 |

**Inductor 编译**

| 字段 | 默认 | 含义 |
|---|---|---|
| `compile_sizes` | `None` | 显式编译尺寸（支持 `"cudagraph_capture_sizes"`） |
| `compile_ranges_endpoints` | `None` | 编译区间端点（`[1,e0],[e0+1,e1],...,[eN+1,max]`） |
| `inductor_compile_config` | `{}` | Inductor 额外配置 |
| `inductor_passes` | `{}` | pass 名→函数全限定名（JSON 友好） |
| `pass_config` | `PassConfig()` | 自定义 pass 开关 |
| `dynamic_shapes_config` | `DynamicShapesConfig()` | 动态形状 |
| `local_cache_dir` | init=False | 每 rank 本地缓存 |
| `fast_moe_cold_start` | `None`(spec off) | MoE 冷启动优化（torch>=2.11 由 OpaqueObject 取代） |

**关键方法**：`compute_hash`、`is_custom_op_enabled(name)`、`custom_op_log_check`、`post_init_cudagraph_sizes`、`set_splitting_ops_for_v1(all2all_backend, data_parallel_size)`、`splitting_ops_contain_kv_cache_update()`、`log_enabled_passes`（在 `PassConfig`）。

## 为什么

- **编译/cudagraph 正交但耦合**：`mode` 控制 torch.compile，`cudagraph_mode` 控制 CUDA graph 捕获。piecewise cudagraph **必须** `mode=VLLM_COMPILE` 且非空 `splitting_ops`；full cudagraph 可独立于编译。`VllmConfig.__post_init__` 末段断言此约束。
- **cudagraph 尺寸 vs inductor 尺寸**：cudagraph 须为每个精确尺寸捕获；inductor 编译的图可覆盖一段形状区间。`compile_ranges_endpoints` 让 inductor 按区间编译（如 SP/allreduce 阈值分段），`cudagraph_capture_sizes` 按 `[1,2,4]+range(8,256,8)+range(256,max+1,16)` 默认推导。
- **PassConfig 隔离**：Inductor pass 不直接访问全 `VllmConfig`（防 PassManager 与 config 循环依赖），独立为 `PassConfig`，由 `VllmConfig.__post_init` 按优化级别/平台填默认。
- **三态融合开关**：`fuse_*` 字段 `None` 让 `OPTIMIZATION_LEVEL_*` dict 的 callable（如 `enable_allreduce_rms_fusion`）按平台/TP 决定，显式 True/False 不被覆盖。
- **动态形状**：`BACKED`(默认) 牺牲 guard 精度换稳定；`UNBACKED` 无 guard 但数据依赖分支可能报错；`BACKED_SIZE_OBLIVIOUS` 实验折中。`evaluate_guards` 调试用。

## 怎么做

- **优化级别**：`-O0`/`-O1`/`-O2`(默认)/`-O3`，`VllmConfig` 自动展开为 `mode`/`cudagraph_mode`/`pass_config`/`kernel_config` 默认。
- **cudagraph 尺寸**：`-cc.cudagraph_capture_sizes=[1,2,4,8,16]` 或 `--max-num-seqs` 影响 `max_cudagraph_capture_size`；`performance_mode=interactivity` 走 1..32 细粒度。
- **关编译**：`enforce_eager` 或 `-cc.mode=none -cc.cudagraph_mode=none`；`TORCH_COMPILE_DISABLE=1` 仅关 inductor。
- **融合开关**：`-cc.pass_config.fuse_norm_quant=true`（点分嵌套，经 `update_config` 下钻）。
- **自定义 pass**：`-cc.inductor_passes='{"my_pass":"mymod.passes:my_pass"}'`。
- **调试**：`-cc.debug_dump_path=/tmp/cc` 或 `VLLM_DEBUG_DUMP_PATH=/tmp/cc`（后者覆盖前者）。

## 与其它模块/系统配合

- **编译子系统（[`09-compilation-ir/`](../09-compilation-ir/README.md)）**：`mode`/`backend`/`splitting_ops`/`use_inductor_graph_partition` 控制 Dynamo trace 与 Inductor 分区；`pass_config` 驱动 `vllm/compilation/passes/` 各融合 pass。
- **ModelRunner（[`02-execution/worker/gpu-model-runner.md`](../02-execution/worker/gpu-model-runner.md) 与 [`02-execution/worker/cudagraph-capture.md`](../02-execution/worker/cudagraph-capture.md)）**：`cudagraph_mode`/`cudagraph_capture_sizes` 驱动 cudagraph 捕获与重放；`compile_sizes`/`compile_ranges_endpoints` 决定 inductor 编译哪些尺寸。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`__post_init` 大量调整——`enforce_eager`/`TORCH_COMPILE_DISABLE`/`VLLM_USE_BREAKABLE_CUDAGRAPH` 重写 `mode`；平台 `apply_config_platform_defaults`；`_set_cudagraph_sizes`/`_set_compile_ranges` 推导；`set_splitting_ops_for_v1` 按 all2all/DP 设拆分点；SP+PP 强制 `+rms_norm`。
- **CustomOp（[`03-model-execution/layers/custom-op.md`](../03-model-execution/layers/custom-op.md)）**：`custom_ops` 列表 + `ir_enable_torch_wrap` 决定 CustomOp 是走 vLLM IR dispatch 还是让 Dynamo 直接 trace 实现。
- **Kernel（[kernel-config.md](kernel-config.md)）**：`pass_config.fuse_*` 的 callable 默认会查 `kernel_config.ir_op_priority`（如 `enable_norm_fusion` 检查 `rms_norm[0] != "native"`）。
- **LoRA（[lora-config.md](lora-config.md)）**：`cudagraph_specialize_lora` 让有无 LoRA 各捕一份图。
- **投机解码（[speculative-config.md](speculative-config.md)）**：`use_v2_model_runner` + 动态 spec 触发 `_maybe_override_dynamic_sd_cudagraph_mode`（强制 PIECEWISE）；V2 不支持 `STOCK_TORCH_COMPILE`/SP。
- **多模态（[multimodal-config.md](multimodal-config.md)）**：`compile_mm_encoder`/`cudagraph_mm_encoder`/`encoder_cudagraph_*` 控制 ViT 编译/捕获；`multimodal_config.compute_hash` 在 `compile_mm_encoder` 时纳入 `VllmConfig.compute_hash`。

## 历史版本演进

- **v0.5/v0.6（v0）**：仅 `enforce_eager` 开关；cudagraph 全图捕获，无分段；无 Inductor pass。
- **v0.7（v1 落地）**：`CompilationConfig` 引入，`CompilationMode` 四档；piecewise 编译 + splitting_ops；`cudagraph_capture_sizes` 推导；`cache_dir` 编译缓存。
- **v0.8**：`PassConfig` 独立；`fuse_norm_quant`/`fuse_act_quant`/`enable_sp`/`fuse_allreduce_rms`；`CUDAGraphMode` 多模式（FULL/PIECEWISE/FULL_DECODE_ONLY/FULL_AND_PIECEWISE）；`DynamicShapesConfig`。
- **v0.9**：`use_inductor_graph_partition`（分区在 codegen 时，支持 full+piecewise 不编译两次）；`cudagraph_specialize_lora`；动态 spec `cudagraph_mode` 强制 PIECEWISE；breakable cudagraph（DeepseekV4/MiniMaxM3Sparse 等，`VLLM_USE_BREAKABLE_CUDAGRAPH`）。
- **v0.10**：`performance_mode`（interactivity 走细粒度 cudagraph 1..32）；`compile_ranges_endpoints` 与 SP/allreduce 阈值联动；`fast_moe_cold_start`；`cudagraph_mm_encoder` + `encoder_cudagraph_*`（ViT cudagraph）。
- **v0.11 / v0.12 / main**：`ir_enable_torch_wrap` + vLLM IR op 体系（`IrOpPriorityConfig` 协同）；`HAS_OPAQUE_TYPE` 关 `fast_moe_cold_start`；`fi_allreduce_fusion_max_size_mb` 字段；`compile_cache_save_format`（binary/unpacked）；MRv2 对 `FULL_AND_PIECEWISE` 的支持细化。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [vllm-config.md](vllm-config.md) — `__post_init` 对本配置的大量调整与优化级别展开。
- [kernel-config.md](kernel-config.md) — `ir_op_priority` 与融合 callable 默认协同。
- [lora-config.md](lora-config.md) — `cudagraph_specialize_lora`。
- [multimodal-config.md](multimodal-config.md) — ViT 编译与 cudagraph。
- [../09-compilation-ir/](../09-compilation-ir/README.md) — 编译子系统消费方。
- [../02-execution/worker/cudagraph-capture.md](../02-execution/worker/cudagraph-capture.md) — cudagraph 捕获消费方。
