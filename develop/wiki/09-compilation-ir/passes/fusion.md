# 融合 Pass 合集（fusion/）

[← Wiki 首页](../../README.md) > [编译与 IR](../README.md) > [Pass 系统](README.md) > Fusion

源码：`vllm/compilation/passes/fusion/`（13 个文件，约 8000+ 行）

## 是什么

`fusion/` 目录收录全部"图融合"Pass，按 `PassConfig` 开关与平台条件在 [`PostGradPassManager`](pass-manager.md) 中启用。它们用 Inductor `pattern_matcher.register_replacement` 把"自定义 op 组合"替换成单个 fused kernel，减少 kernel launch、中间张量与显存带宽。

| Pass 类 | 源文件 | 触发开关 | 融合内容 |
|---|---|---|---|
| `RMSNormQuantFusionPass` | `rms_quant_fusion.py` | `fuse_norm_quant` | `rms_norm`/`fused_add_rms_norm` + FP8/NVFP4 量化（static/dynamic/group/per-block） |
| `ActivationQuantFusionPass` | `act_quant_fusion.py` | `fuse_act_quant` | `silu_and_mul` + FP8/NVFP4/block 量化 |
| `AttnQuantFusionPass` | `attn_quant_fusion.py` | `fuse_attn_quant` | `Attention` 前的 q/kv 量化 + FP8/NVFP4 |
| `MLAAttnQuantFusionPass` | `mla_attn_quant_fusion.py` | `fuse_attn_quant` | `MLAAttention` 专用 attn+quant（含 group quant） |
| `AllReduceFusionPass` / `RocmAiterAllReduceFusionPass` | `allreduce_rms_fusion.py` | `fuse_allreduce_rms` | `tensor_model_parallel_all_reduce` + `fused_add_rms_norm` + 可选 quant |
| `SequenceParallelismPass` | `sequence_parallelism.py` | `enable_sp` | SP 的首段/中段 all_reduce + rms_norm + optional quant |
| `AsyncTPPass` | `collective_fusion.py` | `fuse_gemm_comms` | GEMM↔reduce_scatter / all_gather↔GEMM（含 FP8/Cutlass/FlashInfer BMM/FP4） |
| `RopeKVCacheFusionPass` | `rope_kvcache_fusion.py` | `fuse_rope_kvcache` | RoPE + KV cache 更新（ROCm 限定，小 batch decode） |
| `MLARoPEKVCacheCatFusionPass` | `mla_rope_kvcache_cat_fusion.py` | `fuse_rope_kvcache_cat_mla` | MLA RoPE + `unified_kv_cache_update` + q_pe/k_pe cat |
| `QKNormRoPEFusionPass` | `qk_norm_rope_fusion.py` | `enable_qk_norm_rope_fusion` | Q/K RMSNorm + RoPE → `fused_qk_norm_rope` |
| `RocmAiterRMSNormQuantFusionPass` | `rocm_aiter_fusion.py` | `fuse_norm_quant` | ROCm AITER rms_norm+quant |
| `RocmAiterSiluMulFp8GroupQuantFusionPass` | `rocm_aiter_fusion.py` | `fuse_act_quant` | ROCm AITER silu_mul+fp8 group quant |
| `RocmAiterTritonAddRMSNormPadFusionPass` | `rocm_aiter_fusion.py` | `fuse_act_padding` | ROCm RMSNorm + router-pad |
| `MLADualRMSNormFusionPass` | `rocm_aiter_fusion.py` | `fuse_mla_dual_rms_norm` | ROCm AITER 成对 q/kv RMSNorm |

辅助：`matcher_utils.py` 提供 `MatcherCustomOp` 与子类（`MatcherRotaryEmbedding`/`MatcherRMSNormGated`/`MatcherDeepseekScalingRotaryEmbedding`/`MatcherQuantFP8`/`MatcherSiluAndMul`），封装"custom vs native 两种 forward"供 pattern/replacement trace 时按平台开关切换。

## 为什么

- **减 kernel launch**：Decode 阶段每层多次 norm/quant/silu，融合成单 kernel 显著降 launch 开销，配合 cudagraph 进一步摊薄。
- **减 中间张量/带宽**：`rms_norm → quant` 两步中间是 bf16/f16 全精度张量，融合后中间值留在寄存器/L2，省一次显存往返；FP8 group quant 融合收益更大。
- **平台与算法自适应**：同一 `fuse_norm_quant` 在 CUDA 走 `RMSNormQuantFusionPass`（`torch.ops._C.*`），在 ROCm 额外走 `RocmAiterRMSNormQuantFusionPass`（aiter op），`MatcherQuantFP8.match_rocm_aiter` 切换 QUANT_OP。`PassConfig.__post_init__` 按平台强制关/开相应字段。
- **序列并行与 AsyncTP**：`SequenceParallelismPass` 把首/中段 all_reduce 与 rms_norm+quant 融合，减少 TP 通信与计算串行；`AsyncTPPass` 融合 GEMM↔集合通信（reduce_scatter/all_gather），含 FP8/Cutlass/FlashInfer 多后端，是 TP 大模型的关键优化。
- **MLA 专属融合**：MLA 注意力的 q/kv norm、RoPE、kv cache 更新路径与普通 attention 不同，`mla_*` 系列针对性匹配；`MLARoPEKVCacheCatFusionPass` 把 RoPE 与 `unified_kv_cache_update` 合并并 cat q_pe/k_pe，减少分支。
- **range 感知**：`RopeKVCacheFusionPass`/`AsyncTPPass` 等覆写 `is_applicable_for_range`，仅在大 token 数或特定区间生效（小 batch 用 unfused kernel 更快）。
- **layer name 编码**：`_USE_LAYERNAME` + `_encode_layer_name` 让 attn/rope fusion 的 pattern 能按层名匹配，支持 per-layer 参数差异。

## 怎么做

### 通用 pattern/replacement 范式

以 `RMSNormStaticQuantPattern.register`（`rms_quant_fusion.py:183`）为例：

```python
def pattern(input, weight, scale):
    result_rms = vllm.ir.ops.rms_norm(input, weight, self.epsilon)
    return self.quant_matcher(result_rms, scale)[0]

def replacement(input, weight, scale):
    result = torch.empty(input.shape, device=input.device, dtype=self.quant_dtype)
    at = auto_functionalized(self.FUSED_OP, result=result, input=input,
                             weight=weight, scale=scale, epsilon=self.epsilon)
    return at[1]

pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass,
                        extra_check=_rms_input_weight_dtype_match)
```

- pattern 用 `vllm.ir.ops.rms_norm`（IR op，见 [`ir-ops.md`](../ir-ops.md)）而非 `torch.ops._C.rms_norm`，使 fusion 在 IR 层匹配，lowering 后才落具体实现。
- replacement 用 `auto_functionalized(FUSED_OP, ...)`（functional 形态），由 `FixFunctionalizationPass` 后续 defunctionalize 成 inplace 调用。
- `extra_check` 做运行期 shape/dtype 校验（如 input/weight dtype 必须一致才融合）。

### matcher_utils 的 dual forward

`MatcherCustomOp` 有 `forward_custom`（用 `auto_functionalized(平台 op)`）与 `forward_native`（用层 forward_static），按 `enabled` 选择。pattern 注册时按平台/开关走 custom 或 native，使同一 pattern 在"custom op 可用"与"不可用回退"两种环境都能匹配。

### SequenceParallelismPass 的首/中段差异

`FirstAllReduceRMSNormPattern`（首段：all_reduce 输入是直接输入）与 `MiddleAllReduceRMSNormPattern`（中段：all_reduce 输入是前一段输出）分别注册，区分 TP 段落位置；含 FP8/NVFP4 量化变体。

### AsyncTPPass 的多后端 pattern

`collective_fusion.py` 含 `GEMMReduceScatterPattern`/`AllGatherGEMMPattern`/`ScaledMMReduceScatterPattern`/`AllGatherScaledMMPattern`/`CutlassScaledMMReduceScatterPattern`/`AllGatherCutlassScaledMMPattern`/`FlashInferBMMFP8ReduceScatterPattern`/`FlashInferAllGatherBMMFP8Pattern`/`FlashInferAllGatherFP4Pattern`，按量化方案与 backend 选择匹配。

## 与其它模块/系统配合

- [`pass-manager.md`](pass-manager.md)：`configure()` 按开关与平台 append；`__call__` 跑 `is_applicable_for_range` 过滤。
- [`vllm-inductor-pass.md`](vllm-inductor-pass.md)：`VllmFusionPatternMatcherPass`/`VllmPatternReplacement` 是所有 fusion pass 的基类。
- [`ir.md`](ir.md) / [`ir-ops.md`](../ir-ops.md)：pattern 用 `vllm.ir.ops.rms_norm`/`fused_add_rms_norm`，IR lowering 把它们落成 provider 实现。
- [`utility.md`](utility.md)：`NoOpEliminationPass`（fusion 前去 noop reshape）、`FixFunctionalizationPass`（fusion 后 defunctionalize）、`ScatterSplitReplacementPass`/`SplitCoalescingPass`（rope fusion 前的图整理）是 fusion 的前后置依赖。
- [`模型执行-custom_op`](../../03-model-execution/layers/custom-op.md)：`torch.ops._C.rms_norm_static_fp8_quant` 等 fused op 的定义与启用。
- [`注意力`](../../05-attention/README.md)：`AttnQuantFusionPass`/`MLAAttnQuantFusionPass` 依赖 `Attention`/`MLAAttention` 层结构与 layer name。
- [`分布式`](../../07-distributed/README.md)：`tensor_model_parallel_all_reduce`、TP group、symmetric_memory。
- [`配置-compilation`](../../10-config/README.md)：`PassConfig` 开关 + `rope_kvcache_fusion_max_token_num`/`fi_allreduce_fusion_max_size_mb`/`sp_min_token_num` 阈值。

## 历史版本演进

- **v0.7**：`RMSNormQuantFusionPass` + `NoOpEliminationPass` 首发，仅 FP8 static/dynamic per-token。
- **v0.8**：`ActivationQuantFusionPass`/`AttnQuantFusionPass`/`AllReduceFusionPass`/`QKNormRoPEFusionPass` 扩充；`matcher_utils` 抽象 dual forward；IR op 接入 pattern。
- **v0.9**：`SequenceParallelismPass`/`AsyncTPPass`/`RopeKVCacheFusionPass`/`MLAAttnQuantFusionPass`/`MLARoPEKVCacheCatFusionPass` 引入；ROCm AITER fusion 矩阵扩充（`rocm_aiter_fusion.py` 多 Pass）；`is_applicable_for_range` 区间感知。
- **v0.10 / main**：NVFP4 / block quant / e8m0 scale fusion；`fuse_rope_kvcache_cat_mla`；FlashInfer NVFP4 GEMM backend；`fi_allreduce_fusion_max_size_mb` 扩展到 world_size=16/SM10.3。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [pass-manager.md](pass-manager.md) — `configure()` 顺序与平台条件。
- [vllm-inductor-pass.md](vllm-inductor-pass.md) — pattern 注册基类。
- [ir.md](ir.md) — pattern 用 IR op，lowering 落实。
- [utility.md](utility.md) — fusion 前后的图整理 pass。
- [../ir-ops.md](../ir-ops.md) — pattern 中的 `vllm.ir.ops.*`。
- [../../05-attention/README.md](../../05-attention/README.md) — attn/mla fusion 的层依赖。
