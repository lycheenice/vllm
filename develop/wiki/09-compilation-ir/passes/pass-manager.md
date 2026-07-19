# PostGradPassManager

[← Wiki 首页](../../README.md) > [编译与 IR](../README.md) > [Pass 系统](README.md) > PassManager

源码：`vllm/compilation/passes/pass_manager.py`（约 228 行）

## 是什么

`PostGradPassManager`（`pass_manager.py:86`，继承 `CustomGraphPass`）是 vLLM 注入 Inductor `post_grad_custom_post_pass` 的总编排器。它按固定顺序跑自定义 fusion pass → cleanup → IR lowering → clone 消除 → cleanup → fix_functionalization，并把全部 pass 的 `uuid` 汇总成 cache key。

关键成员：

- `passes: list[InductorPass]`：可配置 fusion passes（`configure()` 按 `PassConfig` 填充）。
- `ir_lowering` / `clone_elimination` / `post_cleanup` / `fix_functionalization`：固定尾段 pass 实例。
- `configure(config)`（`pass_manager.py:138`）：按 `pass_config` + 平台条件构造 fusion passes。
- `uuid()`（`pass_manager.py:206`）：汇总所有 pass uuid + `pass_config.compute_hash()` + `compile_range`。
- `with_pattern_match_debug`（`pass_manager.py:68`）：装饰器，按 `VLLM_PATTERN_MATCH_DEBUG` 设 `TORCHINDUCTOR_PATTERN_MATCH_DEBUG` 环境变量。

## 为什么

- **固定尾段保证图收敛**：fusion pass 会留下死节点/非拓扑序；IR lowering 会插入 clone 保护 inplace；fix_functionalization 必须在 functional 图上跑。固定顺序 `fusion → cleanup → lowering → clone_elim → cleanup → fix_func` 让最终图干净且 functional，交给 Inductor 继续 lowering/codegen。
- **`uuid` 进缓存键**：Inductor code-cache 把 `post_grad_custom_post_pass` 的 `uuid` 纳入 hash。`PostGradPassManager.uuid` 汇总各 pass 的源码 hash + `PassConfig` hash + `compile_range`，保证任何 pass 实现或开关变更都触发重编译，反过来保证缓存命中即图完全一致。
- **`compile_range` 进 uuid**：同一图对不同 batch size 区间可能走不同 fusion（`is_applicable_for_range`），uuid 含 `str(compile_range)` 使各 range 独立缓存。
- **平台条件导入**：`configure()` 顶层按 `is_cuda()`/`is_xpu()`/`is_rocm()`/`rocm_aiter_ops.is_enabled()` 选择 import，非目标平台不实例化对应 pass，避免无关 op 注册失败。
- **PassConfig 隔离**：`PassConfig` 与 `CompilationConfig` 分离（`config/compilation.py:107` 注释），PassManager 只持 `pass_config`，不接触完整 `VllmConfig`（含模型指针），避免循环引用与序列化污染。
- **pre-grad 分离**：`VllmIRInplaceFunctionalizationPass` 是 pre-grad pass（`configure_post_pass` 注入 `pre_grad_custom_pass`），不归 PostGradPassManager 管，因为它须在 AOTAutograd functionalize 之前把 maybe_inplace 转成 default overload。

## 怎么做

### configure 顺序

`pass_manager.py:138` 按 `pass_config` 字段依次 append（节选）：

```
eliminate_noops → NoOpEliminationPass
enable_sp        → SequenceParallelismPass
fuse_gemm_comms  → AsyncTPPass
fuse_act_padding(roc@aiter) → RocmAiterTritonAddRMSNormPadFusionPass
fuse_allreduce_rms → AllReduceFusionPass | RocmAiterAllReduceFusionPass
fuse_norm_quant  → RMSNormQuantFusionPass (+RocmAiterRMSNormQuantFusionPass)
fuse_act_quant   → ActivationQuantFusionPass (+RocmAiterSiluMulFp8GroupQuantFusionPass)
fuse_mla_dual_rms_norm → MLADualRMSNormFusionPass
fuse_rope_kvcache → SplitCoalescingPass + ScatterSplitReplacementPass + RopeKVCacheFusionPass
fuse_rope_kvcache_cat_mla → MLARoPEKVCacheCatFusionPass
fuse_attn_quant  → AttnQuantFusionPass + MLAAttnQuantFusionPass
enable_qk_norm_rope_fusion → SplitCoalescingPass + QKNormRoPEFusionPass
固定尾段：VllmIRLoweringPass / UnsafeCloneEliminationPass / PostCleanupPass / FixFunctionalizationPass
```

顺序有依赖：`fuse_act_padding` 须在 `fuse_allreduce_rms` 前（都消耗 `fused_add_rms_norm`，`pass_manager.py:151`）；rope_kvcache 须先 `SplitCoalescingPass` 合并重复 split 才能匹配 pattern；qk_norm_rope 同理。

### __call__ 执行

```python
# pass_manager.py:104 简化
VllmInductorPass.dump_prefix = 0
compile_range = get_pass_context().compile_range
for pass_ in self.passes:
    if pass_.is_applicable_for_range(compile_range):
        pass_(graph); dump_prefix += 1
    else: debug("Skipping ...")
self.post_cleanup(graph)        # cleanup before lowering
self.ir_lowering(graph)         # lower vllm_ir::*
self.clone_elimination(graph)   # remove redundant clones
self.post_cleanup(graph)        # cleanup after lowering
self.fix_functionalization(graph)  # always last
VllmPatternMatcherPass.log_match_summary()
```

### uuid 汇总

```python
# pass_manager.py:206 简化
state = {"pass_config": self.pass_config.compute_hash()}
passes = [p.uuid() for p in self.passes]
passes += [self.post_cleanup.uuid(), self.ir_lowering.uuid(),
           self.clone_elimination.uuid(), self.post_cleanup.uuid(),
           self.fix_functionalization.uuid()]
state["compile_range"] = str(get_pass_context().compile_range)
state["passes"] = passes
return InductorPass.hash_dict(state)
```

## 与其它模块/系统配合

- [`backends.md`](../backends.md)：`VllmBackend.configure_post_pass`（`backends.py:929`）调 `pass_manager.configure(vllm_config)` 并把 manager 注入 `inductor_config[pass_key]`；平台类经 `get_pass_manager_cls()` 解析。
- [`inductor-pass.md`](inductor-pass.md) / [`vllm-inductor-pass.md`](vllm-inductor-pass.md)：所编排的 pass 基类。
- [`ir.md`](ir.md)：`VllmIRLoweringPass` / `UnsafeCloneEliminationPass` 是固定尾段。
- [`utility.md`](utility.md)：`PostCleanupPass` / `FixFunctionalizationPass` 是固定尾段；`NoOpEliminationPass` 是可配首段。
- [`fusion.md`](fusion.md)：`configure()` 填充的全部 fusion pass。
- [`平台`](../../08-platforms/README.md)：`get_pass_manager_cls()` / `pass_key` 允许厂商替换 manager。
- [`配置-compilation`](../../10-config/README.md)：`PassConfig` 开关 + `inductor_passes` 用户注入。

## 历史版本演进

- **v0.7**：`PostGradPassManager` 引入，手动 `add()` pass；`uuid` 基于源码 hash。
- **v0.8（IR 框架成型）**：固定尾段 `ir_lowering → clone_elim → cleanup → fix_func` 引入；`PassConfig` 抽出；`configure()` 按 pass_config 自动填充。
- **v0.9**：`is_applicable_for_range` 区间感知；`compile_range` 进 uuid；平台条件 import 矩阵完善；rocm aiter fusion 接入。
- **v0.10 / main**：`with_pattern_match_debug` 装饰器；`VllmPatternReplacement` 抽象简化新 fusion 注册；nvfp4 group quant fusion。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [inductor-pass.md](inductor-pass.md) — `InductorPass` 基类与 `PassContext`。
- [vllm-inductor-pass.md](vllm-inductor-pass.md) — `VllmPatternMatcherPass` 与 pattern 注册。
- [ir.md](ir.md) — 固定尾段的 IR lowering / clone 消除。
- [utility.md](utility.md) — `PostCleanupPass` / `FixFunctionalizationPass`。
- [fusion.md](fusion.md) — `configure()` 填充的融合 pass 集合。
