# Utility Pass（passes/utility/）

[← Wiki 首页](../../README.md) > [编译与 IR](../README.md) > [Pass 系统](README.md) > Utility

源码：`vllm/compilation/passes/utility/`（5 个文件）

## 是什么

`utility/` 收录一组"图整理"pass，既被 `PostGradPassManager` 当固定尾段（`PostCleanupPass`/`FixFunctionalizationPass`），也被 fusion 路径当预处理（`NoOpEliminationPass`/`ScatterSplitReplacementPass`/`SplitCoalescingPass`）。

| Pass 类 | 源文件 | 角色 | 作用 |
|---|---|---|---|
| `NoOpEliminationPass` | `noop_elimination.py` | 首段（`eliminate_noops`） | 消除冗余 reshape/slice/slice_scatter 链 |
| `SplitCoalescingPass` | `split_coalescing.py` | rope/qk_norm_rope fusion 前置 | 合并同输入同 split_sizes 的重复 `split_with_sizes` |
| `ScatterSplitReplacementPass` | `scatter_split_replace.py` | rope_kvcache fusion 前置 | 把 `slice_scatter+split_with_sizes` 替换成直接 getitem |
| `PostCleanupPass` | `post_cleanup.py` | 固定尾段（两次） | 拓扑排序 + DCE |
| `FixFunctionalizationPass` | `fix_functionalization.py` | 固定尾段（最后） | 把 `auto_functionalized` 节点 defunctionalize 成 inplace 调用 |

## 为什么

- **`NoOpEliminationPass` 为 fusion 铺路**：`apply_fp8_linear` 在 2D 场景会插一个冗余 reshape，torch 内置 noop 消除又不处理某些 slice 变体，导致 fusion pattern 漏匹配。本 pass 先清掉 reshape 链 / 同形 reshape / 同形 slice / 同形 slice_scatter（`noop_elimination.py:67`），让 fusion 看到干净图。
- **`SplitCoalescingPass` 补 CSE 漏洞**：B200+FP8 等组合下 Inductor 图里会出现同输入同 split_sizes 的多个 `split_with_sizes`，CSE 未合并（PyTorch #174472），导致下游 qk_norm_rope 等只匹配单 split 的 pattern 漏匹配。本 pass 把重复 split 合并到首个 canonical 节点（`split_coalescing.py:36`）。
- **`ScatterSplitReplacementPass` 让 rope fusion 可见**：functionalized `rotary_embedding` 会把 q/k `slice_scatter` 回 qkv 再 `split_with_sizes` 取出，这套冗余结构遮蔽了"qkv split + rotary"子pattern。本 pass 在 defunctionalize 前先把 `slice_scatter+split` 替换成直接从 `auto_functionalized` 结果 `getitem`，使 rope_kvcache fusion 能匹配（`scatter_split_replace.py:60`）。
- **`PostCleanupPass` 收敛图**：pattern matcher 不保证拓扑序、不删死节点。本 pass 跑 `stable_topological_sort` + `eliminate_dead_code`，在 fusion 后、IR lowering 前/后各跑一次，保证 lowering 看到干净图且 lowered 死节点被清。
- **`FixFunctionalizationPass` 去 functionalize**：fusion 的 replacement 用 `auto_functionalized(FUSED_OP, ...)`（functional 形态）写入图，但实际 vLLM fused op 是 inplace 的（写 result/residual 等）。本 pass 把 `auto_functionalized` 节点 defunctionalize：把 `getitem[i]` 用户替换成被 mutate 的原参数、插入直接 `function(**kwargs)` 调用、移除 `auto_functionalized` 节点，并删冗余 slice_scatter/copy（`fix_functionalization.py:28`）。这是 stream/内存优化的关键——defunctionalize 后才真正 inplace 写入 static buffer，配合 cudagraph 才正确。

## 怎么做

### NoOpEliminationPass 三类清理

1. reshape 链折到底：`reshape(reshape(x,s1),s2)` → `reshape(x,s2)`（`noop_elimination.py:74`）。
2. 同形 reshape/slice：输出 shape 与输入完全等价（`statically_known_true` 逐维比较）→ 直接用输入。
3. 同形 slice_scatter：`base_shape == view_shape` → 直接用 view。

### SplitCoalescingPass

按 `arg_node` 聚合所有 `split_with_sizes`（要求全部 user 都是 `getitem`），遇同 `split_sizes` 则把后出现的替换为先出现的 canonical 节点并 erase。

### ScatterSplitReplacementPass

匹配 `auto_functionalized(rotary_embedding)` 且 q/k 来自同一 `split_with_sizes` 的 qkv、q/k 结果被 `slice_scatter` 回 qkv 再 `split_with_sizes` 取出。若 inplace tensor（qkv）无其他用户，把后一个 split 的 `getitem[0/1]` 替换成 `auto_functionalized` 的 `getitem[1/2]`，把 `getitem[2]`（v）指向原 qkv 的 split，删除冗余 slice_scatter/split。

### PostCleanupPass

```python
stable_topological_sort(graph)
graph.eliminate_dead_code()
```

### FixFunctionalizationPass

核心 `defunctionalize(graph, node, mutated_args, args)`（`fix_functionalization.py:257`）：

1. `replace_users_with_mutated_args`：把 `getitem[i]` 用户替换为被 mutate 的原始参数（`node.kwargs[arg_name]`），移除 getitem。
2. `insert_defunctionalized`：在原节点前 `graph.call_function(function, kwargs=node.kwargs)`，若 `getitem[0]`（返回值）存在则替换之。
3. `_remove(node)`：延迟到 pass 末尾批量 erase（避免遍历中改图）。

针对 `rotary_embedding` 的 qkv split+slice_scatter 模式有特化分支（`fix_functionalization.py:55-86`），以及 MLA `fused_rope_unified_mla_kv_cache_update` 的 copy+slice_scatter+view 链特化（`fix_functionalization.py:184`）。XPU 平台因不支持 auto-functionalization 暂跳过（`fix_functionalization.py:32`）。

## 与其它模块/系统配合

- [`pass-manager.md`](pass-manager.md)：`PostCleanupPass`/`FixFunctionalizationPass` 是固定尾段；`NoOpEliminationPass` 是可配首段。
- [`fusion.md`](fusion.md)：`SplitCoalescingPass`/`ScatterSplitReplacementPass` 是 `fuse_rope_kvcache`/`enable_qk_norm_rope_fusion` 的前置依赖；`NoOpEliminationPass` 是 norm/act/attn/allreduce fusion 的前置。
- [`ir.md`](ir.md)：`PostCleanupPass` 在 `VllmIRLoweringPass` 前后各跑一次。
- [`vllm-inductor-pass.md`](vllm-inductor-pass.md)：本组 pass 都继承 `VllmInductorPass`，用 `time_and_log` 计时。
- [`模型执行-custom_op`](../../03-model-execution/layers/custom-op.md)：`FixFunctionalizationPass` 的 target 是 `torch.ops._C.*` 与 `torch.ops.vllm.*` inplace op。

## 历史版本演进

- **v0.7**：`NoOpEliminationPass` 首发（服务 rms+quant fusion）；`FixFunctionalizationPass` 首版（仅 rms_norm/silu_and_mul 等少数 op）。
- **v0.8**：`PostCleanupPass` 引入；`FixFunctionalizationPass` 扩充 rotary_embedding qkv 模式、fused_qk_norm_rope、fused_rope_and_unified_kv_cache_update。
- **v0.9**：`SplitCoalescingPass`（补 CSE 漏洞 #33295/#174472）、`ScatterSplitReplacementPass` 引入，服务 rope_kvcache/qk_norm_rope fusion；MLA `fused_rope_unified_mla_kv_cache_update` 特化分支。
- **v0.10 / main**：`silu_and_mul_nvfp4_quant` / `rms_norm_dynamic_per_token_quant` / `flashinfer_trtllm_fused_allreduce_norm` defunctionalize 支持；XPU 跳过策略。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [pass-manager.md](pass-manager.md) — 固定尾段顺序。
- [fusion.md](fusion.md) — 前置依赖关系。
- [ir.md](ir.md) — cleanup 与 lowering 的穿插。
- [vllm-inductor-pass.md](vllm-inductor-pass.md) — 基类与计时。
