# 参数与权重加载（parameter.py）

[← Wiki 首页](../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > 参数

`vllm/model_executor/parameter.py`（618 行）定义 vLLM 的"特殊 `torch.nn.Parameter` 体系"：把"权重在 TP 下怎么切"、"在 packed 量化下怎么解包"、"在 per-tensor scale 下怎么填到 fused 数组里"等加载期逻辑下沉到 Parameter 自身，让层代码不再关心切片细节。这是 `weight_loader_v2` 的核心载体。

## 是什么

文件公开的全部 Parameter 类（见 `__all__`，`parameter.py:18-27`）：

| 类 | 行 | 用途 |
|---|---|---|
| `BasevLLMParameter` | `:32` | 通用基类，持 `weight_loader`、`tp_rank`、`tp_size`；提供 `load_column_parallel_weight` / `load_row_parallel_weight` / `load_merged_column_weight` / `load_qkv_weight` |
| `RowvLLMParameter` | `:204` | 行并行参数：持 `input_dim`；`load_row_parallel_weight` 按 `input_dim` 切 |
| `_ColumnvLLMParameter`（私有） | `:129` | 列并行参数：持 `output_dim`；实现 `load_merged_column_weight` / `load_qkv_weight` |
| `ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter)` | `:233` | 既可列又可行的通用模型权重（最常见） |
| `GroupQuantScaleParameter` | `:242` | 分组量化 scale（同 ModelWeight 几何） |
| `ChannelQuantScaleParameter` | `:251` | 通道量化 scale（仅列并行） |
| `PerTensorScaleParameter` | `:260` | per-tensor 标量 scale；fused 层中按 shard_id 索引 |
| `PackedColumnParameter` | `:313` | 列并行 packed 参数（如 int4/int8 weight packing） |
| `PackedvLLMParameter` | `:353` | 同时支持列+行的 packed 参数（GPTQ Marlin 等场景） |
| `BlockQuantScaleParameter` | `:397` | 块量化 scale（FP8 block scale 等） |
| `SharedWeightParameter` | `:406` | 跨层共享 memory 的多 partition 参数（Mamba2 等） |

辅助函数：

| 函数 | 行 | 用途 |
|---|---|---|
| `permute_param_layout_` | `:544` | 把参数的 `input_dim`/`output_dim`/`packed_dim` 重排 (`permute`) |
| `_adjust_shard_indexes_for_marlin` | `:602` | Marlin tile size 调整 |
| `_adjust_shard_indexes_for_packing` | `:606` | 通用 packed + marlin 调整 |

## 为什么

把加载逻辑下沉到 Parameter 而不是层 `weight_loader` 方法，是为了：

1. **代码重组**：早期每个层（`ColumnParallelLinear`/`MergedColumnParallelLinear`/`QKVParallelLinear`/`RowParallelLinear`）都有各自的 `weight_loader(self, param, loaded_weight, shard_id)` 方法，重复且易错。`weight_loader_v2` 把"按 dim narrow、按 shard_id offset"的逻辑下沉到 Parameter 的 `load_*_weight`，使层只负责"调对方法 + 传对 shard_id"。
2. **多量化形态覆盖**：per-tensor scale 与 per-channel scale 在 fused 层（QKV/gate_up）下处理方式不同——per-tensor 需要把标量填到 fused 数组对应位置（`PerTensorScaleParameter._load_into_shard_id`，`:291-310`），per-channel 只需按 output_dim 切；这种差异让层级 `weight_loader` 越长越乱，下沉到 Parameter 后每类只关心自己的几何。
3. **Packed 参数的切分规范化**：`PackedColumnParameter` / `PackedvLLMParameter` 持有 `packed_factor`、`packed_dim`、`marlin_tile_size`；`adjust_shard_indexes_for_packing` 统一计算 packed 后的 `shard_size/shard_offset`，Marlin 的 tile 边界对齐也集中处理。
4. **跨层内存共享**：`SharedWeightParameter`（`:406`）通过 `tensors_registry: WeakValueDictionary` 让多个 partition 在不同层间共享同一 storage（Mamba2 的 `in_proj` 两段、Phi 的 gate/up 复用等）。`add_partition(index, data_key, *args, **kwargs)` 按 `data_key` 决定是否复用已存在张量，`local_tensors: set[Tensor]` 持强引用防 GC（torch issue #75932）。
5. **TPU weight loader 同步**：TPU 上 `param.data.narrow(...).copy_(...)` 是 lazy 模式，会导致内存重复。`BasevLLMParameter.__init__` 通过 `current_platform.use_sync_weight_loader()` 包装 loader（`:59-62`），让 TPU 加载期显式 sync，避免 OOM。
6. **运行期参数布局转换**：`permute_param_layout_(param, input_dim=, output_dim=, packed_dim=)`（`:544-599`）用于把参数从一种布局（如 `{input=1, output=0}`）改为另一种（如 `{input=0, output=1, packed=0}`），finetune / 量化场景常用。断言 2D 才支持，且 packed_dim 不变（不支持 repacking）。

## 怎么做

### `BasevLLMParameter`

`parameter.py:32-126`：

- `__new__` 走 `Parameter.__new__(cls, data=data, requires_grad=False)`——vLLM 参数永不训练。
- `__init__(data, weight_loader)`：若 `current_platform.use_sync_weight_loader()` 则包装 loader；记 `tp_rank`/`tp_size`。
- `weight_loader` 是 property + setter + deleter（`:68-86`），允许模型层 override（如 `mamba_mixer2` 注释 `:70-73` 提及）。
- `_assert_and_load(loaded_weight)`：通用 sanity + `copy_`。
- `load_column_parallel_weight` / `load_row_parallel_weight` / `load_merged_column_weight` / `load_qkv_weight`：默认实现都退化为 `_assert_and_load`，由子类 override。
- `_shard_id_as_int("q") -> 0`、`"k" -> 1`、`"v" -> 2`：QKV shard_id 字符串到整数的映射。
- `__torch_function__`：覆盖以让 Parameter 子类参与 torch functional 调用（早期 hack，保留）。

### `_ColumnvLLMParameter`（私有）

`:129-201`：

- 持 `output_dim`。
- `load_column_parallel_weight(loaded_weight)`：按 `tp_rank * shard_size` narrow 后 copy。
- `load_merged_column_weight(loaded_weight, *, shard_offset, shard_size, ...)`：先按 param 的 offset+size `narrow`，再按 `tp_rank * shard_size` narrow loaded，最后 copy。若参数本身是 packed（`isinstance(self, (PackedColumnParameter, PackedvLLMParameter))` 且 `packed_dim == output_dim`）则先调 `adjust_shard_indexes_for_packing` 调整 size/offset。
- `load_qkv_weight(loaded_weight, *, shard_offset, shard_size, shard_id, num_heads, ...)`：逻辑同上，但 `shard_id_int = tp_rank if shard_id == "q" else tp_rank // num_heads`——对应 QKVParallelLinear 中 K/V head 的复制因子。

### `RowvLLMParameter`

`:204-230`：持 `input_dim`；`load_row_parallel_weight` 按 `input_dim` 切。`tp_size==1` 时直接 copy 不 narrow。

### `ModelWeightParameter`

`:233-239`：`class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter): pass`——多继承同时获得列+行加载能力。是 `UnquantizedLinearMethod.create_weights`（[linear.md](linear.md)）创建的默认参数类型。

### `PerTensorScaleParameter`

`:260-310`：per-tensor 标量 scale。`load_merged_column_weight` / `load_qkv_weight` 都走 `_load_into_shard_id`——把标量写到 fused 数组的 `shard_id_int` 位置（`param_data[shard_id]`）。这样 fused 层（QKV 有 3 个 scale）的 per-tensor scale 在加载后变成 `param[0]=q_scale, param[1]=k_scale, param[2]=v_scale`，可被 `process_weights_after_loading` 取 `max()` 操作。

### `PackedColumnParameter` / `PackedvLLMParameter`

`:313-394`：packed 参数。`adjust_shard_indexes_for_packing(shard_size, shard_offset)`（`:344-350` / `:388-394`）调 `_adjust_shard_indexes_for_packing`（`:606-616`）：

```
shard_size = round(shard_size // packed_factor)
shard_offset = round(shard_offset // packed_factor)
if marlin_tile_size is not None:
    shard_size *= marlin_tile_size
    shard_offset *= marlin_tile_size
```

`PackedColumnParameter` 仅列并行；`PackedvLLMParameter` 同时继承 `ModelWeightParameter` 让两类几何共存——主要服务 GPTQ Marlin（按 packed_factor=8 把 int4 pack 到 int32）。

### `BlockQuantScaleParameter`

`:397-403`：FP8 block scale 等。几何同 `ModelWeightParameter`，但加载时 `shard_size/shard_offset` 还要按 `weight_block_size[block_n]` 二次调整（见 [linear.md](linear.md) `adjust_block_scale_shard`）。

### `SharedWeightParameter`

`:406-541`：

- `__new__` 用 `data=None` 构造，因为它内部由多个 `ModelWeightParameter` partition 组成而非直接持数据。
- `tensors_registry: WeakValueDictionary` 全局缓存：同 `data_key` 的张量只 `torch.empty` 一次。
- `add_partition(index, data_key, *args, **kwargs)`：若 data_key 已存在则复用张量，否则建新的 `ModelWeightParameter`，存入 `partitions[index]` 与 `local_tensors.add(data)`。
- `load_*_weight` 把请求转发到对应 `partition` 的 `ModelWeightParameter.load_*_weight`。
- `process_weights_after_loading` 把每个 partition 转回普通 `nn.Parameter`（避免后续 forward 触发 Hook）。
- `data` property 故意 `raise ValueError`——禁止直接访问 `.data`，必须 `get_partition(idx).data`。
- `_fake_weight_loader` 抛错防止误用 partition 的 loader。
- 限制：`tp_size > 1` 时 `NotImplementedError`（`:442-446`，目前只支持 TP=1 的跨层共享）。

### `permute_param_layout_`

`:544-599`：把参数从 `(curr_input_dim, curr_output_dim)` 重排到 `(input_dim, output_dim)`。算法：

1. 取得当前 layout，若任一未定义则用 2D 推断另一维。
2. 构造 `perm`：保留 curr_input/curr_output 的语义，插入到目标位置。
3. 若含 `packed_dim` 则断言 `param.packed_dim == perm[packed_dim]`——目前不支持 repacking。
4. `param.data = param.data.permute(*perm)`，同步更新 `_input_dim`/`_output_dim`/`_packed_dim`。

## 与其它模块/系统配合

- [linear.md](linear.md)：`ColumnParallelLinear`/`RowParallelLinear`/`MergedColumnParallelLinear`/`QKVParallelLinear` 在 `weight_loader_v2` 中把 `narrow + copy` 全部委托给 Parameter 的 `load_*_weight`；`WEIGHT_LOADER_V2_SUPPORTED` 列表决定哪些 quant method 走 v2 路径。
- [custom-op.md](custom-op.md)：参数层不直接 dispatch forward（forward 仍由含 Parameter 的层处理），但 OOT 替换的层常需配套 OOT-specific Parameter（如 HPU 的 `MarlinParameter` 等 `(待核实)`）。
- [fused-moe.md](fused-moe.md)：`RoutedExperts` 持有 `w13`（`MergedColumnParallelLinear` 内含 `ModelWeightParameter`）与 `w2`（`RowParallelLinear` 内含 `RowvLLMParameter`），加载时按 `expert_idx, shard_id` 委托到 Parameter。
- [sampling-decoding #06](../../06-sampling-decoding/README.md)：与参数无直接关联，但 `process_weights_after_loading` 是 vLLM 加载链关键阶段，间接影响所有模型的加载时间。
- [model-loader #03](../../README.md)：DefaultModelLoader 在 `load_weights` 阶段把 `(name, loaded_weight)` 路由到 Parameter 的 `weight_loader` 属性（即 `param.weight_loader(param, loaded_weight, shard_id)`），最终调到 BasevLLMParameter 的 `load_*_weight`。
- [quantization/](quantization/README.md)：各 `LinearMethod`/`FusedMoEMethod` 通过 `create_weights` 创建对应 Parameter 子类（如 `Fp8LinearMethod` 创建 `BlockQuantScaleParameter` + `ModelWeightParameter`），并可能自定义 `process_weights_after_loading` 调 `permute_param_layout_`、`PackedvLLMParameter.adjust_shard_indexes_for_packing` 等。
- [mamba-ssm.md](mamba-ssm.md)：`MambaMixer2` 使用 `SharedWeightParameter`（参见 `mamba_v2_sharded_weight_loader`）让 in_proj 两段共享 storage；`weight_loader` 的允许-delete/override 注释（`:70-73`）就是为这种场景预留。
- [distributed #07](../../07-distributed/README.md)：`get_tensor_model_parallel_rank/world_size` 提供 `tp_rank/tp_size`；`adjust_block_scale_shard` 等 helper 与 `FusedMoEParallelConfig` 协同决定 EP 模式下专家权重切分。

## 历史版本演进

- **早期**：vLLM 全用 `torch.nn.Parameter`，加载逻辑散落各层 `weight_loader` 方法。
- **v0.6–v0.7**：`BasevLLMParameter`、`RowvLLMParameter`、`ModelWeightParameter` 等引入，初步把"narrow"集中化；`weight_loader_v2` 在量化 method 上试点。
- **v0.7–v0.8**：`PerTensorScaleParameter`、`ChannelQuantScaleParameter`、`GroupQuantScaleParameter` 引入，覆盖早期量化方案；`PackedColumnParameter` 引入服务 GPTQ/AWQ。
- **v0.8**：`BlockQuantScaleParameter` 引入服务 FP8 block-quant；`PackedvLLMParameter` 拆出支持同时列+行 packed（Marlin 大量使用）。
- **v0.9**：`SharedWeightParameter` 引入（Mamba2 时）支持跨层共享 storage；`tensors_registry: WeakValueDictionary` + `local_tensors: set` 双重引用机制成型。
- **v0.10**：`weight_loader_v2` 全面铺开，`WEIGHT_LOADER_V2_SUPPORTED` 列表维护成本上升；`PluggableLayer` 让 OOT 层与 Parameter 协同更清晰；`TPU use_sync_weight_loader()` 包装机制引入（`:59-62`）。
- **v0.10末–v0.11**：`permute_param_layout_` 函数化（早期是各 method 自实现）；`_adjust_shard_indexes_for_marlin` 与 `_adjust_shard_indexes_for_packing` 集中化（之前散在各 weight_loader）。
- **v0.11–v0.12**：`mamba_v2_sharded_weight_loader` 与 `SharedWeightParameter` 协同成熟；`weight_loader` property 的 setter/deleter 显式加入支持 `mamba_mixer2` 等模型 override（注释 `:70-73`）；多个 `# TODO: @dsikka - move to parameter.py` 注释（`linear.py:889`、`:1120`）逐步清理。
- **v0.12 / main**：`SharedWeightParameter` 仍 `tp_size > 1` `NotImplementedError`（`:442-446`）—— 跨层共享 + TP 是后续扩展点；`BlockQuantScaleParameter` 配合 FP8 / NVFP4 / MXFP4 / MXINT4 等新量化格式不断扩展使用；`permute_param_layout_` 在更多 method 的 `process_weights_after_loading` 中使用。

[← 返回层库首页](../README.md)

## 参见

- [linear.md](linear.md)：`weight_loader_v2` 调用 Parameter 的入口。
- [fused-moe.md](fused-moe.md)：`RoutedExperts` 通过 `make_expert_params_mapping` 把 expert 维度的权重路由到本文件 Parameter。
- [custom-op.md](custom-op.md)：OOT 替换层与 Parameter 的协同。
- [`./quantization/README.md`](quantization/README.md)：各 `LinearMethod.create_weights` 中创建的具体 Parameter 子类细节。
