# 编译产物缓存与序列化（caching.py）

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > Caching

源码：`vllm/compilation/caching.py`（约 612 行）

## 是什么

`caching.py` 负责"编译产物如何落盘、如何装载、如何去重"。它把 `VllmBackend` 一次冷编译产出的整图 + 子图 artifact 序列化成可移植的 `VllmSerializableFunction`，热启动时直接装载、跳过编译。

关键成员：

- `VllmSerializableFunction(SerializableCallable)`（`caching.py:166`）：编译后 callable 的包装，实现 PyTorch 自定义 backend 序列化协议（`serialize_graph_module`/`deserialize_graph_module`/`serialize_compile_artifacts`/`deserialize_compile_artifacts`）。
- `StandaloneCompiledArtifacts`（`caching.py:37`）：mega-AOT 产物存储，两级去重（`submodule_bytes` → SHA256 → `submodule_bytes_store` 真字节）。
- `reconstruct_serializable_fn_from_mega_artifact(...)`（`caching.py:411`）：mega-AOT 热装载核心，从预编译 artifact 重建 `VllmSerializableFunction`。
- `aot_compile_hash_factors(vllm_config)`（`caching.py:565`）：AOT hash 因子（env + config + inductor factors）。
- `_compute_code_hash` / `_compute_code_hash_with_content`（`caching.py:584`）：traced_files 源码 checksum，供 AOT 装载时校验源码未变。
- `patch_pytree_map_over_slice()`（`caching.py:154`）：注册 `slice` 的 pytree 支持，使 `GraphPickler` 能序列化含 slice 的图。

## 为什么

- **冷启动一次、热启动秒级**：首次启动编译大模型数十秒到数分钟，把产物落盘后，同 hash 重启直接装载，跳过 Dynamo/Inductor。`VllmSerializableFunction` 是这条链路的载体。
- **Layer 级去重**：transformer 各层结构相同，编译产物字节相同。`StandaloneCompiledArtifacts` 按 SHA256 去重，N 层只存 1 份字节，`num_artifacts vs num_entries` 差距显著，磁盘与装载时间大降。
- **两级 hash 隔离**：AOT hash（`aot_compile_hash_factors`）不含 traced_files 源码（装载时才校验），保证"源码不变即可复用"；装载时 `_verify_source_unchanged`（[`decorators.py`](decorators.md)）用 `source_info.inlined_sources` 内容 checksum 校验源码未改。
- **Feic graph + standalone artifact 双轨**：`VllmSerializableFunction` 序列化时若 `vllm_backend` 存在，调 `collect_standalone_compile_artifacts()` 收集每子图每 shape 的字节，存进 `standalone_compile_artifacts`；反序列化时优先走 mega-AOT 路径（`VLLM_USE_MEGA_AOT_ARTIFACT`），否则回落到重建 `VllmBackend`。
- **slice 序列化补丁**：torch `GraphPickler` 原生不注册 `slice`，含 `slice` 参数的图（常见于 attention reshape）会序列化失败，`patch_pytree_map_over_slice` 临时注册。
- **finalize_loading 延迟 backend 构造**：反序列化时尚未跑 `_verify_source_unchanged`，`traced_files` 未知，无法算 cache_dir，故用 lazy closure 包住 `VllmBackend(graph, inputs)`，待 `finalize_loading(vllm_config)` 在源码校验后才真正调 `VllmBackend.__call__`。

## 怎么做

### VllmSerializableFunction 序列化

`serialize_compile_artifacts`（`caching.py:252`）：

1. 复制 `__dict__`，弹出 `optimized_call`/`shape_env`/`vllm_backend`/`_fake_mode`。
2. 清理 `graph_module` 各 node 的 `source_fn_stack`/`nn_module_stack` meta（不可序列化）。
3. 把 `example_inputs` 里的 Tensor 换成 `meta` device 的空张量（数据不需要，只需 meta 给 `make_copy_and_call`）。
4. `serialize_graph_module` 用 `GraphPickler.dumps` + 自定义 `reducer_override`（处理 sympy Function 的 `_torch_unpickler`、把 `FakeTensorMode` 降级为 `type(None)`）。
5. 若 `vllm_backend` 存在，`collect_standalone_compile_artifacts()` 收集 mega artifact + sym_shape_indices_map + returns_tuple_map。
6. `pickle.dumps(state)`。

### 反序列化两条路径

`deserialize_compile_artifacts`（`caching.py:300`）：

- **mega-AOT 路径**（`VLLM_USE_MEGA_AOT_ARTIFACT`）：`standalone_compile_artifacts.load_all()` 多线程 `AOTCompiledArtifact.deserialize` → `reconstruct_serializable_fn_from_mega_artifact` 按 submod 名重建 `PiecewiseBackend(graph=None, compiled_runnables=...)` → `wrap_with_cudagraph_if_needed` → `compile_execution_fn` 生成 callable → `make_copy_and_call`（若 `cudagraph_copy_inputs`）。
- **回退路径**：`deserialize_graph_module` 重建 FX 图 → lazy closure 在 `finalize_loading` 时构造 `VllmBackend` 并 `VllmBackend(graph, inputs)` 重跑拼接（不重编译，靠 Inductor 缓存命中）。

### StandaloneCompiledArtifacts 两级去重

```python
# caching.py:59 简化
def insert(self, submod_name, shape, entry: bytes):
    hex_digest = sha256(entry).hexdigest()
    self.submodule_bytes[f"{submod_name}_{shape}"] = hex_digest
    if hex_digest not in self.submodule_bytes_store:
        self.submodule_bytes_store[hex_digest] = entry  # 唯一份字节
        compilation_counter.num_compiled_artifacts_saved += 1
```

`load_all()`（`caching.py:119`）用 `ThreadPoolExecutor` 并发 `pickle.loads + AOTCompiledArtifact.deserialize`，按 `submodule_bytes_store.keys()` 顺序填 `loaded_submodule_store`。`__getstate__`/`__setstate__` 只序列化两级 dict，`loaded_submodule_store` 不持久化（装载时重建）。

### reconstruct_serializable_fn_from_mega_artifact

`caching.py:411`：

1. `standalone_compile_artifacts.load_all()`。
2. 按 `submodule_bytes` 的 `"{name}_{shape}"` 拆分，填 `compiled_callables[name][shape]`。
3. 构造 `VllmBackend`（仅用于 `compiler_manager` 与 `is_encoder`），`initialize_cache(dummy_cache_dir, disable_cache=True)`。
4. 校验 `piecewise_submod_names ⊆ graph_children`（旧缓存兜底）。
5. 对每个 submod 造 `PiecewiseBackend(graph=None, compiled_runnables=runnables, ...)` + `wrap_with_cudagraph_if_needed`。
6. 用 `execution_code`/`submod_names`/`consts` 调 `compile_execution_fn` 生成 callable（缺失则回落 `GraphPickler.loads`）。
7. `make_copy_and_call`（若 `cudagraph_copy_inputs`）→ `VllmSerializableFunction`。

## 与其它模块/系统配合

- [`backends.py`](backends.md)：`VllmBackend.__call__` 返回 `VllmSerializableFunction`；`collect_standalone_compile_artifacts()` 收集 piecewise 字节。
- [`codegen.py`](codegen.md)：`compile_execution_fn` 在重建 callable 时复用。
- [`piecewise_backend.py`](piecewise-backend.md)：`to_bytes()` 产出每 range 字节；`load_all_ranges()` 从 `compiled_runnables` 装载。
- [`compiler_interface.py`](compiler-interface.md)：`get_inductor_factors()` 参与 AOT hash；`AOTCompiledArtifact.deserialize` 由 `StandaloneCompiledArtifacts.load_all` 调用。
- [`decorators.py`](decorators.md)：`_try_load_aot_compiled_fn` 走 `torch.compiler.load_compiled_function` + `_verify_source_unchanged`；`finalize_loading` 回到 `VllmBackend`。
- [`配置-compilation`](../10-config/README.md)：`cache_dir` / `compile_cache_save_format` / `VLLM_USE_MEGA_AOT_ARTIFACT` / `VLLM_DISABLE_COMPILE_CACHE`。
- [`可观测性`](../16-observability/README.md)：`compile_debug_dump_path()` 供 depyf dump。

## 历史版本演进

- **v0.7**：`VllmSerializableFunction` 引入，序列化 FX 图 + example_inputs，落盘 `~/.cache/vllm/torch_compile_cache`。
- **v0.8**：`GraphPickler` + `reducer_override` 处理 sympy/FakeTensorMode；`patch_pytree_map_over_slice` 补 slice；`finalize_loading` 延迟 backend 构造。
- **v0.9**：`StandaloneCompiledArtifacts` 两级去重引入；`reconstruct_serializable_fn_from_mega_artifact` 与 `PiecewiseCompileInterpreter` 显式对偶（`caching.py:429` 注释要求同步维护）。
- **v0.10**：`VLLM_USE_MEGA_AOT_ARTIFACT` + `VLLM_USE_STANDALONE_COMPILE` 转 pseudo-默认；`aot_compile_hash_factors` + AOT 装载源码校验；多线程 `load_all`。
- **v0.11 / v0.12 / main**：`execution_code`/`consts` 路径成熟，热装载优先 codegen 而非 FX 反序列化；`SerializableCallable` 基类随 torch 演进。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [backends.md](backends.md) — `VllmSerializableFunction` 的产出方。
- [codegen.md](codegen.md) — `execution_code` 的生成与热装载复用。
- [piecewise-backend.md](piecewise-backend.md) — `to_bytes`/`load_all_ranges` 对偶。
- [compiler-interface.md](compiler-interface.md) — `AOTCompiledArtifact` 与 `get_inductor_factors`。
- [decorators.md](decorators.md) — AOT 装载与源码校验。
