# CompilerInterface 与 Inductor 适配器

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > CompilerInterface

源码：`vllm/compilation/compiler_interface.py`（约 810 行）

## 是什么

定义 vLLM 编译后端的抽象基类与三个内置实现，把"把一个 FX Graph 编译成可调用对象 + 缓存句柄"封装成统一接口，供 [`CompilerManager`](backends.md) 调度。

- `CompilerInterface`（`compiler_interface.py:27`）：抽象接口，四个方法 `initialize_cache` / `compute_hash` / `compile` / `load`。
- `InductorStandaloneAdaptor`（`compiler_interface.py:251`）：PyTorch 2.8+ 推荐，走 `torch._inductor.standalone_compile`，`name="inductor_standalone"`。
- `InductorAdaptor`（`compiler_interface.py:449`）：旧路径（2.5–2.7），用 `compile_fx` + 大量 monkey-patch 拿到 cache key，`name="inductor"`。
- `EagerAdaptor`（`compiler_interface.py:796`）：直接返回原图，不做编译，`name="eager"`，仅计数。
- `AlwaysHitShapeEnv`（`compiler_interface.py:117`）：dummy shape env，让 Inductor code-cache 查找永远命中。
- 工具函数：`get_inductor_factors()`（`compiler_interface.py:170`，汇总系统/PyTorch/inductor/functorch 因子用于 hash）、`is_compile_cache_enabled()`（`compiler_interface.py:193`）、`set_inductor_config()` / `set_functorch_config()`、`trigger_inductor_lazy_init()`。

## 为什么

- **解耦"分片策略"与"后端实现"**：`VllmBackend` 只管把图切开，具体怎么交给 Inductor / eager / 厂商后端由 `CompilerInterface` 实现决定，`make_compiler()`（`backends.py:96`）按 `compilation_config.backend` 选择。
- **复用 Inductor 自带缓存**：vLLM 只跑一次 Dynamo 追踪，却要为多个 shape 各编译一次，且这些编译发生在 Dynamo 追踪上下文之外。Inductor 默认会因缺少 shape env 而拒绝缓存命中，`AlwaysHitShapeEnv` 让 `evaluate_guards_expression` 恒真、`produce_guards_expression` 返回空串，绕过该限制（`compiler_interface.py:117-156`）。
- **平滑升级 PyTorch**：`InductorAdaptor` 靠 patch `compiled_fx_graph_hash`/`FxGraphCache._get_shape_env`/`_check_can_cache` 拿 hash，依赖 torch 私有 API；`InductorStandaloneAdaptor` 用官方 `standalone_compile` + `CompiledArtifact.save/load`，无需 patch。通过 `VLLM_USE_STANDALONE_COMPILE` 切换。
- **AOT 与 mega artifact 支持**：`InductorStandaloneAdaptor.compile` 在 `VLLM_USE_MEGA_AOT_ARTIFACT` 时返回 `AOTCompiledArtifact`（不落盘单文件），交由 [`caching.py`](caching.py) 的 `StandaloneCompiledArtifacts` 去重存储。
- **原子写修复**：`_patch_standalone_compile_atomic_save()`（`compiler_interface.py:210`）为 torch < 2.10 回移上游 PR#162432，防止多进程并发编译写出损坏的缓存文件。

## 怎么做

### compile 契约

`compile()` 返回 `(compiled_callable, handle)`：

- `handle` 是可用于直接装载的句柄（字符串/文件路径），`None` 表示不支持缓存。
- `InductorStandaloneAdaptor`：编译后 `compiled_graph.save(path, format)`，handle = `(key, path)`；`load` 用 `CompiledArtifact.load(path)`。`save_format` 由 `compilation_config.compile_cache_save_format` 决定（`binary` / `unpacked`）。
- `InductorAdaptor`：通过 hijack `compiled_fx_graph_hash` 抓 `hash_str`，通过 hijack `compile_fx_inner` 抓 `file_path`，handle = `(hash_str, file_path)`；`load` 走 `FxGraphCache._lookup_graph` + `AlwaysHitShapeEnv`。
- `EagerAdaptor`：直接 `return graph, None`，仅累加 `num_eager_compiles`。

### InductorStandaloneAdaptor 关键流程

```python
# compiler_interface.py:280-414 简化
with pregrad_ctx, fake_mode_ctx:               # 跳过无用的 pre-grad pass（torch<2.12）/ 复用 FakeTensorMode
    compiled = standalone_compile(graph, example_inputs,
        dynamic_shapes="from_example_inputs" if single_size else "from_graph",
        options={"config_patches": current_config})
if use_aot:                                    # VLLM_USE_MEGA_AOT_ARTIFACT
    return compiled, None                      # 由 caching.py 统一序列化
compiled.save(path, format=self.save_format)   # 落盘
return compiled, (key, path)
```

`set_inductor_config()`（`compiler_interface.py:753`）在 `is_single_size()` 时按 `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` / `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING` 开启 triton kernel 调参。

### InductorAdaptor 的 patch 套路

`compile()` 内用 `ExitStack` 临时 patch：

1. `compiled_fx_graph_hash` → 抓 `hash_str`。
2. `FxGraphCache._get_shape_env` → 返回 `AlwaysHitShapeEnv`。
3. `AOTAutogradCache._get_shape_env`（torch 2.8+）→ 同上。
4. `FxGraphCache._check_can_cache` → 恒返回 `None`（强制可缓存）。
5. 关闭 remote cache + `enable_autograd_cache=False`（旧路径依赖）。
6. 清空 `TracingContext`（避免 FakeTensorMode 冲突，`compiler_interface.py:628`）。

> 注释明确写着 "TODO(zou3519): we're going to replace this all with standalone_compile sometime"，即 `InductorAdaptor` 是过渡实现。

## 与其它模块/系统配合

- [`CompilerManager`](backends.md)：唯一持有 `CompilerInterface` 实例的调度者，负责 load→compile→缓存句柄落盘（`vllm_compile_cache.py`）。
- [`VllmBackend`](backends.md)：`make_compiler()` 按配置选适配器；`configure_post_pass()` 把 `PostGradPassManager` 注入 `inductor_config[pass_key]`。
- [`caching.py`](caching.md)：`get_inductor_factors()` 参与 AOT hash；`VllmSerializableFunction` 序列化时调用 `collect_standalone_compile_artifacts()`。
- [`平台`](../08-platforms/README.md)：`compilation_config.backend` 不是 `inductor`/`eager` 时，走 `current_platform.get_compile_back()` 解析 qualname 实例化厂商后端。
- [`env_override`](../17-utils-cross-cutting/README.md)：`_apply_constrain_to_fx_strides_patch()` 在 compile() 开头调用，修正 FX stride 约束。

## 历史版本演进

- **v0.5–v0.7**：仅有 `InductorAdaptor`（旧 monkey-patch 路径），`EagerAdaptor` 作为 fallback。
- **v0.8**：`InductorStandaloneAdaptor` 引入（依赖 torch 2.8 `standalone_compile`），默认不开，`VLLM_USE_STANDALONE_COMPILE` 切换；`AlwaysHitShapeEnv` 增加 `var_to_hint_override` 以兼容 torch 2.11 的 `FxGraphHashDetails`。
- **v0.9**：`_patch_standalone_compile_atomic_save()` 回移上游原子写修复；`pregrad_ctx` patch 跳过无用的 pre-grad pass（torch<2.12 的冷启动 ~1s 开销）。
- **v0.10**：`VLLM_USE_MEGA_AOT_ARTIFACT` + AOT 路径成型，`InductorStandaloneAdaptor.compile` 区分 `use_aot` 分支返回 `AOTCompiledArtifact`。
- **v0.11 / v0.12 / main**：torch 2.10+ 不再 patch 原子写；`donate_graph_module=True`（torch 2.13+）；`FakeTensorMode` 复用 patch（torch issue #176562 workaround）。`InductorAdaptor` 仍保留以支持 torch 2.5–2.7（待核实最低版本）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [backends.md](backends.md) — `CompilerManager` 如何调用 `compile/load` 与缓存句柄。
- [caching.md](caching.md) — `VllmSerializableFunction` 与 mega-AOT 序列化。
- [decorators.md](decorators.md) — `init_backend()` 选择适配器的入口。
- [../10-config/README.md](../10-config/README.md) — `CompileBackend` / `compile_cache_save_format` 字段。
