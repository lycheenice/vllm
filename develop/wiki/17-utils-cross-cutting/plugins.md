# 插件入口（plugins）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 插件入口

本页覆盖 `vllm/plugins/`，描述 vLLM 通过 Python entry points 加载外部插件的机制，以及内置的 LoRA resolver 与 io_processor 子包。

## 是什么

### `vllm/plugins/__init__.py`：入口点加载器

定义四类插件组（entry point group），分别在不同进程/时机加载：

| 常量 | entry point group | 加载进程/时机 |
|---|---|---|
| `DEFAULT_PLUGINS_GROUP` | `vllm.general_plugins` | 所有进程（process0、EngineCore、worker） |
| `IO_PROCESSOR_PLUGINS_GROUP` | `vllm.io_processor_plugins` | 仅 process0 |
| `PLATFORM_PLUGINS_GROUP` | `vllm.platform_plugins` | 所有进程，在 `current_platform` 首次初始化时 |
| `STAT_LOGGER_PLUGINS_GROUP` | `vllm.stat_logger_plugins` | 仅 process0，async serve 时 |

核心函数：

- `load_plugins_by_group(group)`（`vllm/plugins/__init__.py:28`）：用 `importlib.metadata.entry_points(group=...)` 发现插件，按 `envs.VLLM_PLUGINS`（list）过滤——若该 env 设了，只加载名单内插件；未设则全加载。每个 entry point `.load()` 得到可调用对象，返回 `dict[name -> func]`。默认组用 DEBUG 日志，非默认组用 INFO。
- `load_general_plugins()`（`:69`）：加载 `DEFAULT_PLUGINS_GROUP` 并逐个执行；用模块级 `plugins_loaded` 标志保证每进程只加载一次，因此**插件必须可幂等加载**。

### `vllm/plugins/lora_resolvers/`：内置 LoRA 解析器

- `FilesystemResolver`（`filesystem_resolver.py:11`）：基于 `lora_cache_dir` 的文件系统 resolver，从目录加载 LoRA。
- `HfHubResolver`（`hf_hub_resolver.py:16`，继承 `FilesystemResolver`）：从 HuggingFace Hub 解析 LoRA；`register_hf_hub_resolver()`/`register_filesystem_resolver()` 提供注册入口。受 `VLLM_LORA_RESOLVER_CACHE_DIR`/`VLLM_LORA_RESOLVER_HF_REPO_LIST` 控制。

二者都实现 `vllm.lora.resolver.LoRAResolver` 接口（[LoRA 子系统](../12-lora/README.md)）。

### `vllm/plugins/io_processors/`：io_processor 插件

- `interface.py`：定义 `IOProcessor` 抽象接口（`parse_request` 等）。
- `__init__.py`：`has_io_processor(vllm_config, plugin_from_init)` 探测是否启用 io_processor，结合 `VllmConfig` 与插件名决策；`load_plugins_by_group(IO_PROCESSOR_PLUGINS_GROUP)` 加载。
- io_processor 对应 [tasks.md](tasks.md) 中的 `plugin` pooling task——`PoolingParams.verify` 在 `task=="plugin"` 时跳过自身校验，交由 io_processor 处理。

### Platform plugins 集成

`PLATFORM_PLUGINS_GROUP` 在 `vllm.platforms` 的 `current_platform` 首次访问且未初始化时触发加载，允许外部包替换/扩展平台实现（见 [平台子系统](../08-platforms/README.md)）。

## 为什么

- **零侵入扩展**：外部包通过 entry points 注册，无需改 vLLM 源码即可注入：自定义 LoRA 解析、自定义平台、自定义 stat logger、io_processor。
- **进程分流**：不同组在不同进程加载——worker 不需要 io_processor/stat logger，节省 import 与副作用。
- **`VLLM_PLUGINS` 白名单**：多插件环境下让用户精确控制启用哪些，避免环境里无关包被误加载。
- **幂等保证**：多进程 fork 后会重复触发 import，`plugins_loaded` 守护避免重复副作用。

## 怎么做

- **使用插件**：在第三方包的 `pyproject.toml` 声明 entry points：
  ```
  [project.entry-points."vllm.general_plugins"]
  my_plugin = "my_pkg.plugin:register"
  ```
  `register` 被调用一次，可全局 patch 或注册 resolver。
- **控制启用**：设 `VLLM_PLUGINS=my_plugin,other` 仅加载这些；不设则全部加载。
- **LoRA resolver**：调 `register_filesystem_resolver()`/`register_hf_hub_resolver()` 把内置 resolver 注册到 `LoRAResolverRegistry`。
- **io_processor**：实现 `IOProcessor` 接口，注册 entry point；请求 `task="plugin"` 时由其 `parse_request` 接管 `PoolingParams` 装配。

## 与其它模块/系统配合

- [LoRA](../12-lora/README.md)：`LoRAResolver`/`LoRAResolverRegistry` 是 resolver 插件的目标接口；`VLLM_LORA_RESOLVER_*` env。
- [平台](../08-platforms/README.md)：`PLATFORM_PLUGINS_GROUP` 在 `current_platform` 初始化时加载。
- [可观测性](../16-observability/README.md)：`STAT_LOGGER_PLUGINS_GROUP` 注入自定义 stat logger。
- [tasks.md](tasks.md)：`plugin` pooling task 与 io_processor 配套。
- [pooling-params.md](pooling-params.md)：`task=="plugin"` 时 `verify` 让位给 io_processor。
- [配置体系](../10-config/README.md)：`VllmConfig` 携带插件相关配置。
- [envs.md](envs.md)：`VLLM_PLUGINS`、`VLLM_LORA_RESOLVER_*`。

## 历史版本演进

- **v0.5–v0.6**：仅有 `vllm.general_plugins` 入口点，用于环境探针类扩展。
- **v0.7–v0.8**：随 LoRA resolver 抽象引入 `lora_resolvers/` 内置实现；`VLLM_PLUGINS` 白名单上线。
- **v0.9–v0.10**：`PLATFORM_PLUGINS_GROUP`、`STAT_LOGGER_PLUGINS_GROUP` 引入；io_processor 体系初现。
- **v0.11–main**：`io_processors/` 子包就位，`plugin` pooling task 打通；进程分流策略细化（待核实具体版本）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [tasks.md](tasks.md)、[pooling-params.md](pooling-params.md)
- [LoRA 子系统](../12-lora/README.md)
- [平台子系统](../08-platforms/README.md)
- [envs.md](envs.md)（`VLLM_PLUGINS`）
