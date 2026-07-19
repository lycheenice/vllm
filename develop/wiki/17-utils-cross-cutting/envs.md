# 环境变量与导入期覆盖（envs / env_override）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 环境变量

本页覆盖 `vllm/envs.py`（约 290 个环境变量的中央注册表）与 `vllm/env_override.py`（在 `import torch` 前后修改运行环境的副作用模块）。二者共同决定 vLLM 进程的"全局开关面"。

## 是什么

### `vllm/envs.py`：懒求值环境变量注册表

- 顶部用 `if TYPE_CHECKING:` 区块（`vllm/envs.py:14`）集中声明所有变量的**类型与默认值**，供 IDE/mypy 静态检查使用；运行期真实值由 `environment_variables` 字典（`vllm/envs.py:522` 起，`# --8<-- [start:env-vars-definition]`/`[end:...]` 标记供文档抽取）中的 `lambda` / 工厂函数提供。
- 模块级 `__getattr__`（`vllm/envs.py:1972`）拦截 `vllm.envs.VLLM_XXX` 访问，从字典取出对应工厂函数并执行；`enable_envs_cache()`（`vllm/envs.py:1990`）在服务初始化后用 `functools.cache` 包裹 `__getattr__` 并预取所有键，把热路径上的 `os.getenv` 压成 O(1) 字典查表。
- 辅助函数：`env_with_choices`/`env_list_with_choices`/`env_set_with_choices`（`vllm/envs.py:351`/`396`/`451`）做枚举校验；`get_vllm_port`（`:469`）解析 `VLLM_PORT`（URI 形态时给出 Kubernetes 友好报错）；`get_env_or_set_default`（`:498`）允许"首次读时回写默认值"。
- `validate_environ(hard_fail)`（`vllm/envs.py:2035`）扫描 `os.environ` 中所有 `VLLM_` 前缀变量，对未注册项告警或抛错。
- `compile_factors()`（`vllm/envs.py:2044`）把全部已知环境变量（减去 `ignored_factors` 白名单）规范化后作为 torch.compile 缓存键，确保各 worker 编译产物一致。

### `vllm/env_override.py`：导入期 monkeypatch 与 CUDA 兼容

- 模块顶部 `_maybe_set_cuda_compatibility_path()`（`vllm/env_override.py:39`）在 `import torch` **之前**执行：读 `VLLM_ENABLE_CUDA_COMPATIBILITY`/`VLLM_CUDA_COMPATIBILITY_PATH`，找到 `cuda-XX/compat` 目录后将其前置到 `LD_LIBRARY_PATH`，实现"新驱动 + 旧 CUDA runtime"的前向兼容（仅数据中心 GPU 支持）。
- `import torch` 之后设置若干全局环境（`vllm/env_override.py:101` 起）：`PYTORCH_NVML_BASED_CUDA_CHECK=1`（避免误触 CUDA 初始化，#15951）、`TORCHINDUCTOR_COMPILE_THREADS=1`（#10480/#10619）、`TRITON_CACHE_AUTOTUNING=1`、`TILELANG_CLEANUP_TEMP_FILES=1`。
- 针对 PyTorch 2.9 Inductor 的若干 monkeypatch：`memory_plan_reuse_patched`（`#165514`）、`get_graph_partition_signature_patched`（`#165815`）、`Scheduler.should_partition_patched`（`vllm/env_override.py:360`）、`get_raw_stream` 补丁（`:468`）；针对 torch ≥2.11 的 `constrain_to_fx_strides`、`GraphCaptureOutput.get_runtime_env`、`FxGraphCache` pickler、`CppVecKernel.indirect_assert` 等也在此打补丁。

## 为什么

- **集中化**：所有 `VLLM_*` 变量在一处声明、文档化、类型化，避免散落各模块导致"设了不生效/拼错不报错"。
- **懒求值 + 缓存**：worker 进程每个前向都读 env 会放大开销；缓存后一次性固化，并保证同进程内值稳定。
- **导入期副作用必须早**：CUDA compat 改 `LD_LIBRARY_PATH` 必须在 torch dlopen CUDA 之前；Inductor 补丁必须在首图编译前装好，因此 `env_override.py` 被 `vllm/__init__.py` 早期导入。
- **版本漂移容忍**：vLLM 主线常领先 PyTorch release，靠 monkeypatch 在未合入上游的修复落地前先行适配。

## 怎么做

### 读环境变量的正确姿势

```python
import vllm.envs as envs
if envs.VLLM_USE_PRECOMPILED:           # 走 __getattr__ -> 工厂 -> (cache)
    ...
```

- 仅在测试等需要"读后改"场景用 `os.environ`；正式代码一律走 `vllm.envs`。
- 新增变量：在 `TYPE_CHECKING` 区块声明类型+默认，再在 `environment_variables` 字典加 `lambda: os.getenv(...)` 或工厂；运行期自动可用。
- 想校验枚举：用 `env_with_choices("VLLM_XX", default, choices)` 包一层。

### 关键环境变量分组

下面按功能分组枚举常被运维/开发者调整的变量（默认值取自 `vllm/envs.py` 的 `TYPE_CHECKING` 与 `environment_variables` 区块）。完整列表见源码 `environment_variables` 字典，本表只列高频项；标 `(待核实)` 表示默认值随平台/版本变化。

#### 1. 构建/安装（build & install）

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_USE_PRECOMPILED` | `False` | 安装时跳过 C++ 编译，直接用预编译 wheel（AGENTS.md 推荐 `=1`） |
| `VLLM_USE_PRECOMPILED_RUST` | `False` | 同上，针对 Rust 前端 |
| `VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX` | `False` | 跳过版本后缀匹配 |
| `VLLM_DOCKER_BUILD_CONTEXT` | `False` | Docker 构建上下文标志 |
| `VLLM_TARGET_DEVICE` | `cuda` | 编译目标设备（`cuda`/`rocm`/`cpu`/`xpu`/`tpu`...） |
| `VLLM_MAIN_CUDA_VERSION` | `13.0` | 主分支 CUDA 版本基线 |
| `MAX_JOBS` / `NVCC_THREADS` | `None` | 并行编译作业数 / NVCC 线程数 |
| `CMAKE_BUILD_TYPE` | `None` | `Debug`/`Release`/`RelWithDebInfo` |
| `VERBOSE` | `False` | 构建详细日志 |
| `VLLM_USE_RUST_FRONTEND` | `False` | 启用 Rust 实现的前端（`VLLM_RUST_FRONTEND_PATH=auto` 自动解析路径） |

#### 2. 路径/缓存/网络

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | 全局缓存根（`XDG_CACHE_HOME` 可覆盖） |
| `VLLM_CONFIG_ROOT` | `~/.config/vllm` | 全局配置根（`XDG_CONFIG_HOME` 可覆盖） |
| `VLLM_ASSETS_CACHE` | `$VLLM_CACHE_ROOT/assets` | 资产缓存 |
| `VLLM_MEDIA_CACHE` | `""` | 多模态媒体缓存目录（空则在内存） |
| `VLLM_MEDIA_CACHE_MAX_SIZE_MB` | `5120` | 媒体缓存上限 |
| `VLLM_MEDIA_CACHE_TTL_HOURS` | `24` | 媒体缓存 TTL |
| `VLLM_MEDIA_FETCH_MAX_RETRIES` | `3` | 媒体拉取重试次数（被 `connections.py` 使用） |
| `VLLM_HOST_IP` / `VLLM_PORT` | `""` / `None` | 绑定 IP / 端口；`VLLM_PORT` 可被 K8s 注入 URI |
| `VLLM_RPC_BASE_PATH` | `tempfile.gettempdir()` | EngineCore ZMQ IPC 基目录 |
| `VLLM_MODEL_REDIRECT_PATH` | `None` | 模型名重定向到本地路径 |
| `VLLM_API_KEY` | `None` | API server 鉴权密钥 |
| `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL` | `None` | S3 对象存储凭据 |

#### 3. 日志/可观测

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_CONFIGURE_LOGGING` | `True` | 是否由 vLLM 配置 logging |
| `VLLM_LOGGING_LEVEL` | `INFO` | 日志级别（大写） |
| `VLLM_LOGGING_PREFIX` | `""` | 日志前缀 |
| `VLLM_LOGGING_STREAM` | `ext://sys.stdout` | 日志输出流 |
| `VLLM_LOGGING_CONFIG_PATH` | `None` | 自定义 logging 配置文件 |
| `VLLM_LOGGING_COLOR` | `auto` | 颜色 `auto`/`always`/`never`（受 `NO_COLOR` 影响） |
| `VLLM_LOG_STATS_INTERVAL` | `10.0` | 统计日志间隔 |
| `VLLM_LOG_BATCHSIZE_INTERVAL` | `-1` | `forward_context` 批量耗时日志间隔（负数关闭） |
| `VLLM_TRACE_FUNCTION` | `0` | 函数调用追踪开关 |
| `VLLM_USAGE_STATS_SERVER` | `https://stats.vllm.ai` | 遥测上报地址 |
| `VLLM_NO_USAGE_STATS`/`VLLM_DO_NOT_TRACK` | `False` | 关闭遥测 |
| `VLLM_DEBUG_LOG_API_SERVER_RESPONSE` | `False` | 记录 API server 响应 |
| `VLLM_CUSTOM_SCOPES_FOR_PROFILING`/`VLLM_NVTX_SCOPES_FOR_PROFILING` | `False` | profiler 自定义/NVTX scope |

#### 4. 引擎/进程拓扑

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `True` | v1 默认多进程 EngineCore |
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` | worker 多进程启动方式（`fork`/`spawn`） |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | `60` | 单步前向超时 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `600` | 引擎就绪超时 |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | `False` | EngineCore 死后 API server 是否保活 |
| `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` | `128` | 输出处理分块大小 |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `300` | `execute_model` 超时 |
| `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` | `5` | worker 关闭超时 |
| `VLLM_MAX_N_SEQUENCES` | `16384` | 最大序列数上限 |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | `5` | API server keep-alive 超时 |
| `VLLM_SERVER_DEV_MODE` | `False` | API server 开发模式 |

#### 5. 编译 / AOT / CUDAGraph

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_USE_AOT_COMPILE` | 动态（torch≥2.10 且未禁缓存时为 1） | AOT 编译 |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | 动态（torch≥2.12 且 AOT 开） | Mega AOT 工件 |
| `VLLM_FORCE_AOT_LOAD` | `False` | 强制 AOT 加载 |
| `VLLM_USE_BYTECODE_HOOK` | `True` | 字节码 hook |
| `VLLM_DISABLE_COMPILE_CACHE` | `False` | 禁用编译缓存 |
| `VLLM_USE_STANDALONE_COMPILE` | `True` | standalone compile |
| `VLLM_ENABLE_PREGRAD_PASSES` | `True` | grad 前 pass |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `False` | breakable cudagraph（见 [编译](../09-compilation-ir/README.md)） |
| `VLLM_ENABLE_CUDAGRAPH_GC` | `False` | cudagraph 后显式 GC |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | `binary` | `binary`/`unpacked` |
| `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` | `True` | Inductor max-autotune |
| `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING` | `True` | Inductor coordinate descent |
| `VLLM_USE_RUST_FRONTEND` | `False` | Rust 前端 |
| `VLLM_DEBUG_DUMP_PATH` / `VLLM_PATTERN_MATCH_DEBUG` | `None` | 调试产物路径 |
| `VLLM_LOG_MODEL_INSPECTION` | `False` | 打印模型树（见 [model-inspection.md](model-inspection.md)） |

#### 6. 设备/平台专用

CUDA / 通用：
- `VLLM_FLOAT32_MATMUL_PRECISION`=`highest`（`highest`/`high`/`medium`）
- `VLLM_BATCH_INVARIANT`=`False`（batch-invariant 模式）
- `VLLM_GPU_SYNC_CHECK`=`None`（`warn`/`error`，跨流同步检测）
- `VLLM_ENABLE_CUDA_COMPATIBILITY`/`VLLM_CUDA_COMPATIBILITY_PATH`（由 `env_override` 消费）
- `VLLM_CUDART_SO_PATH` / `VLLM_NCCL_SO_PATH` / `VLLM_NCCL_INCLUDE_PATH` / `VLLM_DISABLE_PYNCCL`
- `VLLM_USE_NCCL_SYMM_MEM`=`False`、`VLLM_ALLREDUCE_USE_SYMM_MEM`=`True`、`VLLM_ALLREDUCE_USE_FLASHINFER`=`False`

ROCm / AITER（节选，完整约 30 项见 `vllm/envs.py:121` 区块）：
- `VLLM_ROCM_USE_AITER`=`False`（总开关）
- `VLLM_ROCM_USE_AITER_CUSTOM_AR`/`_PAGED_ATTN`/`_LINEAR`/`_MOE`/`_RMSNORM`/`_MLA`/`_MHA`... 各算子开关
- `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE`、`VLLM_ROCM_FP8_PADDING`、`VLLM_ROCM_MOE_PADDING`
- `VLLM_ROCM_QUICK_REDUCE_*`（quick reduce 量化/阈值族）
- `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`、`VLLM_ROCM_FP8_MFMA_PAGE_ATTN`

CPU：
- `VLLM_CPU_KVCACHE_SPACE`=`0`、`VLLM_CPU_OMP_THREADS_BIND`=`auto`
- `VLLM_CPU_NUM_OF_RESERVED_CPU`、`VLLM_CPU_SGL_KERNEL`、`VLLM_CPU_ATTN_SPLIT_KV`=`True`
- `VLLM_ZENTORCH_WEIGHT_PREPACK`、`VLLM_CPU_INT4_W4A8`

XPU：
- `VLLM_XPU_ENABLE_XPU_GRAPH`、`VLLM_XPU_USE_SAMPLER_KERNEL`

TPU：
- `VLLM_TPU_BUCKET_PADDING_GAP`、`VLLM_TPU_MOST_MODEL_LEN`、`VLLM_TPU_USING_PATHWAYS`

#### 7. 分布式 / 并行 / KV 迁移

- `VLLM_DP_RANK`/`VLLM_DP_RANK_LOCAL`/`VLLM_DP_SIZE`/`VLLM_DP_MASTER_IP`/`VLLM_DP_MASTER_PORT`（DP）
- `VLLM_RANDOMIZE_DP_DUMMY_INPUTS`、`VLLM_RAY_DP_PACK_STRATEGY`=`strict`、`VLLM_RAY_DP_PLACEMENT_NODE_IPS`
- `VLLM_RAY_PER_WORKER_GPUS`=`1.0`、`VLLM_RAY_BUNDLE_INDICES`
- `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`/`VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY`
- `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE`=`auto`、`VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM`、`VLLM_USE_RAY_WRAPPED_PP_COMM`、`VLLM_USE_RAY_V2_EXECUTOR_BACKEND`
- `VLLM_DISTRIBUTED_USE_SPLIT_GROUP`、`VLLM_USE_OINK_OPS`
- `VLLM_SKIP_P2P_CHECK`、`VLLM_GPU_NIC_PCIE_MAPPING`/`VLLM_NIC_SELECTION_VARS`
- `VLLM_DEEPEP_BUFFER_SIZE_MB`、`VLLM_DEEPEP_*`（DeepEP 族）
- `VLLM_NIXL_SIDE_CHANNEL_HOST`/`_PORT`、`VLLM_NIXL_EP_MAX_NUM_RANKS`
- `VLLM_MOONCAKE_BOOTSTRAP_PORT`/`_STORE_TIER_LOG`/`_LOAD_RECV_THREADS`/`_DISK_STAGING_USABLE_RATIO`/`_ABORT_REQUEST_TIMEOUT`、`MOONCAKE_PREFERRED_SEGMENT`/`_REQUESTER_LOCAL_HOSTNAME`
- `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH`/`_DRAIN_REQUESTS`（弹性 EP）
- `VLLM_DBO_COMM_SMS`=`20`（DBO microbatch）

#### 8. 内核/算子选择（节选）

- `VLLM_USE_DEEP_GEMM`=`True`、`VLLM_MOE_USE_DEEP_GEMM`、`VLLM_USE_DEEP_GEMM_E8M0`、`VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES`、`VLLM_DEEP_GEMM_WARMUP`=`relax`
- `VLLM_DEEPEPLL_NVFP4_DISPATCH`
- `VLLM_USE_FUSED_MOE_GROUPED_TOPK`=`True`、`VLLM_MOE_SKIP_PADDING`
- `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`、`VLLM_USE_FLASHINFER_MOE_INT4`、`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`、`VLLM_FLASHINFER_ALLREDUCE_BACKEND`、`VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`、`VLLM_HAS_FLASHINFER_CUBIN`
- `VLLM_USE_TRITON_AWQ`、`VLLM_MARLIN_USE_ATOMIC_ADD`、`VLLM_MARLIN_INPUT_DTYPE`
- `VLLM_HUMMING_*`（Humming 在线量化族）
- `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD`
- `VLLM_USE_FLASHINFER_SAMPLER`=`True`
- `VLLM_KV_CACHE_LAYOUT`（`NHD`/`HND`）、`VLLM_SSM_CONV_STATE_LAYOUT`（`SD`/`DS`）
- `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`、`VLLM_DISABLE_SHARED_EXPERTS_STREAM`、`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`、`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`
- `VLLM_DISABLED_KERNELS`（list）、`VLLM_TUNED_CONFIG_FOLDER`

#### 9. 多模态

- `VLLM_IMAGE_FETCH_TIMEOUT`/`VLLM_VIDEO_FETCH_TIMEOUT`/`VLLM_AUDIO_FETCH_TIMEOUT`
- `VLLM_MEDIA_URL_ALLOW_REDIRECTS`、`VLLM_MEDIA_LOADING_THREAD_COUNT`=`8`
- `VLLM_MAX_IMAGE_PIXELS`、`VLLM_VIDEO_LOADER_BACKEND`=`opencv`、`VLLM_MEDIA_CONNECTOR`=`http`
- `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`/`_DECODE_DURATION_S`/`_PREPROCESS_WORKERS`
- `VLLM_MM_HASHER_ALGORITHM`=`blake3`、`VLLM_ASSETS_CACHE_MODEL_CLEAN`

#### 10. 结构化输出 / 工具调用 / LoRA

- `VLLM_XGRAMMAR_CACHE_MB`、`VLLM_REGEX_COMPILATION_TIMEOUT_S`
- `VLLM_V1_USE_OUTLINES_CACHE`
- `VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS`、`VLLM_ENFORCE_STRICT_TOOL_CALLING`=`True`、`VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY`
- `VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS`、`VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS`、`VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT`
- `VLLM_PLUGINS`（list，过滤 `vllm.general_plugins` 入口点）
- `VLLM_LORA_RESOLVER_CACHE_DIR`/`_HF_REPO_LIST`、`VLLM_ALLOW_RUNTIME_LORA_UPDATING`、`VLLM_LORA_DISABLE_PDL`、`VLLM_LORA_ENABLE_DUAL_STREAM`

#### 11. 杂项/调优

- `VLLM_ALLOW_LONG_MAX_MODEL_LEN`、`VLLM_SKIP_MODEL_NAME_VALIDATION`、`VLLM_ALLOW_INSECURE_SERIALIZATION`
- `VLLM_MSGPACK_ZERO_COPY_THRESHOLD`=`256`、`VLLM_MQ_MAX_CHUNK_BYTES_MB`=`16`
- `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`、`VLLM_ENABLE_RESPONSES_API_STORE`
- `VLLM_COMPUTE_NANS_IN_LOGITS`、`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`=`True`
- `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY`/`_DISABLE_UVA`、`VLLM_WSL2_ENABLE_PIN_MEMORY`
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`、`VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE`=`True`
- `VLLM_USE_V2_MODEL_RUNNER`、`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`、`VLLM_DEBUG_MFU_METRICS`
- `VLLM_MC_*`、`VLLM_SYSTEM_START_DATE`、`VLLM_LOOPBACK_IP`
- `Q_SCALE_CONSTANT`=`200`、`K_SCALE_CONSTANT`=`200`、`V_SCALE_CONSTANT`=`100`
- `VLLM_DEBUG_WORKSPACE`、`VLLM_GC_DEBUG`、`VLLM_USE_LAYERNAME`=`True`
- `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME`、`VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE`

### `env_override.py` 的 CUDA 兼容路径解析顺序

1. 用户显式 `VLLM_CUDA_COMPATIBILITY_PATH`；
2. `$CONDA_PREFIX/cuda-compat`；
3. 由 torch `version.py`（不 import torch）读出 cuda 版本，拼 `/usr/local/cuda-<ver>/compat`。

任一存在且为目录即前置到 `LD_LIBRARY_PATH`，否则跳过（不报错）。

## 与其它模块/系统配合

- [配置体系](../10-config/README.md)：`VllmConfig` 各子配置的默认值大量来自 `envs`；`compile_factors()` 为编译缓存键。
- [平台](../08-platforms/README.md)：`current_platform` 在多处读 `envs` 决定后端；`import_kernels()` 触发 `_custom_ops`/`_aiter_ops`/`_xpu_ops` 注册。
- [编译与 IR](../09-compilation-ir/README.md)：`env_override.py` 的 Inductor monkeypatch 是 torch.compile 集成的前提；`forward_context` 的 `all_moe_layers` 冷启动优化受 `fast_moe_cold_start` 控制。
- [引擎核心](../01-engine-core/README.md)：`VLLM_ENABLE_V1_MULTIPROCESSING`、`VLLM_RPC_BASE_PATH` 决定进程拓扑与 IPC 路径。
- [可观测性](../16-observability/README.md)：日志、profiler、metrics 开关均落在本子系统。
- [多模态](../11-multimodal/README.md)：媒体拉取超时/重试经 `connections.HTTPConnection` 消费。

## 历史版本演进

- **v0.5–v0.6**：`envs.py` 体量较小，变量分散在各模块；无缓存机制。
- **v0.7–v0.8**：引入 `__getattr__` 懒求值与 `environment_variables` 注册表；`env_override.py` 雏形（CUDA compat）从 `_custom_op` 抽出。
- **v0.9–v0.10**：随 torch.compile 大规模落地，Inductor monkeypatch 数量陡增；`enable_envs_cache()`/`disable_envs_cache()` 上线；`VLLM_USE_AOT_COMPILE`/`_MEGA_AOT_ARTIFACT` 引入。
- **v0.11–v0.12/main**：DeepEP/NIXL/Mooncake/Elastic EP 等分布式变量族扩张；ROCm AITER 子开关细化为数十项；`compile_factors()` 引入以稳定多 worker 编译缓存；`VLLM_USE_V2_MODEL_RUNNER`、`VLLM_USE_BREAKABLE_CUDAGRAPH` 等实验开关就位。具体每个变量引入版本（待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [forward-context.md](forward-context.md)：前向上下文如何读 envs 控制批量日志与 DP。
- [custom-ops.md](custom-ops.md)：算子注册受 `VLLM_DISABLED_KERNELS` 等影响。
- [connections.md](connections.md)：HTTP 重试次数取自 `VLLM_MEDIA_FETCH_MAX_RETRIES`。
- 顶层 [Wiki 首页](../README.md) 的"写作与链接规范"。
