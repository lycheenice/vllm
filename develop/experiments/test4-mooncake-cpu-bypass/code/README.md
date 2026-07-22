# test4 CPU 绕行 —— 开发设计与占位

> 状态:🚧 待开发。本目录放该实验的 **vLLM 代码改动**;完成后置于 `code/vllm/`,
> 由 `../config.env` 的 `VLLM_CODE_OVERRIDE=1` + `VLLM_PKG_PATH` 经 bind-mount 覆盖进容器。

## 问题
上游 `MooncakeConnector`(`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`)
是 **P2P GPU 直传**,没有像 NixlConnector 那样的 `kv_buffer_device=cpu` 开关。要做"CPU 绕行"
(KV 经宿主 DRAM 中转,而非 GPU 直传),需自行改动。

## 两条候选路线

### 路线 A:给 MooncakeConnector 加 host-staging(改动较大,最贴近 test3 语义)
- 在 save/load KV 时,先 `cudaMemcpyAsync` D2H 到 host pinned buffer,再由 mooncake 传输;
  对端 H2D 回显存。等价于 nixl 的 `kv_buffer_device=cpu` 路径。
- 参考 nixl 侧 `kv_buffer_device` 的实现:`vllm/config/kv_transfer.py`(字段定义)+
  NixlConnector 里对该字段的分支,把同样的 staging 逻辑移植到 mooncake connector。

### 路线 B:改用 MooncakeStoreConnector + DRAM 后端(改动较小,建议先试)
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py` 的
  `MooncakeStoreConnector` 用 `MooncakeDistributedStore` 作共享 KV 池,P/D 都读写该池。
- 若把 store 段配置为 **DRAM/CPU 内存**,则 KV 天然经宿主内存中转 —— 语义上就是"CPU 绕行"。
- 这条可能 **不需要改 vLLM 源码**,只需 mooncake store 的配置 + connector 名切换;
  若成立,应把它从 test4 拆出为 test4b(配置级),test4 保留真正需要改码的路线 A。

## 落地约定
1. 只拷 **需要改的文件** 到 `code/vllm/` 下与包内一致的相对路径(避免整包 bind-mount 拖慢)。
   例:`code/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`
   —— 对应 `config.env` 里 `VLLM_PKG_PATH` 下同一相对路径,单文件挂载即可。
2. 确认容器内 vllm 包路径:
   `docker exec <c> python -c "import vllm,os;print(os.path.dirname(vllm.__file__))"`,填回 `config.env`。
3. 先跑通(冒烟 + `correct_check` 与 base1 greedy 输出一致),再谈性能;与 **test3**(nixl+cpu)横向对比。

## 参考
- `../../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`(上游实现)
- `../../../examples/disaggregated/mooncake_connector/`(启动/proxy 示例)
- `../../vllm_pd_analysis_and_optimization_20260721.md`(L2/CPU offload 既有分析)
