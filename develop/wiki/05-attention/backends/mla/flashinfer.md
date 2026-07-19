# FlashInfer MLA backend

[← Wiki 首页](../../../README.md) > [注意力](../../README.md) > [Backend 列表](../README.md) > [← MLA 首页](../README.md) > FlashInfer MLA

> 源码：`vllm/v1/attention/backends/mla/flashattn_mla.py`、`flashinfer_mla_sparse.py`、`flashinfer_mla_sparse_sm120.py`、`prefill/flashinfer.py`

## 是什么

基于 FlashInfer 的 MLA backend，覆盖 Blackwell（SM100）/SM120 dense + sparse + prefill：

- `FlashInferMLABackend` / `FlashInferMLAImpl`（`flashinfer_mla.py:54/124`）—— Blackwell dense MLA，用 `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`。
- `FlashInferMLASparseTRTLLMBackend` / `FlashInferMLASparseSM120Backend`（`flashinfer_mla_sparse.py:71/145`）+ `FlashInferMLASparseImpl`（`flashinfer_mla_sparse.py:366`）—— DeepSeek V4 sparse MLA。
- `FlashInferMLASparseSM120Impl`（`flashinfer_mla_sparse_sm120.py:32`）—— SM120 变体（复用 metadata）。
- `FlashInferPrefillBackend`（`prefill/flashinfer.py:36`）—— Blackwell MLA prefill。

`FlashInferMLABackend` 在 Blackwell MLA 优先级**第一**（`platforms/cuda.py:117`）。

## 为什么

- trtllm-gen MLA decode kernel 在 Blackwell 上常胜过 CUTLASS/FlashMLA。
- `lse_base_on_e=False`：trtllm-gen MLA 用 **base 2** LSE，与其它 MLA 不同，DCP 合并 kernel 必须按此分支（见 [backend 抽象层](../../backend-abstraction.md)）。
- 支持 FP8 KV cache，sparse 表中 FP8 KV 第一优先级。
- `QueryLenSupport.UNIFORM`：decode 支持等长多 token（spec decode）。
- SM120 专用 sparse 变体适配新一代 GPU。

## 怎么做

### FlashInferMLABackend 能力

| 项 | 值 | 位置 |
|----|----|------|
| dtypes | fp16/bf16 | `flashinfer_mla.py:55` |
| kv dtype | auto/fp16/bf16/fp8/fp8_e4m3 | `flashinfer_mla.py:56` |
| block sizes | `[32, 64]` | `flashinfer_mla.py:65` |
| `supports_compute_capability` | `major == 10`（仅 Blackwell） | `flashinfer_mla.py:89` |
| cudagraph | `UNIFORM_BATCH` | `flashinfer_mla.py:50` |
| `query_len_support` | `UNIFORM` | `flashinfer_mla.py:51` |
| stride_order | blocks 优先 `(1,0,2,3)` | `flashinfer_mla.py:69` |

workspace：`FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE=128MB`（无 LSE）/ `256MB`（带 LSE，`flashinfer_mla.py:30-31`），`_get_workspace_buffer(return_lse)` 按需扩容。

### sparse 变体

`_FlashInferMLASparseBackendBase`（`flashinfer_mla_sparse.py:47`）公共基类，`get_supported_head_sizes = [576]`。

- `FlashInferMLASparseTRTLLMBackend`（`:71`）—— SM100/通用，trtllm-gen sparse。
- `FlashInferMLASparseSM120Backend`（`:145`）—— SM120，block sizes 不同。
- `FlashInferMLASparseSM120Impl`（`flashinfer_mla_sparse_sm120.py:32`）—— SM120 kernel 实现。
- cudagraph `UNIFORM_BATCH`（`flashinfer_mla_sparse.py:272`）。

`sparse_utils.triton_filter_and_convert_dcp_index`、`triton_convert_req_index_to_global_index` 做 DCP/sparse index 转换。

### prefill

`FlashInferPrefillBackend`（`prefill/flashinfer.py:36`）用 `BatchPrefillWithRaggedKVCacheWrapper`，Blackwell 优先级第 3（在 FlashAttn、TrtllmRagged 后）。`supported_mla_dimensions` 列出支持的 dims 组合。`_DEFAULT_NUM_CHUNKS=32`。

## 与其它模块/系统配合

- **ops**：workspace buffer、`mla/sparse_utils.py`、`mla/indexer.py`。
- **selector**：Blackwell MLA 第一；sparse FP8 KV 第一 / 低 head 数第一，见 [selector](../../selector.md)。
- **DCP**：`lse_base_on_e=False` 必须与 `ops/common.py`/`dcp_alltoall.py` 的 `IS_BASE_E` 分支一致。
- **workspace manager** 与全局 buffer：`_fi_workspace` 单例。
- **MLA 公共层**：见 [MLA 总览](../README.md)。

## 历史版本演进

- **v0.8.x**：`FlashInferMLABackend` 引入（trtllm-gen MLA decode，Blackwell），`lse_base_on_e=False` 确立。
- **v0.9.x**：`FlashInferMLASparseTRTLLMBackend`（DeepSeek V4）；FP8 KV；DCP；`QueryLenSupport.UNIFORM`。
- **v0.10.x（当前）**：`FlashInferMLASparseSM120Backend` + `flashinfer_mla_sparse_sm120.py`（SM120）；`FlashInferPrefillBackend` 独立化；sparse 头数自适应优先级（`platforms/cuda.py:106-115`）。

---

[← 返回 MLA 首页](../README.md)

## 参见

- [FlashInfer 标准 backend](../flashinfer.md)
- [MLA 总览](../README.md)
- [MLA CUTLASS](cutlass.md)
- [MLA FlashMLA](flashmla.md)
- [底层 ops](../../ops.md)
