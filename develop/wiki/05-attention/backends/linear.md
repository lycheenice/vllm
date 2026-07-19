# Linear attention backend

[← Wiki 首页](../../README.md) > [注意力](../../README.md) > [Backend 列表](../README.md) > Linear

> 源码：`vllm/v1/attention/backends/linear_attn.py`

## 是什么

`LinearAttentionBackend`（`linear_attn.py:22`）是为**线性注意力**模型准备的 SSM 族 backend，`is_ssm()` 返回 True，走 `MambaSpec`。当前有两个：

- `LinearAttentionBackend` —— 基础线性注意力。
- `BailingLinearAttentionBackend`（`linear_attn.py:97`）—— 百灵（Bailing）系列模型的线性注意力变体，支持 spec decode metadata 与 `UNIFORM_BATCH` cudagraph。

两者共享 `LinearAttentionMetadata` / `LinearAttentionMetadataBuilder` 基础结构，Bailing 扩展字段。

## 为什么

线性注意力/SSM 变体与标准 softmax attention 执行模型不同：维护递归 state 而非 KV cache，prefill 含状态更新。把它独立成 backend（而非塞进 Mamba）是因为：

- 算子层不同（无 conv1d，或 state 更新公式不同）。
- 但 metadata/batch 拆分逻辑与 Mamba 高度相似，故复用 `split_decodes_and_prefills`、`mamba_get_block_table_tensor` 等 utils。
- Bailing 变体需要 spec decode checkpoint（`num_accepted_tokens`）与统一 batch cudagraph，故单独派生。

## 怎么做

### 能力要点

| 项 | 值 | 位置 |
|----|----|------|
| `is_ssm()` | True | `linear_attn.py:32` |
| `reorder_batch_threshold` | 1（base） | `linear_attn.py:49` |
| `_cudagraph_support`（base） | `UNIFORM_SINGLE_TOKEN_DECODE` | `linear_attn.py:51` |
| Bailing cudagraph | `UNIFORM_BATCH`（override `get_cudagraph_support`） | `linear_attn.py:120` |
| `supports_spec_decode_metadata` | Bailing True | `linear_attn.py:116` |

### LinearAttentionMetadata

`linear_attn.py:37`：

- `num_prefills` / `num_prefill_tokens` / `num_decodes` / `num_decode_tokens`。
- `query_start_loc` / `seq_lens`。
- `state_indices_tensor`（shape `[batch,]`）。

Bailing 扩展（`linear_attn.py:107`）：`state_indices_tensor_d` / `_p`、`num_accepted_tokens`、`query_start_loc_d`，用于 spec decode checkpoint 选择。

### builder build

`build`（`linear_attn.py:63`）读 `mamba_cache_mode`（base 版 `linear_attn.py:76`），拆 prefill/decode。Bailing 的 `build`（`linear_attn.py:183`）额外处理 spec decode 与 chunk。

## 与其它模块/系统配合

- **utils**：`split_decodes_and_prefills`、`mamba_get_block_table_tensor`、`PAD_SLOT_ID`，见 [utils](utils.md)。
- **Mamba backend**：共享 SSM 选择路径 `get_mamba_attn_backend(MambaAttentionBackendEnum.LINEAR)`，见 [Mamba](mamba.md)。
- **KV 管理**：`MambaSpec`，见 [引擎核心-KV 管理](../../01-engine-core/kv-cache-management/README.md)。
- **模型**：百灵系列、其它线性注意力模型。

## 历史版本演进

- **v0.9.x**：`LinearAttentionBackend` 引入，基础 `UNIFORM_SINGLE_TOKEN_DECODE` cudagraph。
- **v0.10.x（当前）**：`BailingLinearAttentionBackend` 加入，支持 spec decode metadata 与 `UNIFORM_BATCH` cudagraph，为百灵模型定制。

---

[← 返回注意力首页](../../README.md)

## 参见

- [Mamba/SSM backend](mamba.md)
- [Utils](utils.md)
- [selector](../selector.md)
