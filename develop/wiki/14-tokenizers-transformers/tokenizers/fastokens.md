[← 分词与转换器](../README.md) > [分词器后端](README.md) > fastokens

# fastokens.py — VLLM_USE_FASTOKENS Rust BPE shim

## 是什么

`vllm/tokenizers/fastokens.py` 只有一个函数 `apply_fastokens_patch()`（`vllm/tokenizers/fastokens.py:20`）。当 `VLLM_USE_FASTOKENS=1` 时，`registry.get_tokenizer` 在加载任何 tokenizer 前调它一次（进程级幂等，`vllm/tokenizers/registry.py:186`）。

它做的事：

1. `import fastokens`，失败抛 `ImportError` 提示安装 `>=0.2.0`。
2. 用 `importlib.metadata.version` 校验版本，低于 `_MIN_FASTOKENS_VERSION = "0.2.0"` 抛错。
3. 调 `fastokens.patch_transformers()`：把 transformers 使用的 Rust BPE 后端换成 fastokens shim，并重绑 `tokenizers.decoders.DecodeStream` 让流式 detokenizer 接受 shim。

模块 docstring（`:1`）明确：patch"applies to any tokenizer mode that ends up loading an HF fast tokenizer (`hf`, `deepseek_v32`, `deepseek_v4`, …)"——即对 Mistral/TikToken 后端无作用（它们不经过 HF fast tokenizer 路径）。

## 为什么

- **性能**：fastokens 是 Rust 实现的更高效 BPE backend，对长 prompt 的 `apply_chat_template`+tokenize 能带来明显加速；线上高频服务尤其受益。
- **流式兼容**：fastokens shim 与 `tokenizers.decoders.DecodeStream` 的接口不完全一致，必须 `patch_transformers` 一起重绑，否则 `detokenize_incrementally` 之类的流式 decode 会失败。
- **进程级幂等**：多次设置 `VLLM_USE_FASTOKENS=1` 不会重复 patch；首次 import 后 `fastokens.patch_transformers()` 自身也是幂等的。
- **可选依赖**：fastokens 不在默认 requirements 中，env 未设时完全不 import，避免影响基础 install 体积。

## 怎么做

用户侧：

```bash
pip install 'fastokens>=0.2.0'
export VLLM_USE_FASTOKENS=1
vllm serve <model>
```

代码侧：除 `registry.get_tokenizer` 外，没有其它入口；patch 一旦生效，所有后续 `CachedHfTokenizer.from_pretrained`/`PreTrainedTokenizerFast.from_pretrained` 调用都自动使用新后端，无需逐个 tokenizer 包装。

与 `_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS` 路径兼容：那条路径用 `TokenizersBackend` 直接加载，仍然会经过 fastokens patch（待核实）。

## 与其它模块/系统配合

- **`tokenizers/registry.py`**：唯一调用点（`:186`）。`get_tokenizer` 在 `cached_resolve_tokenizer_args` 之前就 patch，确保后续任何后端的 HF fast 路径都被覆盖。
- **`tokenizers/hf.py`**：`CachedHfTokenizer` 与 `maybe_make_thread_pool` 不需要任何改动——它们包装的对象已是 patched 后端的产物。
- **`tokenizers/deepseek_v32.py` / `deepseek_v4.py`**：底层 `PreTrainedTokenizerFast.from_pretrained` 也会被 patch 覆盖（与 docstring 一致）。
- **detokenizer**：流式 detokenizer 通过 shim 的 `DecodeStream` 工作；fastokens 已重绑使兼容。

## 历史版本演进

- **v0.11（PR #41741 "tokenizer: Add fastokens support"）**：首次引入 env 与 patch 调用。
- **v0.11（PR #43168 "Rework fastokens integration"）**：把 patch 整合到 `get_tokenizer` 的统一前置步骤（之前的版本可能分散在多处），并加上最低版本校验与友好 ImportError。
- **main**：接口稳定；fastokens 上游若有新版本，本文件只需更新 `_MIN_FASTOKENS_VERSION`。

---

[← 返回分词器后端首页](README.md)

## 参见

- [registry.md](registry.md) — `VLLM_USE_FASTOKENS` 的判定与调用点。
- [hf.md](hf.md) — 默认受益的后端。
