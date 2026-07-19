# hasher.py · 多模态哈希器

[← Wiki 首页](../README.md) > [多模态](../README.md) > hasher

## 是什么

`vllm/multimodal/hasher.py` 定义 `MultiModalHasher`，把异构的多模态输入（`PIL.Image` / `np.ndarray` / `torch.Tensor` / `bytes` / 标量 / `MediaWithBytes` / 嵌套 list/dict）序列化为字节流并哈希，作为 [cache.md](cache.md) 与 v1 `EncoderCacheManager` 的缓存 key。算法可由 `VLLM_MM_HASHER_ALGORITHM` 环境变量切换（`blake3` 默认 / `sha256` / `sha512`，后者用于 FIPS 合规）。

## 为什么

多模态缓存命中要求"输入比特级一致性"判定。然而输入对象类型繁多：`PIL.Image` 同一像素可以有多种 `mode`/`palette`，`torch.Tensor` 的 `bfloat16` numpy 不支持，`np.ndarray` 可能非连续。直接 `pickle.dumps` 既慢又对等价输入产生不同 key（如同一图片 `convert("RGB")` 两次 palette 序列化可能不同）。`MultiModalHasher` 通过：

- **类型专属序列化**：每种类型走最稳的字节表示（如 PIL 取 `np.asarray` + mode + palette；Tensor bfloat16 view 成 uint8；ndarray 记 dtype/shape）。
- **结构化拼接**：用 `iter_item_to_bytes` 把嵌套 dict/list 展开为 `(路径 key bytes, value bytes)` 序列，路径前缀防止不同结构同字节碰撞。
- **PIL EXIF UUID 快速路径**：若图片带 `ImageID` UUID（部分数据集/采集管线植入），直接用 UUID bytes 跳过像素序列化。
- **`MediaWithBytes` 优先原始字节**：从 API 来的图若带 `original_bytes`，直接哈希字节而非 decode 后像素，更快且不受 PIL palette 影响。

哈希还参与 `mm_hash`（处理器缓存 / 跨 LoRA 共享）与 `identifier`（带 LoRA 前缀的编码器缓存 key），二者关系由 `InputProcessor._get_mm_identifier` 决定（详见 [v1-integration.md](v1-integration.md)）。

## 怎么做

### 算法工厂

`_get_hasher_factory(algorithm)`（`:22`）被 `@functools.lru_cache(maxsize=3)` 缓存：

- `"blake3"` → `blake3.blake3`（默认，需 `pip install blake3`）。
- `"sha256"` / `"sha512"` → `hashlib.sha256` / `hashlib.sha512`（FIPS 合规，issue #18334）。
- 其它 raise（`env_with_choices` 已校验）。

### serialize_item

`MultiModalHasher.serialize_item(obj)`（`:52`）返回 `Iterable[bytes | memoryview]`，按类型 dispatch：

| 类型 | 序列化 |
|---|---|
| `bytes` / `memoryview` | 原样 |
| `str` | `.encode("utf-8")` |
| `int` / `float` | `np.array(obj).tobytes()` |
| `PIL.Image` | EXIF UUID 快速路径；否则 `{"mode", np.asarray, palette, palette_rawmode}` 经 `iter_item_to_bytes("image", data)` |
| `MediaWithBytes[Image]` | 同上但优先 `obj.original_bytes` |
| `torch.Tensor` | `.cpu()`；`bfloat16` → view uint8 + 记 `original_dtype/shape`；其它 → `.numpy()` |
| `np.ndarray` | 0-D `.item()`；C-contiguous → `view(uint8).data`（零拷贝）；其它 → `.tobytes()`；记 dtype/shape |
| fallback | `pickle.dumps` + warning |

`iter_item_to_bytes(key, obj)`（`:134`）递归展开 list/tuple/dict，每层把 key 拼成 `key.i` / `key.k`，最终 `yield key.encode("utf-8")` 后 `yield from serialize_item(obj)`。`None` 只 yield key bytes。

### hash_kwargs

`MultiModalHasher.hash_kwargs(**kwargs)`（`:154`）：

1. 用 `_get_hasher_factory(envs.VLLM_MM_HASHER_ALGORITHM)()` 建 hasher。
2. `for k, v in sorted(kwargs.items())`（按 key 字典序，保证调用者传参顺序不影响哈希）。
3. 对每个项 `yield from iter_item_to_bytes(k, v)`，逐块 `hasher.update(bytes_)`。
4. `return hasher.hexdigest()`。

调用约定：`ProcessorInputs.get_mm_hashes`（`processing/inputs.py:25`）以 `hash_kwargs(model_id=model_id, **{modality: item}, **hf_processor_mm_kwargs)` 调用，意味着 `model_id` 与 HF processor kwargs 都进哈希——同图不同 processor 参数会得到不同 hash，避免错误命中。

## 与其它模块/系统配合

- **processing/inputs.py**：`ProcessorInputs.get_mm_hashes` 是 `hash_kwargs` 的唯一规范调用点；若用户传入 `mm_uuids`，且 `hf_processor_mm_kwargs` 为空，则透传 UUID 不哈希，节省 CPU。
- **cache.py**：hash 字符串作 LRU/SHM key；`MultiModalReceiverCache.get_and_update_features` 用 `feature.mm_hash or feature.identifier` 跨 LoRA 共享。
- **v1/engine/input_processor.py**：`_get_mm_identifier`（`:165`）在 `enable_tower_connector_lora` 时把 `mm_hash` 加 `lora_name:` 前缀生成 `identifier`，保证不同 LoRA 下编码器缓存不串。
- **v1/core/encoder_cache_manager.py**：`check_and_update_cache` / `allocate` 用 `request.mm_features[input_id].identifier` 作 GPU encoder cache key。
- **parse.py**：`get_all_items_for_hash` 返回 raw item（保留 `MediaWithBytes`），让 hasher 能优先用 `original_bytes`。
- **media/image.py**：`ImageMediaIO.load_bytes` 把 decoded PIL + raw bytes 包成 `MediaWithBytes`，让 hasher 走快速路径。
- **envs**：`VLLM_MM_HASHER_ALGORITHM` 控制算法；`vllm/envs.py` 中以 `env_with_choices` 形式声明。

## 历史版本演进

- **v0.5（LLaVA 初版）**：无独立 hasher；缓存 key 用 image URL 字符串。
- **v0.7（v1 化）**：`MultiModalHasher` 引入，默认 `sha256`，仅支持 image / ndarray / 标量。
- **v0.8**：加入 `torch.Tensor` 与 `bfloat16` workaround；`iter_item_to_bytes` 支持嵌套 dict/list。
- **v0.9（hash+cache）**：默认算法切到 `blake3`（3-5 倍快）；`MediaWithBytes` 快速路径加入，让原始字节优先于像素。
- **v0.10**：`VLLM_MM_HASHER_ALGORITHM` 加 `sha256` / `sha512` 选项（FIPS 合规，`_get_hasher_factory` lru_cache 化）；PIL EXIF UUID 快速路径加入。
- **main**：`hash_kwargs` 强制 `sorted(kwargs.items())`，使调用顺序不影响哈希；调用约定显式包含 `model_id`，让同图不同模型不串缓存（之前依赖 path-of-model 隐式隔离）。

[← 返回多模态首页](../README.md)

## 参见

- [cache.md](cache.md)：hash 字符串作 cache key。
- [parse.md](parse.md)：`get_all_items_for_hash` 决定哈希输入对象。
- [media.md](media.md)：`MediaWithBytes.original_bytes` 来源。
- [v1-integration.md](v1-integration.md)：`identifier` 与 LoRA 前缀化。
