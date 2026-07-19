# image.py · 图像 mode 转换与归一化

[← Wiki 首页](../README.md) > [多模态](../README.md) > image

## 是什么

`vllm/multimodal/image.py` 是图像侧最薄的 helper 文件，仅 60 行，提供四个纯函数：`rescale_image_size`、`normalize_image`、`rgba_to_rgb`、`convert_image_mode`。它们被 `parse.py`（解析期 EXIF 转正 + RGB 归一）、`media/image.py`（IO 层 mode 转换 + RGBA 背景填充）共用。

## 为什么

PIL 图像进入 vLLM 后有两个"与模型无关"的清洗需求：

1. **EXIF 方向**：手机拍的照片常带 `Orientation` EXIF，像素原始数据是横的但应竖着显示。若不调用 `ImageOps.exif_transpose`，模型看到的像素与人类直觉相反，导致描述错乱。`normalize_image` 用 `contextlib.suppress(Exception)` 容错（部分损坏 EXIF 不应阻塞 ingest）。
2. **mode 归一化**：HF ImageProcessor 几乎都要求 RGB 3 通道；但用户图可能是 RGBA（带透明）、P（palette）、LA（灰度+alpha）、L（灰度）。直接 `convert("RGB")` 会丢失透明度信息（透明区域变黑）；正确做法是先把透明区填背景色再合并。`convert_image_mode` 检测透明度（RGBA/LA/PA 或 tRNS chunk），转 RGBA 后用 mask paste 到指定背景色的 RGB 画布上。

此外 `media/image.py` 允许通过 `--media-io-kwargs '{"image": {"rgba_background_color": [...]}}'` 定制背景色（默认白），多模态配置里集中处理。

`rescale_image_size` 是早期遗留的尺寸缩放 helper（按 factor 而非目标尺寸），主要用于测试与某些 dummy input 路径。

## 怎么做

### normalize_image

`normalize_image(image)`（`:21`）：

```python
with contextlib.suppress(Exception):
    image = ImageOps.exif_transpose(image)
return image
```

任何异常（损坏 EXIF、PIL 版本差异）都被静默吞掉，原图原样返回。代价是极少数错向图片不被修正，但避免 ingest 路径因一张坏图整批失败。

### rgba_to_rgb

`rgba_to_rgb(image, background_color=(255,255,255))`（`:28`）：

```python
assert image.mode == "RGBA"
converted = Image.new("RGB", image.size, background_color)
converted.paste(image, mask=image.split()[3])   # alpha 作 mask
return converted
```

`paste` 的 `mask` 参数让 alpha 非零区露出原图、alpha 零区露背景色、中间值做线性混合，比先 convert 后补色更平滑。

### _has_transparency

`_has_transparency(image)`（`:39`）：`mode in ("RGBA", "LA", "PA")` 或 `"transparency" in image.info`（PNG tRNS chunk 影响 P/L/RGB 模式）。

### convert_image_mode

`convert_image_mode(image, to_mode, background_color=(255,255,255))`（`:47`）：

```python
if image.mode == to_mode:
    return image
if to_mode == "RGB" and _has_transparency(image):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return rgba_to_rgb(image, background_color)
return image.convert(to_mode)
```

非透明场景直接走 PIL `convert`（如 L → RGB 上色、P → RGB 解 palette）。透明场景专用路径。

### rescale_image_size

`rescale_image_size(image, size_factor, transpose=-1)`（`:9`）：按 factor 缩放宽高，可选 PIL `Transpose`（翻转/旋转）。

### MediaWithBytes 解包

注意 `media/image.py` 的 `ImageMediaIO._convert_image_mode` 先把 `MediaWithBytes` 解包出底层 `Image` 再调用本文件的 `convert_image_mode`，避免对包装对象调 PIL 方法。

## 与其它模块/系统配合

- **parse.py**：`_parse_image_data` 对每个 PIL item 做 `convert_image_mode(normalize_image(item), "RGB")`，两步串行。
- **media/image.py**：`ImageMediaIO.load_bytes` 做 `normalize_image` + `_convert_image_mode`，把结果包成 `MediaWithBytes`；`encode_base64` 编码前也走 `_convert_image_mode` 以匹配目标 mode。`rgba_background_color` 从 `--media-io-kwargs` 透传。
- **processing/dummy_inputs.py**：`_get_dummy_images` 用 `Image.new("RGB", (w,h), color=255)` 直接造 RGB 图，不经本文件。
- **video.py**：视频帧的 size/方向归一化在 `video.py` 的 `VideoBackend` 内部处理（如 `resize_video`），不调本文件——视频帧默认已是 RGB 且方向由解码器处理。
- **配置**：`rgba_background_color` 通过 `--media-io-kwargs` 每请求/每服务可配，详见 [media.md](media.md)。

## 历史版本演进

- **v0.5（LLaVA 初版）**：仅有 `rescale_image_size` 与基本 `convert_image_mode`，无透明度处理。
- **v0.6**：`normalize_image`（EXIF transpose）加入，修复手机照片方向问题。
- **v0.7（v1 化）**：`rgba_to_rgb` + `_has_transparency` 加入，处理带 alpha 的 PNG（来自截图、设计稿）。`convert_image_mode` 引入透明分支。
- **v0.9（hash+cache）**：`contextlib.suppress` 容错加强（部分损坏 EXIF 之前会 raise）。
- **v0.10**：`rgba_background_color` 经 `ImageMediaIO` 可配，本文件相关函数接受 `background_color` 参数。
- **main**：极简稳定，未做大改；可定制背景色的能力使透明 PNG 在不同部署（深色 / 浅色背景）下表现可控。

[← 返回多模态首页](../README.md)

## 参见

- [media.md](media.md)：`ImageMediaIO` 调用本文件做 IO 层归一。
- [parse.md](parse.md)：解析期调用本文件做 ingest 归一。
- [video.md](video.md)：视频帧不调本文件，独立处理。
