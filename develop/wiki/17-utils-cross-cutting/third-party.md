# 内嵌第三方代码（third_party）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 内嵌第三方

本页覆盖 `vllm/third_party/`，描述 vLLM 为减少外部 hard pin 而直接内嵌进包的第三方源码。

## 是什么

`vllm/third_party/__init__.py` 为空（仅作为包标记）。实际内容：

- `pynvml.py`：从 PyPI 包 `nvidia-ml-py`（版本 12.570.86）源码复制而来，是 NVIDIA Management Library (NVML) 的 Python 绑定。文件头保留 NVIDIA 的 BSD-3-clause 版权声明与免责条款。提供 GPU 显存/温度/利用率等 NVML 查询接口。
- `flashmla/`：`__init__.py` 头部注明 "Sources copied from FlashMLA"——内嵌 DeepSeek FlashMLA 项目（高效 MLA decode kernel）的 Python 入口/封装，方便 vLLM MLA 注意力后端调用而无需用户单独安装。

## 为什么

- **避免 hard dependency 版本冲突**：`nvidia-ml-py` 等包在用户环境里可能与其它 ML 库冲突，内嵌保证 vLLM 用到确定版本。
- **零配置可用**：FlashMLA 对用户安装不友好，内嵌使启用 MLA 后端即开即用。
- **许可兼容**：NVIDIA BSD-3-clause 与 FlashMLA 许可均与 Apache-2.0 兼容，允许内嵌分发；文件头保留原始版权与许可声明以满足许可条件。

## 怎么做

- 业务代码 `from vllm.third_party import pynvml` 后按 NVML Python API 使用（`pynvml.nvmlInit()`/`nvmlDeviceGet*`）。
- FlashMLA 经由 MLA 注意力后端间接调用（[注意力 · MLA](../05-attention/README.md)），用户通常不直接 import。
- **更新内嵌版本**：替换文件并**保留原始版权/许可头**；同步更新本页版本号注释。

## 与其它模块/系统配合

- [平台 · CUDA](../08-platforms/README.md)：`CudaPlatform`/`nvml` 相关显存与设备查询走 `pynvml`。
- [可观测性](../16-observability/README.md)：metrics 采集 GPU 利用率/显存经 NVML。
- [注意力 · MLA](../05-attention/README.md)：FlashMLA kernel 服务 DeepSeek MLA decode。
- [构建/CI](../18-build-ci-testing/README.md)：内嵌代码随 wheel 一起分发，无需额外构建步骤。

## 历史版本演进

- **早期**：NVML 查询依赖外部 `nvidia-ml-py`，常因版本冲突报错；随后内嵌为 `third_party/pynvml.py`。
- **v0.7–v0.8**：随 MLA 热度引入 `flashmla/` 内嵌（来源 DeepSeek FlashMLA）。
- **v0.9–main**：`pynvml.py` 跟随上游 `nvidia-ml-py` 版本同步更新（当前 12.570.86，待核实是否随主线滚动）；FlashMLA 内嵌版本随 MLA kernel 演进（待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [平台子系统](../08-platforms/README.md)
- [注意力 · MLA](../05-attention/README.md)
- [可观测性](../16-observability/README.md)
