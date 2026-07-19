[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Pooling](README.md) > base

# pooling/base/（PoolingBaseServing + PoolingServing）

> `base/` 定义池化任务的 serving 抽象基类：`PoolingBaseServing` 提供 io_processor 持有与请求执行模板，`PoolingServing` 在其上处理通用 `AnyPoolingRequest`。各具体 task（embed/classify/score/pooling）继承二者。

## 是什么

| 类 | 位置 | 职责 |
|---|---|---|
| `PoolingBaseServing` | `vllm/entrypoints/pooling/base/serving.py:37` | ABC + `BaseServing`：持 io_processor，定义 `do_pooling` 模板 |
| `PoolingServing` | `:263` | 处理通用 pooling 请求，子类化出各 task |

`PoolingBaseServing`（`:37`）持有 io_processor（来自 `base/io_processor.py`），把请求转 `EngineInput`、提交 `engine_client.encode`/`pool`、收集 `PoolingRequestOutput`、经 io_processor 组装响应。`_check_model` 复用 `BaseServing`。

`PoolingServing`（`:263`）在 `PoolingBaseServing` 上加通用 pooling 请求/响应协议（`base/protocol.py`），是 `ServingPooling`（`pooling/serving.py:33`）的父类。

## 为什么

- **io_processor 模板方法**：把"请求→引擎输入"与"引擎输出→响应"两步下沉到 io_processor，`PoolingBaseServing` 只编排，便于多模态/多 task 复用。
- **ABC 强约束**：强制子类实现 task 专属 `protocol` 与字段映射，避免漏实现。
- **通用/专用分层**：`PoolingServing` 给"原始池化向量"端点，`ServingEmbedding`/`ServingScores` 给语义化端点，分层清晰。

## 怎么做

见 [README](README.md) 离线/在线示例。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| PoolingBaseServing | `vllm/entrypoints/pooling/base/serving.py:37` |
| PoolingServing | `vllm/entrypoints/pooling/base/serving.py:263` |
| base io_processor | `vllm/entrypoints/pooling/base/io_processor.py`（待核实行号） |
| base protocol | `vllm/entrypoints/pooling/base/protocol.py` |

## 与其它模块/系统配合

- [serve/engine-serve.md](../serve/engine-serve.md)：`BaseServing`。
- [factories.md](factories.md)：构造子类实例。
- [多模态](../../11-multimodal/README.md)：io_processor 多模态输入。

## 历史版本演进

- **v0.9（base 抽象）**：`PoolingBaseServing`/`PoolingServing` + io_processor 模式。
- **main**：多模态 pooling io_processor 扩展。

## 参见

- [← 返回 Pooling 首页](README.md)
- [serve/engine-serve.md](../serve/engine-serve.md)
