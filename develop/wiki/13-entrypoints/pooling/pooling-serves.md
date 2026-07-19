[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Pooling](README.md) > pooling

# pooling/pooling/（/v1/pooling 通用端点）

> `pooling/pooling/`（注意双重命名）实现 `/v1/pooling`：返回原始池化向量（不附加 embed/classify/score 语义），供高级用户直接取模型池化层输出。

## 是什么

| 文件 | 职责 |
|---|---|
| `serving.py: ServingPooling` | `vllm/entrypoints/pooling/pooling/serving.py:33`，`/v1/pooling` handler，继承 `PoolingBaseServing` |
| `api_router.py: attach_router` | 注册 `POST /v1/pooling` |
| `protocol.py` | 通用 pooling 请求/响应（`data[].data` 原始向量） |
| `io_processor.py` | 请求→引擎、输出→响应 |

请求 → io_processor → `engine_client.pool`（`get_pooling_invocation_types` 返回 `pooling`）→ 返回原始池化结果。

## 为什么

- **语义中性**：embed/classify/score 都对池化输出做了 task 专属包装；`/v1/pooling` 给"裸"向量，便于自定义下游。
- **调试/探索**：查看不同 `PoolingParams`（pooling_type/normalize）下的原始输出。

## 怎么做

```bash
curl -X POST http://localhost:8000/v1/pooling -d '{"model":"...","input":"文本"}'
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| ServingPooling | `vllm/entrypoints/pooling/pooling/serving.py:33` |
| attach_router | `vllm/entrypoints/pooling/pooling/api_router.py` |
| io_processor | `vllm/entrypoints/pooling/pooling/io_processor.py` |

## 与其它模块/系统配合

- [base.md](base.md)：基类。
- [factories.md](factories.md)：注册。
- [配置-scheduler](../../10-config/scheduler-config.md)：`PoolerConfig`（pooling_type/normalize）。

## 历史版本演进

- **v0.11（引入）**：`/v1/pooling` 通用端点。
- **main**：与 `get_pooling_invocation_types` 配合。

## 参见

- [← 返回 Pooling 首页](README.md)
- [base.md](base.md)
