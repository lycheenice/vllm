[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Pooling](README.md) > classify

# pooling/classify/（/v1/classify）

> `classify/` 实现 `/v1/classify`：把文本经分类 head 池化，返回每条输入的类别概率/标签。离线等价 `LLM.classify`。

## 是什么

| 文件 | 职责 |
|---|---|
| `serving.py: ServingClassification` | `/v1/classify` handler，继承 `PoolingServing` |
| `api_router.py: attach_router` | 注册 `POST /v1/classify` |
| `protocol.py` | 分类请求/响应 schema（含 `labels`/`probs`） |
| `io_processor.py` | 请求→引擎输入、输出→分类响应 |

请求经 `_check_model` → io_processor 渲染 → `engine_client.encode`（`get_pooling_invocation_types` 返回 `classify`）→ 收割 → io_processor 组装分类结果（`data[].class_probs`/`label`，待核实字段名）。

## 为什么

- **分类 head 复用**：vLLM 支持带分类 head 的 pooling 模型（如 BERT-classification），`classify` 端点直接出概率，省去客户端后处理。
- **与 embed 共享基建**：同 `PoolingBaseServing`，仅 io_processor/协议不同。
- **离线一致**：`LLM.classify` 同源。

## 怎么做

```bash
curl -X POST http://localhost:8000/v1/classify -d '{"model":"...","input":"文本"}'
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| ServingClassification | `vllm/entrypoints/pooling/classify/serving.py`（待核实行号） |
| attach_router | `vllm/entrypoints/pooling/classify/api_router.py` |
| classify protocol | `vllm/entrypoints/pooling/classify/protocol.py` |
| 离线 classify | `vllm/entrypoints/pooling/offline.py:244` |

## 与其它模块/系统配合

- [base.md](base.md)：基类。
- [factories.md](factories.md)：注册。
- [tokenizers-transformers](../../14-tokenizers-transformers/README.md)：分类 head。

## 历史版本演进

- **v0.10（引入）**：`/v1/classify` + `LLM.classify`。
- **main**：io_processor 完善（待核实）。

## 参见

- [← 返回 Pooling 首页](README.md)
- [base.md](base.md)
