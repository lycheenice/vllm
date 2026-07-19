[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Generate](README.md) > generative scoring

# generate/generative_scoring/（生成式打分）

> `generative_scoring/` 实现"用生成模型给候选答案打分"的端点：模型对每个候选继续生成/算 logprob，把 logprob 转成分数。与 pooling cross-encoder 打分（[scoring.md](../pooling/scoring.md)）语义不同。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `GenerativeScoringRequest` | `vllm/entrypoints/generate/generative_scoring/serving.py:50` | 请求 schema |
| `GenerativeScoringItemResult` | `:106` | 单项结果 |
| `GenerativeScoringResponse` | `:120` | 响应 |
| `ServingGenerativeScoring` | `:145` | handler，继承 `BaseServing` |

`serving.py` 定义请求/响应协议（`GenerativeScoringRequest` 含 `context` + `candidates`），`ServingGenerativeScoring`（`:145`）把 context+每个 candidate 拼成 prompt → 经 renderer tokenize → `engine_client.generate` 取 candidate 部分 logprob → 归一化成 score → `GenerativeScoringResponse`。

## 为什么

- **生成式评测**：评估"模型对哪个候选更可能"用生成 logprob 比 pooling 向量夹角更直接（LM-as-judge 范式）。
- **独立端点**：与 pooling score 区分，避免协议混淆；客户端按场景选 endpoint。
- **复用 generate 基建**：经同一 renderer/`AsyncLLM.generate`，保证 logprob 计算与生成一致。

## 怎么做

```bash
curl -X POST http://localhost:8000/v1/score_generative \
  -d '{"model":"...","context":"问题","candidates":["答A","答B"]}'
```

（端点路径待核实，可能为 `/v1/generative_score` 或类似）。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| GenerativeScoringRequest | `vllm/entrypoints/generate/generative_scoring/serving.py:50` |
| GenerativeScoringResponse | `vllm/entrypoints/generate/generative_scoring/serving.py:120` |
| ServingGenerativeScoring | `vllm/entrypoints/generate/generative_scoring/serving.py:145` |
| generative_scoring api_router | `vllm/entrypoints/generate/generative_scoring/api_router.py` |

## 与其它模块/系统配合

- [serve/engine-serve.md](../serve/engine-serve.md)：`BaseServing`。
- [pooling/scoring.md](../pooling/scoring.md)：cross-encoder 打分对比。
- [采样-结构化](../../06-sampling-decoding/structured-output/README.md)：logprob 取值。

## 历史版本演进

- **v0.10（引入）**：`ServingGenerativeScoring` + 协议；与 pooling score 并列。
- **main**：归一化策略（待核实）；多 candidate 批处理。

## 参见

- [← 返回 Generate 首页](README.md)
- [pooling/scoring.md](../pooling/scoring.md)
