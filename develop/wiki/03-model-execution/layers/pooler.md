# 池化层（pooler/）

[← Wiki 首页](../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > 池化层

`vllm/model_executor/layers/pooler/`（~240 + 子模块文件）实现 vLLM 全部"非生成式"模型（embedding / classify / token_classify / 同时输出多任务如 BgeM3 / reward / score 等）的"hidden_states → 输出向量"收敛层。它与 `Sampler` 互斥——模型 forward 末端要么调 `Sampler`（生成式），要么调 `Pooler`（池化式）。

## 是什么

`pooler/` 目录结构：

| 文件 | 内容 |
|---|---|
| `abstract.py` | `Pooler(nn.Module, ABC)`：定义 `get_supported_tasks` / `get_pooling_updates` / `forward` 抽象 |
| `common.py` | `PoolingParamsUpdate` dataclass：描述 task 切换后的参数补丁；`ClassifierFn` 类型别名 |
| `special.py` | `DispatchPooler` / `IdentityPooler` / `BOSEOSFilter` / `BgeM3Pooler` 等组合型 |
| `seqwise/` | 序列级池化：`SequencePooler`（`poolers.py:44`）+ `EmbeddingPoolerHead` / `ClassifierPoolerHead`（`heads.py`）+ `CLSPool` / `LastPool` / `MeanPool`（`methods.py`）|
| `tokwise/` | Token 级池化：`TokenPooler`（`poolers.py:48`）+ `TokenEmbeddingPoolerHead` / `TokenClassifierPoolerHead`（`heads.py`）+ `AllPool` / `StepPool`（`methods.py`）|

任务类型由 `vllm.tasks.PoolingTask` 枚举决定（`embed` / `classify` / `token_embed` / `token_classify` / `embed&token_classify` / `plugin`）。

## 为什么

把池化抽象成层库子模块，是为了：

1. **支持多任务同一模型**：BgeM3 同时输出 dense embedding 与 token 级 colbert 风格 logits；`BgeM3Pooler`（`special.py:202`）内部组装 `embed_pooler` 与 `token_classify_pooler`，分别 `forward` 后 `cat` 起来返回。
2. **任务调度统一**：`DispatchPooler`（`special.py:25`）持有 `poolers_by_task: Mapping[PoolingTask, Pooler]`，batch 中允许多任务并存（按 task 字段 groupby），把任务子张量切片分发给对应 sub-pooler（`special.py:88-126`）。token_offset 的计算刻意用 CPU `num_scheduled_tokens_cpu.sum()` 避免 GPU→CPU sync。
3. **方法与 head 解耦**：`SequencePooler` = `pooling method`（决定取哪些 token：CLS/Last/Mean）+ `head`（决定后续变换：identity / dense 全连接 / classifier）；`TokenPooler` 同理但作用于每个 token。每种 `*Pool` 方法只关心 token 选择，每种 `*Head` 只关心 head 变换，组合性高。
4. **BOS/EOS 过滤与参数补丁**：`BOSEOSFilter`（`special.py:152`）在子 pooler 输出后按 token id 切掉首 BOS / 尾 EOS；`get_pooling_updates` 通过 `PoolingParamsUpdate(requires_token_ids=True)` 告知上层"需要回传 prompt_token_ids"。
5. **插件扩展**：`IdentityPooler.get_supported_tasks() == {"plugin"}`（`special.py:140-149`）——pooler 也允许通过插件完全外部实现，vLLM 内核只看到 `hidden_states` 原样返回。

## 怎么做

### `Pooler` 抽象

`abstract.py:16-36`：

```python
class Pooler(nn.Module, ABC):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()
    @abstractmethod
    def forward(self, hidden_states, pooling_metadata) -> PoolerOutput: ...
```

`PoolingParamsUpdate`（`common.py:19`）告诉 `Worker`/`Scheduler`：这个 task 是否 `requires_token_ids`、是否需要 `step`/`pooling` 参数覆盖等。

### `DispatchPooler` 的两类工厂

`special.py`：`DispatchPooler.for_embedding(pooler_config)` 与 `DispatchPooler.for_seq_cls(pooler_config, *, pooling=None, classifier=None)`。前者组装 `{token_embed, embed}` 两个 sub-pooler；后者组装 `{token_classify, classify}`，其中 `token_classify` 用 `AllPool()` 把所有 token 输出再分类。`forward` 中按 `pooling_metadata.tasks` groupby 分派。

### SequencePooler 三段流水

`seqwise/poolers.py:44` 的 `SequencePooler`：

1. `pooling(hidden_states, pooling_metadata)` → 选 token：`CLSPool` 取 `first_token_indices_gpu`；`LastPool` 取 `last_token_indices_gpu`；`MeanPool` 取区间均值。
2. `head(pooled, ...)` → 变换：`EmbeddingPoolerHead` 通常 identity + normalize；`ClassifierPoolerHead` 加 `Linear` + softmax/logsoftmax。
3. 返回 `PoolerOutput: list[torch.Tensor]`，每个 request 一个 tensor。

### TokenPooler

`tokwise/poolers.py:48`：对每个 token 应用 head（不再做序列级方法）。`AllPool` 返回全部 token；`StepPool` 是 `AllPool` 的子类用于 stepwise 输出（`methods.py:86`）。

### BgeM3 / BOSEOSFilter / Identity

| Pooler | 行为 |
|---|---|
| `BgeM3Pooler` (`special.py:202`) | 同时调 `embed_pooler` 与 `token_classify_pooler`，把两者输出 `cat([embed.view(-1), token_cls.view(-1)], dim=-1)`，`get_supported_tasks()={'embed&token_classify'}` |
| `BOSEOSFilter` (`special.py:152`) | 包装 sub-pooler，按 `bos_token_id`/`eos_token_id` 切掉首尾 token；通过 `get_pooling_updates` 通知 `requires_token_ids=True` |
| `IdentityPooler` (`special.py:140`) | `forward(hidden_states, pooling_metadata)` 直接 `return hidden_states`；`supported_tasks={'plugin'}` |

## 与其它模块/系统配合

- [sampler-layer.md](sampler-layer.md)：生成式模型走 Sampler，池化模型走 Pooler，二者互斥。模型 `forward` 末端按 `model_config.task` 选择挂哪个。
- [entrypoints #13](../../13-entrypoints/README.md)：OpenAI `/v1/embeddings`、Anthropic embeddings、 pooling serve / classify API 都把请求路由到 `PoolingTask`，最后由 `Pooler` 收敛。
- [model-zoo #04](../../04-model-zoo/README.md)：BgeM3、Qwen3-Embedding、Reward 模型、Classify 模型、GPT-OSS fallback 等都在 `__init__` 中按 `pooler_config` 构造 `DispatchPooler`/`BgeM3Pooler`/`BOSEOSFilter`。
- [config #10](../../10-config/README.md)：`PoolerConfig` 描述 pooling method、normalize、classification head 等；`PoolingParamsUpdate` 是该 config 在 per-task 维度的运行期补丁。
- [multimodal #11](../../11-multimodal/README.md)：跨模态嵌入模型也走 `Pooler`；`pooling_metadata` 与多模态预处理耦合（`requires_token_ids`、`pooling_cursor` 字段）。
- [distributed #07](../../07-distributed/README.md)：`TokenPoolerHead` 的 classifier 权重有时不切 TP（`disable_tp`），具体取决于模型实现 `(待核实)`。

## 历史版本演进

- **早期**：vLLM 只有 generate 任务；embedding 走 `nn.Identity()` 收尾。
- **v0.6–v0.7**：引入 `Pooling` 任务与第一版 `Pooler` 抽象，仅服务 reward/embedding basic。
- **v0.8**：`DispatchPooler` 引入支持多任务混合 batch；`SequencePoolingMethod` 与 `SequencePoolerHead` 分离。
- **v0.9**：BgeM3 模型上线后 `BgeM3Pooler` 引入，支持同模型 dense + colbert；`BOSEOSFilter` 在 Bge 系列与 E5 系列被启用。
- **v0.10**：`PoolingParamsUpdate` 抽象引入，把"task → 参数补丁"的传递从 Worker 硬编码改为 dataclass。
- **v0.10末–v0.11**：`poolers_by_task` + groupby 分派路径重构（`special.py:88-126`），支持同一 step 内多 task 共存；token_offset 显式用 `num_scheduled_tokens_cpu.sum()` 避免 GPU 同步。`IdentityPooler` 支持 plugin 任务。
- **v0.11–v0.12**：`TokenPooler` 与 `StepPool` 引入，支持 token-level embedding（colbert 风格）与 stepwise 输出。
- **v0.12 / main**：DispatchPooler 的 token 切片逻辑稳定；`ClassifierFn` 类型别名明确；与 Reranker / Score 任务的接口在 [entrypoints #13](../../13-entrypoints/README.md) 协同演进。`EmbeddingPoolerHead` 的 normalize 选项扩展到支持多种范数 `(待核实)`。

[← 返回层库首页](../README.md)

## 参见

- [sampler-layer.md](sampler-layer.md)：生成式对应层。
- [entrypoints #13](../../13-entrypoints/README.md)：API 入口与 task 路由。
- [config #10](../../10-config/README.md)：`PoolerConfig` 与 `PoolingParamsUpdate`。
