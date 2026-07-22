# verify-kvrole —— kv_role 语义验证实验

验证 NIXL PD 分离下 `kv_role` 用 **`kv_producer`/`kv_consumer`(pc)** 还是 **`kv_both`(both)**,
是否影响 KV 在 P/D 间的真正传输。这是 test1/test3 能否成立的前提(既有分析怀疑 v0.25.0
官方集成测试统一用 `kv_both`,若用错 decode 可能空等或退化为本地重算)。

## 与其它实验的关系
- 不做 ramp sweep,只做 **小样本正确性 + 传输证据** 验证,快(单次即可)。
- 结论回写:若判定应改 `kv_both`,则把 `test1/test3`(及 test2/test4)的默认改为
  `KV_ROLE_MODE=both`(serve_pd.sh 已支持该开关)。

## 运行(前置:h200-2 8 卡空闲)
```bash
# h200-2 本机一键(自动跑 golden + pc + both 三段并判定):
bash verify.sh
# 或从 a100-2: ssh -l root h200-2 "bash <abs>/verify.sh"
```
也可手动单模式:`KV_ROLE_MODE=both bash run.sh` 起服务,自行验证后 `bash stop.sh`。

## 方法
1. **golden**:单实例 TP4(无 PD),对 5 条固定 prompt 取 greedy(temp=0)输出作标准答案。
2. **pc**:`kv_role=kv_producer/kv_consumer` 起 PD,取同一 prompt 集输出 + 抓 P/D 日志传输证据。
3. **both**:`kv_role=kv_both` 起 PD,同上。
4. 判定并写 `results/verify_<ts>/VERDICT.md`。

## 判据
| 观测 | 含义 |
|------|------|
| 输出与 golden **一致** | 该模式端到端语义正确(**必要**条件;greedy 下 PD 应与单实例逐字一致) |
| 传输证据计数 **>0** | KV 确实在 P/D 间传输,而非 decode 端拿到完整 prompt 后本地重算 |
| pc 证据≈0 且 both>0 | 印证"应使用 `kv_both`",据此改所有 PD 实验默认 |
| 两模式输出都与 golden 不一致 | connector 配置/握手有更深问题,先修再谈性能 |

> 传输证据靠关键字 grep(`EVIDENCE_KEYS` 可覆盖),NIXL 日志措辞可能变化;
> 首跑后按 `results/*/pc_evidence.log` 实际内容校准关键字,必要时补 `/metrics` 计数。

## 结果归档
`results/verify_<ts>/`:`golden.txt` / `pc.txt` / `both.txt` / `*_evidence.log` / `VERDICT.md`。

## 执行记录
> 跑完填:日期 / 结论(pc vs both)/ 是否需要改各实验默认 / 证据样例。
