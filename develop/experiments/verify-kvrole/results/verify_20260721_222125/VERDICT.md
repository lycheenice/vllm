# kv_role 验证结果  (verify_20260721_222125)

- KV_ROLE_MODE=pc: 输出与 golden **一致**,传输证据计数=66
- KV_ROLE_MODE=both: 输出与 golden **一致**,传输证据计数=70

判据:输出一致=语义正确(必要条件);证据>0=KV 真传输。
若 pc 证据≈0 而 both>0 => 应把所有 PD 实验默认改为 kv_both。
