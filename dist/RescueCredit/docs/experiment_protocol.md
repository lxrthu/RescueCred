# 实验协议

1. S0：pytest、replay、audit commit order、provenance zero-gradient、budget identity。
2. S1：Rescue-MDP 精确枚举和 estimator 无偏验证。
3. S2：三方法单 seed pilot；主指标为 `S_off`、First-pass、`G0` 估计。
4. Gate：RescueCredit 至少改善一项核心指标才扩展。
5. S3：42/43/44 三 seed，equal interaction；shadow steps 从总预算扣除。
6. S4：ID 与 tool-OOD 分列；不在结果后改 split、patch 或主要指标。

确认性阶段前必须冻结模型、group size、温度、总交互预算、max steps 和 non-inferiority margin。

