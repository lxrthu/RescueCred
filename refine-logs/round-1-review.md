# Round 1 Review

<details open>
<summary>Raw reviewer response</summary>

## AdaAudit 严格评审

**CALIBRATION: none**

| 维度 | 分数 |
|---|---:|
| Problem Fidelity | 9.5 |
| Method Specificity | 7.0 |
| Contribution Quality | 6.0 |
| Frontier Leverage | 7.0 |
| Feasibility | 8.0 |
| Validation Focus | 8.0 |
| Venue Readiness | 6.0 |

**Weighted overall: 7.23 / 10. Verdict: REVISE.**

方向比继续做 Rescue/Reverse 路由更忠于问题，但当前的 adaptive AIPW、
Horvitz--Thompson 和 Neyman allocation 都接近标准 two-phase sampling。Harness
场景新颖，但换应用语境不足以形成 AAAI 主贡献。

最重要的升级是把中心从 outcome-MSE 最优审计改成 action-replacement 下的
unbiased policy-gradient estimation 与 gradient-variance-optimal auditing。定义

```text
s_t = grad log pi(A_t | x_t)
g_hat_t = s_t [mu_t + Z_t/p_t (G_A - mu_t) - b_t]
```

对应 oracle allocation 应为

```text
p*_t proportional to ||s_t|| sigma_t / sqrt(c_t).
```

必须进一步明确：目标 policy objective 和状态分布、A 的 behavior policy、
REINFORCE/PPO 接口、stop-gradient、多次干预、G_A 是一步 Q 还是完整 return、
Shadow 随机性、expected/hard budget、offline/online allocation。Harness-induced
occupancy 上的单步 A credit 不自动等价于完全关闭 Harness 后的 return。

“budget-optimal”目前只对 oracle 成立。learned plug-in allocator 必须给出
excess-variance/regret bound，或收紧表述。G_B 是免费可见结果，应进入 residual
control variate。Supporting non-identifiability theorem 只能作动机，不能成为平行
主贡献。

与 shielded RL 的区别应落在 missing proposal-return gradient；与 DAgger/AIM
的区别是 AdaAudit 不把 B 当作 A 的负标签；与 verifier training 的区别是 verifier
只在随机审计时观察未执行 proposal 的 counterfactual return，并通过 propensity
correction 形成无偏学习信号。

Simplification: 合并 mu/sigma 双头；删除 conformal/e-process 与 correction
imitation 正文；主实验只保留识别、matched-budget policy pilot 和 fresh confirmation。

Modernization: 预测 gradient-weighted residual scale；对 LLM policy 使用 token-level
score sketch/Fisher norm，避免保存完整梯度。

Drift warning: NONE。

要成为主贡献，必须完成：replacement-aware unbiased gradient theorem、
gradient-variance oracle allocation 与 plug-in bound、matched-budget unassisted
policy gain。

</details>
