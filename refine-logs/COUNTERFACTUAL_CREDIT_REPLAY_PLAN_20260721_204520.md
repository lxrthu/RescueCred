# Counterfactual Credit Replay Pilot Plan

冻结时间：2026-07-21 20:45。该计划在首次运行新 replay 脚本之前冻结，但使用的是历史 RAPG development surrogate，因此结果只能作为机制诊断，不能作为 untouched confirmation。

## 问题

Harness 执行 B 后，直接用 B 的 return 更新 proposal A 会产生 credit leakage。静态 Rescue/Reverse 分类和短程 receipt 已经失败；本实验不再预测标签，而是直接用已有 Full-Shadow A outcome 检查梯度方向。

## 输入

- 已封存 RAPG candidate-policy bank：LoRA score-gradient sketches、proposal/execution identity、task groups、executed-B returns。
- 已封存 Shadow-A returns。
- 已有 task-cross-fitted outcome predictions，仅作为 AIPW control variate。

## 方法

在同一批事件上重放五种 proposal-credit estimator。这里的 `rsc_aipw` 与既有 RAPG uniform AIPW 相同，只作为已知估计器对照，不声明为新算法：

1. `naive_b_to_a`：所有 proposal 都使用 executed-B return。
2. `firewall`：被 Harness 替换的 proposal 不接收 B return；未替换事件正常更新。
3. `rsc_ipw_p30`：对 replacement events 以固定概率 0.30 观察 A，使用 IPW。
4. `rsc_aipw_p30`：相同固定 ledger，使用 cross-fitted AIPW control variate。
5. `full_shadow_oracle`：所有 proposal 使用真实 proposal outcome。

固定 10,000 个随机 ledgers，另以 `p=0.20` 作预声明 secondary sensitivity，不做择优。

## Primary gate

所有条件必须同时满足：

```text
firewall_oracle_distance_ratio_vs_naive <= 0.80
firewall_task_improvement_fraction_vs_naive >= 0.50
rsc_aipw_mse_gain_over_ipw >= 0.20
rsc_aipw_mean_cosine > naive_cosine
rsc_aipw_task_improvement_fraction_vs_naive >= 0.50
firewall_positive_improvement_top1_share <= 0.50
rsc_aipw_positive_improvement_top1_share <= 0.50
task_bootstrap_lower95(firewall vs naive) > 0
task_bootstrap_lower95(rsc_aipw vs naive) > 0
abs(mean_realized_audits - expected_audits) <= 0.10
```

固定 propensity 下 AIPW 的无偏性只作为代数完整性检查，不把 Monte Carlo bias 当结果 Gate。同时报告：基于 `executed_return - proposal_return` 的 credit-error energy、IPW/AIPW MSE、实际 audit rate、task concentration、20,000 次 task bootstrap 和 p=0.20 sensitivity。

## Claim boundary

通过只支持：在历史 replay-valid surrogate 上存在可复现的 Harness credit-leakage 机制，并且 credit firewall/既有 uniform AIPW 对照在任务层面更接近 Full-Shadow oracle gradient。该实验可以作为主诊断图，但不是新的主方法结果，也不证明 autoregressive policy quality、在线训练收益或 deployment generalization。
