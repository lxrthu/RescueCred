# EditCredit 最小实验计划（Latest）

最新完整计划见：`refine-logs/EDITCREDIT_EXPERIMENT_PLAN_20260721_151030.md`。

当前只启动两个阶段：

1. B0：CPU 梯度归属测试，要求干预事件对 A 的 assisted-return 梯度严格为零，并且只更新 A/B 的 changed fields。
2. B1：已有 126 个 exact paired Shadow 事件上的 seed-42、5-fold task-OOF Pilot。

进入三随机种子前必须同时满足：

- `RescueDrop <= 0.02`；
- `ReverseRecall >= Mask + 0.10`；
- `BalancedAccuracy >= full-action preference + 0.05`；
- task-macro improvement fraction `>= 0.50`；
- source-only AUC `<= 0.55` 且 A/B 顺序交换一致率 `>= 0.95`。

不过 Gate 就停止，不重新调 development 门槛。
