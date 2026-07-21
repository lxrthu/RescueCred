# Counterfactual Credit Replay Code Review

时间：2026-07-21 21:04

## 首轮结论

`DEPLOY NO`。首轮实现把既有 RAPG uniform AIPW 误写成新方法候选，并使用了无效的多变量 `projected_bias_upper95` Gate；数据血缘、task cross-fit、任务集中度和实际审计成本检查也不完整。

## 修复

- 明确降级为 historical RAPG surrogate 上的 retrospective main diagnostic；AIPW 不是新方法。
- 删除 bias outcome Gate；固定 propensity 的 AIPW 无偏性仅作为代数 integrity sanity。
- 从原始封存输入验证 behavior ledger、bank/source/prediction hashes、行为先于 outcome、旧 proposal 身份、bank version、task overlap、finite/shape、proposal/replacement encoding 和重复 event IDs。
- credit-error energy 改为基于 `executed_return - proposal_return`。
- outcome Gate 增加 AIPW-vs-IPW MSE、task bootstrap lower bound、top-task concentration 和 realized audit cost。
- primary propensity tag 从冻结 protocol 动态生成。

## 复审结论

`DEPLOY YES`，仅限 retrospective main diagnostic；无剩余 blocker。CPU 运行在 10 分钟预算内可信。

本地验证：`py_compile`、Ruff、`git diff --check` 通过；pytest 为 `ss...`，其中两个 Torch 数学测试因本地环境无 Torch 跳过，服务器 runner 会在有 Torch 的 `.venv` 中重新运行并 fail closed。

