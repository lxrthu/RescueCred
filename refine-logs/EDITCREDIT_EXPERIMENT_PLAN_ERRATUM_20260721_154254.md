# EditCredit Pilot 计划勘误

**日期**：2026-07-21 15:42

本勘误在任何 EditCredit GPU 训练前冻结，处理独立代码审查指出的 source-only 指标定义问题。

## 修正

原计划中的 `source-only ROC-AUC <= 0.55` 不能由当前 edit-value learner 诚实测量，因为模型并未训练输出 A/B source label。Pilot 改为两个可审计条件：

1. **结构条件**：EditCredit prompt 不包含 `original`、`correction`、`Harness` 或方向标签，只包含随机顺序的 `candidate_left/right`、公开上下文和 changed field。
2. **展示侧平衡**：每类内 counterbalance B 出现在 left/right 的次数，并要求每折 `abs(presentation_side_label_auc - 0.5) <= 0.05`。
3. **模型顺序稳健性**：同一 pair 在两个候选展示顺序下独立评分，要求选择一致率至少 0.95；最终 margin 对两个顺序取平均。

这三个条件只排除已知的 source/position shortcut 通道，不声称证明模型不存在所有潜在捷径。

## 新增完整性要求

- GPU 训练前必须通过绑定到 protocol 的 autograd ownership sanity。
- V4.4 `manifest.json`、`data_gate.json` 和 126-pair bank 必须同时 hash-bound。
- evaluator 必须先落盘不含 label/outcome 的 `scores.public.jsonl`，再 join ground truth。
- Gate 必须独立从 frozen bank 重建 task folds、labels 和 calibration threshold，不信任 prediction 文件自带标签。
- 已知 confirmation 集仍不得用于配置选择。
