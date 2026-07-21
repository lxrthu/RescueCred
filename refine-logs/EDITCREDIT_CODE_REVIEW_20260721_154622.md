# EditCredit Experiment Code Review

**日期**：2026-07-21 15:46
**最终结论**：DEPLOY YES

## Review 1：DEPLOY NO

初审发现 Gate 信任 prediction 标签、calibration threshold 与硬编码 isolation，firewall sanity 未绑定真实训练 objective，source-only AUC 定义过宽。

修复：统一 production objective；增加 autograd ownership Gate；绑定 V4.4 manifest/data gate；先落无标签 score artifact；Gate 从 frozen bank 独立重建 labels、folds 和 threshold；将 shortcut claim 缩小为 presentation-side balance 与 swap consistency。

## Review 2：DEPLOY NO

二审剩余两个 blocker：Gate 未独立重算 mean margin/swap flag；run 未完整绑定 base model、train bank、config 和 expected fold adapter。

修复：Gate 逐行重算派生 score，所有最终指标只使用重建值；逐折验证完整训练身份；增加 margin 和 base identity tamper regression，即使外层 hash 被重新绑定也必须拒绝。

## Final Review：DEPLOY YES

BLOCKING：无。

确认通过：

- original/swapped/reported margins 必须 finite，mean 与 swap consistency 由 Gate 独立重算；
- run/eval method、fold、status、protocol、base、train bank、config、presentation budget、adapter 和 artifacts 完整绑定；
- gradient sanity 在 freeze 前执行、进入 protocol hash，并由最终 Gate 再验证；
- V4.4 lineage、exact replay labels、task folds 和 calibration threshold 均独立重建；
- runner 包含 positive/tamper Gate tests，失败时保留日志与中间 artifact。

## Non-blocking claim limits

1. Gradient audit 是合成 autograd implementation invariant，不是完整 LoRA 参数级因果归因证明。
2. AUC 只证明 presentation-side 与 label 平衡，不得称作语义 source-only probe。
3. 当前逐阶段日志足以部署；统一 `failure.json` 可后补。
