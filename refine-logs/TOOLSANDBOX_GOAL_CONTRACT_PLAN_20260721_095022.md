# ToolSandbox Goal Contract 实验计划

## 目标

在 DeltaGuard V2 明确回执仍全部弃权的负结果之后，增加部署前冻结的
Goal Contract。Contract 只能由用户可见指令、固定 A/B 和公开 tool schema
生成；它在任何分支回执出现前写入 protocol lock，之后只把任务目标编译成
确定性的 role、参数来源和 receipt-result 谓词。

## 方法边界

- 禁止读取 Shadow label、branch receipt、official evaluator 或隐藏数据库生成 contract。
- 只接受在可见指令中逐字或规范化出现的参数值作为 grounded goal atom。
- Goal Contract 可以提供新的公开谓词，但不能改变 A/B、probe 选择或 observer plan。
- A 只有在完整 Pareto 证书下优于 B 才能恢复；未知或冲突仍保持 B。
- role alignment 和 grounded-argument coverage 仅作诊断，不能触发路由；至少一个
  post-receipt 或 state witness 才能恢复 A。
- 当前 combined bank 仅作 development feasibility，不产生 paper-facing confirmation claim。

## Milestones

| Milestone | 内容 | Gate | 预算 |
|---|---|---|---:|
| GC000 | deterministic implementation | contract 在 outcome 前可复算；错误值不被 grounding；证书 fail closed | CPU < 5 min |
| GC010 | combined-bank sanity | 全部 integrity checks；至少产生一个 goal predicate；无 collection error | CPU/tool < 15 min |
| GC020 | combined-bank feasibility | 固定 30-event role；原 Gate + Goal AUC 至少高于 receipt-only 0.05 | CPU/tool < 40 min |

## Compared systems

1. Frozen V7 receipt baseline。
2. DeltaGuard V2 explicit receipt validity。
3. Goal Contract + DeltaGuard typed public predicates。

## Ground truth

Contract freeze 和 collection 不读取标签。固定采集完成后才使用 exact Shadow
`rescue_preference` / `reverse_preference` 评估，禁止用另一个模型的输出作标签。

## Stop rule

若 Goal Contract feasibility 仍然全部弃权或 Reverse recall 为 0，则停止在当前
bank 上增加手写 receipt 规则；不放宽 Gate，不把 development 结果写成正向论文结论。
