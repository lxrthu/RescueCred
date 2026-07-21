# Compensation Trap Evidence Code Review

时间：2026-07-21 23:27

## 首轮结论

`DEPLOY NO`。审查发现：approximate nearest-neighbor 在建图时提前使用标签；fresh protocol 缺 `reference_boundary`；历史/fresh ground truth 没有从官方 branch evidence 独立重算；freshness inventory、benchmark verifier、bootstrap 与 missingness Gate 不完整。

## 修复

- Mutual-nearest graph 完全由公开特征和跨任务约束构建，形成 pair 后才检查 Rescue/Reverse。
- Collision features 只使用 visible history、public schemas、A/B；scenario、source、task hash 和内部 prefix index 标为 provenance-only。
- 历史 benchmark 与 fresh confirmation 均验证官方 score provenance/trace/padded AUC，并重算 lexicographic decision、basis、value、weight 和 components。
- Verifier 与 collision Gate 共用完整 package validator；第三标签、task/split mismatch、统计或 schema/source 篡改均 fail closed。
- Fresh freeze 动态绑定全部历史 protocol/audit summaries，识别多种 scenario-hash keys，扫描任何既存 offset-205 artifact，并用 limit-14 sentinel 证明尾部恰好 13 个 scenarios。
- Fresh Gate 绑定 scenario/task/event derivation、runtime、worker、source、config、snapshot、missingness、task bootstrap prevalence/headroom intervals。

## 最终结论

`DEPLOY YES`。本地 `py_compile`、Ruff、8 tests、Bash syntax 和 `git diff --check` 全部通过。

限制：benchmark dataset card 保持 `release_authorized=false`；完成上游许可证审查前只允许内部实验，不得公开发布数据。

