# 评测与统计契约

本项目的 API-Bank 结果是受控、可重放的代理评测，不是 API-Bank 官方 leaderboard 分数。

- 正确 reference action 推进成功谓词。
- schema 合法但偏离 reference 的调用会真实消耗一步，并返回 `no_effect`；episode 不立即终止，策略可以在后续步骤恢复。
- 工具不存在、缺少必需参数或包含 schema 外参数时，programmatic checker 终止 episode。
- `S_off` 是关闭 harness 后的 task-level 终局成功率。
- `first_pass` 是 H0 轨迹中，每个动作首次生成时与当步 reference action 精确相符的比例；它不是 task success。
- Full Shadow 的 `G0` 真值必须跨事件具有非零方差；`Var(G0)==0` 时 estimator 评测直接失败，禁止把不可辨识结果用于论文结论。
- shadow 到达 horizon 但没有终局结果时记为 censored / `replay_valid=false`，不能当作零回报真值。
- 配对比较先在每个 task 内平均三个 seed 的差值，再对 task 做 bootstrap，避免把 task×seed 当作独立样本。
- 聚合器同时检查模型 revision、训练 split hash、评测 split hash、交互预算与关键训练超参数。

事件中的 `state_ref` 指向内容寻址的持久化 snapshot。snapshot 文件包含 task、环境状态和 RNG state；`RescueEvent.metadata` 保存 snapshot digest、generation seed 与 audit seed，可由另一进程恢复并核对 `state_hash`。
