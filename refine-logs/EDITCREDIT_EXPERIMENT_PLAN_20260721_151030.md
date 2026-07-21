# EditCredit 最小实验计划

**问题**：Harness 将策略动作 A 替换为 B 后，执行 B 获得的回报会沿着 A 的 log-probability 回传；现有 Mask 能切断该错误信用，但现有全动作偏好损失容易学成“总信 B”或“总反 B”。

**方法假设**：EditCredit 在干预点建立信用防火墙，并只对 A/B 的最小工具调用差异分配由成对 Shadow 回报确定的有符号信用；Rescue 约束阻止 Reverse 更新破坏已有的有益修正能力。

**日期**：2026-07-21

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：执行 B 的回报不应强化未执行的 A；反事实信用应只落到造成结果差异的动作字段 | 这是 RescueCredit 的核心 action-substitution 错误 | 梯度审计中 A 的污染梯度严格为零；真实 pair 上 balanced accuracy 和 Reverse recall 优于全动作偏好 | B0, B1 |
| C2：Rescue 风险约束能避免 Rescue/Reverse 此消彼长 | 直接回应现有 V5 失败 | Rescue accuracy 不低于 Mask baseline 2 个百分点以上，同时 Reverse recall 有实质提升 | B1, B2 |
| Anti-claim：收益只是位置、A/B 标签或任务族捷径 | 排除“总信 Harness/总反 Harness” | 交换 A/B 展示顺序后预测不变；提升不集中于少数任务；source-only probe 接近随机 | B1 |

## Paper Storyline

- 主文必须证明：错误信用路径真实存在；EditCredit 切断该路径；局部反事实信用比全动作偏好更少产生 Harness-source shortcut。
- 附录可以支持：不同 edit granularity、置信阈值和约束系数。
- 明确删除：新的 Rescue/Reverse 分类器、更长 receipt、ActiveShadow 路由器和 RAPG allocator；这些不再进入本轮。

## Method Frozen for Pilot

对每个干预事件记录策略提出的 A、Harness 执行的 B，以及 exact replay-valid Shadow 的成对结果 `G_A, G_B`：

1. **Credit firewall**：干预步及其之前的 assisted-return policy advantage 置零，禁止 `G_B` 更新 A。
2. **Minimal edit**：在 canonical tool-call AST 上计算 A/B 差异，只保留 tool 名或发生变化的 argument field；相同字段不进入偏好损失。
3. **Signed counterfactual credit**：令 `delta = G_B - G_A`。仅当 exact replay 有效且 `|delta| > epsilon` 时，用 `sign(delta)` 决定 changed fields 的学习方向。
4. **Source invariance**：训练时随机交换 A/B 的序列化位置；标签来自 `delta`，不来自 `original/correction` 身份。
5. **Rescue constraint**：在独立 calibration fold 上选择拉格朗日系数，使 Rescue accuracy 相对 Mask 的经验下降不超过 `delta_R = 0.02`；若无可行系数则弃权并判定方法失败。

本 Pilot 不使用新的分类器，不使用旧 development outcome 调阈值，不访问已知 confirmation 结果选择配置。

## Experiment Blocks

### B0：梯度归属单元实验

- Claim tested：C1。
- Dataset：合成的两步 tool-call pair；包含 shared token、changed tool、changed argument 三种情况。
- Compared systems：Naive B-credit、Mask、现有 full-action pairwise、EditCredit。
- Decisive metrics：A-token contamination gradient；B changed-field gradient；unchanged-field gradient；A/B 序列化交换不变性。
- Success criterion：EditCredit 对 A 的 assisted-return 梯度和 unchanged fields 梯度在数值容差 `1e-8` 内为零；changed-field 方向与 `sign(G_B-G_A)` 完全一致；交换展示顺序不改变决定。
- Failure interpretation：实现没有真正建立信用边界，禁止进入真实数据训练。
- Cost：CPU，约 5–10 分钟。
- Priority：MUST-RUN。

### B1：冻结 126-pair 跨任务 seed-42 Pilot

- Claim tested：C1、C2 和 Anti-claim。
- Dataset：V4.4 的 126 个 exact replay-valid 非零 pair、38 个任务；只使用已有 frozen branch outcomes。按 task group 做 5-fold OOF；每折内部再拆 calibration，不让同任务跨集合。
- Compared systems：
  1. Mask baseline；
  2. 现有 full-action signed counterfactual preference；
  3. EditCredit；
  4. EditCredit + Rescue constraint。
- 公平设置：相同 base checkpoint、LoRA rank、训练 presentation 数、优化器、seed 和 checkpoint selection protocol；不得让 EditCredit 多看事件。
- Primary metrics：Rescue accuracy、Reverse recall、balanced accuracy、相对 Mask 的 Rescue drop、task-macro improvement fraction。
- Shortcut diagnostics：A/B 展示顺序交换一致率；source-only probe AUC；每任务 margin delta；changed-field 与 unchanged-field gradient norm。
- Frozen go gate：
  - `RescueDrop <= 0.02`；
  - `ReverseRecall >= Mask + 0.10`；
  - `BalancedAccuracy >= full-action preference + 0.05`；
  - task-macro improvement fraction `>= 0.50`；
  - source-only ROC-AUC `<= 0.55`，顺序交换一致率 `>= 0.95`。
- Failure interpretation：局部信用仍不能跨任务泛化；停止训练方法，将贡献限制为 credit-contamination 诊断和负边界。
- Cost：1 张 H200/A100，约 30–60 分钟。
- Priority：MUST-RUN。

### B2：三随机种子与真实 rollout

- Claim tested：C1、C2 的稳健性和任务级效用。
- Entry condition：B1 全部 Gate 通过后才能运行。
- Setup：seeds 42/43/44；预注册一个未用于 V4–V9 选择的新任务集合。首先完成 unassisted rollout，再打开标签评分。
- Compared systems：Mask、full-action preference、EditCredit + constraint。
- Metrics：unassisted first-pass success、official task score、Harness intervention rate、Rescue/Reverse、均值与 task bootstrap 95% CI。
- Success criterion：EditCredit 对 Mask 的 unassisted score 置信区间下界不低于 0，点估计至少提升 3 个百分点；Rescue drop 不超过 2 个百分点；三个 seed 方向一致。
- Failure interpretation：离线 action-selection 改善不能转化为策略能力，不作 autonomous-improvement claim。
- Cost：预计 3–6 GPU 小时，加 ToolSandbox rollout 时间。
- Priority：MUST-RUN only after B1。

### B3：最小消融

- Claim tested：创新确实来自 edit-local credit 和约束，而非额外复杂度。
- Variants：去掉 firewall、full-action 替代 minimal edit、去掉 source randomization、去掉 Rescue constraint。
- Entry condition：B2 正向后才运行。
- Target：论文消融表。
- Priority：NICE-TO-HAVE until B2 passes。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 验证信用分配实现 | B0 四种 loss 的确定性梯度测试 | 所有梯度归属断言通过 | CPU <10 min | token/AST 对齐错误；先只做 canonical JSON field span |
| M1 | 快速否证主方法 | B1 seed 42，5-fold task OOF | 五项 frozen gate 全过 | 1 GPU，30–60 min | 126 pair 偏小；只作 feasibility，不作论文确认 |
| M2 | 检查随机性 | B1 seeds 42/43/44 | 三 seed 均满足 Rescue budget，平均指标过 Gate | 3 GPU h 左右 | 训练方差；固定数据顺序和 presentation budget |
| M3 | 验证真实能力 | B2 fresh-task rollout | unassisted score 与 Rescue non-inferiority 同时成立 | 3–6 GPU h | 新任务成本；仅在 M2 后启动 |
| M4 | 论文消融 | B3 | 核心组件删除后主指标显著下降 | 2–4 GPU h | 不提前消耗预算 |

## Compute and Data Budget

- 第一决策点：CPU 10 分钟 + 单卡 30–60 分钟。
- 全部主实验：约 6–10 GPU 小时，不含环境排队。
- 第一阶段不需要重新采集：使用现有 126 个 exact paired branches。
- 只有 M2 通过后，才需要一个新鲜、预注册的 rollout 集合。
- 最大瓶颈：跨任务泛化，而不是 Shadow 收集速度。

## Risks and Mitigations

- **现有 Mask 已经切断部分信用**：主比较必须是 full-action causal preference，而不是只打 Naive。
- **Minimal edit 仍可能等价于分类器**：通过 source randomization、source-only probe 和顺序交换测试排除身份捷径。
- **旧 confirmation 已被观察**：不得再把它作为配置选择或确认集；B2 必须预注册新任务集合。
- **126 pair 统计功效有限**：M1 仅是 go/no-go feasibility；不过 Gate 就停止，不通过反复调参。

## Final Checklist

- [x] 主张与成败 Gate 已冻结
- [x] Mask 和现有 full-action preference 是强基线
- [x] 第一阶段无需重新采集
- [x] shortcut 检验已进入主 Gate
- [x] 新任务仅在 seed-42 和三 seed Gate 后启动
- [ ] 实现 canonical AST edit span 和 source-randomized loss
- [ ] 运行 B0
- [ ] 运行 B1 seed 42
