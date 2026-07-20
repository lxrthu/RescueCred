# DeltaGuard 实验实现计划

## 目标

实现并验证 `FINAL_PROPOSAL.md` 中冻结的 Public Paired Deltas：只使用部署可见公开 observer，对隔离 A/B 分支产生 typed state delta，并在完整 Pareto 证书下才恢复 A。

## Milestones

| Milestone | 内容 | Gate | 预算 |
|---|---|---|---:|
| M0 | compile + deterministic unit tests | observer、certificate、contract、协议测试全部通过 | CPU < 5 min |
| M1 | ToolSandbox sanity | 每类 2 个源事件；标签采集隔离；输出完整 ledger | CPU < 30 min |
| M2 | feasibility | 每类 10 个固定源事件；固定 HMAC probe | CPU < 2 h |
| M3 | full pilot | 3 类 x 80 固定源事件，共 240 | CPU/tool execution；无 GPU 训练 |
| M4 | independent certification | 仅 M3 通过后；独立 episode 单元 | 后续单独冻结 |

## Compared systems

1. 冻结 V7 one-step receipt score。
2. DeltaGuard fixed-Q typed public deltas。
3. 相同 Q、相同调用预算下的 typed deltas + contract-only abstention。

## Ground truth

部署和采集阶段不读取标签。所有指标在固定采集完成后，对照原 paired branch 的 exact Shadow `rescue_preference` / `reverse_preference`。不使用另一个模型的输出作为 ground truth。

## Fixed full protocol

- families：messaging、reminders、settings；
- 每类 80 个源事件，每类 attempt cap 120；
- HMAC-SHA256 public key 固定为 `RescueCredit/DeltaGuard/public-acquisition/v1`；
- acquisition probability 0.25；
- 每类 probe cohort 至少 6 Rescue + 6 Reverse，否则 `INCONCLUSIVE`；
- conditional AUC >= 0.75；
- AUC gain over V7 >= 0.10；
- whole-stream probe rate <= 0.30；
- 不在 pilot 中作形式化 Rescue-risk 声明。

## Stop rules

- 公开 observer 或 prefix replay 访问隐藏数据库/evaluator：立即停止。
- 固定源流或 class coverage 不足：`INCONCLUSIVE`，不得补采到通过。
- Full typed-delta gate 失败：停止 DeltaGuard，不增加 learned scorer、长 horizon 或新 RL loss。
- Contract 不满足零额外 harm、零 Reverse hit 损失、至少避免一个 harm：删除。
