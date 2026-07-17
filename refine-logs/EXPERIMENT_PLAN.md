# RescueCredit 实验计划

## Milestones

| Milestone | Runs | Gate | Budget |
|---|---|---|---:|
| M0 deterministic sanity | pytest + replay + randomization + Toy exact | 全部通过 | CPU < 1h |
| M1 controlled infrastructure | API-Bank prepare + injected patch smoke | replay failure = 0 | CPU < 1h |
| M2 one-seed pilot | Naive / Mask+Correction / RescueCredit, seed 42 | 核心指标至少一项改善 | 4×H200, ≤6h |
| M3 confirmatory | 三方法 × seeds 42/43/44 | equal interaction | 4×H200, ≤15h |
| M4 robustness | audit p / verifier noise / Tool-OOD | 只跑预注册比较 | ≤4h |

## Dataset

- Rescue-MDP：精确枚举，无数据泄漏。
- API-Bank-derived controlled v1：由官方 commit `483554eae102996f5ec1f4feab4e78ef29c2a394` 构建。
- split：按 API family 形成 tool-OOD；ID pool 按 goal template + action signature 分组切分。

## Compared methods

1. Naive H+GRPO：H3 assisted return 直接归因。
2. Mask + Correction：干预 proposal 的 policy advantage 严格置零，加 verified correction preference。
3. RescueCredit：确定性 local label 或随机 residual audit + Patch EMA + correction preference。
4. Full Shadow：作为高成本 oracle，只在估计器/小规模对比使用。

## Metrics and ground truth

`S_on/S_off/First-pass/IR/DG` 对照冻结 reference action sequence 和程序化 checker。估计器指标对照 Full Shadow 或 Rescue-MDP 精确 `G0`，不使用模型输出当标签。

## Stop rules

- replay 不一致：停止训练。
- estimator 在 100k sample gate 明显有偏：修复后重跑。
- RescueCredit 对 Mask + Correction 在 `S_off`、First-pass、G0 MSE/成本均无改善：不扩三 seed。
- 不为正结果修改 test split、patch 子集或主要指标。

