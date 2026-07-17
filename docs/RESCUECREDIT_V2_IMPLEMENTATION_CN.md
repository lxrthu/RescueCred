# RescueCredit-v2 实现说明

## 核心变化

新增两个保留旧版本的实验方法：

- `mask_correction_v2`：Deployable Harness + 三值 validity 普通纠错偏好，不使用 Shadow。
- `rescuecredit_v2`：在完全相同的 Harness 和普通纠错偏好上，增加稀疏审计因果偏好。

旧 `mask_correction` 和 `rescuecredit` 不删除，只作为 Oracle/V1 历史消融。

V2 中干预前和干预位置的 GRPO advantage 与 Mask 一样为零，不再使用 `G0_hat`。只有实际抽中审计、且 replay 有效的事件才能产生因果偏好。

## 偏好定义

令 A 为策略动作，B 为 Harness 修复动作，动作概率使用动作 token 的平均 log-prob：

```text
m = mean_logp(B) - mean_logp(A)
```

审计事件使用真实逐事件概率：

```text
delta = assisted_return - shadow_return
weight = min(abs(delta) / audit_probability, 2.5)
```

Validity 决定普通偏好；Shadow 决定额外因果偏好。任何 validity 为 `unknown` 时不产生偏好。B 语义有效、A 语义无效但 `delta < 0` 时标记为 `trajectory_conflict`，不会把无效 A 学回去。

总损失：

```text
L = L_mask_grpo + lambda_kl * L_kl
    + lambda_corr * L_validity_preference
    + lambda_causal * L_shadow_causal_preference
```

## 公平预算

`--budget-mode main --main-interaction-budget 2000` 表示两种方法都至少达到 2000 个主环境步，最后一个同步 batch 的小幅超量记录为 `budget_overshoot`。Shadow steps 不挤占主训练数据，单独作为额外成本报告。

固定总成本实验仍可使用：

```text
--budget-mode total --total-interaction-budget 2000
```

## 日志

`preference_events_rank*.jsonl` 逐事件记录：

- A/B executable validity 与 semantic validity；
- assisted/shadow return 和 delta；
- audit draw、真实 audit probability；
- causal decision、weight、direction；
- 长度归一化 preference margin；
- correction loss 和 causal loss。

## 第一阶段 GPU Smoke

```bash
export CUDA_VISIBLE_DEVICES=1,3
bash scripts/cloud/run_v2_smoke_2gpu.sh
```

只有 `outputs/rescuecredit_v2_smoke_seed42/smoke_gate.json` 中 `passed=true`，才能运行 2000-main-step 的公平 pilot。
