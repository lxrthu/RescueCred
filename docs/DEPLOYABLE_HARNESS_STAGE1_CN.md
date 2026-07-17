# Deployable Harness Stage 1：Reference 隔离与质量门

## 定位

- `OracleAPIBankHarness`：保留现有 `expected_action` 路径，只作为 Toy、诊断和能力上界。
- `DeployableAPIBankHarness`：论文主路径候选，不接收 `expected_action` 或 `reference_actions`。
- 当前阶段只验证 Harness；质量门通过之前不得接入 RescueCredit-v2 因果偏好损失。

论文可使用的准确表述是：

> Reference-free intervention generation and validation.

训练环境的最终奖励仍可由冻结参考 checker 计算，因此不能声称整个训练过程无 Oracle。

## 可见信息边界

Deployable Harness 只允许读取：

- 当前用户目标；
- 当前公开环境状态；
- 工具 Schema；
- 已经发生的工具回执；
- 当前策略动作 A。

`public_harness_observation()` 会移除 `reference_actions`、标准参数答案和参考动作数量等字段。

## 保守的第一版能力

当前只自动执行：

1. 缺失必需参数；
2. 缺失值能从当前目标的局部上下文唯一确定，或来自可信工具回执；
3. 修复后的动作通过三值语义验证，结果为 `true`。

已有参数不会仅凭目标文本被覆盖，因为多意图任务可能出现同名参数的多个值。无法判断时返回 `unknown`，只给反馈，不改执行动作，也不产生训练 pair。

## 独立质量门

运行：

```bash
bash scripts/cloud/run_deployable_harness_gate.sh
```

默认门槛：

- coverage >= 10%；
- correction precision >= 90%；
- single-step rescue rate >= 10%；
- clean-action harm rate <= 1%。

输出：

- `outputs/deployable_harness_audit_dev/case_results.jsonl`
- `outputs/deployable_harness_audit_dev/harness_metrics.json`
- `outputs/deployable_harness_audit_dev/quality_gate.json`

其中 reference 只在该离线评测脚本中用于构造错误案例和评分，绝不传给 Deployable Harness。

## 当前本地结果

19 个 dev 任务的确定性审计结果：

- 17/17 自动纠正正确；
- correction precision = 100%；
- clean-action harm rate = 0%；
- coverage = 9.29%；
- single-step rescue rate = 9.29%；
- 总质量门未通过，因为效用指标略低于预设 10%。

因此当前结论是：安全性门通过，效用门未通过。下一步应增加一种同样高精度的可部署纠错来源，例如结构化错误回执修复；在总质量门通过之前不实现或运行 V2 训练。

## 冻结 Qwen 补全增强

规则无法补全缺失必填参数时，可以让冻结的 Qwen 根据同一份可见信息生成一个候选 B。该模型：

- 不接收 reference 或 expected action；
- 只能保持 A 的工具名并补全缺失字段；
- 使用确定性解码；
- 输出仍需通过三值语义 validator；
- 不能通过验证时保持 `unknown`，不改变真实执行。

在服务器空闲单卡上运行：

```bash
export CUDA_VISIBLE_DEVICES=2
bash scripts/cloud/run_deployable_harness_qwen_gate.sh
```

结果写入 `outputs/deployable_harness_audit_dev_frozen_qwen/`。只有该目录中的 `quality_gate.json` 显示 `passed: true`，才允许进入 V2 loss 实现。
