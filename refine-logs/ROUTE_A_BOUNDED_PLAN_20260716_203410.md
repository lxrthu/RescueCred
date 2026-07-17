# Route-A 4/8-step Bounded-Horizon 诊断计划

## 目标

判断 RescueCredit-v2 相比同一 correction bank 上的 Mask baseline，是否更常选择在有限后续交互后产生更高 AppWorld 官方 requirement score 的动作。

## 固定协议

- 数据：已冻结的 55 个 seed-42 AppWorld dev 事件。
- 选择：沿用已经冻结的 Mask/V2 A/B 选择结果，不重新训练、不重新打分。
- 分支：每个事件的 A/B 从完全相同的 reference-prefix 状态分别启动。
- continuation：同一 Azure GPT-4o temperature-0 visible-only policy；输入只包含任务说明、公开 schema、事件上下文和当前分支可见历史。
- coupling：continuation 请求按可见输入做内容寻址缓存；相同输入必须复用同一动作。
- horizon：4 和 8 步；4 步轨迹必须是同一 8 步 policy 的前缀。
- 评分：到达 horizon、策略停止或任务完成后，调用 AppWorld 官方 requirement evaluator。
- 禁止：reference suffix、expected action、evaluator output 进入 continuation、test split。

## 主要 Gate（horizon=8）

- 有效配对事件不少于 40；
- 非零 A/B 因果事件不少于 5；
- Mask/V2 选择差异不少于 3；
- V2 平均 bounded score 严格高于 Mask；
- V2 胜场严格多于负场；
- V2 非零事件方向准确率严格高于 Mask；
- continuation cache 无冲突，且无 reference suffix/test access。

Gate 未通过时不扩 seed。horizon=4 只作为延迟效应诊断，不作为主 Gate。
