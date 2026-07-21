# EditCredit Variance and Convergence Preregistration

**冻结时间**：2026-07-21 15:55，在首次 EditCredit 服务器运行和查看任何结果之前。

## Claim boundary

本实验检验 EditCredit 是否在自己的 edit-local objective 下具有更低的 stochastic gradient noise scale，并在 matched event presentations 下更快达到 FullAction preference 的最终 held-out 水平。由于两个方法优化的 objective 不同，variance ratio 不解释为同一 estimand 的无偏估计器方差；它必须与 convergence 和最终 non-inferiority 一起报告。

## Frozen checkpoints

Checkpoint 必须落在 optimizer step 完成后：

```text
presentations = [0, 40, 80, 128, 256, 378]
```

`0` 为相同 base checkpoint；其余 checkpoint 使用完全相同的数据、presentation budget、optimizer 和 LoRA 配置。报告 presentations、optimizer steps 和 wall-clock time。

## Gradient-noise audit

- 在相同 base checkpoint、相同 LoRA 初始化和同一个 126-event frozen bank 上计算每事件梯度。
- FullAction 与 EditCredit 分别使用各自冻结的 production objective。
- 使用同一个 128-bucket、独立 affine universal hash/Rademacher sign 的 deterministic CountSketch，并核验 LoRA 初始参数哈希完全相同。
- 主要指标：
  - `gradient_noise_scale = trace(cov(g_i)) / ||mean(g_i)||^2`；
  - 与实际 `gradient_accumulation=8` 对齐的 event-bootstrap minibatch-gradient MSE 到各自 full-bank mean；
  - EditCredit/FullAction ratio。
- 不以更短 completion 的 raw variance 单独支持结论；必须同时报告 wall-clock cost 和 convergence curve。

## Convergence audit

- 每个 checkpoint 在 5-fold task-disjoint test 上评分。
- 每折 threshold 只由对应 calibration fold 选择，约束 empirical Rescue drop `<= 0.02`。
- Gate 独立从 frozen bank 重建 labels、roles、threshold 和 score derivations。
- 主要指标：balanced-accuracy AULC、扣除各自 p0 后的 baseline-adjusted AULC、达到 FullAction 378-presentation 最终 balanced accuracy 的最早 presentation、最终 Rescue/Reverse/balanced accuracy。
- p0 的评分表示差异单独报告；只有 p0 优势而没有训练后的 adjusted-AULC 增益，不能解释为 optimization convergence。

## Frozen go gate

必须同时满足：

```text
gradient_noise_scale_ratio <= 0.70
minibatch_gradient_mse_ratio <= 0.70
relative_balanced_accuracy_AULC_gain >= 0.10
baseline_adjusted_balanced_accuracy_AULC_gain >= 0.05
presentations_to_full_action_final_ratio <= 0.50
final_balanced_accuracy_editcredit >= final_balanced_accuracy_full_action
final_rescue_drop_vs_default_B <= 0.02
primary_EditCredit_final_gate = PASS
```

若只降低 gradient noise、但收敛曲线或最终 non-inferiority 不通过，只能报告 optimization diagnostic，不能声称 sample-efficiency improvement。
