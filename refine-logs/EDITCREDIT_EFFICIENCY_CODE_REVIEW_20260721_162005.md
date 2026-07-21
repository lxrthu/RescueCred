# EditCredit Efficiency Experiment Code Review

审查时间：2026-07-21 16:20（首次服务器运行前）。

## 结论

`DEPLOY YES`

## 首轮阻断项及修复

1. 原梯度投影每 128 个参数周期性同桶同符号，不能作为合法 CountSketch。现已改为两个独立 affine-mod-prime universal hash，分别生成 bucket 与 Rademacher sign；Gate 独立重建并核验哈希系数。
2. 两方法的 LoRA 初始状态原先只绑定随机种子。现记录所有 trainable parameters 的 SHA-256，并要求 FullAction 与 EditCredit 完全一致。
3. 原 task bootstrap 不对应训练中的 gradient accumulation。现改为 `batch_size=8` 的 event-bootstrap minibatch-gradient MSE。
4. 原收敛 Gate 可能把 p0 评分表示差异误写成优化收益。现同时要求 absolute AULC 与扣除各自 p0 的 baseline-adjusted AULC 增益。
5. efficiency Gate 现强制 primary final Gate 通过；主 Gate 失败时 runner 不会成功退出。
6. 新增 CountSketch 周期、初始化哈希 mismatch 和方差 artifact tamper 回归测试。

## 验证

- Ruff：通过。
- `py_compile`：通过。
- 相关 pytest：通过；本机缺少 torch 的 3 项测试跳过，服务器 runner 会在含 torch 环境重新执行。
- `git diff --check`：通过。

## 剩余边界

本轮仍是单 projection seed、128 buckets、seed-42 frozen-bank feasibility。即使 Gate 通过，也只能触发 seeds 43/44 的确认实验，不能直接形成 paper-facing efficiency claim。
