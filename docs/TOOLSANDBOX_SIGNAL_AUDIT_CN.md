# ToolSandbox Harness/Shadow 信号审计

这一阶段不修改已经冻结的 RescueCredit V3.1，也不直接训练新模型。目标是回答一个更基础的问题：在 ToolSandbox 的单用户多工具场景中，部署可见的 Harness 修复是否能产生足够密集、可复现的因果差值。选择器优先纳入全部可用的状态依赖场景，再用其他多工具场景补足 40 条。

## 两条审计轨道

1. `natural_visible_error_repair`：模型先提出 A。只有 A 的真实工具执行产生可见错误回执时，Harness 才能使用当前对话、公共工具 schema、A 和该回执生成 B；不确定时必须 abstain。
2. `controlled_missing_argument`：先由无 reference 的模型产生 schema 完整动作 B，再仅依据公共 schema 删除一个必填参数构造 A。它用于测试 Shadow 机制的信号密度，不代表自然 Harness 准确率。

A/B 从同一个 ToolSandbox 快照开始，使用同一温度为 0 的 continuation policy，最终差值为：

```text
delta = official ToolSandbox similarity(B branch)
      - official ToolSandbox similarity(A branch)
```

Milestone DAG、minefield 和 evaluator 分数只进入离线分支评分，绝不进入 proposal、repair 或 continuation worker。

## 新窗口完整命令

```bash
cd /data/hxy/projects/new/RescueCredit

source "$(conda info --base)/etc/profile.d/conda.sh"

# 第一次安装/核对官方 ToolSandbox 固定版本
bash scripts/cloud/setup_toolsandbox_stage0.sh

# 3 场景集成 smoke，然后自动跑 40 场景正式信号审计
tmux new-session -d -s toolsandbox_audit42 \
  "cd /data/hxy/projects/new/RescueCredit && \
   bash scripts/cloud/run_toolsandbox_signal_audit.sh \
   2>&1 | tee outputs/toolsandbox_signal_audit_40_seed42.driver.log"
```

查看进度：

```bash
tmux ls | grep toolsandbox_audit42 || echo TMUX_ENDED
tail -f outputs/toolsandbox_signal_audit_40_seed42.driver.log
```

结束后返回：

```bash
cat outputs/toolsandbox_stage0/gate.json
cat outputs/toolsandbox_signal_audit_40_seed42/audit_summary.json
cat outputs/toolsandbox_signal_audit_40_seed42/quality_gate.json
```

## 决策边界

- `controlled nonzero rate >= 20%` 且自然轨道至少产生 3 个可审计事件：冻结 ToolSandbox V3.1 对比协议。
- controlled 信号足够但 natural 覆盖不足：先增强 Harness 触发/证据抽取，不训练。
- controlled 信号也不足：ToolSandbox 不能解决当前 credit-density 问题，停止迁移。

即使 gate 通过，这仍只是机制信号审计，不是端到端任务成功率证据。
