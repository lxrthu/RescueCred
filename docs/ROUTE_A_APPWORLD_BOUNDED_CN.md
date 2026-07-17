# Route-A AppWorld 4/8 步 Bounded-Horizon 诊断

该实验用于判断即时 evaluator 看不到的延迟效果。每个冻结 dev 事件都会从同一 reference-prefix 状态分别执行 A/B，然后由同一个 Azure GPT-4o temperature-0 visible-only policy 最多继续到 4 步和 8 步，再调用 AppWorld 官方 evaluator。

H4 和 H8 不会使用两套策略：策略始终看到相同的最大 8 步预算，continuation 请求按可见输入做内容哈希缓存。因此 H4 是 H8 的严格缓存前缀。模型看不到 reference action、expected value、evaluator output 或 test split，也没有 reference suffix。

## 运行

把 `dist/PASTE_ROUTE_A_BOUNDED_TO_SERVER.sh` 的全部内容粘贴到服务器终端。脚本会检查 `.env` 中已有的 Azure Key，先跑 3 个事件 sanity，然后在 tmux 中运行全部 55 个事件。

```bash
cd /data/hxy/projects/RescueCredit
tail -f outputs/route_a_appworld_dev_bounded_seed42/driver.log
```

预计约 10–25 分钟，不需要 GPU。若 Azure 限流，时间可能更长；缓存会保留，重新运行时相同可见请求不会重复调用 API。

完成标志：

```text
ROUTE_A_BOUNDED_FINISHED
```

完成后返回：

```bash
cat outputs/route_a_appworld_dev_bounded_seed42/bounded_summary.json
cat outputs/route_a_appworld_dev_bounded_seed42/gate.json
```

主 Gate 使用 horizon=8：至少 40 个有效配对、至少 5 个非零因果事件、V2 平均分和因果方向准确率严格提升、胜场多于负场，并通过 cache/reference 边界检查。H4 只用于判断效果是否需要更长轨迹。
