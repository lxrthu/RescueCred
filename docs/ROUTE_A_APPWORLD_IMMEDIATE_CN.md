# Route-A AppWorld 确定性即时因果诊断

这一步不训练、不调用 Azure、不使用 GPU，也不续写轨迹。它在同一个 AppWorld 状态下分别执行候选动作 A 和 B，随后立刻调用 AppWorld 官方 requirement evaluator。这样可以排除续写随机性和参考后缀造成的天花板效应，直接检查 Mask/V2 选中的动作是否带来更好的即时状态。

输入沿用已经完成的 seed-42 dev 实验：

- `outputs/route_a_appworld_dev_seed42_v2/events/dev_events.public.jsonl`
- `outputs/route_a_appworld_dev_seed42_v2/mask/task_results.jsonl`
- `outputs/route_a_appworld_dev_seed42_v2/v2/task_results.jsonl`

## 在服务器运行

把 `dist/PASTE_ROUTE_A_IMMEDIATE_TO_SERVER.sh` 的全部内容粘贴到服务器终端。脚本会先跑 3 个事件的 sanity，再在 tmux 中跑完整 55 个事件。

```bash
cd /data/hxy/projects/RescueCredit
tail -f outputs/route_a_appworld_dev_immediate_seed42/driver.log
```

预计 3–10 分钟。CPU 即可，不占 H200，不读取 `.env`。

跑完后返回：

```bash
cat outputs/route_a_appworld_dev_immediate_seed42/immediate_summary.json
cat outputs/route_a_appworld_dev_immediate_seed42/gate.json
```

## 预注册 Gate

运行前固定以下条件，全部满足才通过：

- 至少 40 个有效 A/B 配对事件；
- 至少 5 个即时官方得分非零差异事件；
- Mask/V2 至少在 3 个事件上选择不同；
- V2 平均即时官方得分严格高于 Mask；
- V2 赢的事件数严格多于输的事件数；
- V2 在非零因果事件上的方向准确率严格高于 Mask；
- 无 Azure、无 continuation、无 reference suffix、无 test split。

该结果是确定性即时效应证据，不等同于完整 AppWorld 任务成功率。
