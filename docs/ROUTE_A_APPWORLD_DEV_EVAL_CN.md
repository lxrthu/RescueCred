# Route-A AppWorld Dev 配对任务评测

该阶段不再训练。它在 AppWorld `dev` 上为每个可用任务冻结一个未见缺参事件，并比较相同 seed-42 Mask/V2 adapter 的官方 requirement score。

修正版协议中，reference-free Harness 先从可见值生成同一个候选 B。Mask/V2 adapter 不自由生成，而是按训练目标计算 A/B 平均 token log-prob 并选择。随后同一个 Azure GPT-4o temperature-0 continuation 只根据可见状态继续任务；不再执行 reference suffix。AppWorld reference trajectory 仅重建候选发生前的受控状态并调用官方 evaluator，不进入 adapter 或 continuation 输入。不会读取 `test_normal` 或 `test_challenge`。

## 运行

把 `dist/PASTE_ROUTE_A_APPWORLD_DEV_TO_SERVER.sh` 的全部内容粘贴到服务器。脚本先跑每种方法 3 个事件的 GPU sanity，再自动选择两张最空闲 GPU 并行跑完整 dev。

```bash
cd /data/hxy/projects/RescueCredit
tail -f outputs/route_a_appworld_dev_seed42_v2/driver.log
```

单独查看：

```bash
tail -f outputs/route_a_appworld_dev_seed42_v2/mask/console.log
tail -f outputs/route_a_appworld_dev_seed42_v2/v2/console.log
```

结束后返回：

```bash
cat outputs/route_a_appworld_dev_seed42_v2/events/manifest.json
cat outputs/route_a_appworld_dev_seed42_v2/mask/eval_summary.json
cat outputs/route_a_appworld_dev_seed42_v2/v2/eval_summary.json
cat outputs/route_a_appworld_dev_seed42_v2/gate.json
```

Gate 在运行前固定：两边使用同一事件集、至少 20 个有效配对事件、V2 task success 不下降，并且 V2 官方平均 requirement score 严格提高。
