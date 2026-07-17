# Route-A Seed-42 同 Bank 偏好 Pilot

这一步用同一个冻结 AppWorld correction bank、公用训练/验证划分和同一个 Qwen2.5-7B 基座，公平比较：

- `mask`：所有事件都固定学习 `B > A`；
- `v2`：仅使用非零 Shadow credit；`delta > 0` 学习 `B > A`，`delta < 0` 反向学习 `A > B`。

动作概率采用平均 token log-prob，训练目标是相对冻结基座的 DPO-style 偏好损失。两种方法分别在两张最空闲 GPU 上并行运行。

## 在服务器执行

将 `dist/PASTE_ROUTE_A_SEED42_TO_SERVER.sh` 的全部内容粘贴进服务器终端即可。脚本会安装本次新增文件、跑单测并在 tmux 中启动实验。

查看总日志：

```bash
cd /data/hxy/projects/RescueCredit
tail -f outputs/route_a_seed42_preference_pair/driver.log
```

查看两边训练：

```bash
tail -f outputs/route_a_seed42_preference_pair/mask/console.log
tail -f outputs/route_a_seed42_preference_pair/v2/console.log
```

结束后返回：

```bash
cat outputs/route_a_seed42_preference_pair/data/manifest.json
cat outputs/route_a_seed42_preference_pair/mask/eval/eval_summary.json
cat outputs/route_a_seed42_preference_pair/v2/eval/eval_summary.json
cat outputs/route_a_seed42_preference_pair/gate.json
```

## 结果边界

这是 held-out frozen-bank 的偏好学习工程 gate，不是 AppWorld 完整任务成功率主结果。只有该 gate 通过，才进入配对 AppWorld dev task-success 评测。
