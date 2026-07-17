# AppWorld Stage 0：环境与接口探针

这一步不是训练，也不使用 GPU。它只确认三件事：

1. AppWorld 官方安装、数据和 train/dev 验证可以在服务器运行；
2. function-calling API 文档可以转换为 RescueCredit 的原子动作 schema；
3. `save_state/load_state` 方法与 train ground-truth 的离线评分结构可用。

注意：AW0 只确认 checkpoint API 可以调用，**不会宣称真实分支回放已经通过**。
AW1 开始前还必须执行一个真实的状态修改 API，并验证数据库和交互计数都精确恢复。

所有大型数据放在数据盘项目根目录的 `data/tasks`（当前服务器为
`/data/hxy/projects/RescueCredit/data/tasks`）。Harness、策略观测和训练标签不会读取
`world.task.ground_truth`；Stage 0 只输出对象类型、字段名和数量，不导出受保护任务内容。

## 服务器执行

```bash
cd /data/hxy/projects/RescueCredit
tmux new-session -d -s appworld_stage0 \
  "cd /data/hxy/projects/RescueCredit && bash scripts/cloud/setup_appworld_stage0.sh 2>&1 | tee outputs/appworld_stage0_console.log"
```

查看进度：

```bash
tail -f outputs/appworld_stage0_console.log
```

结束后返回：

```bash
cat outputs/appworld_contract_probe/gate.json
cat outputs/appworld_contract_probe/contract_probe.json
grep -nE 'Traceback|FAILED|ERROR' outputs/appworld_stage0_console.log | tail -n 30
```

只有 `gate.json` 中 `passed=true`，才进入 30 个 train 任务的 Harness + Shadow 因果审计。
