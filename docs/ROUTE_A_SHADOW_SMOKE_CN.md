# 路线 A：AppWorld Shadow Credit Smoke

本阶段从已经冻结的 150 条 public bank 中取前 20 条，在完全相同的受控状态分别执行动作 A 与纠错动作 B，然后使用同一个冻结 Azure GPT-4o continuation policy 继续最多 6 步。

训练标签或参考动作不会进入 continuation prompt。AppWorld train reference prefix 仅用于重建受控事件状态，因此本实验应表述为 controlled-state causal-credit benchmark，而不是端到端无 Oracle agent training。

## 新窗口进入环境

```bash
cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
source /home/hxy/miniconda3/etc/profile.d/conda.sh
conda activate /data/hxy/venvs/rescuecredit-appworld
unset VIRTUAL_ENV
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit
```

## 执行

```bash
bash scripts/cloud/run_route_a_shadow_smoke.sh
```

## 查看进度

```bash
tail -f outputs/appworld_route_a_shadow_smoke_seed42/console.log
```

每完成一个事件都会输出 `progress`、`valid` 和 `nonzero`，不再出现长时间空日志。

## 返回结果

```bash
cat outputs/appworld_route_a_shadow_smoke_seed42/shadow_summary.json
cat outputs/appworld_route_a_shadow_smoke_seed42/shadow_gate.json
```

只有 `valid_events >= 10` 且 `nonzero_events >= 3` 才进入 Mask/V2 训练。

## Smoke 后的确认集

前 20 条只作为工程调试，不再进入确认或训练。确认实验使用剩余 `20:150` 共 130 条和 12-step Shadow：

```bash
bash scripts/cloud/run_route_a_shadow_confirm.sh
tail -f outputs/appworld_route_a_shadow_confirm_seed42/console.log
```

确认门为 `valid_events >= 100` 且 `nonzero_events >= 5`。
