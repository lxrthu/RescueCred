# 路线 A：冻结 AppWorld 混合纠错库

本阶段只构建论文实验共用的纠错库，不启动主训练。

## 实验边界

- 只读取 AppWorld `train` 的 90 个任务。
- Harness 只接收任务指令、公开 OpenAPI schema、此前可见回执和待修复动作。
- `correction_bank.public.jsonl` 是 Mask+Correction 与 RescueCredit-v2 唯一允许读取的纠错输入。
- `offline_audit.private.jsonl` 仅用于离线统计，训练脚本禁止读取。
- 构建器不会伪造 Shadow 因果收益；后续 V2 pilot 必须通过真实快照分支产生该信号。

## 新窗口进入环境

```bash
cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
source /home/hxy/miniconda3/etc/profile.d/conda.sh
conda activate /data/hxy/venvs/rescuecredit-appworld
unset VIRTUAL_ENV
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit
```

API 密钥只修改：

```text
/data/hxy/projects/RescueCredit/.env
```

必须设置 `AZURE_OPENAI_API_KEY`，不要把密钥写入脚本。

## 构建

```bash
bash scripts/cloud/run_appworld_route_a_bank.sh
```

约每 10 秒查看一次：

```bash
tail -f outputs/appworld_route_a_bank_train90_seed42/console.log
```

完成后返回：

```bash
cat outputs/appworld_route_a_bank_train90_seed42/manifest.json
cat outputs/appworld_route_a_bank_train90_seed42/bank_gate.json
```

只有 `bank_gate.json` 中 `passed=true` 才进入同 bank 的 seed-42 公平训练。
