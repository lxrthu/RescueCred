# ToolSandbox DeltaGuard 运行说明

## 实现边界

DeltaGuard 不训练新模型。它冻结原 Policy、Harness 和 V7 receipt baseline，
只在 HMAC 选中的事件上执行：

1. 重放部署可见前缀；
2. 从同一前缀隔离执行 A/B；
3. 比较两边显式 action receipt，并运行完全相同的公开只读 observer；
4. 根据 receipt validity、typed delta 和 Pareto dominance 决定是否恢复 A；
5. 未知、冲突、重放失败一律保持 B。

Runner 要求事先生成只含 allowlist 字段的 `public_events.jsonl`；它不会在同一次
运行中读取 raw bank 并生成 public bank。Freeze 和 collect 阶段只能读取这个物理
隔离的 public bank，不能读取 `decision`、branch outcome、
official evaluator、reference action 或隐藏数据库。原 raw bank 只在 collection
完成后的 evaluation/gate 阶段作为 sealed label source 打开。

## 当前支持的公开 observer

- settings：cellular、Wi-Fi、location service、low battery mode；
- messaging：`get_cellular_service_status`、`search_messages`；
- contacts：`search_contacts`；
- reminders：`search_reminder`。

工具名被 scramble 时，registry 使用公开 schema description 识别能力。

对于 `search_messages`、`search_reminder` 等不改变状态的读操作，V2 使用 action
receipt 本身作为保守证据：只有一边明确执行成功而另一边出现非幂等异常时才形成
优势；两边都成功、异常含 `already/unchanged/no change` 或证据冲突时一律弃权并
保持 B。该规则不判断语义相关性，也不读取 reference action。

V3 Goal Contract 在任何 branch receipt 产生前，根据可见用户指令、固定 A/B 与
公开 schema 生成并写入 protocol lock。它只加入三类确定性公开谓词：工具 family
是否与指令显式目标一致、动作参数中有多少值可在指令中验证来源、返回结果包含多少
已锁定目标值。未在指令中逐字或规范化出现的值不能进入 contract。Gate 会验证
contract 的 pre-observation 标记、内容哈希及 freeze/collection/certificate 三段
一致性，并在同一批 probe 上独立报告 V2 receipt-only ablation。
其中 role alignment 与 grounded-argument coverage 只用于诊断，不能进入路由
Pareto 向量；恢复 A 必须至少有一个真实 post-receipt 或 state witness。V3 还要求
conditional AUC 比同批 V2 receipt-only 至少高 0.05，否则方法 Gate 失败。

## 一次性准备物理隔离的 public bank

这一步属于已有 Shadow pair bank 的离线封存，不得与新的 DeltaGuard collect
混在同一个 runner 中：

```bash
mkdir -p outputs/toolsandbox_deltaguard_source_bank

$TOOLSANDBOX_PYTHON scripts/export_toolsandbox_deltaguard_public_bank.py \
  --raw-events outputs/toolsandbox_v44_candidate_seed42/full_offset85_h8/candidate_events.jsonl \
  --output outputs/toolsandbox_deltaguard_source_bank/public_events.jsonl \
  --manifest outputs/toolsandbox_deltaguard_source_bank/public_bank_manifest.json
```

生成后 public bank 只含 allowlist 字段；pilot runner 在 collection 完成前不会打开
原 raw/label bank。

## 服务器快速 feasibility

Runner 使用两个隔离运行时：`TOOLSANDBOX_PYTHON` 只负责环境重放与采集，
`RESCUECREDIT_PYTHON` 负责 freeze、Torch baseline、evaluation 和 gate。默认分别为
`/data/hxy/venvs/rescuecredit-toolsandbox/bin/python` 与仓库 `.venv/bin/python`。

```bash
cd /data/hxy/projects/new/RescueCredit

TOOLSANDBOX_PYTHON=/data/hxy/venvs/rescuecredit-toolsandbox/bin/python \
RESCUECREDIT_PYTHON=$PWD/.venv/bin/python \
bash scripts/cloud/run_toolsandbox_deltaguard_seed42.sh \
  feasibility \
  outputs/toolsandbox_deltaguard_feasibility_seed42 \
  outputs/toolsandbox_v7_active_shadow_seed42/model/active_shadow.pt \
  outputs/toolsandbox_deltaguard_source_bank/public_events.jsonl \
  outputs/toolsandbox_v44_candidate_seed42/full_offset85_h8/candidate_events.jsonl
```

查看进度：

```bash
tail -f outputs/toolsandbox_deltaguard_feasibility_seed42/collection.log
```

查看结果：

```bash
cat outputs/toolsandbox_deltaguard_feasibility_seed42/feasibility_gate.json
```

如果 public bank 来源于多个 raw bank，命令末尾必须按 public manifest 中封存的
顺序追加全部 label bank。脚本会校验哈希和重复 `event_id`。

## Full frozen pilot

```bash
bash scripts/cloud/run_toolsandbox_deltaguard_seed42.sh \
  full \
  outputs/toolsandbox_deltaguard_full_seed42 \
  outputs/toolsandbox_v7_active_shadow_seed42/model/active_shadow.pt \
  outputs/toolsandbox_deltaguard_source_bank/public_events.jsonl \
  /path/to/fresh_bank_1/candidate_events.jsonl \
  /path/to/fresh_bank_2/candidate_events.jsonl
```

Full 模式硬编码以下协议，不能通过命令覆盖：

- messaging、reminders、settings 三个 family；
- 每类 80 个源事件，共 240 个；
- 每类最多检查 120 个候选；
- 固定公开 HMAC key，25% 期望 probe rate；
- 每类 probe 后至少 6 个 Rescue、6 个 Reverse，否则 `INCONCLUSIVE`；
- typed-delta AUC 不低于 0.75；
- 相比冻结 V7 receipt baseline 的 AUC 提升不低于 0.10；
- 全源流实际 probe rate 不高于 0.30。

Full 模式还要求 `V7_TRAIN_FILE` 与新源流在 `event_id` 和 `task_id_hash` 上都不
重叠；否则 freeze 会直接失败。默认值是 V4.4 的 `data/train.jsonl`，也可显式设置：

```bash
V7_TRAIN_FILE=/path/to/v7/data/train.jsonl bash scripts/cloud/run_toolsandbox_deltaguard_seed42.sh full ...
```

当前 V4.4 的 126-pair bank 主要适合 feasibility，不应冒充 240-event full
pilot。若 Full freeze 报某个 family 数量不足，应补充预先独立采集的新 raw
bank，而不是根据标签挑事件。

## Contract ablation

当前部署 runner 明确关闭 contract。原因是 contract 必须在任何 observer receipt
产生之前单独生成并哈希锁定；允许 collect 时临时传入 contract 会形成事后解释
泄漏。仓库保留了纯 verifier 和单元测试，但在增加独立的 pre-observation contract
lock 之前，paper-facing pilot 只运行 Public Paired Deltas 主路径。
