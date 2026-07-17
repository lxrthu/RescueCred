# RescueCredit 云服务器执行手册（4×H200）

## 1. 唯一需要修改的 API Key 路径

上传后进入项目根目录：

```bash
cd ~/RescueCredit
cp .env.example .env
nano .env
```

只在 `~/RescueCredit/.env` 中填写：

```dotenv
AZURE_OPENAI_API_KEY=你的新密钥
ENDPOINT_URL=https://scdall3.openai.azure.com/
DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

请使用轮换后的新密钥。用户消息里出现过的旧密钥已视为暴露，项目中没有保存它。

## 2. 上传与安装

本地执行（把服务器地址替换为实际值）：

```bash
scp RescueCredit-cloud-ready.zip user@SERVER:~/
```

服务器执行：

```bash
cd ~
unzip RescueCredit-cloud-ready.zip
cd RescueCredit
bash scripts/cloud/setup_cloud.sh
```

原始 API-Bank 数据已打包在 `data/raw/DAMO-ConvAI/api-bank/`，来源 commit 固定为 `483554eae102996f5ec1f4feab4e78ef29c2a394`。若选择瘦包或需要重下：

```bash
source .venv/bin/activate
python scripts/download_api_bank.py
python scripts/prepare_api_bank_controlled.py
```

## 3. 分阶段执行（推荐）

先跑 deterministic gate：

```bash
bash scripts/cloud/run_sanity.sh
```

验证 Azure 配置并采 5 条 base-agent 轨迹：

```bash
bash scripts/cloud/run_azure_smoke.sh
```

单 seed pilot（依次运行 Naive H+GRPO、Mask + Correction、RescueCredit）：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL=Qwen/Qwen2.5-7B-Instruct
bash scripts/cloud/run_pilot_4gpu.sh 2>&1 | tee outputs/pilot.log
```

只有 `outputs/pilot/gate.json` 中 `passed=true` 才会允许三 seed：

```bash
bash scripts/cloud/run_confirmatory_4gpu.sh 2>&1 | tee outputs/confirmatory.log
```

从零执行全部阶段：

```bash
bash scripts/cloud/run_full_pipeline.sh 2>&1 | tee outputs/full_pipeline.log
```

## 4. 重要输出

- `data/api_bank_controlled_v1/manifest.json`：真实筛选数量、source commit、split hash、泄漏检查。
- `outputs/toy/`：Rescue-MDP 精确 `Q0/QH`、梯度机制和 estimator 曲线。
- `outputs/pilot/*/run_summary.json`：单 seed 训练状态。
- `outputs/pilot/gate.json`：是否允许扩三 seed。
- `outputs/confirmatory/*/eval_*/eval_summary.json`：API-Bank-derived controlled exact-sequence proxy 的 ID/原子 Tool-OOD 评测；不是官方 leaderboard 分数。
- `outputs/tables/`：脚本聚合的 CSV/JSON/Markdown 主表。

`outputs/smoke/` 是注入错误的工程 smoke，带 `not_research_evidence=true`，不能作为论文结果。

## 5. 单独运行一个方法

```bash
source .venv/bin/activate
accelerate launch --config_file configs/accelerate_h200.yaml scripts/run_train.py \
  --method rescuecredit \
  --model Qwen/Qwen2.5-7B-Instruct \
  --seed 42 \
  --max-updates 100 \
  --total-interaction-budget 12000 \
  --output-dir outputs/manual/rescuecredit_seed42
```

## 6. 常见问题

- `source .venv/bin/activate: No such file or directory`：先运行 `bash scripts/cloud/setup_cloud.sh`。
- CUDA OOM：降低 `--group-size` 或 `--max-new-tokens`；所有方法必须使用相同设置。
- Azure `invalid_api_key`：密钥或 endpoint 不匹配；不要通过换模型掩盖该错误。
- Azure `model_not_found`：检查 `DEPLOYMENT_NAME` 是否是 Azure deployment 名，而不是基础模型名。
- Pilot gate 退出码 2：这是预注册停止条件，不应强行启动三 seed。
