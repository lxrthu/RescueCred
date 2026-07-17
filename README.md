# RescueCredit

本仓库实现技术路线中的 Rescue-MDP、随机 residual shadow audit、Patch EMA、token provenance 路由、可回放 API-Bank-derived controlled environment、三类 corrective patch，以及面向 H200 的训练/评测入口。

## 快速验证

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_toy.py --output-dir outputs/toy
python scripts/prepare_api_bank_controlled.py
python scripts/run_api_bank_smoke.py --split dev --limit 30
```

Azure 连通性：先复制 `.env.example` 为 `.env` 并填写轮换后的密钥，然后运行：

```bash
python scripts/check_azure.py
```

完整云端步骤见 `docs/CLOUD_RUN_CN.md`。所有核心输出为 JSON/JSONL/CSV；模拟 smoke 明确标记 `not_research_evidence=true`。

受控环境的 delayed recovery、Full Shadow 可辨识门槛和 task-level bootstrap 定义见 `docs/EVALUATION_CONTRACT_CN.md`。
