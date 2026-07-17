#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
source data_disk_env.sh
source .venv/bin/activate

python scripts/evaluate_deployable_harness.py \
  --tasks data/api_bank_controlled_v1/dev.jsonl \
  --output-dir outputs/deployable_harness_audit_dev

python -m pytest -q tests/test_deployable_harness_no_oracle.py

echo "===== HARNESS METRICS ====="
cat outputs/deployable_harness_audit_dev/harness_metrics.json
echo "===== QUALITY GATE ====="
cat outputs/deployable_harness_audit_dev/quality_gate.json
