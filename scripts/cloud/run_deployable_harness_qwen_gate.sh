#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
source data_disk_env.sh
source .venv/bin/activate

export MODEL="${MODEL:-/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/evaluate_deployable_harness.py \
  --tasks data/api_bank_controlled_v1/dev.jsonl \
  --model "$MODEL" \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --device cuda \
  --generator-max-new-tokens 64 \
  --output-dir outputs/deployable_harness_audit_dev_frozen_qwen

echo "===== HARNESS METRICS ====="
cat outputs/deployable_harness_audit_dev_frozen_qwen/harness_metrics.json
echo "===== QUALITY GATE ====="
cat outputs/deployable_harness_audit_dev_frozen_qwen/quality_gate.json
