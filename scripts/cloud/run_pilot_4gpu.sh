#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"

for method in naive_h_grpo mask_correction rescuecredit; do
  accelerate launch --config_file configs/accelerate_h200.yaml scripts/run_train.py \
    --method "$method" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --seed 42 \
    --max-updates 10000 \
    --total-interaction-budget 12000 \
    --use-lora \
    --output-dir "outputs/pilot/${method}_seed42"

  python scripts/run_eval.py \
    --checkpoint "outputs/pilot/${method}_seed42/checkpoints/final" \
    --method "$method" \
    --seed 42 \
    --split dev \
    --output-dir "outputs/pilot/${method}_seed42/eval_dev"
done

python scripts/evaluate_full_shadow.py \
  --checkpoint outputs/pilot/rescuecredit_seed42/checkpoints/final \
  --output-dir outputs/pilot/rescuecredit_seed42/full_shadow_eval

python scripts/check_pilot_gate.py \
  --mask outputs/pilot/mask_correction_seed42/eval_dev/eval_summary.json \
  --rescue outputs/pilot/rescuecredit_seed42/eval_dev/eval_summary.json \
  --output outputs/pilot/gate.json
echo "PILOT_GATE_PASS"
