#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"

if [[ ! -f outputs/pilot/gate.json ]] || ! grep -q '"passed": true' outputs/pilot/gate.json; then
  echo "Pilot gate is absent or failed; refusing three-seed expansion." >&2
  exit 2
fi

for seed in 42 43 44; do
  for method in naive_h_grpo mask_correction rescuecredit; do
    run_dir="outputs/confirmatory/${method}_seed${seed}"
    accelerate launch --config_file configs/accelerate_h200.yaml scripts/run_train.py \
      --method "$method" \
      --model "$MODEL" \
      --model-revision "$MODEL_REVISION" \
      --seed "$seed" \
      --max-updates 10000 \
      --total-interaction-budget 50000 \
      --use-lora \
      --output-dir "$run_dir"
    for split in test_id test_tool_ood; do
      python scripts/run_eval.py \
        --checkpoint "$run_dir/checkpoints/final" \
        --method "$method" \
        --seed "$seed" \
        --split "$split" \
        --output-dir "$run_dir/eval_${split}"
    done
    if [[ "$method" == "rescuecredit" ]]; then
      python scripts/evaluate_full_shadow.py \
        --checkpoint "$run_dir/checkpoints/final" \
        --output-dir "$run_dir/full_shadow_eval"
    fi
  done
done

mapfile -t summaries < <(find outputs/confirmatory -name eval_summary.json -type f | sort)
python scripts/aggregate_results.py "${summaries[@]}" --output-dir outputs/tables
echo "CONFIRMATORY_COMPLETE"
