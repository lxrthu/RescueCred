#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
source data_disk_env.sh
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/data/hxy/projects/RescueCredit/.venv/bin/python
MODEL=/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct
BANK=outputs/appworld_route_a_bank_train90_seed42
DENSE=outputs/appworld_route_a_dense_credit_seed42
OUT=outputs/route_a_seed42_preference_pair
DATA="$OUT/data"
mkdir -p "$OUT"

"$PY" scripts/prepare_route_a_preference_data.py \
  --bank-dir "$BANK" \
  --dense-dir "$DENSE" \
  --seed 42 \
  --validation-fraction 0.2 \
  --output-dir "$DATA" \
  2>&1 | tee "$OUT/prepare.log"

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}' |
    sort -k2,2n | head -n 2 | awk '{print $1}'
)
if [ "${#GPUS[@]}" -lt 2 ]; then
  echo "Need two visible GPUs for the paired pilot."
  exit 2
fi
echo "PAIR_GPUS=${GPUS[*]}"

run_train() {
  local method="$1"
  local gpu="$2"
  local method_out="$OUT/$method"
  mkdir -p "$method_out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_route_a_preference.py \
    --method "$method" \
    --model "$MODEL" \
    --train-file "$DATA/train.jsonl" \
    --seed 42 \
    --epochs 3 \
    --learning-rate 1e-5 \
    --gradient-accumulation 8 \
    --max-length 2048 \
    --beta 1.0 \
    --max-causal-weight 2.5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --fp32 \
    --output-dir "$method_out" \
    > "$method_out/console.log" 2>&1
}

run_train mask "${GPUS[0]}" &
MASK_PID=$!
run_train v2 "${GPUS[1]}" &
V2_PID=$!
echo "PAIR_TRAIN_STARTED mask_gpu=${GPUS[0]} mask_pid=$MASK_PID v2_gpu=${GPUS[1]} v2_pid=$V2_PID"
echo "DETAIL_LOGS=$OUT/mask/console.log,$OUT/v2/console.log"

set +e
wait "$MASK_PID"; MASK_STATUS=$?
wait "$V2_PID"; V2_STATUS=$?
set -e
if [ "$MASK_STATUS" -ne 0 ] || [ "$V2_STATUS" -ne 0 ]; then
  echo "PAIR_TRAIN_FAILED mask=$MASK_STATUS v2=$V2_STATUS"
  tail -n 80 "$OUT/mask/console.log" || true
  tail -n 80 "$OUT/v2/console.log" || true
  exit 1
fi

run_eval() {
  local method="$1"
  local gpu="$2"
  local method_out="$OUT/$method"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/evaluate_route_a_preference.py \
    --method "$method" \
    --model "$MODEL" \
    --adapter "$method_out/adapter" \
    --validation-file "$DATA/validation.jsonl" \
    --max-length 2048 \
    --fp32 \
    --output-dir "$method_out/eval" \
    > "$method_out/eval_console.log" 2>&1
}

run_eval mask "${GPUS[0]}" &
MASK_EVAL_PID=$!
run_eval v2 "${GPUS[1]}" &
V2_EVAL_PID=$!
wait "$MASK_EVAL_PID"
wait "$V2_EVAL_PID"

set +e
"$PY" scripts/check_route_a_preference_gate.py \
  --mask "$OUT/mask/eval/eval_summary.json" \
  --v2 "$OUT/v2/eval/eval_summary.json" \
  --output "$OUT/gate.json" \
  2>&1 | tee "$OUT/gate_console.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$GATE_STATUS" -eq 0 ]; then
  echo ROUTE_A_SEED42_PREFERENCE_GATE_PASS
else
  echo ROUTE_A_SEED42_PREFERENCE_GATE_FAIL
fi
echo ROUTE_A_SEED42_PREFERENCE_PAIR_FINISHED
