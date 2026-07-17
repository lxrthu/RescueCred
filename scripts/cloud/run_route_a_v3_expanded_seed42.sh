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
DATA_ROOT=outputs/route_a_v21c_balanced_data_seed42
DATA="$DATA_ROOT/data"
OUT=outputs/route_a_v3_expanded_seed42
MASK="$OUT/mask"
V3="$OUT/v3"
LOCK="$OUT/protocol_lock.json"

test ! -e "$OUT"
test -f "$DATA_ROOT/data_gate.json"
test -f "$DATA/train.jsonl"
test -f "$DATA/validation.jsonl"
mkdir -p "$MASK" "$V3"

GPU=$(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}' |
    sort -k2,2n | head -n 1 | awk '{print $1}'
)
echo "V3_EXPANDED_GPU=$GPU"

"$PY" -m pytest -q \
  tests/test_route_a_preference.py \
  tests/test_route_a_v3_expanded.py \
  tests/test_route_a_v21c_repair.py

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_route_a_preference.py \
  --method mask --model "$MODEL" --train-file "$DATA/train.jsonl" \
  --seed 42 --epochs 3 --learning-rate 3e-6 --gradient-accumulation 8 \
  --max-length 2048 --beta 1.0 --max-causal-weight 2.5 \
  --v2-presentations-per-epoch 0 --lora-r 16 --lora-alpha 32 --fp32 \
  --output-dir "$MASK" 2>&1 | tee "$MASK/train_console.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_route_a_preference.py \
  --method mask --model "$MODEL" --adapter "$MASK/adapter" \
  --run-summary "$MASK/run_summary.json" \
  --validation-file "$DATA/validation.jsonl" --max-length 2048 --fp32 \
  --output-dir "$MASK/eval" 2>&1 | tee "$MASK/eval_console.log"

"$PY" scripts/freeze_route_a_v3_expanded_protocol.py \
  --data-root "$DATA_ROOT" --mask-run "$MASK/run_summary.json" \
  --mask-eval "$MASK/eval/eval_summary.json" --model "$MODEL" \
  --output "$LOCK" 2>&1 | tee "$OUT/protocol_console.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_route_a_preference.py \
  --method v3 --model "$MODEL" --train-file "$DATA/train.jsonl" \
  --seed 42 --epochs 3 --learning-rate 3e-6 --gradient-accumulation 8 \
  --max-length 2048 --beta 1.0 --max-causal-weight 2.5 \
  --v2-presentations-per-epoch 0 --absolute-margin-coef 1.0 \
  --target-margin 0.05 --protocol-lock "$LOCK" \
  --lora-r 16 --lora-alpha 32 --fp32 \
  --output-dir "$V3" 2>&1 | tee "$V3/train_console.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_route_a_preference.py \
  --method v3 --model "$MODEL" --adapter "$V3/adapter" \
  --run-summary "$V3/run_summary.json" \
  --validation-file "$DATA/validation.jsonl" --max-length 2048 --fp32 \
  --output-dir "$V3/eval" 2>&1 | tee "$V3/eval_console.log"

set +e
"$PY" scripts/check_route_a_v3_expanded_gate.py \
  --mask "$MASK/eval/eval_summary.json" --mask-run "$MASK/run_summary.json" \
  --v3 "$V3/eval/eval_summary.json" --v3-run "$V3/run_summary.json" \
  --protocol-lock "$LOCK" --output "$OUT/gate.json" \
  2>&1 | tee "$OUT/gate_console.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -eq 0 ]; then
  echo ROUTE_A_V3_EXPANDED_GATE_PASS
else
  echo ROUTE_A_V3_EXPANDED_GATE_FAIL
fi
echo ROUTE_A_V3_EXPANDED_FINISHED
exit "$STATUS"
