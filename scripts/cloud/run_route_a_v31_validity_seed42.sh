#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
source data_disk_env.sh
source .venv/bin/activate
export PYTHONPATH="/data/hxy/projects/RescueCredit${PYTHONPATH:+:$PYTHONPATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/data/hxy/projects/RescueCredit/.venv/bin/python
MODEL=/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct
DATA_ROOT=outputs/route_a_v21c_balanced_data_seed42
DATA="$DATA_ROOT/data"
BASELINE=outputs/route_a_v3_expanded_seed42/mask
OUT=outputs/route_a_v31_validity_seed42
MASK_EVAL="$OUT/mask_eval"
V31="$OUT/v31"
LOCK="$OUT/protocol_lock.json"

if [ -e "$OUT" ]; then
  echo "Refusing to reuse existing V3.1 output root: $OUT" >&2
  exit 1
fi
mkdir -p "$MASK_EVAL" "$V31"

test -x "$PY"
test -f "$DATA/train.jsonl"
test -f "$DATA/validation.jsonl"
test -f "$BASELINE/run_summary.json"
test -d "$BASELINE/adapter"

"$PY" -m py_compile \
  rescuecredit/route_a_preference.py \
  scripts/train_route_a_preference.py \
  scripts/evaluate_route_a_preference.py \
  scripts/freeze_route_a_v31_protocol.py \
  scripts/check_route_a_v31_gate.py
"$PY" -m pytest -q \
  tests/test_route_a_preference.py \
  tests/test_route_a_v31.py \
  tests/test_route_a_v3_expanded.py

GPU=$(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}' |
    sort -k2,2n | head -n 1 | awk '{print $1}'
)
echo "V31_VALIDITY_GPU=$GPU"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_route_a_preference.py \
  --method mask --model "$MODEL" --adapter "$BASELINE/adapter" \
  --run-summary "$BASELINE/run_summary.json" \
  --validation-file "$DATA/validation.jsonl" --max-length 2048 --fp32 \
  --output-dir "$MASK_EVAL" 2>&1 | tee "$MASK_EVAL/console.log"

"$PY" scripts/freeze_route_a_v31_protocol.py \
  --data-root "$DATA_ROOT" \
  --mask-run "$BASELINE/run_summary.json" \
  --mask-eval "$MASK_EVAL/eval_summary.json" \
  --model "$MODEL" --output "$LOCK" \
  2>&1 | tee "$OUT/protocol_console.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_route_a_preference.py \
  --method v31 --model "$MODEL" --train-file "$DATA/train.jsonl" \
  --seed 42 --epochs 3 --learning-rate 3e-6 --gradient-accumulation 8 \
  --max-length 2048 --beta 1.0 --max-causal-weight 2.5 \
  --v2-presentations-per-epoch 0 --absolute-margin-coef 1.0 \
  --target-margin 0.05 --protocol-lock "$LOCK" \
  --lora-r 16 --lora-alpha 32 --fp32 \
  --output-dir "$V31" 2>&1 | tee "$V31/train_console.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_route_a_preference.py \
  --method v31 --model "$MODEL" --adapter "$V31/adapter" \
  --run-summary "$V31/run_summary.json" \
  --validation-file "$DATA/validation.jsonl" --max-length 2048 --fp32 \
  --output-dir "$V31/eval" 2>&1 | tee "$V31/eval_console.log"

set +e
"$PY" scripts/check_route_a_v31_gate.py \
  --mask "$MASK_EVAL/eval_summary.json" \
  --mask-run "$BASELINE/run_summary.json" \
  --v31 "$V31/eval/eval_summary.json" \
  --v31-run "$V31/run_summary.json" \
  --protocol-lock "$LOCK" --output "$OUT/gate.json" \
  2>&1 | tee "$OUT/gate_console.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -eq 0 ]; then
  echo ROUTE_A_V31_VALIDITY_GATE_PASS
else
  echo ROUTE_A_V31_VALIDITY_GATE_FAIL
fi
echo ROUTE_A_V31_VALIDITY_FINISHED
exit "$STATUS"
