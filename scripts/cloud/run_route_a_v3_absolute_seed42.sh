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
PAIR=outputs/route_a_seed42_preference_pair
DATA="$PAIR/data"
OUT=outputs/route_a_v3_absolute_seed42
LOCK="$OUT/protocol_lock.json"
BOUND_MASK="$OUT/frozen_mask_eval"
mkdir -p "$OUT"

test -x "$PY"
test -f "$DATA/train.jsonl"
test -f "$DATA/validation.jsonl"
test -f "$PAIR/mask/eval/eval_summary.json"

GPU=$(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}' |
    sort -k2,2n | head -n 1 | awk '{print $1}'
)
echo "V3_ABSOLUTE_GPU=$GPU"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_route_a_preference.py \
  --method mask \
  --model "$MODEL" \
  --adapter "$PAIR/mask/adapter" \
  --run-summary "$PAIR/mask/run_summary.json" \
  --validation-file "$DATA/validation.jsonl" \
  --max-length 2048 \
  --fp32 \
  --output-dir "$BOUND_MASK" \
  2>&1 | tee "$OUT/frozen_mask_eval_console.log"

"$PY" scripts/freeze_route_a_v3_protocol.py \
  --data-dir "$DATA" \
  --mask-run "$PAIR/mask/run_summary.json" \
  --mask-eval "$BOUND_MASK/eval_summary.json" \
  --model "$MODEL" \
  --output "$LOCK" \
  2>&1 | tee "$OUT/protocol_lock_console.log"

"$PY" -m pytest -q \
  tests/test_route_a_signal_hotfix.py \
  tests/test_route_a_preference.py \
  tests/test_route_a_v3_gate.py

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_route_a_preference.py \
  --method v3 \
  --model "$MODEL" \
  --train-file "$DATA/train.jsonl" \
  --seed 42 \
  --epochs 3 \
  --learning-rate 3e-6 \
  --gradient-accumulation 8 \
  --max-length 2048 \
  --beta 1.0 \
  --max-causal-weight 2.5 \
  --v2-presentations-per-epoch 0 \
  --absolute-margin-coef 1.0 \
  --target-margin 0.05 \
  --protocol-lock "$LOCK" \
  --lora-r 16 \
  --lora-alpha 32 \
  --fp32 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/train_console.log"

"$PY" - "$PAIR/mask/run_summary.json" "$OUT/run_summary.json" <<'PY'
import json
import sys

mask = json.load(open(sys.argv[1], encoding="utf-8"))
v3 = json.load(open(sys.argv[2], encoding="utf-8"))
assert v3["method"] == "v3"
assert v3["absolute_margin_coef"] == 1.0
assert v3["target_margin"] == 0.05
assert v3["presentation_budget_matches_mask"] is True
assert v3["active_event_presentations"] == mask["active_event_presentations"]
assert set(v3["presented_decisions"]) == {
    "rescue_preference",
    "reverse_preference",
}
print("V3_CONFIG_AND_BUDGET_OK")
PY

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_route_a_preference.py \
  --method v3 \
  --model "$MODEL" \
  --adapter "$OUT/adapter" \
  --run-summary "$OUT/run_summary.json" \
  --validation-file "$DATA/validation.jsonl" \
  --max-length 2048 \
  --fp32 \
  --output-dir "$OUT/eval" \
  2>&1 | tee "$OUT/eval_console.log"

set +e
"$PY" scripts/check_route_a_v3_gate.py \
  --mask "$BOUND_MASK/eval_summary.json" \
  --mask-run "$PAIR/mask/run_summary.json" \
  --v3 "$OUT/eval/eval_summary.json" \
  --run-summary "$OUT/run_summary.json" \
  --protocol-lock "$LOCK" \
  --output "$OUT/gate_vs_frozen_mask.json" \
  2>&1 | tee "$OUT/gate_console.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$GATE_STATUS" -eq 0 ]; then
  echo ROUTE_A_V3_ABSOLUTE_GATE_PASS
else
  echo ROUTE_A_V3_ABSOLUTE_GATE_FAIL
fi
echo ROUTE_A_V3_ABSOLUTE_FINISHED
exit "$GATE_STATUS"
