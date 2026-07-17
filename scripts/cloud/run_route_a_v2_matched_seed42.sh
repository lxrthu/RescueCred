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
OUT=outputs/route_a_v2_matched_seed42
mkdir -p "$OUT"

test -x "$PY"
test -f "$DATA/train.jsonl"
test -f "$DATA/validation.jsonl"
test -f "$PAIR/mask/eval/eval_summary.json"

"$PY" -m pytest -q \
  tests/test_route_a_signal_hotfix.py \
  tests/test_route_a_preference.py

GPU=$(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}' |
    sort -k2,2n | head -n 1 | awk '{print $1}'
)
echo "V2_MATCHED_GPU=$GPU"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_route_a_preference.py \
  --method v2 \
  --model "$MODEL" \
  --train-file "$DATA/train.jsonl" \
  --seed 42 \
  --epochs 3 \
  --learning-rate 1e-5 \
  --gradient-accumulation 8 \
  --max-length 2048 \
  --beta 1.0 \
  --max-causal-weight 2.5 \
  --v2-presentations-per-epoch 0 \
  --lora-r 16 \
  --lora-alpha 32 \
  --fp32 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/train_console.log"

"$PY" - "$PAIR/mask/run_summary.json" "$OUT/run_summary.json" <<'PY'
import json
import sys

mask = json.load(open(sys.argv[1], encoding="utf-8"))
v2 = json.load(open(sys.argv[2], encoding="utf-8"))
assert v2["presentation_budget_matches_mask"] is True
assert v2["active_event_presentations"] == mask["active_event_presentations"], (
    mask["active_event_presentations"],
    v2["active_event_presentations"],
)
assert set(v2["presented_decisions"]) == {
    "rescue_preference",
    "reverse_preference",
}
print("MATCHED_PRESENTATION_BUDGET_OK")
PY

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_route_a_preference.py \
  --method v2 \
  --model "$MODEL" \
  --adapter "$OUT/adapter" \
  --validation-file "$DATA/validation.jsonl" \
  --max-length 2048 \
  --fp32 \
  --output-dir "$OUT/eval" \
  2>&1 | tee "$OUT/eval_console.log"

set +e
"$PY" scripts/check_route_a_preference_gate.py \
  --mask "$PAIR/mask/eval/eval_summary.json" \
  --v2 "$OUT/eval/eval_summary.json" \
  --output "$OUT/gate_vs_frozen_mask.json" \
  2>&1 | tee "$OUT/gate_console.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$GATE_STATUS" -eq 0 ]; then
  echo ROUTE_A_V2_MATCHED_TRAIN_GATE_PASS
else
  echo ROUTE_A_V2_MATCHED_TRAIN_GATE_FAIL
fi
echo ROUTE_A_V2_MATCHED_TRAIN_FINISHED
exit "$GATE_STATUS"
