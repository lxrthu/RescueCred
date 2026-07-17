#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
export PYTHONPATH="/data/hxy/projects/RescueCredit${PYTHONPATH:+:$PYTHONPATH}"
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
PY=/data/hxy/projects/RescueCredit/.venv/bin/python
MODEL=/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct
DATA_ROOT=outputs/route_a_v21c_balanced_data_seed42
DATA="$DATA_ROOT/data"
FIXTURE=outputs/route_a_v31_both_valid_appworld_dev_seed42
EVENTS="$FIXTURE/events/both_valid_dev_events.public.jsonl"
MANIFEST="$FIXTURE/events/manifest.json"
OUT=outputs/route_a_v31_confirm_43_44_45

if [ -e "$OUT" ]; then
  echo "Refusing to reuse existing confirmatory output root: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
test -x "$APP_PY"
test -x "$PY"
test -f "$DATA/train.jsonl"
test -f "$DATA/validation.jsonl"
test -f "$EVENTS"
test -f "$MANIFEST"

set -a
source .env
set +a
test -n "${AZURE_OPENAI_API_KEY:-}"
"$PY" scripts/check_azure.py > "$OUT/azure_check.log" 2>&1

"$PY" -m py_compile \
  scripts/freeze_route_a_v31_confirm_protocol.py \
  scripts/check_route_a_v31_confirm_preference.py \
  scripts/freeze_route_a_v31_confirm_bounded_protocol.py \
  scripts/analyze_route_a_v31_confirm.py
"$PY" -m pytest -q \
  tests/test_route_a_v31_confirm.py \
  tests/test_route_a_preference.py \
  tests/test_route_a_v31_both_valid.py

# Freeze all three training protocols before any confirmatory model is trained.
for SEED in 43 44 45; do
  mkdir -p "$OUT/seed$SEED"
  "$PY" scripts/freeze_route_a_v31_confirm_protocol.py \
    --data-root "$DATA_ROOT" --model "$MODEL" --seed "$SEED" \
    --output "$OUT/seed$SEED/training_protocol_lock.json" \
    > "$OUT/seed$SEED/training_protocol_console.log" 2>&1
done
echo ROUTE_A_V31_CONFIRM_PROTOCOLS_FROZEN

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' |
    sort -n | head -n 3 | awk '{print $2}'
)
if [ "${#GPUS[@]}" -lt 3 ]; then
  echo "Need three visible GPUs for the three confirmatory seeds." >&2
  exit 2
fi

train_and_select_seed() {
  local SEED="$1" GPU="$2" ROOT="$OUT/seed$SEED"
  local MASK="$ROOT/mask" V31="$ROOT/v31"
  mkdir -p "$MASK/eval" "$V31/eval" "$ROOT/mask_selection" "$ROOT/v31_selection"
  echo "SEED=$SEED GPU=$GPU START_TRAINING"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_route_a_preference.py \
    --method mask --model "$MODEL" --train-file "$DATA/train.jsonl" \
    --seed "$SEED" --epochs 3 --learning-rate 3e-6 --gradient-accumulation 8 \
    --max-length 2048 --beta 1.0 --max-causal-weight 2.5 \
    --v2-presentations-per-epoch 0 --lora-r 16 --lora-alpha 32 --fp32 \
    --output-dir "$MASK" > "$MASK/train_console.log" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_route_a_preference.py \
    --method mask --model "$MODEL" --adapter "$MASK/adapter" \
    --run-summary "$MASK/run_summary.json" --validation-file "$DATA/validation.jsonl" \
    --max-length 2048 --fp32 --output-dir "$MASK/eval" \
    > "$MASK/eval_console.log" 2>&1

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_route_a_preference.py \
    --method v31 --model "$MODEL" --train-file "$DATA/train.jsonl" \
    --seed "$SEED" --epochs 3 --learning-rate 3e-6 --gradient-accumulation 8 \
    --max-length 2048 --beta 1.0 --max-causal-weight 2.5 \
    --v2-presentations-per-epoch 0 --absolute-margin-coef 1.0 --target-margin 0.05 \
    --protocol-lock "$ROOT/training_protocol_lock.json" \
    --lora-r 16 --lora-alpha 32 --fp32 --output-dir "$V31" \
    > "$V31/train_console.log" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_route_a_preference.py \
    --method v31 --model "$MODEL" --adapter "$V31/adapter" \
    --run-summary "$V31/run_summary.json" --validation-file "$DATA/validation.jsonl" \
    --max-length 2048 --fp32 --output-dir "$V31/eval" \
    > "$V31/eval_console.log" 2>&1

  "$PY" scripts/check_route_a_v31_confirm_preference.py \
    --mask "$MASK/eval/eval_summary.json" --mask-run "$MASK/run_summary.json" \
    --v31 "$V31/eval/eval_summary.json" --v31-run "$V31/run_summary.json" \
    --protocol-lock "$ROOT/training_protocol_lock.json" \
    --output "$ROOT/preference_gate.json" > "$ROOT/preference_gate_console.log" 2>&1

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/select_route_a_frozen_events.py \
    --method mask --event-file "$EVENTS" --worker-python "$PY" \
    --scorer-script scripts/route_a_adapter_scorer_worker.py --model "$MODEL" \
    --adapter "$MASK/adapter" --output-dir "$ROOT/mask_selection" \
    > "$ROOT/mask_selection/console.log" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/select_route_a_frozen_events.py \
    --method v31 --event-file "$EVENTS" --worker-python "$PY" \
    --scorer-script scripts/route_a_adapter_scorer_worker.py --model "$MODEL" \
    --adapter "$V31/adapter" --output-dir "$ROOT/v31_selection" \
    > "$ROOT/v31_selection/console.log" 2>&1

  "$PY" scripts/freeze_route_a_v31_confirm_bounded_protocol.py \
    --event-file "$EVENTS" --event-manifest "$MANIFEST" \
    --mask-results "$ROOT/mask_selection/task_results.jsonl" \
    --v31-results "$ROOT/v31_selection/task_results.jsonl" \
    --mask-selection "$ROOT/mask_selection/selection_summary.json" \
    --v31-selection "$ROOT/v31_selection/selection_summary.json" \
    --mask-run "$MASK/run_summary.json" --v31-run "$V31/run_summary.json" \
    --training-lock "$ROOT/training_protocol_lock.json" \
    --preference-gate "$ROOT/preference_gate.json" \
    --output "$ROOT/bounded_protocol_lock.json" \
    > "$ROOT/bounded_protocol_console.log" 2>&1
  echo "SEED=$SEED TRAINING_AND_SELECTION_FINISHED"
}

declare -A TRAIN_PIDS=()
INDEX=0
for SEED in 43 44 45; do
  train_and_select_seed "$SEED" "${GPUS[$INDEX]}" &
  TRAIN_PIDS[$SEED]=$!
  echo "STARTED_TRAIN seed=$SEED gpu=${GPUS[$INDEX]} pid=${TRAIN_PIDS[$SEED]}"
  INDEX=$((INDEX + 1))
done
for SEED in 43 44 45; do
  wait "${TRAIN_PIDS[$SEED]}"
done
echo ROUTE_A_V31_CONFIRM_TRAINING_FINISHED

run_bounded_seed() {
  local SEED="$1" ROOT="$OUT/seed$SEED"
  "$APP_PY" scripts/evaluate_route_a_bounded.py \
    --development-confirmatory --appworld-root /data/hxy/projects/RescueCredit \
    --event-file "$EVENTS" \
    --mask-results "$ROOT/mask_selection/task_results.jsonl" \
    --v2-results "$ROOT/v31_selection/task_results.jsonl" \
    --protocol-lock "$ROOT/bounded_protocol_lock.json" \
    --seed "$SEED" --horizons 4 8 --worker-python "$PY" \
    --worker-script scripts/appworld_azure_continuation_worker.py \
    --cache-file "$ROOT/continuation_cache.jsonl" --output-dir "$ROOT" \
    > "$ROOT/bounded_console.log" 2>&1
}

declare -A EVAL_PIDS=()
for SEED in 43 44 45; do
  run_bounded_seed "$SEED" &
  EVAL_PIDS[$SEED]=$!
  echo "STARTED_BOUNDED seed=$SEED pid=${EVAL_PIDS[$SEED]}"
done
for SEED in 43 44 45; do
  wait "${EVAL_PIDS[$SEED]}"
done

set +e
"$PY" scripts/analyze_route_a_v31_confirm.py \
  --root "$OUT" --output "$OUT/combined_gate.json" \
  2>&1 | tee "$OUT/combined_gate_console.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -eq 0 ]; then
  echo ROUTE_A_V31_CONFIRM_GATE_PASS
else
  echo ROUTE_A_V31_CONFIRM_GATE_FAIL
fi
echo ROUTE_A_V31_CONFIRM_FINISHED
exit "$STATUS"
