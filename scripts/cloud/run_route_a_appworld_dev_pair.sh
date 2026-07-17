#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
MODEL_PY=/data/hxy/projects/RescueCredit/.venv/bin/python
MODEL=/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct
PAIR=outputs/route_a_seed42_preference_pair
OUT=outputs/route_a_appworld_dev_seed42_v2
EVENTS="$OUT/events"
mkdir -p "$OUT" "$EVENTS"

set -a
source .env
set +a
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "Set AZURE_OPENAI_API_KEY in /data/hxy/projects/RescueCredit/.env"
  exit 2
fi
"$MODEL_PY" scripts/check_azure.py > "$OUT/azure_check.log" 2>&1

"$APP_PY" scripts/build_route_a_appworld_dev_events.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --subset dev \
  --offset 0 \
  --limit 57 \
  --seed 42 \
  --selector-python "$MODEL_PY" \
  --selector-script scripts/appworld_azure_candidate_selector_worker.py \
  --selector-model "$MODEL" \
  --selector-device cpu \
  --output-dir "$EVENTS" \
  2>&1 | tee "$OUT/build_events.log"

head -n 3 "$EVENTS/dev_events.public.jsonl" > "$EVENTS/sanity_events.public.jsonl"

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}' |
    sort -k2,2n | head -n 2 | awk '{print $1}'
)
if [ "${#GPUS[@]}" -lt 2 ]; then
  echo "Need two visible GPUs for paired AppWorld evaluation."
  exit 2
fi
echo "DEV_PAIR_GPUS=${GPUS[*]}"

run_eval() {
  local method="$1"
  local gpu="$2"
  local event_file="$3"
  local output_dir="$4"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$APP_PY" scripts/evaluate_route_a_appworld_dev.py \
    --appworld-root /data/hxy/projects/RescueCredit \
    --method "$method" \
    --event-file "$event_file" \
    --seed 42 \
    --worker-python "$MODEL_PY" \
    --scorer-script scripts/route_a_adapter_scorer_worker.py \
    --continuation-script scripts/appworld_azure_continuation_worker.py \
    --model "$MODEL" \
    --adapter "$PAIR/$method/adapter" \
    --max-continuation-steps 12 \
    --output-dir "$output_dir" \
    > "$output_dir/console.log" 2>&1
}

# GPU sanity: three identical events per method before the full 57-task pass.
run_eval mask "${GPUS[0]}" "$EVENTS/sanity_events.public.jsonl" "$OUT/sanity/mask" &
SANITY_MASK_PID=$!
run_eval v2 "${GPUS[1]}" "$EVENTS/sanity_events.public.jsonl" "$OUT/sanity/v2" &
SANITY_V2_PID=$!
wait "$SANITY_MASK_PID"
wait "$SANITY_V2_PID"
test -f "$OUT/sanity/mask/eval_summary.json"
test -f "$OUT/sanity/v2/eval_summary.json"
echo ROUTE_A_APPWORLD_DEV_SANITY_PASS

run_eval mask "${GPUS[0]}" "$EVENTS/dev_events.public.jsonl" "$OUT/mask" &
MASK_PID=$!
run_eval v2 "${GPUS[1]}" "$EVENTS/dev_events.public.jsonl" "$OUT/v2" &
V2_PID=$!
echo "DEV_EVAL_STARTED mask_gpu=${GPUS[0]} mask_pid=$MASK_PID v2_gpu=${GPUS[1]} v2_pid=$V2_PID"

set +e
wait "$MASK_PID"; MASK_STATUS=$?
wait "$V2_PID"; V2_STATUS=$?
set -e
if [ "$MASK_STATUS" -ne 0 ] || [ "$V2_STATUS" -ne 0 ]; then
  echo "DEV_EVAL_FAILED mask=$MASK_STATUS v2=$V2_STATUS"
  tail -n 100 "$OUT/mask/console.log" || true
  tail -n 100 "$OUT/v2/console.log" || true
  exit 1
fi

set +e
"$MODEL_PY" scripts/check_route_a_appworld_dev_gate.py \
  --mask-dir "$OUT/mask" \
  --v2-dir "$OUT/v2" \
  --min-events 20 \
  --output "$OUT/gate.json" \
  2>&1 | tee "$OUT/gate_console.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$GATE_STATUS" -eq 0 ]; then
  echo ROUTE_A_APPWORLD_DEV_GATE_PASS
else
  echo ROUTE_A_APPWORLD_DEV_GATE_FAIL
fi
echo ROUTE_A_APPWORLD_DEV_PAIR_FINISHED
