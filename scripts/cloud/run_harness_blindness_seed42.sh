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
DATA=data/api_bank_controlled_reference_independent_v1
OUT=outputs/harness_credit_blindness_seed42
LOCK="$OUT/protocol_lock.json"
if [ -e "$OUT" ]; then
  echo "Refusing to reuse existing output root: $OUT" >&2
  echo "Move it aside or delete it explicitly before a fresh frozen run." >&2
  exit 1
fi
mkdir -p "$OUT"

test -x "$PY"
test -f "$DATA/train.jsonl"
test -f "$DATA/dev.jsonl"
test -f "$DATA/manifest.json"

"$PY" -m py_compile \
  scripts/run_train.py \
  scripts/run_eval.py \
  scripts/freeze_harness_blindness_protocol.py \
  scripts/analyze_harness_blindness.py
"$PY" -m pytest -q \
  tests/test_harness_blindness.py \
  tests/test_training_credit.py \
  tests/test_snapshot_replay.py

"$PY" scripts/freeze_harness_blindness_protocol.py \
  --data-dir "$DATA" \
  --model "$MODEL" \
  --output "$LOCK" \
  2>&1 | tee "$OUT/protocol_console.log"

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}' |
    sort -k2,2n | head -n 3 | awk '{print $1}'
)
if [ "${#GPUS[@]}" -lt 3 ]; then
  echo "Need three visible GPUs" >&2
  exit 1
fi
printf 'HARNESS_BLINDNESS_GPUS=%s,%s,%s\n' "${GPUS[0]}" "${GPUS[1]}" "${GPUS[2]}"

# Frozen base checkpoint: establishes pre-training S_on/S_off and intervention rate.
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PY" scripts/run_eval.py \
  --checkpoint "$MODEL" \
  --method base_qwen \
  --seed 42 \
  --split dev \
  --data-dir "$DATA" \
  --max-new-tokens 64 \
  --harness-mode oracle \
  --experiment-protocol-lock "$LOCK" \
  --log-preference-margins \
  --output-dir "$OUT/base_eval" \
  2>&1 | tee "$OUT/base_eval_console.log"

run_arm() {
  local method="$1"
  local gpu="$2"
  local port="$3"
  local arm="$OUT/$method"
  mkdir -p "$arm"

  local extra=()
  if [ "$method" = "rescuecredit" ]; then
    extra+=(--force-shadow-credit)
  fi

  CUDA_VISIBLE_DEVICES="$gpu" accelerate launch \
    --config_file configs/accelerate_h200.yaml \
    --num_processes 1 \
    --main_process_port "$port" \
    scripts/run_train.py \
    --method "$method" \
    --model "$MODEL" \
    --train-file "$DATA/train.jsonl" \
    --manifest "$DATA/manifest.json" \
    --seed 42 \
    --max-updates 10000 \
    --budget-mode main \
    --main-interaction-budget 1200 \
    --strict-main-budget \
    --total-interaction-budget 50000 \
    --group-size 4 \
    --max-new-tokens 64 \
    --max-shadow-steps 12 \
    --policy-epochs 1 \
    --learning-rate 2e-6 \
    --audit-probability 1.0 \
    --audit-warm-start-events 0 \
    --lambda-corr 0.1 \
    --diagnostic-full-shadow \
    --experiment-protocol-lock "$LOCK" \
    --use-lora \
    --fp32 \
    --save-every 5 \
    "${extra[@]}" \
    --output-dir "$arm" \
    2>&1 | tee "$arm/train_console.log"

  for checkpoint in "$arm"/checkpoints/update_* "$arm"/checkpoints/final; do
    [ -d "$checkpoint" ] || continue
    local name
    name=$(basename "$checkpoint")
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_eval.py \
      --checkpoint "$checkpoint" \
      --method "$method" \
      --seed 42 \
      --split dev \
      --data-dir "$DATA" \
      --max-new-tokens 64 \
      --harness-mode oracle \
      --experiment-protocol-lock "$LOCK" \
      --log-preference-margins \
      --output-dir "$arm/eval_$name" \
      2>&1 | tee "$arm/eval_${name}_console.log"
  done
}

run_arm naive_h_grpo "${GPUS[0]}" 29601 &
PID_NAIVE=$!
run_arm mask_correction "${GPUS[1]}" 29602 &
PID_MASK=$!
run_arm rescuecredit "${GPUS[2]}" 29603 &
PID_RESCUE=$!

STATUS=0
wait "$PID_NAIVE" || STATUS=1
wait "$PID_MASK" || STATUS=1
wait "$PID_RESCUE" || STATUS=1
if [ "$STATUS" -ne 0 ]; then
  echo "At least one training arm failed" >&2
  exit 1
fi

set +e
"$PY" scripts/analyze_harness_blindness.py \
  --protocol-lock "$LOCK" \
  --base-eval "$OUT/base_eval/eval_summary.json" \
  --root "$OUT" \
  --output "$OUT/gate.json" \
  2>&1 | tee "$OUT/analysis_console.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$GATE_STATUS" -eq 0 ]; then
  echo HARNESS_CREDIT_BLINDNESS_GATE_PASS
else
  echo HARNESS_CREDIT_BLINDNESS_GATE_FAIL
fi
echo HARNESS_CREDIT_BLINDNESS_FINISHED
exit "$GATE_STATUS"
