#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PY="${TOOLSANDBOX_PYTHON:-/data/hxy/venvs/rescuecredit-toolsandbox/bin/python}"
MODEL_PY="${RESCUECREDIT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
MODEL="${RESCUECREDIT_MODEL:-/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct}"
WORKER="$REPO_ROOT/scripts/toolsandbox_azure_worker.py"
STAGE0="$REPO_ROOT/outputs/toolsandbox_stage0/gate.json"
V4_OLD_LOCK="$REPO_ROOT/outputs/toolsandbox_v4_signal_seed42/protocol_lock.json"
V41_ROOT="$REPO_ROOT/outputs/toolsandbox_v41_toolid_seed42"
V41_DIAGNOSTIC_LOCK="$V41_ROOT/diagnostic_protocol_lock.json"
V41_TRAIN_LOCK="$V41_ROOT/fresh_protocol_lock.json"
SOURCE_AUDIT="$V41_ROOT/fresh_offset85_h8"
V41_PREF_ROOT="$REPO_ROOT/outputs/toolsandbox_v41_preference_seed42"
V41_DEV_LOCK="$V41_PREF_ROOT/evaluation_protocol_lock.json"
DEV_AUDIT="$V41_PREF_ROOT/fresh_eval_offset125_h8"
OLD_V41_GATE="$V41_PREF_ROOT/gate.json"
PLAN="$REPO_ROOT/refine-logs/TOOLSANDBOX_V42_PLAN.md"
OUT="$REPO_ROOT/outputs/toolsandbox_v42_balanced_margin_seed42"
TRAIN_DATA="$OUT/train_data"
DEV_DATA="$OUT/development_data"
CONFIRM_LOCK="$OUT/confirmation_protocol_lock.json"
PREF_LOCK="$OUT/preference_protocol_lock.json"
CONFIRM_AUDIT="$OUT/fresh_confirm_offset165_h8"
CONFIRM_DATA="$OUT/confirmation_data"

cd "$REPO_ROOT"
export PROMPT_COMMAND=
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -x "$MODEL_PY"
test -x "$APP_PY"
for path in \
  "$WORKER" "$STAGE0" "$V4_OLD_LOCK" \
  "$V41_DIAGNOSTIC_LOCK" "$V41_TRAIN_LOCK" "$V41_DEV_LOCK" \
  "$SOURCE_AUDIT/signal_events.jsonl" "$SOURCE_AUDIT/audit_summary.json" \
  "$SOURCE_AUDIT/quality_gate.json" "$DEV_AUDIT/signal_events.jsonl" \
  "$DEV_AUDIT/audit_summary.json" "$DEV_AUDIT/quality_gate.json" \
  "$OLD_V41_GATE" "$PLAN"; do
  test -e "$path"
done
test -d "$MODEL"
if [ -e "$OUT" ]; then
  echo "Refusing to reuse V4.2 output root: $OUT" >&2
  exit 1
fi
mkdir -p "$TRAIN_DATA" "$DEV_DATA" "$CONFIRM_DATA" \
  "$OUT/mask" "$OUT/v42"

"$MODEL_PY" -m py_compile \
  scripts/train_toolsandbox_v42_preference.py \
  scripts/evaluate_toolsandbox_v42_preference.py \
  scripts/freeze_toolsandbox_v42_protocol.py \
  scripts/check_toolsandbox_v42_gate.py
"$MODEL_PY" -m pytest -q \
  tests/test_toolsandbox_v42.py \
  tests/test_toolsandbox_preference.py \
  tests/test_toolsandbox_v4_protocol.py \
  tests/test_toolsandbox_v41.py

set -a
source .env
set +a
test "${TOOLSANDBOX_LLM_PROVIDER:-}" = "deepseek"
test -n "${DEEPSEEK_API_KEY:-}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://zhi-api.com/v1}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
test "$DEEPSEEK_THINKING" = "disabled"
"$MODEL_PY" scripts/check_llm.py --provider deepseek

# Reconstruct the exact prior training and development data without new model
# calls.  The offset-125 outcomes are explicitly development-only in V4.2.
"$APP_PY" scripts/prepare_toolsandbox_v41_preference_data.py \
  --signal-events "$SOURCE_AUDIT/signal_events.jsonl" \
  --audit-summary "$SOURCE_AUDIT/audit_summary.json" \
  --quality-gate "$SOURCE_AUDIT/quality_gate.json" \
  --role train --output-dir "$TRAIN_DATA" \
  2>&1 | tee "$OUT/prepare_train.log"
"$APP_PY" scripts/prepare_toolsandbox_v41_preference_data.py \
  --signal-events "$DEV_AUDIT/signal_events.jsonl" \
  --audit-summary "$DEV_AUDIT/audit_summary.json" \
  --quality-gate "$DEV_AUDIT/quality_gate.json" \
  --role evaluation --output-dir "$DEV_DATA" \
  2>&1 | tee "$OUT/prepare_development.log"

# Freeze the untouched offset-165 confirmation identity before training.  The
# four earlier locks cover V4, V4.1 diagnostic, V4.1 training, and offset-125.
"$APP_PY" scripts/freeze_toolsandbox_v4_protocol.py \
  --output "$CONFIRM_LOCK" --plan "$PLAN" --stage0-gate "$STAGE0" \
  --seed 42 --scenario-offset 165 --limit 40 --minimum-scenarios 40 \
  --horizon 8 --event-search-steps 8 --worker-timeout-sec 600 \
  --harness-interface tool_id_v2 \
  --exclude-protocol "$V4_OLD_LOCK" \
  --exclude-protocol "$V41_DIAGNOSTIC_LOCK" \
  --exclude-protocol "$V41_TRAIN_LOCK" \
  --exclude-protocol "$V41_DEV_LOCK"

"$MODEL_PY" scripts/freeze_toolsandbox_v42_protocol.py \
  --data-dir "$TRAIN_DATA" --source-audit-root "$SOURCE_AUDIT" \
  --development-data-dir "$DEV_DATA" --development-audit-root "$DEV_AUDIT" \
  --v41-development-gate "$OLD_V41_GATE" \
  --confirmation-protocol "$CONFIRM_LOCK" --model "$MODEL" \
  --output "$PREF_LOCK" 2>&1 | tee "$OUT/freeze_preference_protocol.log"

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' |
    sort -n | head -n 2 | awk '{print $2}'
)
if [ "${#GPUS[@]}" -lt 2 ]; then
  echo "Need two GPUs for the matched Mask/V4.2 pair." >&2
  exit 2
fi
echo "MASK_GPU=${GPUS[0]} V42_GPU=${GPUS[1]}"

train_one() {
  local method="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" scripts/train_toolsandbox_v42_preference.py \
    --method "$method" --model "$MODEL" --train-file "$TRAIN_DATA/train.jsonl" \
    --protocol-lock "$PREF_LOCK" --seed 42 --epochs 3 --learning-rate 3e-6 \
    --gradient-accumulation 8 --max-length 2048 --beta 1.0 \
    --absolute-margin-coef 1.0 --target-margin 0.05 \
    --presentations-per-epoch 36 --lora-r 16 --lora-alpha 32 --fp32 \
    --output-dir "$OUT/$method" > "$OUT/$method/train_console.log" 2>&1
}
train_one mask "${GPUS[0]}" & MASK_PID=$!
train_one v42 "${GPUS[1]}" & V42_PID=$!
cleanup_train() { kill "$MASK_PID" "$V42_PID" 2>/dev/null || true; }
trap cleanup_train EXIT INT TERM
set +e
wait "$MASK_PID"; MASK_STATUS=$?
wait "$V42_PID"; V42_STATUS=$?
set -e
trap - EXIT INT TERM
if [ "$MASK_STATUS" -ne 0 ] || [ "$V42_STATUS" -ne 0 ]; then
  echo "Matched training failed: mask=$MASK_STATUS v42=$V42_STATUS" >&2
  exit 1
fi
echo TOOLSANDBOX_V42_TRAINING_FINISHED

eval_pair() {
  local role="$1" data="$2" subdir="$3"
  local method gpu
  for method in mask v42; do
    if [ "$method" = mask ]; then gpu="${GPUS[0]}"; else gpu="${GPUS[1]}"; fi
    CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" scripts/evaluate_toolsandbox_v42_preference.py \
      --method "$method" --model "$MODEL" --adapter "$OUT/$method/adapter" \
      --run-summary "$OUT/$method/run_summary.json" --protocol-lock "$PREF_LOCK" \
      --public-events "$data/events.public.jsonl" \
      --private-outcomes "$data/outcomes.private.jsonl" \
      --evaluation-role "$role" --max-length 2048 --fp32 \
      --output-dir "$OUT/$method/$subdir" \
      > "$OUT/$method/${subdir}_console.log" 2>&1 &
    if [ "$method" = mask ]; then MASK_EVAL_PID=$!; else V42_EVAL_PID=$!; fi
  done
  cleanup_eval() { kill "$MASK_EVAL_PID" "$V42_EVAL_PID" 2>/dev/null || true; }
  trap cleanup_eval EXIT INT TERM
  set +e
  wait "$MASK_EVAL_PID"; MASK_EVAL_STATUS=$?
  wait "$V42_EVAL_PID"; V42_EVAL_STATUS=$?
  set -e
  trap - EXIT INT TERM
  if [ "$MASK_EVAL_STATUS" -ne 0 ] || [ "$V42_EVAL_STATUS" -ne 0 ]; then
    echo "Matched $role evaluation failed: mask=$MASK_EVAL_STATUS v42=$V42_EVAL_STATUS" >&2
    exit 1
  fi
}

# Development uses only the already observed offset-125 artifacts.  It makes
# no continuation-model calls and gates all confirmatory spending.
eval_pair development "$DEV_DATA" dev_eval
set +e
"$MODEL_PY" scripts/check_toolsandbox_v42_gate.py \
  --role development \
  --mask-eval "$OUT/mask/dev_eval/eval_summary.json" \
  --v42-eval "$OUT/v42/dev_eval/eval_summary.json" \
  --mask-run "$OUT/mask/run_summary.json" --v42-run "$OUT/v42/run_summary.json" \
  --mask-results "$OUT/mask/dev_eval/task_results.jsonl" \
  --v42-results "$OUT/v42/dev_eval/task_results.jsonl" \
  --protocol-lock "$PREF_LOCK" --eval-manifest "$DEV_DATA/manifest.json" \
  --eval-audit "$DEV_AUDIT/audit_summary.json" \
  --eval-audit-gate "$DEV_AUDIT/quality_gate.json" \
  --output "$OUT/development_gate.json" 2>&1 | tee "$OUT/development_gate_console.log"
DEV_STATUS=${PIPESTATUS[0]}
set -e
if [ "$DEV_STATUS" -ne 0 ]; then
  echo TOOLSANDBOX_V42_DEVELOPMENT_GATE_FAIL
  echo TOOLSANDBOX_V42_FINISHED_BEFORE_CONFIRMATION
  exit "$DEV_STATUS"
fi
echo TOOLSANDBOX_V42_DEVELOPMENT_GATE_PASS

# Only a passed development gate authorizes the untouched offset-165 audit.
set +e
"$APP_PY" scripts/audit_toolsandbox_signal.py \
  --limit 40 --scenario-offset 165 --seed 42 --horizon 8 \
  --event-search-steps 8 --credit-mode lexicographic_v4 \
  --worker-timeout-sec 600 --harness-interface tool_id_v2 \
  --protocol-lock "$CONFIRM_LOCK" --worker-python "$MODEL_PY" \
  --worker-script "$WORKER" --output-dir "$CONFIRM_AUDIT" \
  2>&1 | tee "$OUT/confirmation_audit_console.log"
AUDIT_STATUS=${PIPESTATUS[0]}
set -e
"$MODEL_PY" - "$CONFIRM_AUDIT/quality_gate.json" "$AUDIT_STATUS" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert gate.get("mechanism_passed") is True, (gate, sys.argv[2])
PY
echo "TOOLSANDBOX_V42_CONFIRM_AUDIT_MECHANISM_PASS status=$AUDIT_STATUS"

"$APP_PY" scripts/prepare_toolsandbox_v41_preference_data.py \
  --signal-events "$CONFIRM_AUDIT/signal_events.jsonl" \
  --audit-summary "$CONFIRM_AUDIT/audit_summary.json" \
  --quality-gate "$CONFIRM_AUDIT/quality_gate.json" \
  --role evaluation --output-dir "$CONFIRM_DATA" \
  2>&1 | tee "$OUT/prepare_confirmation.log"

eval_pair confirmation "$CONFIRM_DATA" confirm_eval
set +e
"$MODEL_PY" scripts/check_toolsandbox_v42_gate.py \
  --role confirmation \
  --mask-eval "$OUT/mask/confirm_eval/eval_summary.json" \
  --v42-eval "$OUT/v42/confirm_eval/eval_summary.json" \
  --mask-run "$OUT/mask/run_summary.json" --v42-run "$OUT/v42/run_summary.json" \
  --mask-results "$OUT/mask/confirm_eval/task_results.jsonl" \
  --v42-results "$OUT/v42/confirm_eval/task_results.jsonl" \
  --protocol-lock "$PREF_LOCK" --eval-manifest "$CONFIRM_DATA/manifest.json" \
  --eval-audit "$CONFIRM_AUDIT/audit_summary.json" \
  --eval-audit-gate "$CONFIRM_AUDIT/quality_gate.json" \
  --output "$OUT/confirmation_gate.json" \
  2>&1 | tee "$OUT/confirmation_gate_console.log"
CONFIRM_STATUS=${PIPESTATUS[0]}
set -e
if [ "$CONFIRM_STATUS" -eq 0 ]; then
  echo TOOLSANDBOX_V42_CONFIRMATION_GATE_PASS
else
  echo TOOLSANDBOX_V42_CONFIRMATION_GATE_FAIL
fi
echo TOOLSANDBOX_V42_FINISHED
exit "$CONFIRM_STATUS"
