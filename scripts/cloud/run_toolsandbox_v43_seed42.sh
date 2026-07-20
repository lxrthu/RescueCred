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
V41_PREF_ROOT="$REPO_ROOT/outputs/toolsandbox_v41_preference_seed42"
V41_DEV_LOCK="$V41_PREF_ROOT/evaluation_protocol_lock.json"
DEV_AUDIT="$V41_PREF_ROOT/fresh_eval_offset125_h8"
V42_ROOT="$REPO_ROOT/outputs/toolsandbox_v42_balanced_margin_seed42"
V42_DEV_DATA="$V42_ROOT/development_data"
V42_DEV_GATE="$V42_ROOT/development_gate.json"
V42_CONFIRM_LOCK="$V42_ROOT/confirmation_protocol_lock.json"
PLAN="$REPO_ROOT/refine-logs/TOOLSANDBOX_V43_PLAN.md"

OUT="$REPO_ROOT/outputs/toolsandbox_v43_multi_prefix_anchor_seed42"
MINING_LOCK="$OUT/mining_protocol_lock.json"
MINING_AUDIT="$OUT/multi_prefix_offset85_h8"
TRAIN_DATA="$OUT/train_data"
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

test -x "$APP_PY"
test -x "$MODEL_PY"
test -d "$MODEL"
for path in \
  "$WORKER" "$STAGE0" "$V4_OLD_LOCK" "$V41_DIAGNOSTIC_LOCK" \
  "$V41_TRAIN_LOCK" "$V41_DEV_LOCK" "$DEV_AUDIT/audit_summary.json" \
  "$DEV_AUDIT/quality_gate.json" "$V42_DEV_DATA/manifest.json" \
  "$V42_DEV_DATA/events.public.jsonl" "$V42_DEV_DATA/outcomes.private.jsonl" \
  "$V42_DEV_GATE" "$V42_CONFIRM_LOCK" "$PLAN"; do
  test -e "$path"
done
if [ -e "$V42_ROOT/fresh_confirm_offset165_h8/audit_summary.json" ]; then
  echo "Refusing V4.3: offset-165 confirmation outcomes already exist." >&2
  exit 1
fi
if [ -e "$OUT" ]; then
  echo "Refusing to reuse V4.3 output root: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT" "$TRAIN_DATA" "$CONFIRM_DATA" "$OUT/mask" "$OUT/v43"

"$MODEL_PY" -m py_compile \
  scripts/audit_toolsandbox_signal.py \
  scripts/freeze_toolsandbox_v4_protocol.py \
  scripts/prepare_toolsandbox_v43_training_data.py \
  scripts/train_toolsandbox_v43_preference.py \
  scripts/evaluate_toolsandbox_v43_preference.py \
  scripts/freeze_toolsandbox_v43_protocol.py \
  scripts/check_toolsandbox_v43_gate.py
"$MODEL_PY" -m pytest -q \
  tests/test_toolsandbox_v43.py \
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

# Freeze both scenario identities before mining, training, or scoring.  Offset
# 85 intentionally reuses only the prior training tasks; offset 165 remains the
# untouched confirmation set.  The final 13 scenarios after offset 205 are held out.
"$APP_PY" scripts/freeze_toolsandbox_v4_protocol.py \
  --output "$MINING_LOCK" --plan "$PLAN" --stage0-gate "$STAGE0" \
  --seed 42 --scenario-offset 85 --limit 40 --minimum-scenarios 40 \
  --horizon 8 --event-search-steps 8 --max-events-per-scenario 4 \
  --worker-timeout-sec 600 --harness-interface tool_id_v2 \
  --exclude-protocol "$V4_OLD_LOCK" \
  --exclude-protocol "$V41_DIAGNOSTIC_LOCK" \
  --exclude-protocol "$V41_DEV_LOCK" \
  --exclude-protocol "$V42_CONFIRM_LOCK"
"$APP_PY" scripts/freeze_toolsandbox_v4_protocol.py \
  --output "$CONFIRM_LOCK" --plan "$PLAN" --stage0-gate "$STAGE0" \
  --seed 42 --scenario-offset 165 --limit 40 --minimum-scenarios 40 \
  --horizon 8 --event-search-steps 8 --max-events-per-scenario 1 \
  --worker-timeout-sec 600 --harness-interface tool_id_v2 \
  --exclude-protocol "$V4_OLD_LOCK" \
  --exclude-protocol "$V41_DIAGNOSTIC_LOCK" \
  --exclude-protocol "$V41_TRAIN_LOCK" \
  --exclude-protocol "$V41_DEV_LOCK"

set +e
"$APP_PY" scripts/audit_toolsandbox_signal.py \
  --limit 40 --scenario-offset 85 --seed 42 --horizon 8 \
  --event-search-steps 8 --max-events-per-scenario 4 \
  --credit-mode lexicographic_v4 --worker-timeout-sec 600 \
  --harness-interface tool_id_v2 --protocol-lock "$MINING_LOCK" \
  --worker-python "$MODEL_PY" --worker-script "$WORKER" \
  --output-dir "$MINING_AUDIT" 2>&1 | tee "$OUT/mining_console.log"
MINING_STATUS=${PIPESTATUS[0]}
set -e
"$MODEL_PY" - "$MINING_AUDIT/quality_gate.json" "$MINING_STATUS" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert gate.get("mechanism_passed") is True, (gate, sys.argv[2])
PY

"$APP_PY" scripts/prepare_toolsandbox_v43_training_data.py \
  --audit-root "$MINING_AUDIT" --output-dir "$TRAIN_DATA" \
  2>&1 | tee "$OUT/prepare_train.log"

"$MODEL_PY" scripts/freeze_toolsandbox_v43_protocol.py \
  --data-dir "$TRAIN_DATA" --mining-audit-root "$MINING_AUDIT" \
  --mining-protocol "$MINING_LOCK" --old-training-protocol "$V41_TRAIN_LOCK" \
  --development-data-dir "$V42_DEV_DATA" --development-audit-root "$DEV_AUDIT" \
  --v42-development-gate "$V42_DEV_GATE" \
  --confirmation-protocol "$CONFIRM_LOCK" \
  --old-v42-confirmation-protocol "$V42_CONFIRM_LOCK" --v42-root "$V42_ROOT" \
  --model "$MODEL" --output "$PREF_LOCK" \
  2>&1 | tee "$OUT/freeze_preference_protocol.log"

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' |
    sort -n | head -n 2 | awk '{print $2}'
)
if [ "${#GPUS[@]}" -lt 2 ]; then
  echo "Need two GPUs for the matched Mask/V4.3 pair." >&2
  exit 2
fi
echo "MASK_GPU=${GPUS[0]} V43_GPU=${GPUS[1]}"

train_one() {
  local method="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" scripts/train_toolsandbox_v43_preference.py \
    --method "$method" --model "$MODEL" --train-file "$TRAIN_DATA/train.jsonl" \
    --protocol-lock "$PREF_LOCK" --seed 42 --epochs 3 --learning-rate 3e-6 \
    --gradient-accumulation 8 --max-length 2048 --beta 1.0 \
    --absolute-margin-coef 1.0 --target-margin 0.05 \
    --reference-anchor-coef 0.25 --presentations-per-epoch 60 \
    --lora-r 16 --lora-alpha 32 --fp32 --output-dir "$OUT/$method" \
    > "$OUT/$method/train_console.log" 2>&1
}
train_one mask "${GPUS[0]}" & MASK_PID=$!
train_one v43 "${GPUS[1]}" & V43_PID=$!
cleanup_train() { kill "$MASK_PID" "$V43_PID" 2>/dev/null || true; }
trap cleanup_train EXIT INT TERM
set +e
wait "$MASK_PID"; MASK_STATUS=$?
wait "$V43_PID"; V43_STATUS=$?
set -e
trap - EXIT INT TERM
if [ "$MASK_STATUS" -ne 0 ] || [ "$V43_STATUS" -ne 0 ]; then
  echo "Matched training failed: mask=$MASK_STATUS v43=$V43_STATUS" >&2
  exit 1
fi
echo TOOLSANDBOX_V43_TRAINING_FINISHED

eval_pair() {
  local role="$1" data="$2" subdir="$3"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$MODEL_PY" scripts/evaluate_toolsandbox_v43_preference.py \
    --method mask --model "$MODEL" --adapter "$OUT/mask/adapter" \
    --run-summary "$OUT/mask/run_summary.json" --protocol-lock "$PREF_LOCK" \
    --public-events "$data/events.public.jsonl" \
    --private-outcomes "$data/outcomes.private.jsonl" \
    --evaluation-role "$role" --max-length 2048 --fp32 \
    --output-dir "$OUT/mask/$subdir" > "$OUT/mask/${subdir}_console.log" 2>&1 &
  MASK_EVAL_PID=$!
  CUDA_VISIBLE_DEVICES="${GPUS[1]}" "$MODEL_PY" scripts/evaluate_toolsandbox_v43_preference.py \
    --method v43 --model "$MODEL" --adapter "$OUT/v43/adapter" \
    --run-summary "$OUT/v43/run_summary.json" --protocol-lock "$PREF_LOCK" \
    --public-events "$data/events.public.jsonl" \
    --private-outcomes "$data/outcomes.private.jsonl" \
    --evaluation-role "$role" --max-length 2048 --fp32 \
    --output-dir "$OUT/v43/$subdir" > "$OUT/v43/${subdir}_console.log" 2>&1 &
  V43_EVAL_PID=$!
  cleanup_eval() { kill "$MASK_EVAL_PID" "$V43_EVAL_PID" 2>/dev/null || true; }
  trap cleanup_eval EXIT INT TERM
  set +e
  wait "$MASK_EVAL_PID"; MASK_EVAL_STATUS=$?
  wait "$V43_EVAL_PID"; V43_EVAL_STATUS=$?
  set -e
  trap - EXIT INT TERM
  if [ "$MASK_EVAL_STATUS" -ne 0 ] || [ "$V43_EVAL_STATUS" -ne 0 ]; then
    echo "Matched $role evaluation failed: mask=$MASK_EVAL_STATUS v43=$V43_EVAL_STATUS" >&2
    exit 1
  fi
}

gate_pair() {
  local role="$1" data="$2" audit="$3" subdir="$4" output="$5"
  "$MODEL_PY" scripts/check_toolsandbox_v43_gate.py \
    --role "$role" --mask-eval "$OUT/mask/$subdir/eval_summary.json" \
    --v43-eval "$OUT/v43/$subdir/eval_summary.json" \
    --mask-run "$OUT/mask/run_summary.json" --v43-run "$OUT/v43/run_summary.json" \
    --mask-results "$OUT/mask/$subdir/task_results.jsonl" \
    --v43-results "$OUT/v43/$subdir/task_results.jsonl" \
    --protocol-lock "$PREF_LOCK" --eval-manifest "$data/manifest.json" \
    --eval-audit "$audit/audit_summary.json" \
    --eval-audit-gate "$audit/quality_gate.json" --output "$output"
}

# Offset 125 is known development data.  A failed gate stops before any call
# on offset 165, preserving the confirmatory evidence.
eval_pair development "$V42_DEV_DATA" dev_eval
set +e
gate_pair development "$V42_DEV_DATA" "$DEV_AUDIT" dev_eval \
  "$OUT/development_gate.json" 2>&1 | tee "$OUT/development_gate_console.log"
DEV_STATUS=${PIPESTATUS[0]}
set -e
if [ "$DEV_STATUS" -ne 0 ]; then
  echo TOOLSANDBOX_V43_DEVELOPMENT_GATE_FAIL
  echo TOOLSANDBOX_V43_FINISHED_BEFORE_CONFIRMATION
  exit "$DEV_STATUS"
fi
echo TOOLSANDBOX_V43_DEVELOPMENT_GATE_PASS

set +e
"$APP_PY" scripts/audit_toolsandbox_signal.py \
  --limit 40 --scenario-offset 165 --seed 42 --horizon 8 \
  --event-search-steps 8 --max-events-per-scenario 1 \
  --credit-mode lexicographic_v4 --worker-timeout-sec 600 \
  --harness-interface tool_id_v2 --protocol-lock "$CONFIRM_LOCK" \
  --worker-python "$MODEL_PY" --worker-script "$WORKER" \
  --output-dir "$CONFIRM_AUDIT" 2>&1 | tee "$OUT/confirmation_audit_console.log"
CONFIRM_AUDIT_STATUS=${PIPESTATUS[0]}
set -e
"$MODEL_PY" - "$CONFIRM_AUDIT/quality_gate.json" "$CONFIRM_AUDIT_STATUS" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert gate.get("mechanism_passed") is True, (gate, sys.argv[2])
PY

"$APP_PY" scripts/prepare_toolsandbox_v41_preference_data.py \
  --signal-events "$CONFIRM_AUDIT/signal_events.jsonl" \
  --audit-summary "$CONFIRM_AUDIT/audit_summary.json" \
  --quality-gate "$CONFIRM_AUDIT/quality_gate.json" \
  --role evaluation --output-dir "$CONFIRM_DATA" \
  2>&1 | tee "$OUT/prepare_confirmation.log"

eval_pair confirmation "$CONFIRM_DATA" confirm_eval
set +e
gate_pair confirmation "$CONFIRM_DATA" "$CONFIRM_AUDIT" confirm_eval \
  "$OUT/confirmation_gate.json" 2>&1 | tee "$OUT/confirmation_gate_console.log"
CONFIRM_STATUS=${PIPESTATUS[0]}
set -e
if [ "$CONFIRM_STATUS" -eq 0 ]; then
  echo TOOLSANDBOX_V43_CONFIRMATION_GATE_PASS
else
  echo TOOLSANDBOX_V43_CONFIRMATION_GATE_FAIL
fi
echo TOOLSANDBOX_V43_FINISHED
exit "$CONFIRM_STATUS"
