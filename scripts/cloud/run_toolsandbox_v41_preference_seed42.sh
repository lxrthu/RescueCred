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
PLAN="$REPO_ROOT/refine-logs/TOOLSANDBOX_V41_PREFERENCE_PLAN.md"
OUT="$REPO_ROOT/outputs/toolsandbox_v41_preference_seed42"
TRAIN_DATA="$OUT/train_data"
EVAL_AUDIT="$OUT/fresh_eval_offset125_h8"
EVAL_DATA="$OUT/eval_data"
EVAL_LOCK="$OUT/evaluation_protocol_lock.json"
PREF_LOCK="$OUT/preference_protocol_lock.json"

cd "$REPO_ROOT"
export PROMPT_COMMAND=
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -x "$APP_PY"
test -x "$MODEL_PY"
test -d "$MODEL"
test -f "$WORKER"
test -f "$STAGE0"
test -f "$V4_OLD_LOCK"
test -f "$V41_DIAGNOSTIC_LOCK"
test -f "$V41_TRAIN_LOCK"
test -f "$SOURCE_AUDIT/signal_events.jsonl"
test -f "$SOURCE_AUDIT/audit_summary.json"
test -f "$SOURCE_AUDIT/quality_gate.json"
test -f "$PLAN"
if [ -e "$OUT" ]; then
  echo "Refusing to reuse V4.1 preference output root: $OUT" >&2
  exit 1
fi
mkdir -p "$TRAIN_DATA" "$EVAL_DATA" "$OUT/mask" "$OUT/v4"

"$MODEL_PY" -m py_compile \
  scripts/prepare_toolsandbox_v41_preference_data.py \
  scripts/train_toolsandbox_v41_preference.py \
  scripts/evaluate_toolsandbox_v41_preference.py \
  scripts/freeze_toolsandbox_v41_preference_protocol.py \
  scripts/check_toolsandbox_v41_preference_gate.py
"$MODEL_PY" -m pytest -q \
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

# Convert the already passed V4.1 audit into a public-prompt/private-credit
# training set.  This does not call the continuation model again.
"$APP_PY" scripts/prepare_toolsandbox_v41_preference_data.py \
  --signal-events "$SOURCE_AUDIT/signal_events.jsonl" \
  --audit-summary "$SOURCE_AUDIT/audit_summary.json" \
  --quality-gate "$SOURCE_AUDIT/quality_gate.json" \
  --role train --output-dir "$TRAIN_DATA" \
  2>&1 | tee "$OUT/prepare_train.log"

# Freeze the untouched evaluation scenarios before either adapter is trained.
"$APP_PY" scripts/freeze_toolsandbox_v4_protocol.py \
  --output "$EVAL_LOCK" --plan "$PLAN" --stage0-gate "$STAGE0" \
  --seed 42 --scenario-offset 125 --limit 40 --minimum-scenarios 40 \
  --horizon 8 --event-search-steps 8 --worker-timeout-sec 600 \
  --harness-interface tool_id_v2 \
  --exclude-protocol "$V4_OLD_LOCK" \
  --exclude-protocol "$V41_DIAGNOSTIC_LOCK" \
  --exclude-protocol "$V41_TRAIN_LOCK"

"$MODEL_PY" scripts/freeze_toolsandbox_v41_preference_protocol.py \
  --data-dir "$TRAIN_DATA" --source-audit-root "$SOURCE_AUDIT" \
  --evaluation-protocol "$EVAL_LOCK" --model "$MODEL" --output "$PREF_LOCK" \
  2>&1 | tee "$OUT/freeze_preference_protocol.log"

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' |
    sort -n | head -n 2 | awk '{print $2}'
)
if [ "${#GPUS[@]}" -lt 2 ]; then
  echo "Need two GPUs for the matched Mask/V4 pair." >&2
  exit 2
fi
echo "MASK_GPU=${GPUS[0]} V4_GPU=${GPUS[1]}"

train_one() {
  local method="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" scripts/train_toolsandbox_v41_preference.py \
    --method "$method" --model "$MODEL" --train-file "$TRAIN_DATA/train.jsonl" \
    --protocol-lock "$PREF_LOCK" --seed 42 --epochs 3 --learning-rate 3e-6 \
    --gradient-accumulation 8 --max-length 2048 --beta 1.0 \
    --lora-r 16 --lora-alpha 32 --fp32 --output-dir "$OUT/$method" \
    > "$OUT/$method/train_console.log" 2>&1
}
train_one mask "${GPUS[0]}" & MASK_PID=$!
train_one v4 "${GPUS[1]}" & V4_PID=$!
cleanup_train() { kill "$MASK_PID" "$V4_PID" 2>/dev/null || true; }
trap cleanup_train EXIT INT TERM
set +e
wait "$MASK_PID"; MASK_STATUS=$?
wait "$V4_PID"; V4_STATUS=$?
set -e
trap - EXIT INT TERM
if [ "$MASK_STATUS" -ne 0 ] || [ "$V4_STATUS" -ne 0 ]; then
  echo "Matched training failed: mask=$MASK_STATUS v4=$V4_STATUS" >&2
  exit 1
fi
echo TOOLSANDBOX_V41_PREFERENCE_TRAINING_FINISHED

# Generate outcomes only for the pre-frozen untouched evaluation scenarios.
set +e
"$APP_PY" scripts/audit_toolsandbox_signal.py \
  --limit 40 --scenario-offset 125 --seed 42 --horizon 8 \
  --event-search-steps 8 --credit-mode lexicographic_v4 \
  --worker-timeout-sec 600 --harness-interface tool_id_v2 \
  --protocol-lock "$EVAL_LOCK" --worker-python "$MODEL_PY" \
  --worker-script "$WORKER" --output-dir "$EVAL_AUDIT" \
  2>&1 | tee "$OUT/eval_audit_console.log"
AUDIT_STATUS=${PIPESTATUS[0]}
set -e
"$MODEL_PY" - "$EVAL_AUDIT/quality_gate.json" "$AUDIT_STATUS" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert gate.get("mechanism_passed") is True, (gate, sys.argv[2])
PY
echo "TOOLSANDBOX_V41_PREFERENCE_EVAL_AUDIT_MECHANISM_PASS status=$AUDIT_STATUS"

"$APP_PY" scripts/prepare_toolsandbox_v41_preference_data.py \
  --signal-events "$EVAL_AUDIT/signal_events.jsonl" \
  --audit-summary "$EVAL_AUDIT/audit_summary.json" \
  --quality-gate "$EVAL_AUDIT/quality_gate.json" \
  --role evaluation --output-dir "$EVAL_DATA" \
  2>&1 | tee "$OUT/prepare_eval.log"

eval_one() {
  local method="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" scripts/evaluate_toolsandbox_v41_preference.py \
    --method "$method" --model "$MODEL" --adapter "$OUT/$method/adapter" \
    --run-summary "$OUT/$method/run_summary.json" --protocol-lock "$PREF_LOCK" \
    --public-events "$EVAL_DATA/events.public.jsonl" \
    --private-outcomes "$EVAL_DATA/outcomes.private.jsonl" \
    --max-length 2048 --fp32 --output-dir "$OUT/$method/eval" \
    > "$OUT/$method/eval_console.log" 2>&1
}
eval_one mask "${GPUS[0]}" & MASK_EVAL_PID=$!
eval_one v4 "${GPUS[1]}" & V4_EVAL_PID=$!
cleanup_eval() { kill "$MASK_EVAL_PID" "$V4_EVAL_PID" 2>/dev/null || true; }
trap cleanup_eval EXIT INT TERM
set +e
wait "$MASK_EVAL_PID"; MASK_EVAL_STATUS=$?
wait "$V4_EVAL_PID"; V4_EVAL_STATUS=$?
set -e
trap - EXIT INT TERM
if [ "$MASK_EVAL_STATUS" -ne 0 ] || [ "$V4_EVAL_STATUS" -ne 0 ]; then
  echo "Matched evaluation failed: mask=$MASK_EVAL_STATUS v4=$V4_EVAL_STATUS" >&2
  exit 1
fi

set +e
"$MODEL_PY" scripts/check_toolsandbox_v41_preference_gate.py \
  --mask-eval "$OUT/mask/eval/eval_summary.json" \
  --v4-eval "$OUT/v4/eval/eval_summary.json" \
  --mask-run "$OUT/mask/run_summary.json" --v4-run "$OUT/v4/run_summary.json" \
  --mask-results "$OUT/mask/eval/task_results.jsonl" \
  --v4-results "$OUT/v4/eval/task_results.jsonl" \
  --protocol-lock "$PREF_LOCK" --eval-manifest "$EVAL_DATA/manifest.json" \
  --eval-audit "$EVAL_AUDIT/audit_summary.json" \
  --eval-audit-gate "$EVAL_AUDIT/quality_gate.json" --output "$OUT/gate.json" \
  2>&1 | tee "$OUT/gate_console.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -eq 0 ]; then
  echo TOOLSANDBOX_V41_PREFERENCE_GATE_PASS
else
  echo TOOLSANDBOX_V41_PREFERENCE_GATE_FAIL
fi
echo TOOLSANDBOX_V41_PREFERENCE_FINISHED
exit "$STATUS"
