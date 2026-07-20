#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PY="${TOOLSANDBOX_PYTHON:-/data/hxy/venvs/rescuecredit-toolsandbox/bin/python}"
MODEL_PY="${RESCUECREDIT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
MODEL="${RESCUECREDIT_MODEL:-/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct}"
WORKER="$REPO_ROOT/scripts/toolsandbox_azure_worker.py"
STAGE0="$REPO_ROOT/outputs/toolsandbox_stage0/gate.json"
V44="$REPO_ROOT/outputs/toolsandbox_v44_candidate_diversity_seed42"
PLAN="$REPO_ROOT/refine-logs/TOOLSANDBOX_V45_PLAN.md"
OUT="$REPO_ROOT/outputs/toolsandbox_v45_matched_anchor_seed42"
DEV_LOCK="$OUT/development_candidate_protocol.json"
CONFIRM_LOCK="$OUT/confirmation_candidate_protocol.json"
LEARNER_LOCK="$OUT/learner_protocol.json"

cd "$REPO_ROOT"
export PROMPT_COMMAND=
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
test -x "$APP_PY"; test -x "$MODEL_PY"; test -d "$MODEL"
test -f "$V44/data/data_gate.json"; test -f "$V44/data/train.jsonl"
if [ -e "$OUT" ]; then echo "Refusing to reuse V4.5 output root: $OUT" >&2; exit 1; fi
mkdir -p "$OUT" "$OUT/mask" "$OUT/v45"

"$MODEL_PY" -m py_compile scripts/freeze_toolsandbox_v45_candidate_protocol.py \
  scripts/freeze_toolsandbox_v45_learner_protocol.py scripts/check_toolsandbox_v45_gate.py \
  scripts/train_toolsandbox_v43_preference.py scripts/evaluate_toolsandbox_v43_preference.py
"$MODEL_PY" -m pytest -q tests/test_toolsandbox_v45.py tests/test_toolsandbox_v44.py \
  tests/test_toolsandbox_v43.py tests/test_toolsandbox_preference.py

set -a; source .env; set +a
test "${TOOLSANDBOX_LLM_PROVIDER:-}" = deepseek
test -n "${DEEPSEEK_API_KEY:-}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://zhi-api.com/v1}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
test "$DEEPSEEK_THINKING" = disabled
"$MODEL_PY" scripts/check_llm.py --provider deepseek

freeze_candidate() {
  local role="$1" offset="$2" output="$3"
  "$APP_PY" scripts/freeze_toolsandbox_v45_candidate_protocol.py \
    --evaluation-role "$role" --output "$output" --plan "$PLAN" \
    --stage0-gate "$STAGE0" --v44-root "$V44" --seed 42 \
    --scenario-offset "$offset" --limit 40 --horizon 8 --event-search-steps 8 \
    --candidate-count 3 --max-pairs-per-scenario 4 --worker-timeout-sec 600 \
    --harness-interface tool_id_v2
}
# Both evaluation identities and the learner are frozen before any V4.5 outcome.
freeze_candidate development 125 "$DEV_LOCK" | tee "$OUT/freeze_development.log"
freeze_candidate confirmation 165 "$CONFIRM_LOCK" | tee "$OUT/freeze_confirmation.log"
"$MODEL_PY" scripts/freeze_toolsandbox_v45_learner_protocol.py \
  --data-dir "$V44/data" --source-audit-root "$V44/full_offset85_h8" \
  --source-protocol "$V44/full_protocol_lock.json" \
  --development-protocol "$DEV_LOCK" --confirmation-protocol "$CONFIRM_LOCK" \
  --model "$MODEL" --output "$LEARNER_LOCK" | tee "$OUT/freeze_learner.log"

run_candidate_audit() {
  local role="$1" offset="$2" lock="$3" audit="$4" data="$5"
  "$APP_PY" scripts/audit_toolsandbox_v44_candidates.py --role full \
    --protocol-lock "$lock" --worker-python "$MODEL_PY" --worker-script "$WORKER" \
    --worker-timeout-sec 600 --harness-interface tool_id_v2 --seed 42 \
    --scenario-offset "$offset" --limit 40 --horizon 8 --event-search-steps 8 \
    --candidate-count 3 --max-pairs-per-scenario 4 --output-dir "$audit"
  "$APP_PY" scripts/prepare_toolsandbox_v44_candidate_data.py --audit-root "$audit" \
    --protocol-lock "$lock" --data-role evaluation --output-dir "$data"
  echo "TOOLSANDBOX_V45_${role^^}_CANDIDATES_READY"
}

DEV_AUDIT="$OUT/development_candidates_offset125_h8"; DEV_DATA="$OUT/development_data"
run_candidate_audit development 125 "$DEV_LOCK" "$DEV_AUDIT" "$DEV_DATA" \
  2>&1 | tee "$OUT/development_candidate_console.log"

mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
  awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2);print $2,$1}' | sort -n | head -n2 | awk '{print $2}')
[ "${#GPUS[@]}" -ge 2 ] || { echo "Need two GPUs" >&2; exit 2; }
echo "MASK_GPU=${GPUS[0]} V45_GPU=${GPUS[1]}"
train_one() {
  local method="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" scripts/train_toolsandbox_v43_preference.py \
    --method "$method" --model "$MODEL" --train-file "$V44/data/train.jsonl" \
    --protocol-lock "$LEARNER_LOCK" --seed 42 --epochs 3 --learning-rate 3e-6 \
    --gradient-accumulation 8 --max-length 2048 --beta 1.0 --absolute-margin-coef 1.0 \
    --target-margin 0.05 --reference-anchor-coef 0.25 --presentations-per-epoch 126 \
    --lora-r 16 --lora-alpha 32 --fp32 --output-dir "$OUT/$method" \
    > "$OUT/$method/train_console.log" 2>&1
}
train_one mask "${GPUS[0]}" & P1=$!; train_one v45 "${GPUS[1]}" & P2=$!
trap 'kill "$P1" "$P2" 2>/dev/null || true' EXIT INT TERM
set +e; wait "$P1"; S1=$?; wait "$P2"; S2=$?; set -e; trap - EXIT INT TERM
[ "$S1" -eq 0 ] && [ "$S2" -eq 0 ] || { echo "training failed mask=$S1 v45=$S2" >&2; exit 1; }
echo TOOLSANDBOX_V45_TRAINING_FINISHED

eval_pair() {
  local role="$1" data="$2" sub="$3"
  for spec in "mask:${GPUS[0]}" "v45:${GPUS[1]}"; do
    method=${spec%%:*}; gpu=${spec##*:}
    CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" scripts/evaluate_toolsandbox_v43_preference.py \
      --method "$method" --model "$MODEL" --adapter "$OUT/$method/adapter" \
      --run-summary "$OUT/$method/run_summary.json" --protocol-lock "$LEARNER_LOCK" \
      --public-events "$data/events.public.jsonl" --private-outcomes "$data/outcomes.private.jsonl" \
      --evaluation-role "$role" --max-length 2048 --fp32 --output-dir "$OUT/$method/$sub" \
      > "$OUT/$method/${sub}_console.log" 2>&1 &
    [ "$method" = mask ] && E1=$! || E2=$!
  done
  set +e; wait "$E1"; S1=$?; wait "$E2"; S2=$?; set -e
  [ "$S1" -eq 0 ] && [ "$S2" -eq 0 ] || { echo "$role eval failed mask=$S1 v45=$S2" >&2; exit 1; }
}
gate_pair() {
  local role="$1" lock="$2" audit="$3" data="$4" sub="$5" output="$6"
  "$MODEL_PY" scripts/check_toolsandbox_v45_gate.py --role "$role" \
    --mask-eval "$OUT/mask/$sub/eval_summary.json" --v45-eval "$OUT/v45/$sub/eval_summary.json" \
    --mask-run "$OUT/mask/run_summary.json" --v45-run "$OUT/v45/run_summary.json" \
    --mask-results "$OUT/mask/$sub/task_results.jsonl" --v45-results "$OUT/v45/$sub/task_results.jsonl" \
    --protocol-lock "$LEARNER_LOCK" --candidate-protocol "$lock" \
    --eval-manifest "$data/manifest.json" --eval-data-gate "$data/data_gate.json" \
    --eval-audit "$audit/audit_summary.json" --output "$output"
}

eval_pair development "$DEV_DATA" dev_eval
set +e; gate_pair development "$DEV_LOCK" "$DEV_AUDIT" "$DEV_DATA" dev_eval "$OUT/development_gate.json" | tee "$OUT/development_gate_console.log"; DEV_STATUS=${PIPESTATUS[0]}; set -e
if [ "$DEV_STATUS" -ne 0 ]; then echo TOOLSANDBOX_V45_DEVELOPMENT_GATE_FAIL; echo TOOLSANDBOX_V45_FINISHED_BEFORE_CONFIRMATION; exit "$DEV_STATUS"; fi
echo TOOLSANDBOX_V45_DEVELOPMENT_GATE_PASS

CONFIRM_AUDIT="$OUT/confirmation_candidates_offset165_h8"; CONFIRM_DATA="$OUT/confirmation_data"
run_candidate_audit confirmation 165 "$CONFIRM_LOCK" "$CONFIRM_AUDIT" "$CONFIRM_DATA" \
  2>&1 | tee "$OUT/confirmation_candidate_console.log"
eval_pair confirmation "$CONFIRM_DATA" confirm_eval
set +e; gate_pair confirmation "$CONFIRM_LOCK" "$CONFIRM_AUDIT" "$CONFIRM_DATA" confirm_eval "$OUT/confirmation_gate.json" | tee "$OUT/confirmation_gate_console.log"; STATUS=${PIPESTATUS[0]}; set -e
[ "$STATUS" -eq 0 ] && echo TOOLSANDBOX_V45_CONFIRMATION_GATE_PASS || echo TOOLSANDBOX_V45_CONFIRMATION_GATE_FAIL
echo TOOLSANDBOX_V45_FINISHED
exit "$STATUS"
