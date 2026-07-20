#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${RESCUECREDIT_PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${RESCUECREDIT_MODEL:-/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct}"
V44="$ROOT/outputs/toolsandbox_v44_candidate_diversity_seed42/data"
V45="$ROOT/outputs/toolsandbox_v45_matched_anchor_seed42"
V46="$ROOT/outputs/toolsandbox_v46_selective_residual_seed42"
OUT="$ROOT/outputs/toolsandbox_v5_causal_router_seed42"
LOCK="$OUT/protocol_lock.json"
FEATURES="$OUT/features"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
test -x "$PY"
test -d "$MODEL"
test -f "$V46/development_gate.json"
[ ! -e "$OUT" ] || {
  echo "Refusing to reuse V5 output root: $OUT" >&2
  exit 1
}
mkdir -p "$FEATURES" "$OUT/margin_control" "$OUT/causal_router_v5"

"$PY" -m py_compile \
  rescuecredit/toolsandbox_router.py \
  scripts/freeze_toolsandbox_v5_protocol.py \
  scripts/build_toolsandbox_v5_features.py \
  scripts/train_toolsandbox_v5_router.py \
  scripts/score_toolsandbox_v5_router.py \
  scripts/evaluate_toolsandbox_v5_router.py \
  scripts/check_toolsandbox_v5_gate.py
"$PY" -m pytest -q \
  tests/test_toolsandbox_v5.py \
  tests/test_toolsandbox_v46.py \
  tests/test_toolsandbox_v45.py \
  tests/test_toolsandbox_preference.py

"$PY" scripts/freeze_toolsandbox_v5_protocol.py \
  --v44-data "$V44" \
  --v45-root "$V45" \
  --v46-root "$V46" \
  --model "$MODEL" \
  --output "$LOCK" | tee "$OUT/freeze.log"

GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2);print $2,$1}' | sort -n | head -n1 | awk '{print $2}')"
test -n "$GPU"
echo "V5_FEATURE_GPU=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/build_toolsandbox_v5_features.py \
  --model "$MODEL" \
  --mask-adapter "$V45/mask/adapter" \
  --train-file "$V44/train.jsonl" \
  --protocol-lock "$LOCK" \
  --output-dir "$FEATURES" | tee "$OUT/feature_console.log"

for METHOD in margin_control causal_router_v5; do
  "$PY" scripts/train_toolsandbox_v5_router.py \
    --method "$METHOD" \
    --feature-cache "$FEATURES/train_features.pt" \
    --feature-manifest "$FEATURES/feature_manifest.json" \
    --protocol-lock "$LOCK" \
    --output-dir "$OUT/$METHOD" | tee "$OUT/$METHOD/train_console.log"
done
echo TOOLSANDBOX_V5_ROUTER_TRAINING_FINISHED

score_split() {
  local role="$1" data="$2" sub="$3"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/score_toolsandbox_v5_router.py \
    --model "$MODEL" \
    --mask-adapter "$V45/mask/adapter" \
    --control-router "$OUT/margin_control/router.pt" \
    --v5-router "$OUT/causal_router_v5/router.pt" \
    --protocol-lock "$LOCK" \
    --public-events "$data/events.public.jsonl" \
    --evaluation-role "$role" \
    --output-dir "$OUT/$sub" | tee "$OUT/${sub}_score_console.log"
}

evaluate_split() {
  local role="$1" data="$2" sub="$3"
  for METHOD in mask margin_control causal_router_v5; do
    "$PY" scripts/evaluate_toolsandbox_v5_router.py \
      --method "$METHOD" \
      --predictions "$OUT/$sub/$METHOD.predictions.jsonl" \
      --scoring-summary "$OUT/$sub/scoring_summary.json" \
      --public-events "$data/events.public.jsonl" \
      --private-outcomes "$data/outcomes.private.jsonl" \
      --protocol-lock "$LOCK" \
      --evaluation-role "$role" \
      --output-dir "$OUT/$METHOD/$sub" | tee "$OUT/$METHOD/${sub}_eval_console.log"
  done
}

score_split development "$V45/development_data" dev_eval
evaluate_split development "$V45/development_data" dev_eval

set +e
"$PY" scripts/check_toolsandbox_v5_gate.py \
  --mask-eval "$OUT/mask/dev_eval/eval_summary.json" \
  --control-eval "$OUT/margin_control/dev_eval/eval_summary.json" \
  --v5-eval "$OUT/causal_router_v5/dev_eval/eval_summary.json" \
  --mask-results "$OUT/mask/dev_eval/task_results.jsonl" \
  --control-results "$OUT/margin_control/dev_eval/task_results.jsonl" \
  --v5-results "$OUT/causal_router_v5/dev_eval/task_results.jsonl" \
  --control-run "$OUT/margin_control/run_summary.json" \
  --v5-run "$OUT/causal_router_v5/run_summary.json" \
  --control-router "$OUT/margin_control/router.pt" \
  --v5-router "$OUT/causal_router_v5/router.pt" \
  --feature-cache "$FEATURES/train_features.pt" \
  --feature-manifest "$FEATURES/feature_manifest.json" \
  --scoring-summary "$OUT/dev_eval/scoring_summary.json" \
  --protocol-lock "$LOCK" \
  --output "$OUT/development_gate.json" | tee "$OUT/gate_console.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -eq 0 ]; then
  echo TOOLSANDBOX_V5_DEVELOPMENT_GATE_PASS
else
  echo TOOLSANDBOX_V5_DEVELOPMENT_GATE_FAIL
fi

# The old V4.5 confirmation split is known and remains post-hoc for V5.
score_split posthoc "$V45/confirmation_data" posthoc_confirm
evaluate_split posthoc "$V45/confirmation_data" posthoc_confirm
echo TOOLSANDBOX_V5_FINISHED
exit "$STATUS"
