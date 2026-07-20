#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${RESCUECREDIT_PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${RESCUECREDIT_MODEL:-/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct}"
V5="$ROOT/outputs/toolsandbox_v5_causal_router_seed42"
V45="$ROOT/outputs/toolsandbox_v45_matched_anchor_seed42"
OUT="$ROOT/outputs/toolsandbox_v6_reverse_diagnostic_seed42"
LOCK="$OUT/protocol_lock.json"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
test -x "$PY"
test -d "$MODEL"
test -f "$V5/development_gate.json"
[ ! -e "$OUT" ] || {
  echo "Refusing to reuse V6 output root: $OUT" >&2
  exit 1
}
mkdir -p "$OUT/margin_probe" "$OUT/semantic_probe"

"$PY" -m py_compile \
  rescuecredit/toolsandbox_selective_router.py \
  scripts/freeze_toolsandbox_v6_protocol.py \
  scripts/train_toolsandbox_v6_diagnostic.py \
  scripts/score_toolsandbox_v6_diagnostic.py \
  scripts/evaluate_toolsandbox_v6_diagnostic.py \
  scripts/check_toolsandbox_v6_gate.py
"$PY" -m pytest -q tests/test_toolsandbox_v6.py tests/test_toolsandbox_v5.py

"$PY" scripts/freeze_toolsandbox_v6_protocol.py \
  --v5-root "$V5" \
  --output "$LOCK" | tee "$OUT/freeze.log"

for METHOD in margin_probe semantic_probe; do
  "$PY" scripts/train_toolsandbox_v6_diagnostic.py \
    --method "$METHOD" \
    --feature-cache "$V5/features/train_features.pt" \
    --feature-manifest "$V5/features/feature_manifest.json" \
    --protocol-lock "$LOCK" \
    --output-dir "$OUT/$METHOD" | tee "$OUT/$METHOD/train_console.log"
done
echo TOOLSANDBOX_V6_CROSS_TASK_DIAGNOSTIC_FINISHED

GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2);print $2,$1}' | sort -n | head -n1 | awk '{print $2}')"
test -n "$GPU"

score_split() {
  local role="$1" data="$2" sub="$3"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/score_toolsandbox_v6_diagnostic.py \
    --model "$MODEL" \
    --mask-adapter "$V45/mask/adapter" \
    --margin-probe "$OUT/margin_probe/probe.pt" \
    --semantic-probe "$OUT/semantic_probe/probe.pt" \
    --protocol-lock "$LOCK" \
    --public-events "$data/events.public.jsonl" \
    --evaluation-role "$role" \
    --output-dir "$OUT/$sub" | tee "$OUT/${sub}_score_console.log"
}

evaluate_split() {
  local role="$1" data="$2" sub="$3"
  for METHOD in default_b margin_probe semantic_probe; do
    mkdir -p "$OUT/$METHOD/$sub"
    "$PY" scripts/evaluate_toolsandbox_v6_diagnostic.py \
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

DEV_DATA="$V45/development_data"
score_split development "$DEV_DATA" dev_eval
evaluate_split development "$DEV_DATA" dev_eval

set +e
"$PY" scripts/check_toolsandbox_v6_gate.py \
  --default-eval "$OUT/default_b/dev_eval/eval_summary.json" \
  --control-eval "$OUT/margin_probe/dev_eval/eval_summary.json" \
  --semantic-eval "$OUT/semantic_probe/dev_eval/eval_summary.json" \
  --default-results "$OUT/default_b/dev_eval/task_results.jsonl" \
  --control-results "$OUT/margin_probe/dev_eval/task_results.jsonl" \
  --semantic-results "$OUT/semantic_probe/dev_eval/task_results.jsonl" \
  --control-run "$OUT/margin_probe/run_summary.json" \
  --semantic-run "$OUT/semantic_probe/run_summary.json" \
  --control-probe "$OUT/margin_probe/probe.pt" \
  --semantic-probe "$OUT/semantic_probe/probe.pt" \
  --scoring-summary "$OUT/dev_eval/scoring_summary.json" \
  --protocol-lock "$LOCK" \
  --output "$OUT/development_gate.json" | tee "$OUT/gate_console.log"
STATUS=${PIPESTATUS[0]}
set -e

# The inherited V4.5 confirmation split is known and remains post-hoc.
POSTHOC_DATA="$V45/confirmation_data"
score_split posthoc "$POSTHOC_DATA" posthoc_confirm
evaluate_split posthoc "$POSTHOC_DATA" posthoc_confirm
echo TOOLSANDBOX_V6_FINISHED
exit "$STATUS"
