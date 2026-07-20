#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PY="${TOOLSANDBOX_PYTHON:-/data/hxy/venvs/rescuecredit-toolsandbox/bin/python}"
MODEL_PY="${RESCUECREDIT_PYTHON:-$ROOT/.venv/bin/python}"
WORKER="$ROOT/scripts/toolsandbox_azure_worker.py"
V44="$ROOT/outputs/toolsandbox_v44_candidate_diversity_seed42"
V5="$ROOT/outputs/toolsandbox_v5_causal_router_seed42"
V7="$ROOT/outputs/toolsandbox_v7_active_shadow_seed42"
OUT="$ROOT/outputs/toolsandbox_v8_visible_state_seed42"
LOCK="$OUT/protocol_lock.json"
COLLECTION="$OUT/collection"
FEATURES="$OUT/features"
MODEL="$OUT/model"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -x "$APP_PY"
test -x "$MODEL_PY"
test -f "$V44/full_protocol_lock.json"
test -f "$V44/full_offset85_h8/candidate_events.jsonl"
test -f "$V44/data/train.jsonl"
test -f "$V5/features/train_features.pt"
test -f "$V7/feasibility_gate.json"
[ ! -e "$OUT" ] || {
  echo "Refusing to reuse V8 output root: $OUT" >&2
  exit 1
}
mkdir -p "$COLLECTION" "$FEATURES" "$MODEL"

"$MODEL_PY" -m py_compile \
  rescuecredit/toolsandbox_active_shadow_v8.py \
  scripts/freeze_toolsandbox_v8_protocol.py \
  scripts/collect_toolsandbox_v8_visible_state.py \
  scripts/build_toolsandbox_v8_features.py \
  scripts/train_toolsandbox_v8_active_shadow.py \
  scripts/check_toolsandbox_v8_gate.py
"$MODEL_PY" -m pytest -q tests/test_toolsandbox_v8.py tests/test_toolsandbox_v7.py

set -a
source .env
set +a
test "${TOOLSANDBOX_LLM_PROVIDER:-}" = "deepseek"
test -n "${DEEPSEEK_API_KEY:-}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://zhi-api.com/v1}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
test "$DEEPSEEK_THINKING" = "disabled"

"$APP_PY" scripts/freeze_toolsandbox_v8_protocol.py \
  --v44-root "$V44" \
  --v5-root "$V5" \
  --v7-root "$V7" \
  --worker-script "$WORKER" \
  --output "$LOCK" | tee "$OUT/freeze.log"

"$APP_PY" scripts/collect_toolsandbox_v8_visible_state.py \
  --protocol-lock "$LOCK" \
  --raw-events "$V44/full_offset85_h8/candidate_events.jsonl" \
  --train-file "$V44/data/train.jsonl" \
  --worker-python "$MODEL_PY" \
  --worker-script "$WORKER" \
  --worker-model "$DEEPSEEK_MODEL" \
  --output-dir "$COLLECTION" | tee "$OUT/collection_console.log"

"$MODEL_PY" scripts/build_toolsandbox_v8_features.py \
  --state-events "$COLLECTION/visible_state_events.jsonl" \
  --collection-summary "$COLLECTION/collection_summary.json" \
  --train-file "$V44/data/train.jsonl" \
  --v5-feature-cache "$V5/features/train_features.pt" \
  --protocol-lock "$LOCK" \
  --output-dir "$FEATURES" | tee "$OUT/feature_console.log"

"$MODEL_PY" scripts/train_toolsandbox_v8_active_shadow.py \
  --feature-cache "$FEATURES/active_shadow_features.pt" \
  --feature-manifest "$FEATURES/feature_manifest.json" \
  --protocol-lock "$LOCK" \
  --output-dir "$MODEL" | tee "$OUT/train_console.log"

set +e
"$MODEL_PY" scripts/check_toolsandbox_v8_gate.py \
  --run-summary "$MODEL/run_summary.json" \
  --oof-predictions "$MODEL/oof_predictions.jsonl" \
  --checkpoint "$MODEL/active_shadow.pt" \
  --feature-cache "$FEATURES/active_shadow_features.pt" \
  --feature-manifest "$FEATURES/feature_manifest.json" \
  --protocol-lock "$LOCK" \
  --output "$OUT/feasibility_gate.json" | tee "$OUT/gate_console.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -eq 0 ]; then
  echo TOOLSANDBOX_V8_FEASIBILITY_PASS
else
  echo TOOLSANDBOX_V8_FEASIBILITY_FAIL
fi
echo TOOLSANDBOX_V8_FINISHED
exit "$STATUS"
