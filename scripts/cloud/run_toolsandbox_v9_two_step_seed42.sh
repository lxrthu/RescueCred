#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${RESCUECREDIT_PYTHON:-$ROOT/.venv/bin/python}"
V44="$ROOT/outputs/toolsandbox_v44_candidate_diversity_seed42"
V5="$ROOT/outputs/toolsandbox_v5_causal_router_seed42"
V7="$ROOT/outputs/toolsandbox_v7_active_shadow_seed42"
OUT="$ROOT/outputs/toolsandbox_v9_two_step_seed42"
LOCK="$OUT/protocol_lock.json"
FEATURES="$OUT/features"
MODEL="$OUT/model"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PY"
test -f "$V44/full_offset85_h8/candidate_events.jsonl"
test -f "$V44/data/train.jsonl"
test -f "$V5/features/train_features.pt"
test -f "$V7/feasibility_gate.json"
[ ! -e "$OUT" ] || {
  echo "Refusing to reuse V9 output root: $OUT" >&2
  exit 1
}
mkdir -p "$FEATURES" "$MODEL"

"$PY" -m py_compile \
  rescuecredit/toolsandbox_active_shadow_v9.py \
  scripts/freeze_toolsandbox_v9_protocol.py \
  scripts/build_toolsandbox_v9_features.py \
  scripts/train_toolsandbox_v9_active_shadow.py \
  scripts/check_toolsandbox_v9_gate.py
"$PY" -m pytest -q tests/test_toolsandbox_v9.py tests/test_toolsandbox_v7.py

"$PY" scripts/freeze_toolsandbox_v9_protocol.py \
  --v44-root "$V44" \
  --v5-root "$V5" \
  --v7-root "$V7" \
  --output "$LOCK" | tee "$OUT/freeze.log"

"$PY" scripts/build_toolsandbox_v9_features.py \
  --raw-events "$V44/full_offset85_h8/candidate_events.jsonl" \
  --train-file "$V44/data/train.jsonl" \
  --v5-feature-cache "$V5/features/train_features.pt" \
  --protocol-lock "$LOCK" \
  --output-dir "$FEATURES" | tee "$OUT/feature_console.log"

"$PY" scripts/train_toolsandbox_v9_active_shadow.py \
  --feature-cache "$FEATURES/active_shadow_features.pt" \
  --feature-manifest "$FEATURES/feature_manifest.json" \
  --protocol-lock "$LOCK" \
  --output-dir "$MODEL" | tee "$OUT/train_console.log"

set +e
"$PY" scripts/check_toolsandbox_v9_gate.py \
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
  echo TOOLSANDBOX_V9_FEASIBILITY_PASS
else
  echo TOOLSANDBOX_V9_FEASIBILITY_FAIL
fi
echo TOOLSANDBOX_V9_FINISHED
exit "$STATUS"
