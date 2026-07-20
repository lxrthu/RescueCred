#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${RESCUECREDIT_PYTHON:-$ROOT/.venv/bin/python}"
V7="$ROOT/outputs/toolsandbox_v7_active_shadow_seed42"
V9="$ROOT/outputs/toolsandbox_v9_two_step_seed42"
OUT="$ROOT/outputs/toolsandbox_receipt_horizon_stats_seed42"
LOCK="$OUT/protocol_lock.json"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PY"
test -f "$V7/model/oof_predictions.jsonl"
test -f "$V9/model/oof_predictions.jsonl"
[ ! -e "$OUT" ] || {
  echo "Refusing to reuse statistical-audit output root: $OUT" >&2
  exit 1
}
mkdir -p "$OUT"

"$PY" -m py_compile \
  rescuecredit/paired_task_statistics.py \
  scripts/freeze_toolsandbox_receipt_horizon_stats.py \
  scripts/analyze_toolsandbox_receipt_horizon_stats.py
"$PY" -m pytest -q tests/test_toolsandbox_receipt_horizon_stats.py

"$PY" scripts/freeze_toolsandbox_receipt_horizon_stats.py \
  --v7-root "$V7" \
  --v9-root "$V9" \
  --output "$LOCK" | tee "$OUT/freeze.log"

"$PY" scripts/analyze_toolsandbox_receipt_horizon_stats.py \
  --protocol-lock "$LOCK" \
  --v7-oof "$V7/model/oof_predictions.jsonl" \
  --v9-oof "$V9/model/oof_predictions.jsonl" \
  --output "$OUT/statistical_audit.json" | tee "$OUT/analysis.log"

echo TOOLSANDBOX_RECEIPT_HORIZON_STATS_FINISHED
