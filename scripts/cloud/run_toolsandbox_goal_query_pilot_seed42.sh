#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

APP_PY="${TOOLSANDBOX_PYTHON:-/data/hxy/venvs/rescuecredit-toolsandbox/bin/python}"
MODEL_PY="${RESCUECREDIT_PYTHON:-$ROOT/.venv/bin/python}"
PUBLIC_BANK="${GOAL_QUERY_PUBLIC_BANK:-$ROOT/outputs/toolsandbox_deltaguard_combined_bank/public_events.jsonl}"
PUBLIC_MANIFEST="${GOAL_QUERY_PUBLIC_MANIFEST:-$ROOT/outputs/toolsandbox_deltaguard_combined_bank/public_bank_manifest.json}"
OUT="${GOAL_QUERY_OUTPUT:-$ROOT/outputs/toolsandbox_goal_query_pilot_seed42}"
LABEL_EVENTS=(
  "$ROOT/outputs/toolsandbox_v44_candidate_diversity_seed42/full_offset85_h8/candidate_events.jsonl"
  "$ROOT/outputs/toolsandbox_v45_matched_anchor_seed42/development_candidates_offset125_h8/candidate_events.jsonl"
  "$ROOT/outputs/toolsandbox_v45_matched_anchor_seed42/confirmation_candidates_offset165_h8/candidate_events.jsonl"
)

test -x "$APP_PY"
test -x "$MODEL_PY"
test -f "$PUBLIC_BANK"
test -f "$PUBLIC_MANIFEST"
for path in "${LABEL_EVENTS[@]}"; do
  test -f "$path"
done
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite Goal Query output: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$MODEL_PY" -m py_compile \
  rescuecredit/goal_directed_query.py \
  scripts/freeze_toolsandbox_goal_query_pilot.py \
  scripts/collect_toolsandbox_goal_query_pilot.py \
  scripts/evaluate_toolsandbox_goal_query_pilot.py
"$MODEL_PY" -m pytest -q tests/test_goal_directed_query.py

"$MODEL_PY" scripts/freeze_toolsandbox_goal_query_pilot.py \
  --public-events "$PUBLIC_BANK" \
  --public-bank-manifest "$PUBLIC_MANIFEST" \
  --target-events 30 --minimum-events 12 --seed 42 \
  --output "$OUT/protocol_lock.json" \
  2>&1 | tee "$OUT/freeze.log"

"$APP_PY" scripts/collect_toolsandbox_goal_query_pilot.py \
  --protocol-lock "$OUT/protocol_lock.json" \
  --output-dir "$OUT/collection" \
  2>&1 | tee "$OUT/collection.log"

"$MODEL_PY" scripts/evaluate_toolsandbox_goal_query_pilot.py \
  --protocol-lock "$OUT/protocol_lock.json" \
  --collection-dir "$OUT/collection" \
  --label-events "${LABEL_EVENTS[@]}" \
  --output-dir "$OUT/evaluation" \
  2>&1 | tee "$OUT/evaluate.log"

cp "$OUT/evaluation/feasibility_gate.json" "$OUT/feasibility_gate.json"
echo GOAL_QUERY_PILOT_COMPLETE
