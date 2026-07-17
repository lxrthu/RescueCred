#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit

APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
MODEL_PY=/data/hxy/projects/RescueCredit/.venv/bin/python
SOURCE=outputs/route_a_appworld_dev_seed42_v2
EVENTS="$SOURCE/events/dev_events.public.jsonl"
MASK_RESULTS="$SOURCE/mask/task_results.jsonl"
V2_RESULTS="$SOURCE/v2/task_results.jsonl"
OUT=outputs/route_a_appworld_dev_immediate_seed42
mkdir -p "$OUT/sanity"

test -x "$APP_PY"
test -f "$EVENTS"
test -f "$MASK_RESULTS"
test -f "$V2_RESULTS"

head -n 3 "$EVENTS" > "$OUT/sanity/events.jsonl"

"$APP_PY" scripts/evaluate_route_a_immediate.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --event-file "$OUT/sanity/events.jsonl" \
  --mask-results "$MASK_RESULTS" \
  --v2-results "$V2_RESULTS" \
  --seed 42 \
  --output-dir "$OUT/sanity" \
  > "$OUT/sanity/console.log" 2>&1

test -f "$OUT/sanity/immediate_summary.json"
echo ROUTE_A_IMMEDIATE_SANITY_PASS

"$APP_PY" scripts/evaluate_route_a_immediate.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --event-file "$EVENTS" \
  --mask-results "$MASK_RESULTS" \
  --v2-results "$V2_RESULTS" \
  --seed 42 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

set +e
"$MODEL_PY" scripts/check_route_a_immediate_gate.py \
  --summary "$OUT/immediate_summary.json" \
  --min-valid 40 \
  --min-nonzero 5 \
  --output "$OUT/gate.json" \
  2>&1 | tee "$OUT/gate_console.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$GATE_STATUS" -eq 0 ]; then
  echo ROUTE_A_IMMEDIATE_GATE_PASS
else
  echo ROUTE_A_IMMEDIATE_GATE_FAIL
fi
echo ROUTE_A_IMMEDIATE_FINISHED
