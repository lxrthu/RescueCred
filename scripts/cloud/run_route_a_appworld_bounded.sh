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
OUT=outputs/route_a_appworld_dev_bounded_seed42
CACHE="$OUT/continuation_cache.jsonl"
LOCK="$OUT/protocol_lock.json"
mkdir -p "$OUT/sanity"

test -x "$APP_PY"
test -x "$MODEL_PY"
test -f "$EVENTS"
test -f "$MASK_RESULTS"
test -f "$V2_RESULTS"

set -a
source .env
set +a
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "Set AZURE_OPENAI_API_KEY in /data/hxy/projects/RescueCredit/.env"
  exit 2
fi
"$MODEL_PY" scripts/check_azure.py > "$OUT/azure_check.log" 2>&1

"$MODEL_PY" scripts/freeze_route_a_bounded_protocol.py \
  --event-file "$EVENTS" \
  --mask-results "$MASK_RESULTS" \
  --v2-results "$V2_RESULTS" \
  --output "$LOCK" \
  > "$OUT/protocol_lock_console.log" 2>&1

"$APP_PY" scripts/evaluate_route_a_bounded.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --event-file "$EVENTS" \
  --mask-results "$MASK_RESULTS" \
  --v2-results "$V2_RESULTS" \
  --protocol-lock "$LOCK" \
  --seed 42 \
  --horizons 4 8 \
  --worker-python "$MODEL_PY" \
  --worker-script scripts/appworld_azure_continuation_worker.py \
  --cache-file "$CACHE" \
  --limit 3 \
  --output-dir "$OUT/sanity" \
  > "$OUT/sanity/console.log" 2>&1

test -f "$OUT/sanity/bounded_summary.json"
"$MODEL_PY" - "$OUT/sanity/bounded_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
primary = summary["primary"]
assert summary["events"] == 3
assert primary["valid_paired_events"] == 3
assert summary["horizon_prefix_mismatches"] == 0
assert summary["horizon_prefix_unverifiable"] == 0
assert summary["cache_conflicts"] == 0
PY
echo ROUTE_A_BOUNDED_SANITY_PASS

"$APP_PY" scripts/evaluate_route_a_bounded.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --event-file "$EVENTS" \
  --mask-results "$MASK_RESULTS" \
  --v2-results "$V2_RESULTS" \
  --protocol-lock "$LOCK" \
  --seed 42 \
  --horizons 4 8 \
  --worker-python "$MODEL_PY" \
  --worker-script scripts/appworld_azure_continuation_worker.py \
  --cache-file "$CACHE" \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

set +e
"$MODEL_PY" scripts/check_route_a_bounded_gate.py \
  --summary "$OUT/bounded_summary.json" \
  --output "$OUT/gate.json" \
  2>&1 | tee "$OUT/gate_console.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$GATE_STATUS" -eq 0 ]; then
  echo ROUTE_A_BOUNDED_GATE_PASS
else
  echo ROUTE_A_BOUNDED_GATE_FAIL
fi
echo ROUTE_A_BOUNDED_FINISHED
exit "$GATE_STATUS"
