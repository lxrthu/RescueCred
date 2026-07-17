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
OUT=outputs/route_a_appworld_bounded_confirm_43_44_45

if [ -e "$OUT" ]; then
  echo "Refusing to reuse existing confirmatory output root: $OUT" >&2
  echo "Move it aside explicitly before a fresh frozen run." >&2
  exit 2
fi
mkdir -p "$OUT"

test -x "$APP_PY"
test -x "$MODEL_PY"
test -f "$EVENTS"
test -f "$MASK_RESULTS"
test -f "$V2_RESULTS"

set -a
source .env
set +a
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "Set AZURE_OPENAI_API_KEY in /data/hxy/projects/RescueCredit/.env" >&2
  exit 2
fi
"$MODEL_PY" scripts/check_azure.py > "$OUT/azure_check.log" 2>&1

# Freeze every seed and the aggregate gate before any confirmatory outcome exists.
for SEED in 43 44 45; do
  SEED_OUT="$OUT/seed$SEED"
  mkdir -p "$SEED_OUT"
  "$APP_PY" scripts/freeze_route_a_bounded_confirm_protocol.py \
    --appworld-root /data/hxy/projects/RescueCredit \
    --event-file "$EVENTS" \
    --mask-results "$MASK_RESULTS" \
    --v2-results "$V2_RESULTS" \
    --seed "$SEED" \
    --worker-script scripts/appworld_azure_continuation_worker.py \
    --output "$SEED_OUT/protocol_lock.json" \
    > "$SEED_OUT/protocol_lock_console.log" 2>&1
done

# Cheap contract sanity check. Its outcomes are excluded from the final aggregate.
"$APP_PY" scripts/evaluate_route_a_bounded.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --event-file "$EVENTS" \
  --mask-results "$MASK_RESULTS" \
  --v2-results "$V2_RESULTS" \
  --protocol-lock "$OUT/seed43/protocol_lock.json" \
  --seed 43 \
  --horizons 4 8 \
  --confirmatory \
  --worker-python "$MODEL_PY" \
  --worker-script scripts/appworld_azure_continuation_worker.py \
  --cache-file "$OUT/seed43/continuation_cache.jsonl" \
  --limit 3 \
  --output-dir "$OUT/sanity_seed43" \
  > "$OUT/sanity_seed43.log" 2>&1

"$MODEL_PY" - "$OUT/sanity_seed43/bounded_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["events"] == 3
assert summary["primary"]["valid_paired_events"] == 3
assert summary["horizon_prefix_mismatches"] == 0
assert summary["horizon_prefix_unverifiable"] == 0
assert summary["cache_conflicts"] == 0
assert summary["confirmatory"] is True
PY
echo ROUTE_A_CONFIRM_SANITY_PASS

run_seed() {
  local SEED="$1"
  local SEED_OUT="$OUT/seed$SEED"
  "$APP_PY" scripts/evaluate_route_a_bounded.py \
    --appworld-root /data/hxy/projects/RescueCredit \
    --event-file "$EVENTS" \
    --mask-results "$MASK_RESULTS" \
    --v2-results "$V2_RESULTS" \
    --protocol-lock "$SEED_OUT/protocol_lock.json" \
    --seed "$SEED" \
    --horizons 4 8 \
    --confirmatory \
    --worker-python "$MODEL_PY" \
    --worker-script scripts/appworld_azure_continuation_worker.py \
    --cache-file "$SEED_OUT/continuation_cache.jsonl" \
    --output-dir "$SEED_OUT" \
    > "$SEED_OUT/console.log" 2>&1
}

# Three independent deterministic seeds. They use Azure, not local GPUs.
declare -A PIDS=()
for SEED in 43 44 45; do
  run_seed "$SEED" &
  PIDS[$SEED]=$!
  echo "STARTED seed=$SEED pid=${PIDS[$SEED]}"
done

FAILED=0
for SEED in 43 44 45; do
  if wait "${PIDS[$SEED]}"; then
    echo "FINISHED seed=$SEED"
  else
    echo "FAILED seed=$SEED" >&2
    FAILED=1
  fi
done
if [ "$FAILED" -ne 0 ]; then
  echo "At least one seed failed; aggregate analysis was not run." >&2
  exit 1
fi

set +e
"$MODEL_PY" scripts/analyze_route_a_bounded_confirm.py \
  --root "$OUT" \
  --output "$OUT/combined_gate.json" \
  2>&1 | tee "$OUT/combined_gate_console.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$GATE_STATUS" -eq 0 ]; then
  echo ROUTE_A_BOUNDED_CONFIRM_GATE_PASS
else
  echo ROUTE_A_BOUNDED_CONFIRM_GATE_FAIL
fi
echo ROUTE_A_BOUNDED_CONFIRM_FINISHED
exit "$GATE_STATUS"
