#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
export PYTHONPATH="/data/hxy/projects/RescueCredit${PYTHONPATH:+:$PYTHONPATH}"
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
MODEL_PY=/data/hxy/projects/RescueCredit/.venv/bin/python
MODEL=/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct
MASK_TRAIN=outputs/route_a_v3_expanded_seed42/mask
V31_TRAIN=outputs/route_a_v31_validity_seed42/v31
V31_PREF_GATE=outputs/route_a_v31_validity_seed42/gate.json
OUT=outputs/route_a_v31_both_valid_appworld_dev_seed42
EVENT_DIR="$OUT/events"
EVENTS="$EVENT_DIR/both_valid_dev_events.public.jsonl"
MANIFEST="$EVENT_DIR/manifest.json"
MASK_RESULTS="$OUT/mask/task_results.jsonl"
V31_RESULTS="$OUT/v31/task_results.jsonl"
LOCK="$OUT/protocol_lock.json"
CACHE="$OUT/continuation_cache.jsonl"

if [ -e "$OUT" ]; then
  echo "Refusing to reuse existing output root: $OUT" >&2
  exit 1
fi
mkdir -p "$EVENT_DIR" "$OUT/mask" "$OUT/v31" "$OUT/sanity"

test -x "$APP_PY"
test -x "$MODEL_PY"
test -d "$MASK_TRAIN/adapter"
test -d "$V31_TRAIN/adapter"
test -f "$MASK_TRAIN/run_summary.json"
test -f "$V31_TRAIN/run_summary.json"
test -f "$V31_PREF_GATE"

"$MODEL_PY" -m py_compile \
  scripts/build_route_a_both_valid_dev_events.py \
  scripts/select_route_a_frozen_events.py \
  scripts/evaluate_route_a_bounded.py \
  scripts/freeze_route_a_v31_both_valid_protocol.py \
  scripts/audit_route_a_v31_both_valid_bounded.py
"$MODEL_PY" -m pytest -q \
  tests/test_route_a_v31_both_valid.py \
  tests/test_route_a_bounded.py \
  tests/test_route_a_bounded_contract.py

"$MODEL_PY" - "$V31_PREF_GATE" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert gate["passed"] is True
assert gate["stage"] == "route_a_seed42_v31_validity_first_gate"
PY

set -a
source .env
set +a
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "Set AZURE_OPENAI_API_KEY in /data/hxy/projects/RescueCredit/.env" >&2
  exit 2
fi
"$MODEL_PY" scripts/check_azure.py > "$OUT/azure_check.log" 2>&1

"$APP_PY" scripts/build_route_a_both_valid_dev_events.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --subset dev --offset 0 --limit 57 --seed 42 \
  --max-events 60 --max-events-per-task 2 --max-alternatives-per-field 2 \
  --selector-python "$MODEL_PY" \
  --selector-script scripts/appworld_azure_candidate_selector_worker.py \
  --selector-model "$MODEL" --selector-device cpu \
  --output-dir "$EVENT_DIR" \
  2>&1 | tee "$OUT/build_events.log"

"$MODEL_PY" - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
assert m["events"] >= 30, m
assert m["tasks_with_events"] >= 15, m
assert m["max_task_event_share"] <= 0.10, m
assert m["both_actions_schema_complete"] is True, m
assert m["reference_execution_failure_rate"] <= 0.05, m
PY
echo ROUTE_A_V31_BOTH_VALID_EVENT_GATE_PASS

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' |
    sort -n | head -n 2 | awk '{print $2}'
)
if [ "${#GPUS[@]}" -lt 2 ]; then
  echo "Need two GPUs for paired adapter scoring." >&2
  exit 2
fi
echo "MASK_GPU=${GPUS[0]} V31_GPU=${GPUS[1]}"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$MODEL_PY" scripts/select_route_a_frozen_events.py \
  --method mask --event-file "$EVENTS" --worker-python "$MODEL_PY" \
  --scorer-script scripts/route_a_adapter_scorer_worker.py --model "$MODEL" \
  --adapter "$MASK_TRAIN/adapter" --output-dir "$OUT/mask" \
  > "$OUT/mask/console.log" 2>&1 &
MASK_PID=$!

CUDA_VISIBLE_DEVICES="${GPUS[1]}" "$MODEL_PY" scripts/select_route_a_frozen_events.py \
  --method v31 --event-file "$EVENTS" --worker-python "$MODEL_PY" \
  --scorer-script scripts/route_a_adapter_scorer_worker.py --model "$MODEL" \
  --adapter "$V31_TRAIN/adapter" --output-dir "$OUT/v31" \
  > "$OUT/v31/console.log" 2>&1 &
V31_PID=$!

cleanup_scorers() {
  kill "$MASK_PID" "$V31_PID" 2>/dev/null || true
}
trap cleanup_scorers EXIT INT TERM
set +e
wait "$MASK_PID"; MASK_STATUS=$?
wait "$V31_PID"; V31_STATUS=$?
set -e
trap - EXIT INT TERM
if [ "$MASK_STATUS" -ne 0 ] || [ "$V31_STATUS" -ne 0 ]; then
  echo "Adapter scoring failed: mask=$MASK_STATUS v31=$V31_STATUS" >&2
  exit 1
fi

"$MODEL_PY" scripts/freeze_route_a_v31_both_valid_protocol.py \
  --event-file "$EVENTS" --event-manifest "$MANIFEST" \
  --mask-results "$MASK_RESULTS" --v31-results "$V31_RESULTS" \
  --mask-selection "$OUT/mask/selection_summary.json" \
  --v31-selection "$OUT/v31/selection_summary.json" \
  --mask-run "$MASK_TRAIN/run_summary.json" \
  --v31-run "$V31_TRAIN/run_summary.json" \
  --v31-preference-gate "$V31_PREF_GATE" --output "$LOCK" \
  2>&1 | tee "$OUT/protocol_console.log"

"$APP_PY" scripts/evaluate_route_a_bounded.py \
  --development-protocol \
  --appworld-root /data/hxy/projects/RescueCredit \
  --event-file "$EVENTS" --mask-results "$MASK_RESULTS" \
  --v2-results "$V31_RESULTS" --protocol-lock "$LOCK" \
  --seed 42 --horizons 4 8 --worker-python "$MODEL_PY" \
  --worker-script scripts/appworld_azure_continuation_worker.py \
  --cache-file "$CACHE" --limit 3 --output-dir "$OUT/sanity" \
  > "$OUT/sanity/console.log" 2>&1

"$MODEL_PY" - "$OUT/sanity/bounded_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
assert s["events"] == 3
assert s["primary"]["valid_paired_events"] == 3
assert s["horizon_prefix_mismatches"] == 0
assert s["horizon_prefix_unverifiable"] == 0
assert s["cache_conflicts"] == 0
assert s["development_protocol"] is True
PY
echo ROUTE_A_V31_BOTH_VALID_SANITY_PASS

"$APP_PY" scripts/evaluate_route_a_bounded.py \
  --development-protocol \
  --appworld-root /data/hxy/projects/RescueCredit \
  --event-file "$EVENTS" --mask-results "$MASK_RESULTS" \
  --v2-results "$V31_RESULTS" --protocol-lock "$LOCK" \
  --seed 42 --horizons 4 8 --worker-python "$MODEL_PY" \
  --worker-script scripts/appworld_azure_continuation_worker.py \
  --cache-file "$CACHE" --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

set +e
"$MODEL_PY" scripts/audit_route_a_v31_both_valid_bounded.py \
  --raw-summary "$OUT/bounded_summary.json" \
  --bounded-results "$OUT/bounded_results.jsonl" --protocol-lock "$LOCK" \
  --event-file "$EVENTS" --event-manifest "$MANIFEST" \
  --mask-results "$MASK_RESULTS" --v31-results "$V31_RESULTS" \
  --mask-selection "$OUT/mask/selection_summary.json" \
  --v31-selection "$OUT/v31/selection_summary.json" \
  --mask-run "$MASK_TRAIN/run_summary.json" \
  --v31-run "$V31_TRAIN/run_summary.json" \
  --preference-gate "$V31_PREF_GATE" \
  --summary-output "$OUT/bounded_summary_v31.json" \
  --gate-output "$OUT/gate.json" \
  2>&1 | tee "$OUT/gate_console.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -eq 0 ]; then
  echo ROUTE_A_V31_BOTH_VALID_DEV_GATE_PASS
else
  echo ROUTE_A_V31_BOTH_VALID_DEV_GATE_FAIL
fi
echo ROUTE_A_V31_BOTH_VALID_DEV_FINISHED
exit "$STATUS"
