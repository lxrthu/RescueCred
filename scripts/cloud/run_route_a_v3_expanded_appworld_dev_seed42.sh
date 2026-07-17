#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
MODEL_PY=/data/hxy/projects/RescueCredit/.venv/bin/python
MODEL=/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct
EVENTS=outputs/route_a_appworld_dev_seed42_v2/events/dev_events.public.jsonl
TRAIN=outputs/route_a_v3_expanded_seed42
OUT=outputs/route_a_v3_expanded_appworld_dev_seed42
MASK_RESULTS="$OUT/mask/task_results.jsonl"
V3_RESULTS="$OUT/v3/task_results.jsonl"
LOCK="$OUT/protocol_lock.json"
CACHE="$OUT/continuation_cache.jsonl"

if [ -e "$OUT" ]; then
  echo "Refusing to reuse existing output root: $OUT" >&2
  echo "Move it aside before a fresh frozen run." >&2
  exit 1
fi
mkdir -p "$OUT/mask" "$OUT/v3" "$OUT/sanity"

test -x "$APP_PY"
test -x "$MODEL_PY"
test -f "$EVENTS"
test -d "$TRAIN/mask/adapter"
test -d "$TRAIN/v3/adapter"
test -f "$TRAIN/gate_erratum.json"
echo "fcfd0de213e044c7ae54bd5a4f340b50ade39a4d1f508627adceb5f02a683f3c  $EVENTS" | sha256sum -c -

"$MODEL_PY" - "$TRAIN/gate_erratum.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert gate["passed"] is True
assert gate["status"] == "audited_arithmetic_erratum"
PY

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
  awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' |
  sort -n | head -n 2 | awk '{print $2}'
)
if [ "${#GPUS[@]}" -lt 2 ]; then
  echo "Need two visible GPUs for the paired scorer run." >&2
  exit 2
fi
echo "MASK_GPU=${GPUS[0]} V3_GPU=${GPUS[1]}"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$MODEL_PY" scripts/select_route_a_frozen_events.py \
  --method mask \
  --event-file "$EVENTS" \
  --worker-python "$MODEL_PY" \
  --scorer-script scripts/route_a_adapter_scorer_worker.py \
  --model "$MODEL" \
  --adapter "$TRAIN/mask/adapter" \
  --output-dir "$OUT/mask" \
  > "$OUT/mask/console.log" 2>&1 &
MASK_PID=$!

CUDA_VISIBLE_DEVICES="${GPUS[1]}" "$MODEL_PY" scripts/select_route_a_frozen_events.py \
  --method v3 \
  --event-file "$EVENTS" \
  --worker-python "$MODEL_PY" \
  --scorer-script scripts/route_a_adapter_scorer_worker.py \
  --model "$MODEL" \
  --adapter "$TRAIN/v3/adapter" \
  --output-dir "$OUT/v3" \
  > "$OUT/v3/console.log" 2>&1 &
V3_PID=$!

wait "$MASK_PID"
wait "$V3_PID"

"$MODEL_PY" scripts/freeze_route_a_bounded_protocol.py \
  --event-file "$EVENTS" \
  --mask-results "$MASK_RESULTS" \
  --v2-results "$V3_RESULTS" \
  --output "$LOCK" \
  > "$OUT/protocol_lock_console.log" 2>&1

set -a
source .env
set +a
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "Set AZURE_OPENAI_API_KEY in /data/hxy/projects/RescueCredit/.env" >&2
  exit 2
fi
"$MODEL_PY" scripts/check_azure.py > "$OUT/azure_check.log" 2>&1

# Reusing the old exact-policy cache is valid: the frozen events, A/B actions,
# and reference-free continuation policy are unchanged. The evaluator verifies
# its policy fingerprint before accepting any row.
OLD_CACHE=outputs/route_a_appworld_dev_bounded_seed42/continuation_cache.jsonl
if [ -f "$OLD_CACHE" ]; then
  cp "$OLD_CACHE" "$CACHE"
fi

"$APP_PY" scripts/evaluate_route_a_bounded.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --event-file "$EVENTS" \
  --mask-results "$MASK_RESULTS" \
  --v2-results "$V3_RESULTS" \
  --protocol-lock "$LOCK" \
  --seed 42 \
  --horizons 4 8 \
  --worker-python "$MODEL_PY" \
  --worker-script scripts/appworld_azure_continuation_worker.py \
  --cache-file "$CACHE" \
  --limit 3 \
  --output-dir "$OUT/sanity" \
  > "$OUT/sanity/console.log" 2>&1

"$MODEL_PY" - "$OUT/sanity/bounded_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
assert s["events"] == 3
assert s["primary"]["valid_paired_events"] == 3
assert s["horizon_prefix_mismatches"] == 0
assert s["horizon_prefix_unverifiable"] == 0
assert s["cache_conflicts"] == 0
PY
echo ROUTE_A_V3_BOUNDED_SANITY_PASS

"$APP_PY" scripts/evaluate_route_a_bounded.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --event-file "$EVENTS" \
  --mask-results "$MASK_RESULTS" \
  --v2-results "$V3_RESULTS" \
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
  --output "$OUT/gate_raw_legacy_v2_slot.json" \
  > "$OUT/gate_raw_console.log" 2>&1
set -e

set +e
"$MODEL_PY" scripts/audit_route_a_v3_bounded.py \
  --raw-summary "$OUT/bounded_summary.json" \
  --raw-gate "$OUT/gate_raw_legacy_v2_slot.json" \
  --erratum-gate "$TRAIN/gate_erratum.json" \
  --mask-selection-summary "$OUT/mask/selection_summary.json" \
  --v3-selection-summary "$OUT/v3/selection_summary.json" \
  --mask-results "$MASK_RESULTS" \
  --v3-results "$V3_RESULTS" \
  --summary-output "$OUT/bounded_summary_v3.json" \
  --gate-output "$OUT/gate.json" \
  2>&1 | tee "$OUT/gate_console.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -eq 0 ]; then
  echo ROUTE_A_V3_APPWORLD_DEV_GATE_PASS
else
  echo ROUTE_A_V3_APPWORLD_DEV_GATE_FAIL
fi
echo ROUTE_A_V3_APPWORLD_DEV_FINISHED
exit "$STATUS"
