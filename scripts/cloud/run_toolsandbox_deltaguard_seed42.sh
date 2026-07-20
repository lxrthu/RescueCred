#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ROLE="${1:-feasibility}"
OUT="${2:-$ROOT/outputs/toolsandbox_deltaguard_${ROLE}_seed42}"
V7_CHECKPOINT="${3:-$ROOT/outputs/toolsandbox_v7_active_shadow_seed42/model/active_shadow.pt}"
PUBLIC_EVENTS="${4:-$ROOT/outputs/toolsandbox_deltaguard_source_bank/public_events.jsonl}"
PUBLIC_BANK_MANIFEST="${PUBLIC_BANK_MANIFEST:-$(dirname "$PUBLIC_EVENTS")/public_bank_manifest.json}"
V7_TRAIN_FILE="${V7_TRAIN_FILE:-$ROOT/outputs/toolsandbox_v44_candidate_seed42/data/train.jsonl}"
V7_RUN_SUMMARY="${V7_RUN_SUMMARY:-$ROOT/outputs/toolsandbox_v7_active_shadow_seed42/model/run_summary.json}"
V7_PROTOCOL_LOCK="${V7_PROTOCOL_LOCK:-$ROOT/outputs/toolsandbox_v7_active_shadow_seed42/protocol_lock.json}"
V7_OOF="${V7_OOF:-$ROOT/outputs/toolsandbox_v7_active_shadow_seed42/model/oof_predictions.jsonl}"
shift $(( $# >= 4 ? 4 : $# ))
LABEL_EVENTS=("$@")

if [[ ${#LABEL_EVENTS[@]} -eq 0 ]]; then
  LABEL_EVENTS=(
    "$ROOT/outputs/toolsandbox_v44_candidate_seed42/full_offset85_h8/candidate_events.jsonl"
  )
fi

APP_PY="${TOOLSANDBOX_PYTHON:-/data/hxy/venvs/rescuecredit-toolsandbox/bin/python}"
MODEL_PY="${RESCUECREDIT_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$APP_PY" ]]; then
  echo "ToolSandbox Python not found: $APP_PY" >&2
  exit 2
fi
if [[ ! -x "$MODEL_PY" ]]; then
  echo "RescueCredit model Python not found: $MODEL_PY" >&2
  exit 2
fi
if ! "$MODEL_PY" -c 'import torch' >/dev/null 2>&1; then
  echo "RescueCredit model Python cannot import torch: $MODEL_PY" >&2
  exit 2
fi
if [[ ! -f "$V7_CHECKPOINT" ]]; then
  echo "V7 checkpoint not found: $V7_CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$PUBLIC_EVENTS" || ! -f "$PUBLIC_BANK_MANIFEST" ]]; then
  echo "Pre-sealed public bank missing: $PUBLIC_EVENTS / $PUBLIC_BANK_MANIFEST" >&2
  echo "Prepare it in a separate prior step; this runner never reads labels before collection." >&2
  exit 2
fi
for path in "${LABEL_EVENTS[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "raw event bank not found: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUT" ]]; then
  echo "Output already exists; choose a new OUT path: $OUT" >&2
  exit 2
fi

mkdir -p "$OUT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$MODEL_PY" -m compileall -q \
  "$ROOT/rescuecredit" \
  "$ROOT/scripts/freeze_toolsandbox_deltaguard_protocol.py" \
  "$ROOT/scripts/collect_toolsandbox_deltaguard.py" \
  "$ROOT/scripts/evaluate_toolsandbox_deltaguard.py" \
  "$ROOT/scripts/check_toolsandbox_deltaguard_gate.py"

V7_TRAIN_ARGS=()
if [[ -f "$V7_TRAIN_FILE" ]]; then
  V7_TRAIN_ARGS+=(--v7-train-file "$V7_TRAIN_FILE")
fi
if [[ -f "$V7_RUN_SUMMARY" ]]; then
  V7_TRAIN_ARGS+=(--v7-run-summary "$V7_RUN_SUMMARY")
fi
if [[ -f "$V7_PROTOCOL_LOCK" ]]; then
  V7_TRAIN_ARGS+=(--v7-protocol-lock "$V7_PROTOCOL_LOCK")
fi
if [[ -f "$V7_OOF" ]]; then
  V7_TRAIN_ARGS+=(--v7-oof "$V7_OOF")
fi

"$MODEL_PY" "$ROOT/scripts/freeze_toolsandbox_deltaguard_protocol.py" \
  --role "$ROLE" \
  --public-events "$PUBLIC_EVENTS" \
  --public-bank-manifest "$PUBLIC_BANK_MANIFEST" \
  --v7-checkpoint "$V7_CHECKPOINT" \
  "${V7_TRAIN_ARGS[@]}" \
  --output "$OUT/protocol_lock.json" \
  2>&1 | tee "$OUT/freeze.log"

"$APP_PY" "$ROOT/scripts/collect_toolsandbox_deltaguard.py" \
  --protocol-lock "$OUT/protocol_lock.json" \
  --output-dir "$OUT/collection" \
  2>&1 | tee "$OUT/collection.log"

"$MODEL_PY" "$ROOT/scripts/evaluate_toolsandbox_deltaguard.py" \
  --protocol-lock "$OUT/protocol_lock.json" \
  --collection-dir "$OUT/collection" \
  --label-events "${LABEL_EVENTS[@]}" \
  --output-dir "$OUT/evaluation" \
  2>&1 | tee "$OUT/evaluation.log"

"$MODEL_PY" "$ROOT/scripts/check_toolsandbox_deltaguard_gate.py" \
  --protocol-lock "$OUT/protocol_lock.json" \
  --collection-dir "$OUT/collection" \
  --evaluation-dir "$OUT/evaluation" \
  --label-events "${LABEL_EVENTS[@]}" \
  --output "$OUT/feasibility_gate.json" \
  2>&1 | tee "$OUT/gate.log"

echo "DeltaGuard complete: $OUT/feasibility_gate.json"
