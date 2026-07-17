#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit
source data_disk_env.sh

APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
MODEL_PY=/data/hxy/projects/RescueCredit/.venv/bin/python
MODEL=/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct
ROOT=outputs/route_a_v21b_expanded_data_seed42
BANK="$ROOT/bank"
SHADOW="$ROOT/shadow"
DENSE="$ROOT/dense"
DATA="$ROOT/data"
SEED=421

if [ -e "$ROOT" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "Refusing to reuse existing V2.1 output root: $ROOT" >&2
  echo "For an interrupted identical run use: RESUME=1 bash $0" >&2
  exit 2
fi
mkdir -p "$BANK" "$SHADOW" "$DENSE" "$DATA"

test -x "$APP_PY"
test -x "$MODEL_PY"
test -d "$MODEL"
test -f .env
set -a
source .env
set +a
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "Set AZURE_OPENAI_API_KEY in /data/hxy/projects/RescueCredit/.env" >&2
  exit 2
fi
"$MODEL_PY" scripts/check_azure.py > "$ROOT/azure_check.log" 2>&1

if [ ! -f "$BANK/manifest.json" ]; then
  "$APP_PY" scripts/build_appworld_route_a_bank_v21.py \
    --appworld-root /data/hxy/projects/RescueCredit \
    --subset train \
    --offset 0 \
    --limit 90 \
    --seed 42 \
    --max-missing-per-variant 3 \
    --max-variants-per-call 20 \
    --max-cases-per-task 80 \
    --max-wrong-value-variants-per-field 2 \
    --selector-python "$MODEL_PY" \
    --selector-script scripts/appworld_azure_candidate_selector_worker.py \
    --selector-model "$MODEL" \
    --selector-device cpu \
    --output-dir "$BANK" \
    2>&1 | tee "$BANK/console.log"
fi

"$APP_PY" scripts/check_route_a_bank.py \
  --bank-dir "$BANK" \
  --min-events 300 \
  2>&1 | tee "$BANK/gate_console.log"

EVENTS=$("$APP_PY" -c \
  'import json; print(json.load(open("outputs/route_a_v21b_expanded_data_seed42/bank/manifest.json"))["events"])')
echo "V21_BANK_EVENTS=$EVENTS"

"$APP_PY" scripts/attach_appworld_shadow_credit_v21.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --bank-dir "$BANK" \
  --offset 0 \
  --limit "$EVENTS" \
  --seed "$SEED" \
  --max-shadow-steps 12 \
  --worker-python "$MODEL_PY" \
  --worker-script scripts/appworld_azure_continuation_worker.py \
  --output-dir "$SHADOW" \
  2>&1 | tee "$SHADOW/console.log"

"$APP_PY" scripts/recompute_route_a_dense_credit.py \
  --bank-dir "$BANK" \
  --binary-credit-dir "$SHADOW" \
  --experiments-root experiments/outputs \
  --bank-offset 0 \
  --limit "$EVENTS" \
  --seed "$SEED" \
  --output-dir "$DENSE" \
  2>&1 | tee "$DENSE/console.log"

"$APP_PY" scripts/prepare_route_a_v21_data.py \
  --bank-dir "$BANK" \
  --dense-dir "$DENSE" \
  --seed 42 \
  --validation-task-fraction 0.2 \
  --min-abs-delta 0.05 \
  --output-dir "$DATA" \
  2>&1 | tee "$DATA/console.log"

set +e
"$APP_PY" scripts/check_route_a_v21_data.py \
  --bank-dir "$BANK" \
  --shadow-dir "$SHADOW" \
  --data-dir "$DATA" \
  --output "$ROOT/data_gate.json" \
  --min-bank-events 300 \
  --min-nonzero-events 100 \
  --min-nonzero-tasks 30 \
  --min-rescue 25 \
  --min-reverse 25 \
  --min-validation-nonzero 15 \
  --min-replay-valid-rate 0.90 \
  --max-task-nonzero-share 0.10 \
  2>&1 | tee "$ROOT/data_gate_console.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$GATE_STATUS" -eq 0 ]; then
  echo ROUTE_A_V21_DATA_GATE_PASS
else
  echo ROUTE_A_V21_DATA_GATE_FAIL
fi
echo ROUTE_A_V21_DATA_EXPANSION_FINISHED
exit "$GATE_STATUS"
