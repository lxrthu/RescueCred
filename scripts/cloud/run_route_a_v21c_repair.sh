#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
SOURCE=outputs/route_a_v21b_expanded_data_seed42
ROOT=outputs/route_a_v21c_balanced_data_seed42

test -s "$SOURCE/shadow/shadow_credit.train.jsonl"
test -s "$SOURCE/dense/dense_shadow_credit.train.jsonl"
test -s "$SOURCE/data_gate.json"
test ! -e "$ROOT"

"$APP_PY" scripts/repair_route_a_v21_balanced_data.py \
  --source-root "$SOURCE" \
  --output-root "$ROOT" \
  --max-events-per-task 10 \
  2>&1 | tee outputs/route_a_v21c_repair.log

"$APP_PY" scripts/prepare_route_a_v21_data.py \
  --bank-dir "$ROOT/bank" \
  --dense-dir "$ROOT/dense" \
  --seed 42 \
  --validation-task-fraction 0.2 \
  --min-abs-delta 0.05 \
  --output-dir "$ROOT/data" \
  2>&1 | tee "$ROOT/data_console.log"

"$APP_PY" scripts/check_route_a_v21_data.py \
  --bank-dir "$ROOT/bank" \
  --shadow-dir "$ROOT/shadow" \
  --data-dir "$ROOT/data" \
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

echo ROUTE_A_V21C_REPAIR_PASS
