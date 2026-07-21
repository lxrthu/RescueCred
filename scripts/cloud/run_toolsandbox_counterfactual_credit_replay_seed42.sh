#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${RESCUECREDIT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
RAPG_ROOT="${CREDIT_REPLAY_RAPG_ROOT:-$REPO_ROOT/outputs/toolsandbox_rapg_pilot0_retry1_seed42}"
OUT="${CREDIT_REPLAY_OUTPUT:-$REPO_ROOT/outputs/toolsandbox_counterfactual_credit_replay_seed42}"
BANK="$RAPG_ROOT/bank/rapg_bank.pt"
BANK_MANIFEST="$RAPG_ROOT/bank/bank_manifest.json"
SOURCE_MANIFEST="$RAPG_ROOT/source/source_manifest.json"
SHADOW_A="$RAPG_ROOT/source/shadow_a_returns.private.jsonl"
PREDICTIONS="$RAPG_ROOT/evaluation/crossfit_predictions.pt"
BEHAVIOR_LEDGER="$RAPG_ROOT/bank/behavior_ledger.jsonl"
LOCK="$OUT/protocol_lock.json"
REPLAY="$OUT/replay"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PYTHON"
test -f "$BANK"
test -f "$BANK_MANIFEST"
test -f "$SOURCE_MANIFEST"
test -f "$SHADOW_A"
test -f "$PREDICTIONS"
test -f "$BEHAVIOR_LEDGER"
if [ -e "$OUT" ]; then
  echo "Refusing to reuse counterfactual credit replay output: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

"$PYTHON" -m py_compile \
  rescuecredit/counterfactual_credit_replay.py \
  scripts/freeze_toolsandbox_counterfactual_credit_replay.py \
  scripts/run_toolsandbox_counterfactual_credit_replay.py \
  scripts/check_toolsandbox_counterfactual_credit_replay.py
"$PYTHON" -m pytest -q tests/test_counterfactual_credit_replay.py

"$PYTHON" scripts/freeze_toolsandbox_counterfactual_credit_replay.py \
  --bank "$BANK" --bank-manifest "$BANK_MANIFEST" \
  --source-manifest "$SOURCE_MANIFEST" --predictions "$PREDICTIONS" \
  --behavior-ledger "$BEHAVIOR_LEDGER" \
  --output "$LOCK" | tee "$OUT/freeze.log"

"$PYTHON" scripts/run_toolsandbox_counterfactual_credit_replay.py \
  --protocol-lock "$LOCK" --bank "$BANK" --bank-manifest "$BANK_MANIFEST" \
  --source-manifest "$SOURCE_MANIFEST" --shadow-a-returns "$SHADOW_A" \
  --predictions "$PREDICTIONS" --behavior-ledger "$BEHAVIOR_LEDGER" \
  --output-dir "$REPLAY" \
  | tee "$OUT/replay.log"

set +e
"$PYTHON" scripts/check_toolsandbox_counterfactual_credit_replay.py \
  --protocol-lock "$LOCK" --bank "$BANK" --bank-manifest "$BANK_MANIFEST" \
  --source-manifest "$SOURCE_MANIFEST" --shadow-a-returns "$SHADOW_A" \
  --predictions "$PREDICTIONS" --behavior-ledger "$BEHAVIOR_LEDGER" \
  --replay-summary "$REPLAY/replay_summary.json" \
  --replay-artifact "$REPLAY/replay_artifact.pt" \
  --output "$OUT/feasibility_gate.json" | tee "$OUT/gate.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -eq 0 ]; then
  echo COUNTERFACTUAL_CREDIT_REPLAY_GATE_PASS
else
  echo COUNTERFACTUAL_CREDIT_REPLAY_GATE_FAIL
fi
exit "$STATUS"
