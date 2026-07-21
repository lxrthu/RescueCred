#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${RESCUECREDIT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
RAPG_ROOT="${RAPG_ROOT:-$REPO_ROOT/outputs/toolsandbox_rapg_pilot0_retry1_seed42}"
BANK="$RAPG_ROOT/bank"
SOURCE="$RAPG_ROOT/source"
EVAL="$RAPG_ROOT/evaluation"
OUT="${RAPG_AUDIT_OUTPUT:-$RAPG_ROOT/task_stability_audit}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PYTHON"
test -f "$BANK/rapg_bank.pt"
test -f "$BANK/bank_manifest.json"
test -f "$SOURCE/source_manifest.json"
test -f "$SOURCE/shadow_a_returns.private.jsonl"
test -f "$EVAL/crossfit_predictions.pt"
test -f "$EVAL/audit_estimates.pt"
test -f "$EVAL/simulation_summary.json"
test -f "$RAPG_ROOT/pilot_gate.json"
test ! -e "$OUT"

"$PYTHON" -m py_compile \
  rescuecredit/rapg_stability.py \
  scripts/analyze_toolsandbox_rapg_task_stability.py
"$PYTHON" -m pytest -q tests/test_rapg_stability.py

"$PYTHON" scripts/analyze_toolsandbox_rapg_task_stability.py \
  --bank "$BANK/rapg_bank.pt" \
  --bank-manifest "$BANK/bank_manifest.json" \
  --source-manifest "$SOURCE/source_manifest.json" \
  --shadow-a-returns "$SOURCE/shadow_a_returns.private.jsonl" \
  --predictions "$EVAL/crossfit_predictions.pt" \
  --estimates "$EVAL/audit_estimates.pt" \
  --simulation-summary "$EVAL/simulation_summary.json" \
  --pilot-gate "$RAPG_ROOT/pilot_gate.json" \
  --output-dir "$OUT" \
  --bootstrap-replicates 20000 \
  --seed 42 \
  2>&1 | tee "$RAPG_ROOT/task_stability_audit.log"

echo RAPG_TASK_STABILITY_AUDIT_COMPLETE
