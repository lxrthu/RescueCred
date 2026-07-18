#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PY="${TOOLSANDBOX_PYTHON:-/data/hxy/venvs/rescuecredit-toolsandbox/bin/python}"
MODEL_PY="${RESCUECREDIT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
WORKER="$REPO_ROOT/scripts/toolsandbox_azure_worker.py"
STAGE0="$REPO_ROOT/outputs/toolsandbox_stage0/gate.json"
OUT="$REPO_ROOT/outputs/toolsandbox_signal_audit_40_seed42"
SANITY="$REPO_ROOT/outputs/toolsandbox_signal_sanity_3_seed42"

cd "$REPO_ROOT"
export PROMPT_COMMAND=
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -x "$APP_PY"
test -x "$MODEL_PY"
test -f "$WORKER"
test -f "$STAGE0"

"$MODEL_PY" -m py_compile \
  environments/toolsandbox/adapter.py \
  rescuecredit/toolsandbox_audit.py \
  scripts/toolsandbox_azure_worker.py \
  scripts/audit_toolsandbox_signal.py
"$MODEL_PY" -m pytest -q tests/test_toolsandbox_audit.py

"$MODEL_PY" - "$STAGE0" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert gate["passed"] is True, gate
PY

set -a
source .env
set +a
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "Set AZURE_OPENAI_API_KEY in $REPO_ROOT/.env" >&2
  exit 2
fi
"$MODEL_PY" scripts/check_azure.py > "$REPO_ROOT/outputs/toolsandbox_azure_check.log" 2>&1

if [ -e "$SANITY" ] || [ -e "$OUT" ]; then
  echo "Refusing to reuse ToolSandbox output directories." >&2
  echo "Move old outputs aside before a fresh frozen run." >&2
  exit 1
fi
mkdir -p "$SANITY" "$OUT"

set +e
"$APP_PY" scripts/audit_toolsandbox_signal.py \
  --limit 3 --seed 42 --horizon 4 \
  --worker-python "$MODEL_PY" --worker-script "$WORKER" \
  --output-dir "$SANITY" 2>&1 | tee "$SANITY/console.log"
SANITY_STATUS=${PIPESTATUS[0]}
set -e

# The 3-scenario run is a transport/integration smoke. Its statistical gate is
# expected to fail the 30-scenario threshold; require artifacts and no crash.
test -f "$SANITY/audit_summary.json"
test -f "$SANITY/signal_events.jsonl"
if [ "$SANITY_STATUS" -ne 0 ]; then
  "$MODEL_PY" - "$SANITY/audit_summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["scenarios_selected"] == 3, summary
assert summary["worker_failure_rate"] <= 0.34, summary
PY
fi
echo TOOLSANDBOX_SIGNAL_SANITY_COMPLETE

set +e
"$APP_PY" scripts/audit_toolsandbox_signal.py \
  --limit 40 --seed 42 --horizon 8 \
  --worker-python "$MODEL_PY" --worker-script "$WORKER" \
  --output-dir "$OUT" 2>&1 | tee "$OUT/console.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -eq 0 ]; then
  echo TOOLSANDBOX_SIGNAL_GATE_PASS
else
  echo TOOLSANDBOX_SIGNAL_GATE_FAIL
fi
echo TOOLSANDBOX_SIGNAL_AUDIT_FINISHED
exit "$STATUS"
