#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PY="${TOOLSANDBOX_PYTHON:-/data/hxy/venvs/rescuecredit-toolsandbox/bin/python}"
MODEL_PY="${RESCUECREDIT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
WORKER="$REPO_ROOT/scripts/toolsandbox_azure_worker.py"
STAGE0="$REPO_ROOT/outputs/toolsandbox_stage0/gate.json"
ROOT="$REPO_ROOT/outputs/toolsandbox_v4_signal_seed42"
LOCK="$ROOT/protocol_lock.json"
SANITY="$ROOT/sanity_offset0_h4"
HOLDOUT="$ROOT/fresh_offset40_h8"

cd "$REPO_ROOT"
export PROMPT_COMMAND=
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -x "$APP_PY"
test -x "$MODEL_PY"
test -f "$WORKER"
test -f "$STAGE0"
if [ -e "$ROOT" ]; then
  echo "Refusing to reuse V4 output root: $ROOT" >&2
  exit 1
fi

"$MODEL_PY" -m py_compile \
  environments/toolsandbox/adapter.py \
  rescuecredit/toolsandbox_audit.py \
  rescuecredit/toolsandbox_credit.py \
  scripts/toolsandbox_azure_worker.py \
  scripts/audit_toolsandbox_signal.py \
  scripts/freeze_toolsandbox_v4_protocol.py \
  scripts/check_llm.py
"$MODEL_PY" -m pytest -q \
  tests/test_toolsandbox_audit.py \
  tests/test_toolsandbox_credit.py \
  tests/test_toolsandbox_v4_protocol.py \
  tests/test_llm_provider.py

"$MODEL_PY" - "$STAGE0" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert gate["passed"] is True, gate
PY

set -a
source .env
set +a
test "${TOOLSANDBOX_LLM_PROVIDER:-}" = "deepseek"
test -n "${DEEPSEEK_API_KEY:-}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://zhi-api.com/v1}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
test "$DEEPSEEK_THINKING" = "disabled"
"$MODEL_PY" scripts/check_llm.py --provider deepseek

mkdir -p "$ROOT"

"$APP_PY" scripts/freeze_toolsandbox_v4_protocol.py \
  --output "$LOCK" \
  --stage0-gate "$STAGE0" \
  --seed 42 --scenario-offset 40 --limit 40 \
  --horizon 8 --event-search-steps 8 \
  > "$ROOT/protocol_console.log"

set +e
"$APP_PY" scripts/audit_toolsandbox_signal.py \
  --limit 3 --scenario-offset 0 --seed 42 --horizon 4 \
  --event-search-steps 4 --credit-mode lexicographic_v4 \
  --worker-python "$MODEL_PY" --worker-script "$WORKER" \
  --output-dir "$SANITY" 2>&1 | tee "$SANITY.console.log"
SANITY_STATUS=${PIPESTATUS[0]}
set -e
test -f "$SANITY/audit_summary.json"
test -f "$SANITY/signal_events.jsonl"
test -f "$SANITY/quality_gate.json"
if [ "$SANITY_STATUS" -ne 0 ]; then
  "$MODEL_PY" - "$SANITY/audit_summary.json" "$SANITY/quality_gate.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
gate = json.load(open(sys.argv[2], encoding="utf-8"))
assert summary["scenarios_selected"] == 3, summary
assert summary["worker_failure_rate"] <= 0.34, summary
assert summary["credit_mode"] == "lexicographic_v4", summary
assert summary["controlled"]["valid_events"] >= 1, summary
assert summary["snapshot_audit"]["exact"] is True, summary
assert gate["checks"]["official_evaluator_used"] is True, gate
PY
fi
echo TOOLSANDBOX_V4_SANITY_COMPLETE

set +e
"$APP_PY" scripts/audit_toolsandbox_signal.py \
  --limit 40 --scenario-offset 40 --seed 42 --horizon 8 \
  --event-search-steps 8 --credit-mode lexicographic_v4 \
  --protocol-lock "$LOCK" \
  --worker-python "$MODEL_PY" --worker-script "$WORKER" \
  --output-dir "$HOLDOUT" 2>&1 | tee "$HOLDOUT.console.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -eq 0 ]; then
  echo TOOLSANDBOX_V4_DEPLOYABLE_GATE_PASS
else
  "$MODEL_PY" - "$HOLDOUT/quality_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
if gate.get("mechanism_passed"):
    print("TOOLSANDBOX_V4_MECHANISM_GATE_PASS_DEPLOYABLE_GATE_FAIL")
else:
    print("TOOLSANDBOX_V4_MECHANISM_GATE_FAIL")
PY
fi
echo TOOLSANDBOX_V4_SIGNAL_AUDIT_FINISHED
exit "$STATUS"
