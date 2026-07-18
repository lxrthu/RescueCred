#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PY="${TOOLSANDBOX_PYTHON:-/data/hxy/venvs/rescuecredit-toolsandbox/bin/python}"
MODEL_PY="${RESCUECREDIT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
WORKER="$REPO_ROOT/scripts/toolsandbox_azure_worker.py"
STAGE0="$REPO_ROOT/outputs/toolsandbox_stage0/gate.json"
OLD_LOCK="$REPO_ROOT/outputs/toolsandbox_v4_signal_seed42/protocol_lock.json"
PLAN="$REPO_ROOT/refine-logs/TOOLSANDBOX_V41_PLAN.md"
ROOT="$REPO_ROOT/outputs/toolsandbox_v41_toolid_seed42"
DIAGNOSTIC="$ROOT/diagnostic_offset80_h4"
FRESH="$ROOT/fresh_offset85_h8"
DIAGNOSTIC_LOCK="$ROOT/diagnostic_protocol_lock.json"
FRESH_LOCK="$ROOT/fresh_protocol_lock.json"

cd "$REPO_ROOT"
export PROMPT_COMMAND=
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -x "$APP_PY"
test -x "$MODEL_PY"
test -f "$WORKER"
test -f "$STAGE0"
test -f "$OLD_LOCK"
test -f "$PLAN"
if [ -e "$ROOT" ]; then
  echo "Refusing to reuse V4.1 output root: $ROOT" >&2
  exit 1
fi

"$MODEL_PY" -m py_compile \
  scripts/audit_toolsandbox_signal.py \
  scripts/check_toolsandbox_v41_diagnostic_gate.py \
  scripts/freeze_toolsandbox_v4_protocol.py \
  scripts/toolsandbox_azure_worker.py
"$MODEL_PY" -m pytest -q \
  tests/test_toolsandbox_audit.py \
  tests/test_toolsandbox_credit.py \
  tests/test_toolsandbox_v4_protocol.py \
  tests/test_toolsandbox_v41.py \
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

# Freeze both partitions before observing any V4.1 diagnostic outcome.
"$APP_PY" scripts/freeze_toolsandbox_v4_protocol.py \
  --output "$DIAGNOSTIC_LOCK" --plan "$PLAN" --stage0-gate "$STAGE0" \
  --seed 42 --scenario-offset 80 --limit 5 --minimum-scenarios 5 \
  --horizon 4 --event-search-steps 4 --worker-timeout-sec 600 \
  --harness-interface tool_id_v2 --exclude-protocol "$OLD_LOCK"

"$APP_PY" scripts/freeze_toolsandbox_v4_protocol.py \
  --output "$FRESH_LOCK" --plan "$PLAN" --stage0-gate "$STAGE0" \
  --seed 42 --scenario-offset 85 --limit 40 --minimum-scenarios 30 \
  --horizon 8 --event-search-steps 8 --worker-timeout-sec 600 \
  --harness-interface tool_id_v2 \
  --exclude-protocol "$OLD_LOCK" --exclude-protocol "$DIAGNOSTIC_LOCK"

set +e
"$APP_PY" scripts/audit_toolsandbox_signal.py \
  --limit 5 --scenario-offset 80 --seed 42 --horizon 4 \
  --event-search-steps 4 --credit-mode lexicographic_v4 \
  --worker-timeout-sec 600 --harness-interface tool_id_v2 \
  --protocol-lock "$DIAGNOSTIC_LOCK" \
  --worker-python "$MODEL_PY" --worker-script "$WORKER" \
  --output-dir "$DIAGNOSTIC" 2>&1 | tee "$DIAGNOSTIC.console.log"
DIAGNOSTIC_AUDIT_STATUS=${PIPESTATUS[0]}
set -e
test -f "$DIAGNOSTIC/audit_summary.json"
test -f "$DIAGNOSTIC/quality_gate.json"
"$MODEL_PY" scripts/check_toolsandbox_v41_diagnostic_gate.py \
  --summary "$DIAGNOSTIC/audit_summary.json" \
  --audit-gate "$DIAGNOSTIC/quality_gate.json" \
  --output "$ROOT/diagnostic_gate.json"
echo "TOOLSANDBOX_V41_DIAGNOSTIC_PASS audit_status=$DIAGNOSTIC_AUDIT_STATUS"

set +e
"$APP_PY" scripts/audit_toolsandbox_signal.py \
  --limit 40 --scenario-offset 85 --seed 42 --horizon 8 \
  --event-search-steps 8 --credit-mode lexicographic_v4 \
  --worker-timeout-sec 600 --harness-interface tool_id_v2 \
  --protocol-lock "$FRESH_LOCK" \
  --worker-python "$MODEL_PY" --worker-script "$WORKER" \
  --output-dir "$FRESH" 2>&1 | tee "$FRESH.console.log"
FRESH_STATUS=${PIPESTATUS[0]}
set -e

if [ "$FRESH_STATUS" -eq 0 ]; then
  echo TOOLSANDBOX_V41_FRESH_GATE_PASS
else
  "$MODEL_PY" - "$FRESH/quality_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
if gate.get("mechanism_passed"):
    print("TOOLSANDBOX_V41_MECHANISM_PASS_DEPLOYABLE_FAIL")
else:
    print("TOOLSANDBOX_V41_MECHANISM_FAIL")
PY
fi
echo TOOLSANDBOX_V41_FINISHED
exit "$FRESH_STATUS"
