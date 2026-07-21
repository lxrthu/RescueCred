#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PY="${TOOLSANDBOX_PYTHON:-/data/hxy/venvs/rescuecredit-toolsandbox/bin/python}"
MODEL_PY="${RESCUECREDIT_PYTHON:-$ROOT/.venv/bin/python}"
WORKER="$ROOT/scripts/toolsandbox_azure_worker.py"
STAGE0="$ROOT/outputs/toolsandbox_stage0/gate.json"
PLAN="$ROOT/refine-logs/COMPENSATION_TRAP_EXPERIMENT_PLAN_20260721_230022.md"
OUT="${COMPENSATION_CONFIRM_OUTPUT:-$ROOT/outputs/toolsandbox_compensation_confirmation_seed42}"
LOCK="$OUT/protocol_lock.json"
AUDIT="$OUT/fresh_offset205_h8"

cd "$ROOT"
export PROMPT_COMMAND=
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -x "$APP_PY"
test -x "$MODEL_PY"
test -f "$WORKER"
test -f "$STAGE0"
test -f "$PLAN"
if [ -e "$OUT" ]; then
  echo "Refusing to reuse fresh Compensation Trap output: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"
mapfile -t HISTORICAL < <(
  find "$ROOT/outputs" -type f \
    \( -name '*protocol*.json' -o -name 'audit_summary.json' \) \
    ! -path "$OUT/*" | sort
)
[ "${#HISTORICAL[@]}" -ge 5 ] || {
  echo "Historical ToolSandbox inventory is unexpectedly incomplete" >&2
  exit 1
}

"$MODEL_PY" -m py_compile \
  scripts/freeze_toolsandbox_compensation_confirmation.py \
  scripts/check_toolsandbox_compensation_confirmation.py
"$MODEL_PY" -m pytest -q tests/test_compensation_trap.py

set -a
source .env
set +a
test "${TOOLSANDBOX_LLM_PROVIDER:-}" = deepseek
test -n "${DEEPSEEK_API_KEY:-}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://zhi-api.com/v1}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
test "$DEEPSEEK_THINKING" = disabled
"$MODEL_PY" scripts/check_llm.py --provider deepseek

FREEZE_ARGS=()
for path in "${HISTORICAL[@]}"; do
  FREEZE_ARGS+=(--historical-protocol "$path")
done
"$APP_PY" scripts/freeze_toolsandbox_compensation_confirmation.py \
  --stage0-gate "$STAGE0" --historical-output-root "$ROOT/outputs" --plan "$PLAN" \
  "${FREEZE_ARGS[@]}" --output "$LOCK" | tee "$OUT/freeze.log"

"$APP_PY" scripts/audit_toolsandbox_v44_candidates.py --role full \
  --protocol-lock "$LOCK" --worker-python "$MODEL_PY" --worker-script "$WORKER" \
  --worker-timeout-sec 600 --harness-interface tool_id_v2 --seed 42 \
  --scenario-offset 205 --limit 13 --horizon 8 --event-search-steps 8 \
  --candidate-count 3 --max-pairs-per-scenario 4 --output-dir "$AUDIT" \
  2>&1 | tee "$OUT/audit.log"

set +e
"$MODEL_PY" scripts/check_toolsandbox_compensation_confirmation.py \
  --protocol-lock "$LOCK" --audit-summary "$AUDIT/audit_summary.json" \
  --raw-events "$AUDIT/candidate_events.jsonl" \
  --output "$OUT/confirmation_gate.json" | tee "$OUT/gate.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -eq 0 ]; then
  echo COMPENSATION_TRAP_FRESH_CONFIRMATION_PASS
else
  echo COMPENSATION_TRAP_FRESH_CONFIRMATION_FAIL
fi
exit "$STATUS"
