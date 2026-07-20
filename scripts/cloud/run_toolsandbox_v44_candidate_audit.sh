#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PY="${TOOLSANDBOX_PYTHON:-/data/hxy/venvs/rescuecredit-toolsandbox/bin/python}"
MODEL_PY="${RESCUECREDIT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
WORKER="$REPO_ROOT/scripts/toolsandbox_azure_worker.py"
STAGE0="$REPO_ROOT/outputs/toolsandbox_stage0/gate.json"
V41_ROOT="$REPO_ROOT/outputs/toolsandbox_v41_toolid_seed42"
OLD_TRAIN_LOCK="$V41_ROOT/fresh_protocol_lock.json"
V41_PREF_ROOT="$REPO_ROOT/outputs/toolsandbox_v41_preference_seed42"
DEV_LOCK="$V41_PREF_ROOT/evaluation_protocol_lock.json"
V42_ROOT="$REPO_ROOT/outputs/toolsandbox_v42_balanced_margin_seed42"
CONFIRM_LOCK="$V42_ROOT/confirmation_protocol_lock.json"
V43_ROOT="$REPO_ROOT/outputs/toolsandbox_v43_multi_prefix_anchor_seed42"
V43_MINING_LOCK="$V43_ROOT/mining_protocol_lock.json"
V43_DATA="$V43_ROOT/train_data"
PLAN="$REPO_ROOT/refine-logs/TOOLSANDBOX_V44_PLAN.md"

OUT="$REPO_ROOT/outputs/toolsandbox_v44_candidate_diversity_seed42"
SANITY_LOCK="$OUT/sanity_protocol_lock.json"
FULL_LOCK="$OUT/full_protocol_lock.json"
SANITY="$OUT/sanity_offset85_h8"
FULL="$OUT/full_offset85_h8"
DATA="$OUT/data"

cd "$REPO_ROOT"
export PROMPT_COMMAND=
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

test -x "$APP_PY"
test -x "$MODEL_PY"
for path in \
  "$WORKER" "$STAGE0" "$OLD_TRAIN_LOCK" "$DEV_LOCK" "$CONFIRM_LOCK" \
  "$V43_MINING_LOCK" "$V43_DATA/manifest.json" "$V43_DATA/data_gate.json" \
  "$PLAN"; do
  test -e "$path"
done
if [ -e "$V42_ROOT/fresh_confirm_offset165_h8/audit_summary.json" ] || \
   [ -e "$V43_ROOT/fresh_confirm_offset165_h8/audit_summary.json" ]; then
  echo "Refusing V4.4: offset-165 confirmation outcomes exist." >&2
  exit 1
fi
if [ -e "$OUT" ]; then
  echo "Refusing to reuse V4.4 output root: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT" "$DATA"

"$MODEL_PY" -m py_compile \
  scripts/toolsandbox_azure_worker.py \
  scripts/freeze_toolsandbox_v44_candidate_protocol.py \
  scripts/audit_toolsandbox_v44_candidates.py \
  scripts/prepare_toolsandbox_v44_candidate_data.py
"$MODEL_PY" -m pytest -q \
  tests/test_toolsandbox_v44.py \
  tests/test_toolsandbox_v43.py \
  tests/test_toolsandbox_audit.py

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

# Freeze sanity and full identities before either audit observes outcomes.
freeze_protocol() {
  local role="$1" output="$2" limit="$3" candidates="$4" pairs="$5"
  "$APP_PY" scripts/freeze_toolsandbox_v44_candidate_protocol.py \
    --role "$role" --output "$output" --plan "$PLAN" \
    --stage0-gate "$STAGE0" --old-training-protocol "$OLD_TRAIN_LOCK" \
    --v43-mining-protocol "$V43_MINING_LOCK" --v43-data-dir "$V43_DATA" \
    --development-protocol "$DEV_LOCK" --confirmation-protocol "$CONFIRM_LOCK" \
    --v42-root "$V42_ROOT" --v43-root "$V43_ROOT" \
    --seed 42 --scenario-offset 85 --limit "$limit" --horizon 8 \
    --event-search-steps 8 --candidate-count "$candidates" \
    --max-pairs-per-scenario "$pairs" --worker-timeout-sec 600 \
    --harness-interface tool_id_v2
}
freeze_protocol sanity "$SANITY_LOCK" 3 2 1 \
  2>&1 | tee "$OUT/freeze_sanity.log"
freeze_protocol full "$FULL_LOCK" 40 3 4 \
  2>&1 | tee "$OUT/freeze_full.log"

run_audit() {
  local role="$1" lock="$2" limit="$3" candidates="$4" pairs="$5" output="$6"
  "$APP_PY" scripts/audit_toolsandbox_v44_candidates.py \
    --role "$role" --protocol-lock "$lock" --worker-python "$MODEL_PY" \
    --worker-script "$WORKER" --worker-timeout-sec 600 \
    --harness-interface tool_id_v2 --seed 42 --scenario-offset 85 \
    --limit "$limit" --horizon 8 --event-search-steps 8 \
    --candidate-count "$candidates" --max-pairs-per-scenario "$pairs" \
    --output-dir "$output"
}

run_audit sanity "$SANITY_LOCK" 3 2 1 "$SANITY" \
  2>&1 | tee "$OUT/sanity_console.log"
"$MODEL_PY" - "$SANITY/audit_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
assert s["protocol_validated"] is True, s
assert s["snapshot_audit"]["exact"] is True, s
assert s["valid_pairs"] >= 1, s
assert s["worker_failure_rate"] <= 0.67, s
PY
echo TOOLSANDBOX_V44_SANITY_PASS

run_audit full "$FULL_LOCK" 40 3 4 "$FULL" \
  2>&1 | tee "$OUT/full_console.log"

set +e
"$APP_PY" scripts/prepare_toolsandbox_v44_candidate_data.py \
  --audit-root "$FULL" --protocol-lock "$FULL_LOCK" --output-dir "$DATA" \
  2>&1 | tee "$OUT/prepare_data.log"
DATA_STATUS=${PIPESTATUS[0]}
set -e
if [ "$DATA_STATUS" -eq 0 ]; then
  echo TOOLSANDBOX_V44_DATA_GATE_PASS
else
  echo TOOLSANDBOX_V44_DATA_GATE_FAIL
fi
echo TOOLSANDBOX_V44_CANDIDATE_AUDIT_FINISHED
exit "$DATA_STATUS"
