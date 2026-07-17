#!/usr/bin/env bash
set -uo pipefail

cd /data/hxy/projects/RescueCredit
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit

if [ ! -f .env ]; then
  echo "MISSING /data/hxy/projects/RescueCredit/.env"
  exit 2
fi
set -a
source .env
set +a
if [ -z "${AZURE_OPENAI_API_KEY:-}" ] || [ "$AZURE_OPENAI_API_KEY" = "REPLACE_WITH_ROTATED_KEY" ]; then
  echo "Set a rotated AZURE_OPENAI_API_KEY in /data/hxy/projects/RescueCredit/.env"
  exit 2
fi

PY_APPWORLD=/data/hxy/venvs/rescuecredit-appworld/bin/python
PY_AZURE=/data/hxy/projects/RescueCredit/.venv/bin/python
ROOT_OUT=outputs/appworld_harness_audit_30_v8_provenance
mkdir -p "$ROOT_OUT"

if ! "$PY_AZURE" scripts/check_azure.py > "$ROOT_OUT/azure_check.log" 2>&1; then
  echo "AZURE_CHECK_FAILED"
  tail -n 40 "$ROOT_OUT/azure_check.log"
  exit 2
fi
echo "AZURE_CHECK_OK"

run_partition() {
  local name="$1"
  local offset="$2"
  local out="$ROOT_OUT/$name"
  mkdir -p "$out"
  set +e
  "$PY_APPWORLD" scripts/audit_appworld_deployable_harness.py \
    --appworld-root /data/hxy/projects/RescueCredit \
    --subset train \
    --offset "$offset" \
    --limit 30 \
    --max-cases-per-task 10 \
    --min-selector-candidates 1 \
    --min-supported-coverage 0.10 \
    --selector-python "$PY_AZURE" \
    --selector-script scripts/appworld_azure_candidate_selector_worker.py \
    --selector-model /data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct \
    --selector-device cpu \
    --output-dir "$out" \
    > "$out/console.log" 2>&1
  local exit_code=$?
  set -e
  echo "$exit_code" > "$out/exit_code.txt"
  return 0
}

run_partition development 0
run_partition final_fresh_holdout 90

"$PY_APPWORLD" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/appworld_harness_audit_30_v8_provenance")
gates = {}
for partition in ("development", "final_fresh_holdout"):
    path = root / partition / "quality_gate.json"
    gates[partition] = json.loads(path.read_text()) if path.exists() else {"passed": False, "missing": True}
passed = bool(gates["development"].get("passed") and gates["final_fresh_holdout"].get("passed"))
combined = {
    "passed": passed,
    "selector": "Azure GPT-4o with reference-free candidate provenance",
    "confidence_threshold_frozen_before_holdout": 0.90,
    "provenance_inputs": ["visible receipt field path", "visible instruction span"],
    "reference_boundary": "no reference action or protected value enters Azure prompts",
    "partitions": {
        "development": "train tasks 0:30",
        "final_fresh_holdout": "train tasks 90:120, unused by AW2b through AW2g",
    },
    "gates": gates,
    "authorizes_v2_training_smoke": passed,
}
(root / "combined_gate.json").write_text(
    json.dumps(combined, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps(combined, indent=2, ensure_ascii=False))
PY

echo AW2H_PROVENANCE_FINISHED
