#!/usr/bin/env bash
set -uo pipefail

cd /data/hxy/projects/RescueCredit
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit

PY_APPWORLD=/data/hxy/venvs/rescuecredit-appworld/bin/python
PY_SELECTOR=/data/hxy/projects/RescueCredit/.venv/bin/python
MODEL=/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct
ROOT_OUT=outputs/appworld_harness_audit_30_v5
mkdir -p "$ROOT_OUT"

run_split() {
  local split="$1"
  local out="$ROOT_OUT/$split"
  mkdir -p "$out"
  set +e
  "$PY_APPWORLD" scripts/audit_appworld_deployable_harness.py \
    --appworld-root /data/hxy/projects/RescueCredit \
    --subset "$split" \
    --limit 30 \
    --max-cases-per-task 10 \
    --min-selector-candidates 6 \
    --min-supported-coverage 0.10 \
    --selector-python "$PY_SELECTOR" \
    --selector-model "$MODEL" \
    --selector-device cuda:0 \
    --output-dir "$out" \
    > "$out/console.log" 2>&1
  local exit_code=$?
  set -e
  echo "$exit_code" > "$out/exit_code.txt"
  return 0
}

run_split train
run_split dev

"$PY_APPWORLD" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/appworld_harness_audit_30_v5")
gates = {}
for split in ("train", "dev"):
    path = root / split / "quality_gate.json"
    gates[split] = json.loads(path.read_text()) if path.exists() else {"passed": False, "missing": True}
combined = {
    "passed": bool(gates["train"].get("passed") and gates["dev"].get("passed")),
    "rule_frozen_before_dev": True,
    "development_rule": "selector requires at least 6 visible candidates",
    "hard_gates": {
        "correction_precision": 0.90,
        "single_step_rescue_rate": 0.10,
        "harm_rate": 0.01,
        "coverage": 0.10,
        "supported_coverage": 0.10,
    },
    "splits": gates,
    "authorizes_v2_training_smoke": bool(
        gates["train"].get("passed") and gates["dev"].get("passed")
    ),
}
(root / "combined_gate.json").write_text(
    json.dumps(combined, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps(combined, indent=2, ensure_ascii=False))
PY

echo AW2E_FINISHED
