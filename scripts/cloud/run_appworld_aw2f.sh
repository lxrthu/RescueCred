#!/usr/bin/env bash
set -uo pipefail

cd /data/hxy/projects/RescueCredit
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit

PY_APPWORLD=/data/hxy/venvs/rescuecredit-appworld/bin/python
PY_SELECTOR=/data/hxy/projects/RescueCredit/.venv/bin/python
MODEL=/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct
ROOT_OUT=outputs/appworld_harness_audit_30_v6_pointwise
mkdir -p "$ROOT_OUT"

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

run_partition development 0
run_partition holdout 30

"$PY_APPWORLD" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/appworld_harness_audit_30_v6_pointwise")
gates = {}
for partition in ("development", "holdout"):
    path = root / partition / "quality_gate.json"
    gates[partition] = json.loads(path.read_text()) if path.exists() else {"passed": False, "missing": True}
passed = bool(gates["development"].get("passed") and gates["holdout"].get("passed"))
combined = {
    "passed": passed,
    "selector": "order-independent pointwise Yes/No mean-token log-likelihood",
    "thresholds_frozen_before_holdout": {
        "minimum_yes_probability": 0.80,
        "minimum_probability_gap": 0.20,
    },
    "partitions": {
        "development": "train tasks 0:30",
        "holdout": "train tasks 30:60, not used for rule design",
    },
    "gates": gates,
    "authorizes_v2_training_smoke": passed,
}
(root / "combined_gate.json").write_text(
    json.dumps(combined, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps(combined, indent=2, ensure_ascii=False))
PY

echo AW2F_FINISHED
