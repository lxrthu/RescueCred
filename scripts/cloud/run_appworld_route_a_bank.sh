#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
source /home/hxy/miniconda3/etc/profile.d/conda.sh
conda activate /data/hxy/venvs/rescuecredit-appworld
unset VIRTUAL_ENV
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit

if [ ! -f .env ]; then
  echo "MISSING: /data/hxy/projects/RescueCredit/.env"
  exit 2
fi
set -a
source .env
set +a
if [ -z "${AZURE_OPENAI_API_KEY:-}" ] || [ "$AZURE_OPENAI_API_KEY" = "REPLACE_WITH_ROTATED_KEY" ]; then
  echo "Edit /data/hxy/projects/RescueCredit/.env and set AZURE_OPENAI_API_KEY"
  exit 2
fi

PY_APPWORLD=/data/hxy/venvs/rescuecredit-appworld/bin/python
PY_SELECTOR=/data/hxy/projects/RescueCredit/.venv/bin/python
MODEL=/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct
OUT=outputs/appworld_route_a_bank_train90_seed42
mkdir -p "$OUT"

"$PY_SELECTOR" scripts/check_azure.py > "$OUT/azure_check.log" 2>&1

"$PY_APPWORLD" scripts/build_appworld_route_a_bank.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --subset train \
  --offset 0 \
  --limit 90 \
  --seed 42 \
  --max-cases-per-task 20 \
  --selector-python "$PY_SELECTOR" \
  --selector-script scripts/appworld_azure_candidate_selector_worker.py \
  --selector-model "$MODEL" \
  --selector-device cpu \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

"$PY_APPWORLD" scripts/check_route_a_bank.py \
  --bank-dir "$OUT" \
  --min-events 30 \
  2>&1 | tee "$OUT/gate_console.log"

echo ROUTE_A_BANK_FINISHED
