#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
source /home/hxy/miniconda3/etc/profile.d/conda.sh
conda activate /data/hxy/venvs/rescuecredit-appworld
unset VIRTUAL_ENV
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit

set -a
source .env
set +a
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "Set AZURE_OPENAI_API_KEY in /data/hxy/projects/RescueCredit/.env"
  exit 2
fi

PY_APPWORLD=/data/hxy/venvs/rescuecredit-appworld/bin/python
PY_AZURE=/data/hxy/projects/RescueCredit/.venv/bin/python
BANK=outputs/appworld_route_a_bank_train90_seed42
OUT=outputs/appworld_route_a_shadow_confirm_seed42
mkdir -p "$OUT"

"$PY_AZURE" scripts/check_azure.py > "$OUT/azure_check.log" 2>&1

# Events 0:20 are permanently reserved as engineering smoke. Confirmation and
# downstream training use the untouched 20:150 partition only.
"$PY_APPWORLD" scripts/attach_appworld_shadow_credit.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --bank-dir "$BANK" \
  --offset 20 \
  --limit 130 \
  --seed 42 \
  --max-shadow-steps 12 \
  --worker-python "$PY_AZURE" \
  --worker-script scripts/appworld_azure_continuation_worker.py \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

"$PY_APPWORLD" scripts/check_route_a_shadow_gate.py \
  --shadow-dir "$OUT" \
  --min-valid 100 \
  --min-nonzero 5 \
  2>&1 | tee "$OUT/gate_console.log"

echo ROUTE_A_SHADOW_CONFIRM_FINISHED
