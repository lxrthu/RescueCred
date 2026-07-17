#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=
source /home/hxy/miniconda3/etc/profile.d/conda.sh
conda activate /data/hxy/venvs/rescuecredit-appworld
unset VIRTUAL_ENV

PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
OUT=outputs/appworld_route_a_dense_credit_seed42
mkdir -p "$OUT"

"$PY" scripts/recompute_route_a_dense_credit.py \
  --bank-dir outputs/appworld_route_a_bank_train90_seed42 \
  --binary-credit-dir outputs/appworld_route_a_shadow_confirm_seed42 \
  --experiments-root experiments/outputs \
  --bank-offset 20 \
  --limit 130 \
  --seed 42 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

set +e
"$PY" scripts/check_route_a_dense_gate.py \
  --dense-dir "$OUT" \
  --min-valid 100 \
  --min-nonzero 15 \
  --min-rescue 5 \
  --min-reverse 3 \
  2>&1 | tee "$OUT/gate_console.log"
GATE_EXIT=$?
set -e

echo "DENSE_GATE_EXIT=$GATE_EXIT"
echo ROUTE_A_DENSE_RECOMPUTE_FINISHED
