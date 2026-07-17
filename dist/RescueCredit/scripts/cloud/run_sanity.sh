#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
pytest -q -p no:cacheprovider
python scripts/run_toy.py --samples 100000 --output-dir outputs/toy
python scripts/run_api_bank_smoke.py --split dev --limit 24 --output-dir outputs/smoke/api_bank
python scripts/validate_g0_support.py
python scripts/run_train.py --method rescuecredit --output-dir outputs/dry_run --dry-run
echo "SANITY_PASS_REAL_GATE_PENDING"
