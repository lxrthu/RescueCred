#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -e '.[dev,train]'

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created $(pwd)/.env — fill AZURE_OPENAI_API_KEY with a rotated key."
fi

python -m compileall -q rescuecredit environments scripts tests
pytest -q -p no:cacheprovider
python scripts/prepare_api_bank_controlled.py
echo "SETUP_PASS"

