#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/check_azure.py
python scripts/collect_azure_trajectories.py --split dev --limit 5 --output outputs/azure/base_trajectories_smoke.jsonl

