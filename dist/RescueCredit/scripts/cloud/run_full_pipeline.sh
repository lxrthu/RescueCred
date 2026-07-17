#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/cloud/setup_cloud.sh
bash scripts/cloud/run_sanity.sh

set -a
source .env
set +a
if [[ -z "${AZURE_OPENAI_API_KEY:-}" || "${AZURE_OPENAI_API_KEY}" == "REPLACE_WITH_ROTATED_KEY" ]]; then
  echo "Azure key not configured; skipping Azure-only base trajectory smoke."
else
  bash scripts/cloud/run_azure_smoke.sh
fi

bash scripts/cloud/run_pilot_4gpu.sh
bash scripts/cloud/run_confirmatory_4gpu.sh
