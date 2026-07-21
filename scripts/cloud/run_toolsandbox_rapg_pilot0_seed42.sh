#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${RESCUECREDIT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
MODEL="${RESCUECREDIT_MODEL:-/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct}"
ADAPTER="${RAPG_ADAPTER:-$REPO_ROOT/outputs/toolsandbox_v41_preference_seed42/mask/adapter}"
V44_ROOT="${RAPG_V44_ROOT:-$REPO_ROOT/outputs/toolsandbox_v44_candidate_diversity_seed42}"
RAW="$V44_ROOT/full_offset85_h8/candidate_events.jsonl"
AUDIT_SUMMARY="$V44_ROOT/full_offset85_h8/audit_summary.json"
PROTOCOL_LOCK="$V44_ROOT/full_protocol_lock.json"
OUT="${RAPG_OUTPUT_ROOT:-$REPO_ROOT/outputs/toolsandbox_rapg_pilot0_seed42}"
BANK="$OUT/bank"
EVAL="$OUT/evaluation"
SOURCE="$OUT/source"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -x "$PYTHON"
test -d "$MODEL"
test -d "$ADAPTER"
test -f "$RAW"
test -f "$AUDIT_SUMMARY"
test -f "$PROTOCOL_LOCK"
if [ -e "$OUT" ]; then
  echo "Refusing to reuse RAPG output root: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

"$PYTHON" -m py_compile \
  rescuecredit/rapg.py \
  scripts/prepare_toolsandbox_rapg_preflight_data.py \
  scripts/build_toolsandbox_rapg_bank.py \
  scripts/evaluate_toolsandbox_rapg_pilot0.py \
  scripts/check_toolsandbox_rapg_gate.py
"$PYTHON" -c 'import torch; print(torch.__version__)'
"$PYTHON" -m pytest -q -rs tests/test_rapg.py
echo RAPG_M0_SANITY_PASS

"$PYTHON" scripts/prepare_toolsandbox_rapg_preflight_data.py \
  --raw-events "$RAW" --audit-summary "$AUDIT_SUMMARY" \
  --protocol-lock "$PROTOCOL_LOCK" --output-dir "$SOURCE" \
  2>&1 | tee "$OUT/prepare_source.log"
echo RAPG_SOURCE_SPLIT_COMPLETE

CUDA_VISIBLE_DEVICES="${RAPG_GPU:-0}" "$PYTHON" \
  scripts/build_toolsandbox_rapg_bank.py \
  --public-events "$SOURCE/events.public.jsonl" \
  --executed-b-returns "$SOURCE/executed_b_returns.jsonl" \
  --source-manifest "$SOURCE/source_manifest.json" \
  --model "$MODEL" --adapter "$ADAPTER" --output-dir "$BANK" \
  --seed 42 --temperature 1.0 --max-length 2048 --hash-dimension 128 \
  2>&1 | tee "$OUT/build_bank.log"
echo RAPG_M1_BANK_COMPLETE

"$PYTHON" scripts/evaluate_toolsandbox_rapg_pilot0.py \
  --bank "$BANK/rapg_bank.pt" --bank-manifest "$BANK/bank_manifest.json" \
  --shadow-a-returns "$SOURCE/shadow_a_returns.private.jsonl" \
  --source-manifest "$SOURCE/source_manifest.json" \
  --output-dir "$EVAL" --seed 42 --folds 5 --ridge-alpha 10 \
  --audit-rate 0.20 --p-min 0.05 --replicates 1000 \
  2>&1 | tee "$OUT/evaluate.log"
echo RAPG_M2_SIMULATION_COMPLETE

set +e
"$PYTHON" scripts/check_toolsandbox_rapg_gate.py \
  --bank "$BANK/rapg_bank.pt" --bank-manifest "$BANK/bank_manifest.json" \
  --source-manifest "$SOURCE/source_manifest.json" \
  --shadow-a-returns "$SOURCE/shadow_a_returns.private.jsonl" \
  --simulation-summary "$EVAL/simulation_summary.json" \
  --predictions "$EVAL/crossfit_predictions.pt" \
  --estimates "$EVAL/audit_estimates.pt" \
  --propensity-ledger "$EVAL/propensity_ledger.jsonl" \
  --behavior-ledger "$BANK/behavior_ledger.jsonl" \
  --output "$OUT/pilot_gate.json" \
  2>&1 | tee "$OUT/gate.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -eq 0 ]; then
  echo RAPG_PILOT0_GATE_PASS
else
  echo RAPG_PILOT0_GATE_FAIL
fi
exit "$STATUS"
