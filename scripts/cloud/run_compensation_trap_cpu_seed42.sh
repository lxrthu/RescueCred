#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${RESCUECREDIT_PYTHON:-$ROOT/.venv/bin/python}"
OUT="${COMPENSATION_TRAP_OUTPUT:-$ROOT/outputs/compensation_trap_cpu_seed42}"
V44="$ROOT/outputs/toolsandbox_v44_candidate_diversity_seed42/full_offset85_h8/candidate_events.jsonl"
V45_DEV="$ROOT/outputs/toolsandbox_v45_matched_anchor_seed42/development_candidates_offset125_h8/candidate_events.jsonl"
V45_CONFIRM="$ROOT/outputs/toolsandbox_v45_matched_anchor_seed42/confirmation_candidates_offset165_h8/candidate_events.jsonl"
BENCHMARK="$OUT/benchmark"
COLLISIONS="$OUT/collisions"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PY"
test -f "$V44"
test -f "$V45_DEV"
test -f "$V45_CONFIRM"
if [ -e "$OUT" ]; then
  echo "Refusing to reuse Compensation Trap output: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

"$PY" -m py_compile \
  rescuecredit/compensation_trap.py \
  scripts/build_compensation_trap_benchmark.py \
  scripts/audit_compensation_trap_collisions.py \
  scripts/verify_compensation_trap_benchmark.py
"$PY" -m pytest -q tests/test_compensation_trap.py

"$PY" scripts/build_compensation_trap_benchmark.py \
  --raw-events "$V44" --source-name toolsandbox_offset85 \
  --raw-events "$V45_DEV" --source-name toolsandbox_offset125 \
  --raw-events "$V45_CONFIRM" --source-name toolsandbox_offset165 \
  --output-dir "$BENCHMARK" | tee "$OUT/build.log"

"$PY" scripts/verify_compensation_trap_benchmark.py \
  --benchmark-dir "$BENCHMARK" | tee "$OUT/verify.log"

set +e
"$PY" scripts/audit_compensation_trap_collisions.py \
  --benchmark-dir "$BENCHMARK" --similarity-threshold 0.90 \
  --output-dir "$COLLISIONS" | tee "$OUT/collision.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -eq 0 ]; then
  echo COMPENSATION_TRAP_COLLISION_GATE_PASS
else
  echo COMPENSATION_TRAP_COLLISION_GATE_FAIL
fi
exit "$STATUS"
