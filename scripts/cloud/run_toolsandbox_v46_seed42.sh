#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${RESCUECREDIT_PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${RESCUECREDIT_MODEL:-/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct}"
V44="$ROOT/outputs/toolsandbox_v44_candidate_diversity_seed42/data"
V45="$ROOT/outputs/toolsandbox_v45_matched_anchor_seed42"
OUT="$ROOT/outputs/toolsandbox_v46_selective_residual_seed42"
LOCK="$OUT/protocol_lock.json"
cd "$ROOT"; export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
test -x "$PY"; test -d "$MODEL"; test -f "$V45/confirmation_gate.json"
[ ! -e "$OUT" ] || { echo "Refusing to reuse V4.6 output root" >&2; exit 1; }
mkdir -p "$OUT/control" "$OUT/v46"
"$PY" -m py_compile scripts/train_toolsandbox_v46_residual.py scripts/freeze_toolsandbox_v46_protocol.py scripts/check_toolsandbox_v46_gate.py
"$PY" -m pytest -q tests/test_toolsandbox_v46.py tests/test_toolsandbox_v45.py tests/test_toolsandbox_preference.py
"$PY" scripts/freeze_toolsandbox_v46_protocol.py --v44-data "$V44" --v45-root "$V45" --model "$MODEL" --output "$LOCK" | tee "$OUT/freeze.log"
mapfile -t G < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2);print $2,$1}' | sort -n | head -n2 | awk '{print $2}')
[ "${#G[@]}" -ge 2 ] || exit 2; echo "CONTROL_GPU=${G[0]} V46_GPU=${G[1]}"
train() { local method="$1" gpu="$2"; CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_toolsandbox_v46_residual.py --method "$method" --model "$MODEL" --mask-adapter "$V45/mask/adapter" --train-file "$V44/train.jsonl" --protocol-lock "$LOCK" --seed 42 --epochs 3 --learning-rate 3e-6 --gradient-accumulation 8 --max-length 2048 --beta 1.0 --target-residual 0.05 --confidence-margin 0.05 --retention-coef 1.0 --reference-anchor-coef 0.25 --fp32 --output-dir "$OUT/$method" > "$OUT/$method/train_console.log" 2>&1; }
train control "${G[0]}" & P1=$!; train v46 "${G[1]}" & P2=$!
set +e; wait "$P1"; S1=$?; wait "$P2"; S2=$?; set -e
[ "$S1" -eq 0 ] && [ "$S2" -eq 0 ] || { echo "V4.6 training failed control=$S1 v46=$S2" >&2; exit 1; }
echo TOOLSANDBOX_V46_TRAINING_FINISHED
eval_one() { local method="$1" gpu="$2" role="$3" data="$4" sub="$5"; CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/evaluate_toolsandbox_v43_preference.py --method "$method" --model "$MODEL" --adapter "$OUT/$method/adapter" --run-summary "$OUT/$method/run_summary.json" --protocol-lock "$LOCK" --public-events "$data/events.public.jsonl" --private-outcomes "$data/outcomes.private.jsonl" --evaluation-role "$role" --max-length 2048 --fp32 --output-dir "$OUT/$method/$sub" > "$OUT/$method/${sub}_console.log" 2>&1; }
eval_one control "${G[0]}" development "$V45/development_data" dev_eval & E1=$!
eval_one v46 "${G[1]}" development "$V45/development_data" dev_eval & E2=$!
set +e; wait "$E1"; S1=$?; wait "$E2"; S2=$?; set -e
[ "$S1" -eq 0 ] && [ "$S2" -eq 0 ] || exit 1
set +e
"$PY" scripts/check_toolsandbox_v46_gate.py --mask-eval "$V45/mask/dev_eval/eval_summary.json" --control-eval "$OUT/control/dev_eval/eval_summary.json" --v46-eval "$OUT/v46/dev_eval/eval_summary.json" --control-run "$OUT/control/run_summary.json" --v46-run "$OUT/v46/run_summary.json" --mask-results "$V45/mask/dev_eval/task_results.jsonl" --control-results "$OUT/control/dev_eval/task_results.jsonl" --v46-results "$OUT/v46/dev_eval/task_results.jsonl" --protocol-lock "$LOCK" --output "$OUT/development_gate.json" | tee "$OUT/gate_console.log"
STATUS=${PIPESTATUS[0]}; set -e
[ "$STATUS" -eq 0 ] && echo TOOLSANDBOX_V46_DEVELOPMENT_GATE_PASS || echo TOOLSANDBOX_V46_DEVELOPMENT_GATE_FAIL
# Known confirmation outcomes are scored only as a post-hoc diagnostic and never gate V4.6.
eval_one control "${G[0]}" confirmation "$V45/confirmation_data" posthoc_confirm & E1=$!
eval_one v46 "${G[1]}" confirmation "$V45/confirmation_data" posthoc_confirm & E2=$!
wait "$E1"; wait "$E2"
echo TOOLSANDBOX_V46_FINISHED
exit "$STATUS"
