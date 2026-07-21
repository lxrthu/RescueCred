#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${RESCUECREDIT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
MODEL="${RESCUECREDIT_MODEL:-/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct}"
V44="${EDITCREDIT_V44_ROOT:-$REPO_ROOT/outputs/toolsandbox_v44_candidate_diversity_seed42}"
TRAIN_FILE="$V44/data/train.jsonl"
DATA_MANIFEST="$V44/data/manifest.json"
DATA_GATE="$V44/data/data_gate.json"
OUT="${EDITCREDIT_OUTPUT:-$REPO_ROOT/outputs/toolsandbox_editcredit_seed42}"
LOCK="$OUT/protocol_lock.json"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
test -x "$PYTHON"
test -d "$MODEL"
test -f "$TRAIN_FILE"
test -f "$DATA_MANIFEST"
test -f "$DATA_GATE"
if [ -e "$OUT" ]; then
  echo "Refusing to reuse EditCredit output root: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

"$PYTHON" -m py_compile rescuecredit/edit_credit.py \
  scripts/freeze_toolsandbox_editcredit_protocol.py \
  scripts/audit_editcredit_gradients.py \
  scripts/audit_toolsandbox_editcredit_gradients.py \
  scripts/check_toolsandbox_editcredit_variance.py \
  scripts/check_toolsandbox_editcredit_efficiency.py \
  scripts/train_toolsandbox_editcredit.py \
  scripts/evaluate_toolsandbox_editcredit.py \
  scripts/check_toolsandbox_editcredit_gate.py
"$PYTHON" -m pytest -q tests/test_edit_credit.py tests/test_editcredit_gate.py \
  tests/test_editcredit_efficiency.py \
  tests/test_toolsandbox_preference.py \
  | tee "$OUT/sanity.log"
"$PYTHON" scripts/audit_editcredit_gradients.py \
  --output "$OUT/gradient_sanity.json" | tee "$OUT/gradient_sanity.log"

"$PYTHON" scripts/freeze_toolsandbox_editcredit_protocol.py \
  --train-file "$TRAIN_FILE" --data-manifest "$DATA_MANIFEST" \
  --data-gate "$DATA_GATE" --gradient-sanity "$OUT/gradient_sanity.json" \
  --model "$MODEL" --output "$LOCK" \
  | tee "$OUT/freeze.log"

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2);print $2,$1}' \
    | sort -n | head -n2 | awk '{print $2}'
)
[ "${#GPUS[@]}" -ge 2 ] || { echo "EditCredit requires two visible GPUs" >&2; exit 2; }
echo "FULL_ACTION_GPU=${GPUS[0]} EDITCREDIT_GPU=${GPUS[1]}"

gradient_audit() {
  local method="$1" gpu="$2" directory="$OUT/gradient/$method"
  mkdir -p "$directory"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/audit_toolsandbox_editcredit_gradients.py \
    --method "$method" --model "$MODEL" --train-file "$TRAIN_FILE" \
    --protocol-lock "$LOCK" --seed 42 --max-length 2048 --buckets 128 \
    --output-dir "$directory" > "$directory/console.log" 2>&1
}

echo EDITCREDIT_GRADIENT_AUDIT_START
gradient_audit full_action "${GPUS[0]}" & G1=$!
gradient_audit editcredit "${GPUS[1]}" & G2=$!
trap 'kill "$G1" "$G2" 2>/dev/null || true' EXIT INT TERM
set +e
wait "$G1"; S1=$?
wait "$G2"; S2=$?
set -e
trap - EXIT INT TERM
[ "$S1" -eq 0 ] && [ "$S2" -eq 0 ] || {
  echo "gradient audit failed full_action=$S1 editcredit=$S2" >&2
  exit 1
}
set +e
"$PYTHON" scripts/check_toolsandbox_editcredit_variance.py \
  --protocol-lock "$LOCK" --train-file "$TRAIN_FILE" \
  --full-summary "$OUT/gradient/full_action/summary.json" \
  --full-sketches "$OUT/gradient/full_action/gradient_sketches.jsonl" \
  --edit-summary "$OUT/gradient/editcredit/summary.json" \
  --edit-sketches "$OUT/gradient/editcredit/gradient_sketches.jsonl" \
  --output "$OUT/variance_gate.json" | tee "$OUT/variance_gate.log"
VARIANCE_STATUS=${PIPESTATUS[0]}
set -e
echo "EDITCREDIT_VARIANCE_GATE_STATUS=$VARIANCE_STATUS"

train_fold() {
  local method="$1" fold="$2" gpu="$3" directory="$OUT/$method/fold$fold"
  mkdir -p "$directory"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_toolsandbox_editcredit.py \
    --method "$method" --fold "$fold" --model "$MODEL" \
    --train-file "$TRAIN_FILE" --protocol-lock "$LOCK" \
    --seed 42 --folds 5 --epochs 3 --learning-rate 3e-6 \
    --gradient-accumulation 8 --max-length 2048 --beta 1.0 \
    --absolute-margin-coef 1.0 --target-margin 0.05 \
    --reference-anchor-coef 0.25 --presentations-per-epoch 126 \
    --lora-r 16 --lora-alpha 32 --fp32 --rescue-delta 0.02 \
    --output-dir "$directory" > "$directory/train_console.log" 2>&1
}

eval_fold() {
  local method="$1" fold="$2" gpu="$3" directory="$OUT/$method/fold$fold"
  mkdir -p "$directory/eval"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/evaluate_toolsandbox_editcredit.py \
    --method "$method" --fold "$fold" --model "$MODEL" \
    --adapter "$directory/adapter" --run-summary "$directory/run_summary.json" \
    --train-file "$TRAIN_FILE" --protocol-lock "$LOCK" \
    --max-length 2048 --fp32 --output-dir "$directory/eval" \
    > "$directory/eval_console.log" 2>&1
}

eval_curve() {
  local method="$1" fold="$2" gpu="$3" presentations="$4"
  local directory="$OUT/$method/fold$fold"
  local tag; printf -v tag 'p%06d' "$presentations"
  local curve="$directory/curve/$tag"
  mkdir -p "$curve"
  local checkpoint_args=()
  if [ "$presentations" -eq 0 ]; then
    checkpoint_args=(--base-only --checkpoint-presentations 0)
  else
    checkpoint_args=(
      --adapter "$directory/checkpoints/$tag/adapter"
      --checkpoint-presentations "$presentations"
    )
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/evaluate_toolsandbox_editcredit.py \
    --method "$method" --fold "$fold" --model "$MODEL" \
    "${checkpoint_args[@]}" --run-summary "$directory/run_summary.json" \
    --train-file "$TRAIN_FILE" --protocol-lock "$LOCK" \
    --max-length 2048 --fp32 --output-dir "$curve" \
    > "$curve/console.log" 2>&1
}

for fold in 0 1 2 3 4; do
  echo "EDITCREDIT_FOLD_${fold}_TRAIN_START"
  train_fold full_action "$fold" "${GPUS[0]}" & P1=$!
  train_fold editcredit "$fold" "${GPUS[1]}" & P2=$!
  trap 'kill "$P1" "$P2" 2>/dev/null || true' EXIT INT TERM
  set +e
  wait "$P1"; S1=$?
  wait "$P2"; S2=$?
  set -e
  trap - EXIT INT TERM
  [ "$S1" -eq 0 ] && [ "$S2" -eq 0 ] || {
    echo "fold $fold training failed full_action=$S1 editcredit=$S2" >&2
    exit 1
  }

  echo "EDITCREDIT_FOLD_${fold}_EVAL_START"
  eval_fold full_action "$fold" "${GPUS[0]}" & E1=$!
  eval_fold editcredit "$fold" "${GPUS[1]}" & E2=$!
  trap 'kill "$E1" "$E2" 2>/dev/null || true' EXIT INT TERM
  set +e
  wait "$E1"; S1=$?
  wait "$E2"; S2=$?
  set -e
  trap - EXIT INT TERM
  [ "$S1" -eq 0 ] && [ "$S2" -eq 0 ] || {
    echo "fold $fold evaluation failed full_action=$S1 editcredit=$S2" >&2
    exit 1
  }
  for presentations in 0 40 80 128 256; do
    echo "EDITCREDIT_FOLD_${fold}_CURVE_${presentations}_START"
    eval_curve full_action "$fold" "${GPUS[0]}" "$presentations" & C1=$!
    eval_curve editcredit "$fold" "${GPUS[1]}" "$presentations" & C2=$!
    trap 'kill "$C1" "$C2" 2>/dev/null || true' EXIT INT TERM
    set +e
    wait "$C1"; S1=$?
    wait "$C2"; S2=$?
    set -e
    trap - EXIT INT TERM
    [ "$S1" -eq 0 ] && [ "$S2" -eq 0 ] || {
      echo "fold $fold curve $presentations failed full_action=$S1 editcredit=$S2" >&2
      exit 1
    }
  done
  echo "EDITCREDIT_FOLD_${fold}_DONE"
done

set +e
"$PYTHON" scripts/check_toolsandbox_editcredit_gate.py \
  --protocol-lock "$LOCK" --root "$OUT" --train-file "$TRAIN_FILE" \
  --data-manifest "$DATA_MANIFEST" --data-gate "$DATA_GATE" \
  --gradient-sanity "$OUT/gradient_sanity.json" \
  --output "$OUT/feasibility_gate.json" \
  | tee "$OUT/gate_console.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -eq 0 ]; then
  echo EDITCREDIT_SEED42_GATE_PASS
else
  echo EDITCREDIT_SEED42_GATE_FAIL
fi
echo "EDITCREDIT_FINAL_GATE_STATUS=$STATUS"

set +e
"$PYTHON" scripts/check_toolsandbox_editcredit_efficiency.py \
  --protocol-lock "$LOCK" --train-file "$TRAIN_FILE" --root "$OUT" \
  --variance-gate "$OUT/variance_gate.json" --final-gate "$OUT/feasibility_gate.json" \
  --output "$OUT/efficiency_gate.json" | tee "$OUT/efficiency_gate.log"
EFFICIENCY_STATUS=${PIPESTATUS[0]}
set -e
[ "$EFFICIENCY_STATUS" -eq 0 ] \
  && echo EDITCREDIT_EFFICIENCY_GATE_PASS \
  || echo EDITCREDIT_EFFICIENCY_GATE_FAIL
exit "$EFFICIENCY_STATUS"
