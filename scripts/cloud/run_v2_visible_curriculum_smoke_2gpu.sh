#!/usr/bin/env bash
set -euo pipefail

cd /data/hxy/projects/RescueCredit
source data_disk_env.sh
source .venv/bin/activate

export MODEL="${MODEL:-/data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=data/api_bank_controlled_reference_independent_v1
OUT=outputs/rescuecredit_v2_visible_curriculum_smoke_seed42

if [[ ! -f "$DATA/manifest.json" ]]; then
  python scripts/prepare_api_bank_controlled.py --output-dir "$DATA"
fi

if [[ -d "$OUT" && ! -f "$OUT/run_summary.json" ]]; then
  mv "$OUT" "${OUT}_failed_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT"

accelerate launch \
  --config_file configs/accelerate_h200.yaml \
  --num_processes 2 \
  scripts/run_train.py \
  --method rescuecredit_v2 \
  --model "$MODEL" \
  --harness-generator-model "$MODEL" \
  --train-file "$DATA/train.jsonl" \
  --manifest "$DATA/manifest.json" \
  --seed 42 \
  --max-updates 10000 \
  --budget-mode main \
  --main-interaction-budget 512 \
  --total-interaction-budget 50000 \
  --group-size 4 \
  --max-new-tokens 64 \
  --max-shadow-steps 12 \
  --policy-epochs 1 \
  --learning-rate 2e-6 \
  --audit-probability 1.0 \
  --audit-warm-start-events 0 \
  --lambda-corr 0.1 \
  --lambda-causal 0.1 \
  --preference-beta 1.0 \
  --max-causal-weight 2.5 \
  --visible-curriculum-fraction 0.75 \
  --use-lora \
  --fp32 \
  --save-every 1000 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

python - <<'PY'
import json
from collections import Counter
from pathlib import Path

root = Path("outputs/rescuecredit_v2_visible_curriculum_smoke_seed42")
summary = json.loads((root / "run_summary.json").read_text())
events = []
for path in root.glob("preference_events_rank*.jsonl"):
    events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
nonzero = [
    event for event in events
    if event.get("causal_direction") is not None
    and abs(float(event.get("causal_weight") or 0.0)) > 1e-12
]
decisions = Counter(str(event.get("causal_decision")) for event in events)
audit = summary.get("audit_stats", {})
sampling = summary.get("sampling", {})
gate = {
    "passed": bool(
        summary.get("status") == "completed"
        and summary.get("main_steps", 0) >= 512
        and summary.get("failed_replay_steps", 0) == 0
        and sampling.get("reference_free_selection") is True
        and sampling.get("visible_curriculum_fraction") == 0.75
        and audit.get("eligible_events", 0) >= 5
        and audit.get("valid_audits", 0) >= 5
        and len(nonzero) >= 3
    ),
    "main_steps": summary.get("main_steps"),
    "shadow_steps": summary.get("shadow_steps"),
    "failed_replay_steps": summary.get("failed_replay_steps"),
    "sampling": sampling,
    "audit_stats": audit,
    "preference_events": len(events),
    "nonzero_causal_events": len(nonzero),
    "causal_decisions": dict(decisions),
}
(root / "smoke_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
print(json.dumps(gate, indent=2))
PY

echo V2_VISIBLE_CURRICULUM_SMOKE_FINISHED
