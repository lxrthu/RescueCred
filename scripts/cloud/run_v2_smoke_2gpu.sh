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

OUT=outputs/rescuecredit_v2_smoke_seed42
mkdir -p "$OUT"

accelerate launch \
  --config_file configs/accelerate_h200.yaml \
  --num_processes 2 \
  scripts/run_train.py \
  --method rescuecredit_v2 \
  --model "$MODEL" \
  --harness-generator-model "$MODEL" \
  --seed 42 \
  --max-updates 10000 \
  --budget-mode main \
  --main-interaction-budget 256 \
  --group-size 4 \
  --max-new-tokens 64 \
  --policy-epochs 1 \
  --learning-rate 2e-6 \
  --audit-probability 1.0 \
  --audit-warm-start-events 0 \
  --lambda-corr 0.1 \
  --lambda-causal 0.1 \
  --preference-beta 1.0 \
  --max-causal-weight 2.5 \
  --use-lora \
  --fp32 \
  --save-every 1000 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/rescuecredit_v2_smoke_seed42")
summary = json.loads((root / "run_summary.json").read_text())
events = []
for path in root.glob("preference_events_rank*.jsonl"):
    events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
gate = {
    "passed": bool(
        summary.get("status") == "completed"
        and summary.get("main_steps", 0) >= 256
        and summary.get("failed_replay_steps", 0) == 0
        and summary.get("comparability", {}).get("harness_mode") == "deployable"
        and events
        and any(float(event.get("causal_loss", 0.0)) > 0 for event in events)
    ),
    "main_steps": summary.get("main_steps"),
    "shadow_steps": summary.get("shadow_steps"),
    "budget_overshoot": summary.get("budget_overshoot"),
    "failed_replay_steps": summary.get("failed_replay_steps"),
    "preference_events": len(events),
    "nonzero_causal_events": sum(float(event.get("causal_loss", 0.0)) > 0 for event in events),
    "causal_decisions": sorted({event.get("causal_decision") for event in events}),
}
(root / "smoke_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
print(json.dumps(gate, indent=2))
PY
