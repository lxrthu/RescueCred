#!/usr/bin/env python3
from __future__ import annotations

import base64
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "dist/rescuecredit_h0_shadow_paste_bundle.tar.gz"
OUTPUT = ROOT / "dist/REMOTE_APPLY_H0_SHADOW_FIX.txt"

payload = base64.b64encode(BUNDLE.read_bytes()).decode("ascii")
content = f"""cd /data/hxy/projects/RescueCredit
printf '%s' '{payload}' | base64 -d > /tmp/rescuecredit_h0_shadow_bundle.tar.gz
tar -xzf /tmp/rescuecredit_h0_shadow_bundle.tar.gz
source data_disk_env.sh
source .venv/bin/activate
python scripts/cloud/apply_h0_shadow_hotfix.py
python -m py_compile scripts/run_train.py scripts/evaluate_full_shadow.py
chmod +x scripts/cloud/run_v2_causal_subset_smoke_2gpu.sh
tmux kill-session -t rescue_v2_h0 2>/dev/null || true
tmux new-session -d -s rescue_v2_h0 "cd /data/hxy/projects/RescueCredit && CUDA_VISIBLE_DEVICES=1,3 bash scripts/cloud/run_v2_causal_subset_smoke_2gpu.sh"
sleep 5
tail -f outputs/rescuecredit_v2_causal_smoke_h0_shadow_seed42/console.log

# Run finishes in about 20-30 minutes. Press Ctrl+C, then run:
cat outputs/rescuecredit_v2_causal_smoke_h0_shadow_seed42/smoke_gate.json
cat outputs/rescuecredit_v2_causal_smoke_h0_shadow_seed42/run_summary.json
"""
OUTPUT.write_text(content, encoding="utf-8", newline="\n")
print(f"WROTE {OUTPUT} bytes={OUTPUT.stat().st_size} payload_chars={len(payload)}")
