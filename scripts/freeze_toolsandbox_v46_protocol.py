#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash, matched_epoch_order
from scripts.train_toolsandbox_v43_preference import V46_PROTOCOL_STATUS

CONFIG = {
    "seed": 42,
    "epochs": 3,
    "learning_rate": 3e-6,
    "gradient_accumulation": 8,
    "max_length": 2048,
    "beta": 1.0,
    "target_residual": 0.05,
    "confidence_margin": 0.05,
    "retention_coef": 1.0,
    "reference_anchor_coef": 0.25,
    "fp32": True,
    "sampling": "all_unique_events_identical_order",
}
THRESHOLDS = {
    "min_events": 100,
    "min_disagreements": 3,
    "min_reverse_margin_decrease": 0.02,
    "require_rescue_noninferiority": True,
    "require_reverse_improvement": True,
    "require_overall_improvement": True,
    "require_wins_over_losses": True,
}
SOURCE_PATHS = (
    "rescuecredit/frozen_bank.py",
    "rescuecredit/toolsandbox_preference.py",
    "scripts/train_route_a_preference.py",
    "scripts/train_toolsandbox_v43_preference.py",
    "scripts/train_toolsandbox_v46_residual.py",
    "scripts/evaluate_toolsandbox_v43_preference.py",
    "scripts/freeze_toolsandbox_v46_protocol.py",
    "scripts/check_toolsandbox_v46_gate.py",
    "scripts/cloud/run_toolsandbox_v46_seed42.sh",
    "refine-logs/TOOLSANDBOX_V46_PLAN.md",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v44-data", type=Path, required=True)
    p.add_argument("--v45-root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    train = a.v44_data / "train.jsonl"
    manifest = a.v44_data / "manifest.json"
    mask_run = a.v45_root / "mask/run_summary.json"
    mask_adapter = a.v45_root / "mask/adapter"
    dev_gate = a.v45_root / "development_gate.json"
    confirm_gate = a.v45_root / "confirmation_gate.json"
    dev_data = a.v45_root / "development_data"
    confirm_data = a.v45_root / "confirmation_data"
    required = [
        train,
        manifest,
        mask_run,
        dev_gate,
        confirm_gate,
        dev_data / "manifest.json",
        confirm_data / "manifest.json",
    ]
    missing = [str(x) for x in required if not x.exists()] + [
        x for x in SOURCE_PATHS if not Path(x).is_file()
    ]
    if missing:
        raise FileNotFoundError(missing)
    rows = read_jsonl(train)
    man, run = load(manifest), load(mask_run)
    dg, cg = load(dev_gate), load(confirm_gate)
    sequence = [
        row for epoch in range(3) for row in matched_epoch_order(rows, 42, epoch)
    ]
    seq_hash = hashlib.sha256(
        "\n".join(str(r["event_id"]) for r in sequence).encode()
    ).hexdigest()
    checks = {
        "v44_data_bound": man.get("status") == "frozen"
        and man.get("events") == 126
        and man.get("train_sha256") == file_sha256(train)
        and man.get("event_set_hash") == event_set_hash(rows),
        "common_mask_bound": run.get("method") == "mask"
        and run.get("status") == "completed"
        and run.get("adapter_sha256") == directory_sha256(mask_adapter),
        "v45_negative_result_preserved": dg.get("passed") is True
        and cg.get("passed") is False
        and cg.get("causal_accuracy_improvement", 0) < 0,
        "known_development_only": load(dev_data / "manifest.json").get("role")
        == "evaluation"
        and load(confirm_data / "manifest.json").get("role") == "evaluation",
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    protocol = {
        "status": V46_PROTOCOL_STATUS,
        "stage": "toolsandbox_v46_selective_residual_seed42",
        "methods": ["control", "v46"],
        "checks": checks,
        "config": CONFIG,
        "thresholds": THRESHOLDS,
        "train_events": len(rows),
        "train_sha256": file_sha256(train),
        "train_event_set_hash": event_set_hash(rows),
        "mask_run_sha256": file_sha256(mask_run),
        "mask_adapter_sha256": directory_sha256(mask_adapter),
        "v45_development_gate_sha256": file_sha256(dev_gate),
        "v45_confirmation_gate_sha256": file_sha256(confirm_gate),
        "base_model_sha256": directory_sha256(a.model),
        "development": {
            "data_dir": str(dev_data),
            "manifest_sha256": file_sha256(dev_data / "manifest.json"),
        },
        "posthoc_confirmation": {
            "data_dir": str(confirm_data),
            "manifest_sha256": file_sha256(confirm_data / "manifest.json"),
            "gating_role": False,
        },
        "source_sha256": {x: file_sha256(Path(x)) for x in SOURCE_PATHS},
        "expected_presented_event_sequence_sha256": seq_hash,
        "reference_boundary": "V4.6 routing uses frozen train causal labels and frozen Mask margins; known V4.5 evaluation outcomes never enter training",
        "scope": "development-only ToolSandbox selective residual diagnostic; no fresh confirmation claim",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
