#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash


PROTOCOL_STATUS = "frozen_before_toolsandbox_v5_router_training"
CONFIG = {
    "seed": 42,
    "max_length": 2048,
    "fp32": True,
    "projection_dim": 128,
    "projection_seed": 20260720,
    "group_folds": 5,
    "head_steps": 800,
    "head_learning_rate": 0.03,
    "head_weight_decay": 0.01,
    "threshold_candidates": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "min_oof_flips": 5,
}
THRESHOLDS = {
    "min_events": 100,
    "min_router_flips": 3,
    "require_oof_improvement": True,
    "require_overall_improvement": True,
    "require_margin_control_improvement": True,
    "require_rescue_noninferiority": True,
    "require_reverse_improvement": True,
    "require_wins_over_losses": True,
    "require_terminal_noninferiority": True,
    "require_progress_noninferiority": True,
}
SOURCE_PATHS = (
    "rescuecredit/frozen_bank.py",
    "rescuecredit/toolsandbox_preference.py",
    "rescuecredit/toolsandbox_router.py",
    "scripts/freeze_toolsandbox_v5_protocol.py",
    "scripts/build_toolsandbox_v5_features.py",
    "scripts/train_toolsandbox_v5_router.py",
    "scripts/score_toolsandbox_v5_router.py",
    "scripts/evaluate_toolsandbox_v5_router.py",
    "scripts/check_toolsandbox_v5_gate.py",
    "scripts/cloud/run_toolsandbox_v5_seed42.sh",
    "refine-logs/TOOLSANDBOX_V5_PLAN.md",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v44-data", type=Path, required=True)
    parser.add_argument("--v45-root", type=Path, required=True)
    parser.add_argument("--v46-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    train_path = args.v44_data / "train.jsonl"
    data_manifest_path = args.v44_data / "manifest.json"
    mask_run_path = args.v45_root / "mask/run_summary.json"
    mask_adapter = args.v45_root / "mask/adapter"
    v46_gate_path = args.v46_root / "development_gate.json"
    v46_run_path = args.v46_root / "v46/run_summary.json"
    dev_data = args.v45_root / "development_data"
    confirm_data = args.v45_root / "confirmation_data"
    required = [
        train_path,
        data_manifest_path,
        mask_run_path,
        v46_gate_path,
        v46_run_path,
        dev_data / "manifest.json",
        confirm_data / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    missing += [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing)

    rows = read_jsonl(train_path)
    data_manifest = load(data_manifest_path)
    mask_run = load(mask_run_path)
    v46_gate = load(v46_gate_path)
    v46_run = load(v46_run_path)
    checks = {
        "v44_training_data_bound": len(rows) == 126
        and data_manifest.get("status") == "frozen"
        and data_manifest.get("train_sha256") == file_sha256(train_path)
        and data_manifest.get("event_set_hash") == event_set_hash(rows),
        "frozen_mask_bound": mask_run.get("status") == "completed"
        and mask_run.get("method") == "mask"
        and mask_run.get("adapter_sha256") == directory_sha256(mask_adapter),
        "v46_negative_result_preserved": v46_gate.get("passed") is False
        and abs(float(v46_gate.get("v46_vs_mask", 1.0))) <= 1e-12
        and v46_gate.get("outcome_checks", {}).get("reverse_improvement") is False
        and v46_run.get("routing_counts") == {"correct": 186, "preserve": 192},
        "known_evaluation_sets_are_development_only": load(
            dev_data / "manifest.json"
        ).get("role")
        == "evaluation"
        and load(confirm_data / "manifest.json").get("role") == "evaluation",
    }
    if not all(checks.values()):
        raise RuntimeError(checks)

    protocol = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_v5_frozen_causal_router_seed42",
        "methods": ["mask", "margin_control", "causal_router_v5"],
        "checks": checks,
        "config": CONFIG,
        "thresholds": THRESHOLDS,
        "train_events": len(rows),
        "train_sha256": file_sha256(train_path),
        "train_event_set_hash": event_set_hash(rows),
        "train_manifest_sha256": file_sha256(data_manifest_path),
        "mask_run_sha256": file_sha256(mask_run_path),
        "mask_adapter_sha256": directory_sha256(mask_adapter),
        "base_model_sha256": directory_sha256(args.model),
        "v46_gate_sha256": file_sha256(v46_gate_path),
        "v46_run_sha256": file_sha256(v46_run_path),
        "development": {
            "data_dir": str(dev_data),
            "manifest_sha256": file_sha256(dev_data / "manifest.json"),
        },
        "posthoc_confirmation": {
            "data_dir": str(confirm_data),
            "manifest_sha256": file_sha256(confirm_data / "manifest.json"),
            "gating_role": False,
        },
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "reference_boundary": "router features contain only frozen Mask representations, Mask margin, visible prompts, public schemas, and candidates; causal direction is a train-only label; evaluation outcomes join after public-only scoring",
        "scope": "development-only ToolSandbox independent causal-router diagnostic; no fresh confirmation or autonomous task-success claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
