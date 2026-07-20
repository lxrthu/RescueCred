#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_active_shadow import minimum_zero_harm_calibration_size


PROTOCOL_STATUS = "frozen_before_toolsandbox_v7_active_shadow"
CONFIG = {
    "seed": 42,
    "group_folds": 5,
    "hash_dimension": 256,
    "head_steps": 1000,
    "head_learning_rate": 0.003,
    "head_weight_decay": 0.03,
    "hidden_dim": 64,
    "calibration_steps": 500,
    "calibration_learning_rate": 0.02,
    "max_probe_rate": 0.30,
    "rescue_delta": 0.02,
    "risk_alpha": 0.05,
    "route_threshold_candidates": [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.975,
        0.99,
    ],
}
GATE = {
    "min_active_cross_task_roc_auc": 0.75,
    "max_empirical_rescue_drop": 0.02,
    "min_reverse_recall": 0.20,
    "max_probe_rate": 0.30,
    "require_active_auc_above_static": True,
}
SOURCE_PATHS = (
    "rescuecredit/toolsandbox_active_shadow.py",
    "rescuecredit/toolsandbox_selective_router.py",
    "scripts/freeze_toolsandbox_v7_protocol.py",
    "scripts/build_toolsandbox_v7_features.py",
    "scripts/train_toolsandbox_v7_active_shadow.py",
    "scripts/check_toolsandbox_v7_gate.py",
    "scripts/cloud/run_toolsandbox_v7_active_shadow_seed42.sh",
    "refine-logs/TOOLSANDBOX_V7_ACTIVESHADOW_PLAN.md",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v44-root", type=Path, required=True)
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    raw_events = args.v44_root / "full_offset85_h8/candidate_events.jsonl"
    raw_summary_path = args.v44_root / "full_offset85_h8/audit_summary.json"
    train_file = args.v44_root / "data/train.jsonl"
    train_manifest_path = args.v44_root / "data/manifest.json"
    v5_cache = args.v5_root / "features/train_features.pt"
    v5_manifest_path = args.v5_root / "features/feature_manifest.json"
    v6_gate_path = args.v6_root / "development_gate.json"
    required = [
        raw_events,
        raw_summary_path,
        train_file,
        train_manifest_path,
        v5_cache,
        v5_manifest_path,
        v6_gate_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    missing += [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing)

    raw_summary = load(raw_summary_path)
    train_manifest = load(train_manifest_path)
    v5_manifest = load(v5_manifest_path)
    v6_gate = load(v6_gate_path)
    checks = {
        "raw_shadow_events_bound": raw_summary.get("event_file_sha256")
        == file_sha256(raw_events),
        "v44_train_derived_from_raw": train_manifest.get("source_event_sha256")
        == file_sha256(raw_events)
        and train_manifest.get("train_sha256") == file_sha256(train_file),
        "v44_snapshot_exact": raw_summary.get("snapshot_audit", {}).get("exact")
        is True,
        "v5_static_features_public_only": v5_manifest.get(
            "private_branch_outcomes_cached"
        )
        is False
        and v5_manifest.get("feature_file_sha256") == file_sha256(v5_cache),
        "v6_static_diagnostic_preserved": v6_gate.get("passed") is False
        and float(v6_gate.get("semantic_cross_task_roc_auc", 1.0)) < 0.70,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)

    minimum_rescue = minimum_zero_harm_calibration_size(
        CONFIG["rescue_delta"], alpha=CONFIG["risk_alpha"]
    )
    protocol = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_v7_active_shadow_seed42",
        "checks": checks,
        "config": CONFIG,
        "gate": GATE,
        "raw_events": str(raw_events),
        "raw_events_sha256": file_sha256(raw_events),
        "raw_summary_sha256": file_sha256(raw_summary_path),
        "train_file": str(train_file),
        "train_file_sha256": file_sha256(train_file),
        "train_manifest_sha256": file_sha256(train_manifest_path),
        "v5_feature_cache": str(v5_cache),
        "v5_feature_cache_sha256": file_sha256(v5_cache),
        "v5_feature_manifest_sha256": file_sha256(v5_manifest_path),
        "v6_gate_sha256": file_sha256(v6_gate_path),
        "minimum_zero_harm_rescue_calibration_events": minimum_rescue,
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "router_inputs": [
            "candidate A and Harness correction B",
            "first visible receipt after isolated A execution",
            "first visible receipt after isolated B execution",
            "static public-only V5 representation for probe acquisition",
        ],
        "prohibited_router_inputs": [
            "reference actions",
            "full-trajectory branch outcomes",
            "official score_trace",
            "progress_auc",
            "ending_context_digest",
            "decision label at deployment",
        ],
        "scope": "cross-task ActiveShadow feasibility on existing V4.4 one-step receipts; formal two-point risk certification requires an enlarged independent calibration set",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
