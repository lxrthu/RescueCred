#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_selective_router import reverse_target


PROTOCOL_STATUS = "frozen_before_toolsandbox_v6_reverse_diagnostic"
CONFIG = {
    "seed": 42,
    "group_folds": 5,
    "head_steps": 1000,
    "head_learning_rate": 0.003,
    "head_weight_decay": 0.03,
    "hidden_dim": 64,
    "calibration_steps": 500,
    "calibration_learning_rate": 0.02,
    "rescue_delta": 0.02,
    "threshold_candidates": [
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
    "min_cross_task_roc_auc": 0.70,
    "min_pr_auc_lift_over_prevalence": 0.10,
    "min_reverse_recall_at_rescue_budget": 0.10,
    "require_semantic_auc_above_margin_control": True,
    "require_development_rescue_noninferiority": True,
}
SOURCE_PATHS = (
    "rescuecredit/toolsandbox_router.py",
    "rescuecredit/toolsandbox_selective_router.py",
    "scripts/freeze_toolsandbox_v6_protocol.py",
    "scripts/train_toolsandbox_v6_diagnostic.py",
    "scripts/score_toolsandbox_v6_diagnostic.py",
    "scripts/evaluate_toolsandbox_v6_diagnostic.py",
    "scripts/check_toolsandbox_v6_gate.py",
    "scripts/cloud/run_toolsandbox_v6_diagnostic_seed42.sh",
    "refine-logs/TOOLSANDBOX_V6_DIAGNOSTIC_PLAN.md",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    v5_lock_path = args.v5_root / "protocol_lock.json"
    feature_cache = args.v5_root / "features/train_features.pt"
    feature_manifest_path = args.v5_root / "features/feature_manifest.json"
    v5_gate_path = args.v5_root / "development_gate.json"
    required = [v5_lock_path, feature_cache, feature_manifest_path, v5_gate_path]
    missing = [str(path) for path in required if not path.is_file()]
    missing += [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing)

    import torch

    v5_lock = load(v5_lock_path)
    feature_manifest = load(feature_manifest_path)
    v5_gate = load(v5_gate_path)
    cache = torch.load(feature_cache, map_location="cpu", weights_only=True)
    decisions = [str(value) for value in cache["decisions"]]
    labels = [reverse_target(value) for value in decisions]
    task_ids = [str(value) for value in cache["task_ids"]]
    checks = {
        "v5_result_preserved": v5_gate.get("passed") is False,
        "v5_public_feature_cache_bound": feature_manifest.get("private_branch_outcomes_cached")
        is False
        and feature_manifest.get("feature_file_sha256") == file_sha256(feature_cache)
        and feature_manifest.get("protocol_lock_sha256") == file_sha256(v5_lock_path),
        "training_labels_complete": len(labels) == len(task_ids) > 0
        and set(labels) == {0, 1},
        "enough_task_groups": len(set(task_ids)) >= CONFIG["group_folds"],
        "frozen_policy_reused": bool(v5_lock.get("base_model_sha256"))
        and bool(v5_lock.get("mask_adapter_sha256")),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)

    protocol = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_v6_reverse_only_diagnostic_seed42",
        "checks": checks,
        "methods": ["default_b", "margin_probe", "semantic_probe"],
        "config": CONFIG,
        "gate": GATE,
        "training_events": len(labels),
        "training_tasks": len(set(task_ids)),
        "reverse_events": sum(labels),
        "rescue_events": len(labels) - sum(labels),
        "v5_protocol_lock_sha256": file_sha256(v5_lock_path),
        "v5_development_gate_sha256": file_sha256(v5_gate_path),
        "feature_cache": str(feature_cache),
        "feature_cache_sha256": file_sha256(feature_cache),
        "feature_manifest": str(feature_manifest_path),
        "feature_manifest_sha256": file_sha256(feature_manifest_path),
        "base_model_sha256": v5_lock["base_model_sha256"],
        "mask_adapter_sha256": v5_lock["mask_adapter_sha256"],
        "development": v5_lock["development"],
        "posthoc_confirmation": v5_lock["posthoc_confirmation"],
        "feature_config": {
            key: v5_lock["config"][key]
            for key in ("max_length", "fp32", "projection_dim", "projection_seed")
        },
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "deployment_rule": "select A only when calibrated P(Reverse) meets the frozen threshold; otherwise abstain to Harness correction B",
        "reference_boundary": "probe inputs are the V5 public-only frozen representations and margin; Rescue/Reverse outcomes are training labels only and join evaluation after public-only scoring",
        "scope": "cross-task diagnostic and conservative routing test; known development and confirmation sets remain non-confirmatory",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
