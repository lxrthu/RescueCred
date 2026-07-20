#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256
from rescuecredit.logging import write_json


PROTOCOL_STATUS = "frozen_before_receipt_horizon_statistical_audit"
CONFIG = {
    "seed": 42,
    "bootstrap_replicates": 20000,
    "permutation_replicates": 20000,
    "alpha": 0.05,
    "resampling_unit": "task_id_hash",
    "comparison": "V9 two-step minus V7 one-step",
}
SOURCE_PATHS = (
    "rescuecredit/paired_task_statistics.py",
    "scripts/freeze_toolsandbox_receipt_horizon_stats.py",
    "scripts/analyze_toolsandbox_receipt_horizon_stats.py",
    "scripts/cloud/run_toolsandbox_receipt_horizon_stats_seed42.sh",
    "refine-logs/TOOLSANDBOX_RECEIPT_HORIZON_STATS_PLAN.md",
)
EXPECTED_INTEGRITY_KEYS = {
    "checkpoint_bound",
    "cross_task_group_isolation",
    "feature_cache_bound",
    "feature_manifest_bound",
    "frozen_policy_unchanged",
    "metrics_recomputed",
    "oof_predictions_bound",
    "protocol_frozen",
    "public_active_features_only",
    "run_bound",
    "source_identity",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--v9-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = {
        "v7_gate": args.v7_root / "feasibility_gate.json",
        "v7_protocol": args.v7_root / "protocol_lock.json",
        "v7_summary": args.v7_root / "model/run_summary.json",
        "v7_oof": args.v7_root / "model/oof_predictions.jsonl",
        "v9_gate": args.v9_root / "feasibility_gate.json",
        "v9_protocol": args.v9_root / "protocol_lock.json",
        "v9_summary": args.v9_root / "model/run_summary.json",
        "v9_oof": args.v9_root / "model/oof_predictions.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    missing += [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    v7_gate = load(paths["v7_gate"])
    v9_gate = load(paths["v9_gate"])
    v7_summary = load(paths["v7_summary"])
    v9_summary = load(paths["v9_summary"])
    v7_integrity = v7_gate.get("integrity_checks")
    v9_integrity = v9_gate.get("integrity_checks")
    checks = {
        "v7_gate_bound": v7_gate.get("stage")
        == "toolsandbox_v7_active_shadow_feasibility_gate_seed42"
        and v7_gate.get("passed") is False
        and isinstance(v7_integrity, dict)
        and set(v7_integrity) == EXPECTED_INTEGRITY_KEYS
        and all(v7_integrity.values()),
        "v9_gate_bound": v9_gate.get("stage")
        == "toolsandbox_v9_two_step_feasibility_gate_seed42"
        and v9_gate.get("feature_variant") == "first_two_branch_receipts"
        and v9_gate.get("passed") is False
        and isinstance(v9_integrity, dict)
        and set(v9_integrity) == EXPECTED_INTEGRITY_KEYS
        and all(v9_integrity.values()),
        "v7_summary_bound": v7_summary.get("status") == "completed"
        and v7_summary.get("oof_predictions_sha256") == file_sha256(paths["v7_oof"])
        and v7_summary.get("protocol_lock_sha256")
        == file_sha256(paths["v7_protocol"]),
        "v9_summary_bound": v9_summary.get("status") == "completed"
        and v9_summary.get("oof_predictions_sha256") == file_sha256(paths["v9_oof"])
        and v9_summary.get("protocol_lock_sha256")
        == file_sha256(paths["v9_protocol"]),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    protocol = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_receipt_horizon_statistical_audit_seed42",
        "checks": checks,
        "config": CONFIG,
        "artifact_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "decision_rule": {
            "two_step_better": "observed AUC delta > 0, 95% task-bootstrap lower bound > 0, and one-sided task-swap p <= 0.05",
            "two_step_worse": "observed AUC delta < 0, 95% task-bootstrap upper bound < 0, and one-sided task-swap p <= 0.05",
            "otherwise": "no significant two-step difference",
        },
        "scope": "paired task-level uncertainty audit; descriptive closure only; no model selection or claim expansion",
        "uncertainty_scope": "conditional on frozen OOF predictions; excludes algorithm-retraining and broader population uncertainty",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
