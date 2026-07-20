#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_active_shadow import active_decision_metrics
from rescuecredit.toolsandbox_selective_router import roc_auc
from scripts.freeze_toolsandbox_v7_protocol import GATE, PROTOCOL_STATUS


TOL = 1e-12


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load(args.protocol_lock)
    summary = load(args.run_summary)
    manifest = load(args.feature_manifest)
    rows = read_jsonl(args.oof_predictions)
    expected_lock = file_sha256(args.protocol_lock)
    source_ok = bool(protocol.get("source_sha256")) and all(
        Path(path).is_file() and file_sha256(Path(path)) == digest
        for path, digest in protocol.get("source_sha256", {}).items()
    )
    labels = [int(row["label"]) for row in rows]
    static_scores = [float(row["static_score"]) for row in rows]
    active_raw_scores = [float(row["active_raw_score"]) for row in rows]
    probed = [bool(row["probed"]) for row in rows]
    routed_to_a = [bool(row["routed_to_a"]) for row in rows]
    recomputed = active_decision_metrics(
        labels,
        probed,
        routed_to_a,
        alpha=float(protocol["config"]["risk_alpha"]),
    )
    task_folds: dict[str, set[int]] = {}
    for row in rows:
        task_folds.setdefault(str(row["task_id_hash"]), set()).add(
            int(row["fold_id"])
        )
    integrity = {
        "protocol_frozen": protocol.get("status") == PROTOCOL_STATUS
        and protocol.get("gate") == GATE,
        "source_identity": source_ok,
        "run_bound": summary.get("protocol_lock_sha256") == expected_lock,
        "feature_cache_bound": summary.get("feature_cache_sha256")
        == manifest.get("feature_cache_sha256")
        == file_sha256(args.feature_cache),
        "feature_manifest_bound": summary.get("feature_manifest_sha256")
        == file_sha256(args.feature_manifest),
        "checkpoint_bound": summary.get("checkpoint_sha256")
        == file_sha256(args.checkpoint),
        "oof_predictions_bound": summary.get("oof_predictions_sha256")
        == file_sha256(args.oof_predictions),
        "cross_task_group_isolation": all(
            fold.get("task_overlap") == 0 for fold in summary.get("fold_audit", [])
        )
        and len(summary.get("fold_audit", []))
        == protocol["config"]["group_folds"]
        and all(len(folds) == 1 for folds in task_folds.values())
        and {int(row["fold_id"]) for row in rows}
        == set(range(protocol["config"]["group_folds"])),
        "public_active_features_only": manifest.get("protected_fields_used") == []
        and manifest.get("full_trajectory_fields_used") is False
        and summary.get("full_trajectory_outcomes_used_as_features") is False,
        "frozen_policy_unchanged": summary.get("base_and_mask_parameters_updated")
        is False,
        "metrics_recomputed": abs(
            roc_auc(labels, static_scores)
            - float(summary["static_cross_task_roc_auc"])
        )
        <= TOL
        and abs(
            roc_auc(labels, active_raw_scores)
            - float(summary["active_cross_task_roc_auc"])
        )
        <= TOL
        and all(
            abs(float(recomputed[key]) - float(summary["pipeline_metrics"][key]))
            <= TOL
            for key in (
                "probe_rate",
                "reverse_recall",
                "rescue_drop",
                "rescue_risk_upper_bound",
            )
        ),
    }
    outcomes = {
        "active_auc_gate": float(summary["active_cross_task_roc_auc"])
        >= GATE["min_active_cross_task_roc_auc"] - TOL,
        "active_beats_static": float(summary["active_cross_task_roc_auc"])
        > float(summary["static_cross_task_roc_auc"]) + TOL,
        "empirical_rescue_budget": recomputed["rescue_drop"]
        <= GATE["max_empirical_rescue_drop"] + TOL,
        "reverse_recall_gate": recomputed["reverse_recall"]
        >= GATE["min_reverse_recall"] - TOL,
        "probe_budget_gate": recomputed["probe_rate"]
        <= GATE["max_probe_rate"] + TOL,
    }
    feasibility_passed = all(integrity.values()) and all(outcomes.values())
    risk_certified = (
        summary.get("formal_risk_certification_possible") is True
        and recomputed["rescue_risk_upper_bound"]
        <= float(protocol["config"]["rescue_delta"]) + TOL
    )
    gate = {
        "passed": feasibility_passed,
        "feasibility_passed": feasibility_passed,
        "risk_certified": risk_certified,
        "stage": "toolsandbox_v7_active_shadow_feasibility_gate_seed42",
        "integrity_checks": integrity,
        "outcome_checks": outcomes,
        "thresholds": GATE,
        "static_cross_task_roc_auc": summary["static_cross_task_roc_auc"],
        "active_cross_task_roc_auc": summary["active_cross_task_roc_auc"],
        "active_cross_task_pr_auc": summary["active_cross_task_pr_auc"],
        "probe_rate": recomputed["probe_rate"],
        "reverse_recall": recomputed["reverse_recall"],
        "empirical_rescue_drop": recomputed["rescue_drop"],
        "descriptive_uncertified_rescue_risk_upper_bound": recomputed[
            "rescue_risk_upper_bound"
        ],
        "rescue_calibration_events": summary["current_rescue_calibration_events"],
        "minimum_zero_harm_rescue_calibration_events": summary[
            "minimum_zero_harm_rescue_calibration_events"
        ],
        "scope": protocol["scope"],
        "next_step": (
            "collect an independent calibration set with enough Rescue events for formal risk certification"
            if feasibility_passed and not risk_certified
            else (
                "freeze a fresh confirmation profile"
                if feasibility_passed
                else "if ActiveShadow AUC is weak, add an explicit one-step visible state summary before extending to two steps"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if feasibility_passed else 1)


if __name__ == "__main__":
    main()
