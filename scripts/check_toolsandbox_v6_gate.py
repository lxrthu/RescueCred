#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_router import summarize_router_predictions
from scripts.freeze_toolsandbox_v6_protocol import GATE, PROTOCOL_STATUS


TOL = 1e-12


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_exact(summary: dict, rows: list[dict]) -> bool:
    expected = summarize_router_predictions(rows)
    for key, value in expected.items():
        actual = summary.get(key)
        if isinstance(value, float):
            if actual is None or abs(float(actual) - value) > TOL:
                return False
        elif actual != value:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "default-eval",
        "control-eval",
        "semantic-eval",
        "default-results",
        "control-results",
        "semantic-results",
        "control-run",
        "semantic-run",
        "control-probe",
        "semantic-probe",
        "scoring-summary",
        "protocol-lock",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    default_eval = load(args.default_eval)
    control_eval = load(args.control_eval)
    semantic_eval = load(args.semantic_eval)
    control_run = load(args.control_run)
    semantic_run = load(args.semantic_run)
    scoring = load(args.scoring_summary)
    protocol = load(args.protocol_lock)
    default_rows = read_jsonl(args.default_results)
    control_rows = read_jsonl(args.control_results)
    semantic_rows = read_jsonl(args.semantic_results)
    default_ids = {str(row["event_id"]) for row in default_rows}
    control_ids = {str(row["event_id"]) for row in control_rows}
    semantic_ids = {str(row["event_id"]) for row in semantic_rows}
    exact_ids = (
        default_ids == control_ids == semantic_ids
        and len(default_ids) == len(default_rows)
        and len(control_ids) == len(control_rows)
        and len(semantic_ids) == len(semantic_rows)
    )
    source_ok = bool(protocol.get("source_sha256")) and all(
        Path(path).is_file() and file_sha256(Path(path)) == digest
        for path, digest in protocol.get("source_sha256", {}).items()
    )
    expected_lock = file_sha256(args.protocol_lock)
    integrity = {
        "protocol_frozen": protocol.get("status") == PROTOCOL_STATUS
        and protocol.get("gate") == GATE,
        "source_identity": source_ok,
        "probe_runs_bound": control_run.get("protocol_lock_sha256")
        == semantic_run.get("protocol_lock_sha256")
        == expected_lock,
        "probe_files_bound": control_run.get("probe_sha256")
        == file_sha256(args.control_probe)
        and semantic_run.get("probe_sha256") == file_sha256(args.semantic_probe),
        "method_roles": control_run.get("method") == "margin_probe"
        and semantic_run.get("method") == "semantic_probe",
        "cross_task_group_isolation": all(
            fold.get("task_overlap") == 0 for fold in semantic_run.get("fold_audit", [])
        )
        and len(semantic_run.get("fold_audit", []))
        == protocol["config"]["group_folds"],
        "frozen_policy_unchanged": control_run.get("base_and_mask_parameters_updated")
        is False
        and semantic_run.get("base_and_mask_parameters_updated") is False,
        "public_only_features": control_run.get("private_outcomes_used_as_features")
        is False
        and semantic_run.get("private_outcomes_used_as_features") is False
        and scoring.get("private_outcomes_read") is False,
        "public_only_scoring": default_eval.get("public_only_model_scoring") is True
        and control_eval.get("public_only_model_scoring") is True
        and semantic_eval.get("public_only_model_scoring") is True,
        "same_development_events": exact_ids
        and default_eval.get("event_set_hash")
        == control_eval.get("event_set_hash")
        == semantic_eval.get("event_set_hash")
        and default_eval.get("evaluation_role")
        == control_eval.get("evaluation_role")
        == semantic_eval.get("evaluation_role")
        == "development",
        "raw_metrics_recomputed": summary_exact(default_eval, default_rows)
        and summary_exact(control_eval, control_rows)
        and summary_exact(semantic_eval, semantic_rows),
    }
    semantic_auc = float(semantic_run["cross_task_roc_auc"])
    control_auc = float(control_run["cross_task_roc_auc"])
    reverse_recall = float(
        semantic_run["oof_selective_metrics"]["reverse_recall"]
    )
    rescue_delta = float(protocol["config"]["rescue_delta"])
    outcomes = {
        "cross_task_auc_signal": semantic_auc
        >= GATE["min_cross_task_roc_auc"] - TOL,
        "cross_task_pr_lift_signal": float(semantic_run["cross_task_pr_auc_lift"])
        >= GATE["min_pr_auc_lift_over_prevalence"] - TOL,
        "semantic_beats_margin_control": semantic_auc > control_auc + TOL,
        "oof_rescue_budget_met": float(
            semantic_run["oof_selective_metrics"]["rescue_drop"]
        )
        <= rescue_delta + TOL,
        "oof_reverse_recall_useful": reverse_recall
        >= GATE["min_reverse_recall_at_rescue_budget"] - TOL,
        "development_rescue_noninferiority": float(semantic_eval["rescue_accuracy"])
        >= float(default_eval["rescue_accuracy"]) - rescue_delta - TOL,
    }
    if outcomes["cross_task_auc_signal"] and outcomes["cross_task_pr_lift_signal"]:
        if outcomes["oof_reverse_recall_useful"] and outcomes["oof_rescue_budget_met"]:
            diagnosis = "information_sufficient_and_safe_route_feasible"
        else:
            diagnosis = "information_detectable_but_safe_operating_point_weak"
    elif semantic_auc <= 0.60 + TOL:
        diagnosis = "deployment_visible_information_insufficient"
    else:
        diagnosis = "diagnostic_inconclusive"
    passed = all(integrity.values()) and all(outcomes.values())
    gate = {
        "passed": passed,
        "stage": "toolsandbox_v6_reverse_diagnostic_development_gate_seed42",
        "diagnosis": diagnosis,
        "integrity_checks": integrity,
        "outcome_checks": outcomes,
        "thresholds": GATE,
        "rescue_delta": rescue_delta,
        "semantic_cross_task_roc_auc": semantic_auc,
        "margin_cross_task_roc_auc": control_auc,
        "semantic_cross_task_pr_auc": semantic_run["cross_task_pr_auc"],
        "semantic_cross_task_pr_auc_lift": semantic_run["cross_task_pr_auc_lift"],
        "oof_reverse_recall_at_budget": reverse_recall,
        "oof_rescue_drop": semantic_run["oof_selective_metrics"]["rescue_drop"],
        "selected_threshold": semantic_run["selected_threshold"],
        "development_default_b_rescue_accuracy": default_eval["rescue_accuracy"],
        "development_semantic_rescue_accuracy": semantic_eval["rescue_accuracy"],
        "development_default_b_reverse_accuracy": default_eval["reverse_accuracy"],
        "development_semantic_reverse_accuracy": semantic_eval["reverse_accuracy"],
        "scope": protocol["scope"],
        "next_step": (
            "freeze a fresh cross-scenario confirmation profile"
            if passed
            else "follow diagnosis: revise objective if detectable, otherwise add deployment-visible feedback or short-horizon simulation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
