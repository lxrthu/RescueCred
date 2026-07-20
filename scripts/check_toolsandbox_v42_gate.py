#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import summarize_evaluation_rows
from scripts.freeze_toolsandbox_v42_protocol import (
    CONFIRMATION_THRESHOLDS,
    DEVELOPMENT_THRESHOLDS,
)
from scripts.train_toolsandbox_v42_preference import PROTOCOL_STATUS


TOLERANCE = 1e-12


def _summary_matches(summary: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    recomputed = summarize_evaluation_rows(rows)
    fields = (
        "events",
        "valid_events",
        "decisions",
        "causal_accuracy",
        "rescue_accuracy",
        "reverse_accuracy",
        "selected_b_rate",
        "mean_selected_terminal_similarity",
        "mean_selected_progress_auc",
    )
    for field in fields:
        left, right = summary.get(field), recomputed.get(field)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if abs(float(left) - float(right)) > TOLERANCE:
                return False
        elif left != right:
            return False
    return True


def build_gate(
    *,
    role: str,
    mask_eval: dict[str, Any],
    v42_eval: dict[str, Any],
    mask_run: dict[str, Any],
    v42_run: dict[str, Any],
    mask_rows: list[dict[str, Any]],
    v42_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    eval_manifest: dict[str, Any],
    eval_audit: dict[str, Any],
    eval_audit_gate: dict[str, Any],
    identity: dict[str, bool],
) -> dict[str, Any]:
    if role not in {"development", "confirmation"}:
        raise ValueError(f"unsupported evaluation role: {role}")
    thresholds = (
        DEVELOPMENT_THRESHOLDS if role == "development" else CONFIRMATION_THRESHOLDS
    )
    mask_by_id = {str(row["event_id"]): row for row in mask_rows}
    v42_by_id = {str(row["event_id"]): row for row in v42_rows}
    same_ids = (
        len(mask_by_id) == len(mask_rows)
        and len(v42_by_id) == len(v42_rows)
        and set(mask_by_id) == set(v42_by_id)
    )
    disagreements = [
        event_id
        for event_id in sorted(mask_by_id)
        if same_ids and mask_by_id[event_id]["selected"] != v42_by_id[event_id]["selected"]
    ]
    wins = sum(
        v42_by_id[event_id]["causal_correct"]
        and not mask_by_id[event_id]["causal_correct"]
        for event_id in disagreements
    )
    losses = sum(
        mask_by_id[event_id]["causal_correct"]
        and not v42_by_id[event_id]["causal_correct"]
        for event_id in disagreements
    )
    accuracy_improvement = float(v42_eval["causal_accuracy"]) - float(
        mask_eval["causal_accuracy"]
    )
    terminal_improvement = float(
        v42_eval["mean_selected_terminal_similarity"]
    ) - float(mask_eval["mean_selected_terminal_similarity"])
    progress_improvement = float(v42_eval["mean_selected_progress_auc"]) - float(
        mask_eval["mean_selected_progress_auc"]
    )
    reverse_events = int(v42_eval.get("decisions", {}).get("reverse_preference", 0))

    expected_source = protocol.get("expected_presented_source_decisions")
    expected_labels = protocol.get("expected_presented_decisions", {})
    expected_presentations = int(protocol["config"]["presentations_per_epoch"])
    expected_total = expected_presentations * int(protocol["config"]["epochs"])
    integrity_checks = {
        **identity,
        "same_eval_event_set": same_ids
        and mask_eval.get("event_set_hash")
        == v42_eval.get("event_set_hash")
        == eval_manifest.get("event_set_hash"),
        "raw_metrics_independently_recomputed": _summary_matches(
            mask_eval, mask_rows
        )
        and _summary_matches(v42_eval, v42_rows),
        "same_training_data_and_budget": mask_run.get("train_file_sha256")
        == v42_run.get("train_file_sha256")
        == protocol.get("train_sha256")
        and mask_run.get("presentations_per_epoch")
        == v42_run.get("presentations_per_epoch")
        == expected_presentations
        and mask_run.get("active_event_presentations")
        == v42_run.get("active_event_presentations")
        == expected_total,
        "identical_balanced_event_sequence": mask_run.get(
            "presented_event_sequence_sha256"
        )
        == v42_run.get("presented_event_sequence_sha256")
        == protocol.get("expected_presented_event_sequence_sha256")
        and mask_run.get("presented_source_decisions")
        == v42_run.get("presented_source_decisions")
        == expected_source,
        "expected_method_labels": mask_run.get("presented_decisions")
        == expected_labels.get("mask")
        and v42_run.get("presented_decisions") == expected_labels.get("v42"),
        "identical_absolute_margin_objective": mask_run.get("absolute_margin_coef")
        == v42_run.get("absolute_margin_coef")
        == protocol["config"]["absolute_margin_coef"]
        and mask_run.get("target_margin")
        == v42_run.get("target_margin")
        == protocol["config"]["target_margin"]
        and mask_run.get("loss_definition")
        == v42_run.get("loss_definition")
        == "unit_weight*(dpo_shift+absolute_margin)",
        "method_roles_bound": mask_run.get("method") == mask_eval.get("method")
        == "mask"
        and v42_run.get("method") == v42_eval.get("method") == "v42",
        "evaluation_role_bound": mask_eval.get("evaluation_role")
        == v42_eval.get("evaluation_role")
        == role,
        "public_only_model_scoring": mask_eval.get(
            "worker_receives_public_prompt_and_candidates_only"
        )
        is True
        and v42_eval.get("worker_receives_public_prompt_and_candidates_only") is True
        and mask_eval.get("offline_outcomes_joined_after_scoring") is True
        and v42_eval.get("offline_outcomes_joined_after_scoring") is True,
        "audit_mechanism_passed": eval_audit_gate.get("mechanism_passed") is True,
        "audit_protocol_bound": eval_audit.get("protocol_validated") is True
        and eval_audit.get("harness_interface") == "tool_id_v2",
        "evaluation_reference_boundary": eval_manifest.get("role") == "evaluation"
        and eval_manifest.get("protected_outcomes_in_prompt") is False
        and eval_manifest.get("official_branch_metrics_in_training_file") is False
        and eval_manifest.get("branch_receipts_exported") is False
        and eval_manifest.get("reference_actions_read_or_exported") is False,
        "evaluation_covers_all_nonzero_pairs": int(eval_manifest.get("events", -1))
        == int(eval_audit.get("controlled", {}).get("nonzero_events", 0))
        + int(eval_audit.get("natural", {}).get("nonzero_events", 0)),
        "thresholds_frozen": protocol.get("gate_thresholds", {}).get(role)
        == thresholds,
    }
    outcome_checks = {
        "enough_eval_events": int(v42_eval["valid_events"])
        >= thresholds["min_eval_events"],
        "enough_eval_reverse_events": reverse_events
        >= thresholds["min_eval_reverse_events"],
        "enough_selection_disagreements": len(disagreements)
        >= thresholds["min_selection_disagreements"],
        "v42_improves_causal_accuracy": (
            accuracy_improvement > TOLERANCE
            if role == "development"
            else accuracy_improvement
            >= thresholds["min_causal_accuracy_improvement"] - TOLERANCE
        ),
        "v42_wins_over_losses": wins > losses,
        "terminal_noninferiority": terminal_improvement >= -TOLERANCE,
        "progress_noninferiority": progress_improvement >= -TOLERANCE,
    }
    passed = all(integrity_checks.values()) and all(outcome_checks.values())
    return {
        "passed": passed,
        "stage": f"toolsandbox_v42_{role}_gate_seed42",
        "evaluation_role": role,
        "integrity_checks": integrity_checks,
        "outcome_checks": outcome_checks,
        "thresholds": thresholds,
        "train_events": protocol["train_events"],
        "eval_events": v42_eval["valid_events"],
        "eval_reverse_events": reverse_events,
        "selection_disagreements": len(disagreements),
        "v42_wins": wins,
        "v42_losses": losses,
        "ties": int(v42_eval["valid_events"]) - len(disagreements),
        "mask_causal_accuracy": mask_eval["causal_accuracy"],
        "v42_causal_accuracy": v42_eval["causal_accuracy"],
        "causal_accuracy_improvement": accuracy_improvement,
        "mask_mean_terminal_similarity": mask_eval[
            "mean_selected_terminal_similarity"
        ],
        "v42_mean_terminal_similarity": v42_eval[
            "mean_selected_terminal_similarity"
        ],
        "terminal_similarity_improvement": terminal_improvement,
        "mask_mean_progress_auc": mask_eval["mean_selected_progress_auc"],
        "v42_mean_progress_auc": v42_eval["mean_selected_progress_auc"],
        "progress_auc_improvement": progress_improvement,
        "scope": protocol["scope"],
        "next_step": (
            "run the frozen offset-165 confirmation"
            if passed and role == "development"
            else "freeze multi-seed replication; retain AppWorld as secondary evidence"
            if passed
            else "stop expansion and inspect the frozen comparison"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("development", "confirmation"), required=True)
    parser.add_argument("--mask-eval", type=Path, required=True)
    parser.add_argument("--v42-eval", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--v42-run", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v42-results", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--eval-audit", type=Path, required=True)
    parser.add_argument("--eval-audit-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    mask_eval = load(args.mask_eval)
    v42_eval = load(args.v42_eval)
    mask_run = load(args.mask_run)
    v42_run = load(args.v42_run)
    protocol = load(args.protocol_lock)
    eval_manifest = load(args.eval_manifest)
    eval_audit = load(args.eval_audit)
    eval_audit_gate = load(args.eval_audit_gate)
    source_identity = bool(protocol.get("source_sha256")) and all(
        Path(path).is_file() and file_sha256(Path(path)) == expected
        for path, expected in protocol.get("source_sha256", {}).items()
    )
    role_lock = protocol[args.role]
    expected_scenarios = (
        role_lock["scenario_hashes"]
        if args.role == "development"
        else role_lock["scenario_identity"]["fresh_hashes"]
    )
    expected_audit_protocol = (
        role_lock["audit_protocol_sha256"]
        if args.role == "development"
        else role_lock["protocol_sha256"]
    )
    identity = {
        "protocol_status": protocol.get("status") == PROTOCOL_STATUS,
        "protocol_source_identity": source_identity,
        "protocol_bound_to_runs": mask_run.get("protocol_lock_sha256")
        == v42_run.get("protocol_lock_sha256")
        == file_sha256(args.protocol_lock),
        "run_summaries_bound_to_evaluations": mask_eval.get("run_summary_sha256")
        == file_sha256(args.mask_run)
        and v42_eval.get("run_summary_sha256") == file_sha256(args.v42_run),
        "adapters_bound": mask_eval.get("adapter_sha256")
        == mask_run.get("adapter_sha256")
        and v42_eval.get("adapter_sha256") == v42_run.get("adapter_sha256"),
        "evaluation_files_bound": mask_eval.get("public_events_sha256")
        == v42_eval.get("public_events_sha256")
        == eval_manifest.get("public_sha256")
        and mask_eval.get("private_outcomes_sha256")
        == v42_eval.get("private_outcomes_sha256")
        == eval_manifest.get("private_sha256"),
        "result_files_bound": mask_eval.get("results_sha256")
        == file_sha256(args.mask_results)
        and v42_eval.get("results_sha256") == file_sha256(args.v42_results),
        "audit_bound_to_eval_manifest": eval_manifest.get("source_summary_sha256")
        == file_sha256(args.eval_audit)
        and eval_manifest.get("source_gate_sha256")
        == file_sha256(args.eval_audit_gate)
        and eval_manifest.get("source_signal_sha256")
        == eval_audit.get("event_file_sha256"),
        "evaluation_protocol_identity": eval_audit.get("protocol_lock_sha256")
        == expected_audit_protocol
        and eval_audit.get("selected_scenario_hashes") == expected_scenarios,
        "development_artifacts_frozen_before_training": (
            args.role != "development"
            or role_lock.get("known_before_v42_training") is True
            and role_lock.get("manifest_sha256") == file_sha256(args.eval_manifest)
            and role_lock.get("public_sha256") == file_sha256(
                args.eval_manifest.parent / "events.public.jsonl"
            )
            and role_lock.get("private_sha256") == file_sha256(
                args.eval_manifest.parent / "outcomes.private.jsonl"
            )
        ),
        "base_model_bound": mask_run.get("base_model_sha256")
        == v42_run.get("base_model_sha256")
        == mask_eval.get("base_model_sha256")
        == v42_eval.get("base_model_sha256")
        == protocol.get("base_model_sha256"),
    }
    gate = build_gate(
        role=args.role,
        mask_eval=mask_eval,
        v42_eval=v42_eval,
        mask_run=mask_run,
        v42_run=v42_run,
        mask_rows=read_jsonl(args.mask_results),
        v42_rows=read_jsonl(args.v42_results),
        protocol=protocol,
        eval_manifest=eval_manifest,
        eval_audit=eval_audit,
        eval_audit_gate=eval_audit_gate,
        identity=identity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
