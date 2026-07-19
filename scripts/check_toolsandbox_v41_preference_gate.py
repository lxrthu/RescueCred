#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import summarize_evaluation_rows
from scripts.freeze_toolsandbox_v41_preference_protocol import GATE_THRESHOLDS


TOLERANCE = 1e-12


def build_gate(
    *,
    mask_eval: dict[str, Any],
    v4_eval: dict[str, Any],
    mask_run: dict[str, Any],
    v4_run: dict[str, Any],
    mask_rows: list[dict[str, Any]],
    v4_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    eval_manifest: dict[str, Any],
    eval_audit: dict[str, Any],
    eval_audit_gate: dict[str, Any],
    identity: dict[str, bool],
) -> dict[str, Any]:
    mask_by_id = {str(row["event_id"]): row for row in mask_rows}
    v4_by_id = {str(row["event_id"]): row for row in v4_rows}
    same_ids = set(mask_by_id) == set(v4_by_id)
    disagreements = []
    if same_ids:
        disagreements = [
            event_id
            for event_id in sorted(mask_by_id)
            if mask_by_id[event_id]["selected"] != v4_by_id[event_id]["selected"]
        ]
    wins = sum(
        v4_by_id[event_id]["causal_correct"]
        and not mask_by_id[event_id]["causal_correct"]
        for event_id in disagreements
    )
    losses = sum(
        mask_by_id[event_id]["causal_correct"]
        and not v4_by_id[event_id]["causal_correct"]
        for event_id in disagreements
    )
    mask_recomputed = summarize_evaluation_rows(mask_rows)
    v4_recomputed = summarize_evaluation_rows(v4_rows)

    def summary_matches(summary: dict[str, Any], recomputed: dict[str, Any]) -> bool:
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
    accuracy_improvement = float(v4_eval["causal_accuracy"]) - float(
        mask_eval["causal_accuracy"]
    )
    terminal_improvement = float(v4_eval["mean_selected_terminal_similarity"]) - float(
        mask_eval["mean_selected_terminal_similarity"]
    )
    progress_improvement = float(v4_eval["mean_selected_progress_auc"]) - float(
        mask_eval["mean_selected_progress_auc"]
    )
    reverse_events = int(v4_eval.get("decisions", {}).get("reverse_preference", 0))
    integrity_checks = {
        **identity,
        "same_eval_event_set": same_ids
        and len(mask_by_id) == len(mask_rows)
        and len(v4_by_id) == len(v4_rows)
        and mask_eval.get("event_set_hash") == v4_eval.get("event_set_hash")
        == eval_manifest.get("event_set_hash"),
        "raw_metrics_independently_recomputed": summary_matches(
            mask_eval, mask_recomputed
        )
        and summary_matches(v4_eval, v4_recomputed),
        "same_training_budget": mask_run.get("train_file_sha256")
        == v4_run.get("train_file_sha256")
        == protocol.get("train_sha256")
        and mask_run.get("presentations_per_epoch")
        == v4_run.get("presentations_per_epoch")
        == protocol.get("train_events")
        and mask_run.get("active_event_presentations")
        == v4_run.get("active_event_presentations")
        == protocol.get("train_events") * protocol.get("config", {}).get("epochs", 0),
        "identical_presented_event_sequence": mask_run.get(
            "presented_event_sequence_sha256"
        )
        == v4_run.get("presented_event_sequence_sha256")
        == protocol.get("expected_presented_event_sequence_sha256"),
        "method_roles_bound": mask_run.get("method") == mask_eval.get("method")
        == "mask"
        and v4_run.get("method") == v4_eval.get("method") == "v4",
        "public_only_model_scoring": mask_eval.get(
            "worker_receives_public_prompt_and_candidates_only"
        )
        is True
        and v4_eval.get("worker_receives_public_prompt_and_candidates_only") is True
        and mask_eval.get("offline_outcomes_joined_after_scoring") is True
        and v4_eval.get("offline_outcomes_joined_after_scoring") is True,
        "fresh_audit_mechanism_passed": eval_audit_gate.get("mechanism_passed")
        is True,
        "fresh_audit_protocol_bound": eval_audit.get("protocol_validated") is True
        and eval_audit.get("harness_interface") == "tool_id_v2"
        and eval_audit.get("protocol_lock_sha256")
        == protocol.get("evaluation_protocol_sha256"),
        "fresh_scenarios_match_protocol": sorted(
            eval_audit.get("selected_scenario_hashes", [])
        )
        == sorted(
            protocol.get("evaluation_scenario_identity", {}).get("fresh_hashes", [])
        ),
        "eval_data_reference_boundary": eval_manifest.get("role") == "evaluation"
        and eval_manifest.get("protected_outcomes_in_prompt") is False
        and eval_manifest.get("official_branch_metrics_in_training_file") is False
        and eval_manifest.get("branch_receipts_exported") is False
        and eval_manifest.get("reference_actions_read_or_exported") is False,
        "eval_data_covers_all_nonzero_pairs": int(eval_manifest.get("events", -1))
        == int(eval_audit.get("controlled", {}).get("nonzero_events", 0))
        + int(eval_audit.get("natural", {}).get("nonzero_events", 0)),
        "thresholds_frozen": protocol.get("gate_thresholds") == GATE_THRESHOLDS,
    }
    outcome_checks = {
        "enough_eval_events": int(v4_eval["valid_events"])
        >= GATE_THRESHOLDS["min_eval_events"],
        "enough_eval_reverse_events": reverse_events
        >= GATE_THRESHOLDS["min_eval_reverse_events"],
        "enough_selection_disagreements": len(disagreements)
        >= GATE_THRESHOLDS["min_selection_disagreements"],
        "v4_improves_causal_accuracy": accuracy_improvement
        >= GATE_THRESHOLDS["min_causal_accuracy_improvement"],
        "v4_wins_over_losses": wins > losses,
        "terminal_noninferiority": terminal_improvement >= -TOLERANCE,
        "progress_noninferiority": progress_improvement >= -TOLERANCE,
    }
    passed = all(integrity_checks.values()) and all(outcome_checks.values())
    return {
        "passed": passed,
        "stage": "toolsandbox_v41_same_data_preference_seed42_gate",
        "integrity_checks": integrity_checks,
        "outcome_checks": outcome_checks,
        "thresholds": GATE_THRESHOLDS,
        "train_events": protocol["train_events"],
        "eval_events": v4_eval["valid_events"],
        "eval_reverse_events": reverse_events,
        "selection_disagreements": len(disagreements),
        "v4_wins": wins,
        "v4_losses": losses,
        "ties": int(v4_eval["valid_events"]) - len(disagreements),
        "mask_causal_accuracy": mask_eval["causal_accuracy"],
        "v4_causal_accuracy": v4_eval["causal_accuracy"],
        "causal_accuracy_improvement": accuracy_improvement,
        "mask_mean_terminal_similarity": mask_eval[
            "mean_selected_terminal_similarity"
        ],
        "v4_mean_terminal_similarity": v4_eval[
            "mean_selected_terminal_similarity"
        ],
        "terminal_similarity_improvement": terminal_improvement,
        "mask_mean_progress_auc": mask_eval["mean_selected_progress_auc"],
        "v4_mean_progress_auc": v4_eval["mean_selected_progress_auc"],
        "progress_auc_improvement": progress_improvement,
        "scope": protocol["scope"],
        "next_step": (
            "freeze multi-seed confirmation, then autonomous ToolSandbox evaluation"
            if passed
            else "stop expansion and inspect the frozen seed-42 preference comparison"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-eval", type=Path, required=True)
    parser.add_argument("--v4-eval", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--v4-run", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v4-results", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--eval-audit", type=Path, required=True)
    parser.add_argument("--eval-audit-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    mask_eval = load(args.mask_eval)
    v4_eval = load(args.v4_eval)
    mask_run = load(args.mask_run)
    v4_run = load(args.v4_run)
    protocol = load(args.protocol_lock)
    eval_manifest = load(args.eval_manifest)
    eval_audit = load(args.eval_audit)
    eval_audit_gate = load(args.eval_audit_gate)
    source_identity = bool(protocol.get("source_sha256")) and all(
        Path(path).is_file() and file_sha256(Path(path)) == expected
        for path, expected in protocol.get("source_sha256", {}).items()
    )
    identity = {
        "protocol_status": protocol.get("status")
        == "frozen_before_toolsandbox_v41_preference_outcomes",
        "protocol_source_identity": source_identity,
        "protocol_bound_to_runs": mask_run.get("protocol_lock_sha256")
        == v4_run.get("protocol_lock_sha256")
        == file_sha256(args.protocol_lock),
        "run_summaries_bound_to_evaluations": mask_eval.get("run_summary_sha256")
        == file_sha256(args.mask_run)
        and v4_eval.get("run_summary_sha256") == file_sha256(args.v4_run),
        "adapters_bound": mask_eval.get("adapter_sha256")
        == mask_run.get("adapter_sha256")
        and v4_eval.get("adapter_sha256") == v4_run.get("adapter_sha256"),
        "evaluation_files_bound": mask_eval.get("public_events_sha256")
        == v4_eval.get("public_events_sha256")
        == eval_manifest.get("public_sha256")
        and mask_eval.get("private_outcomes_sha256")
        == v4_eval.get("private_outcomes_sha256")
        == eval_manifest.get("private_sha256"),
        "result_files_bound": mask_eval.get("results_sha256")
        == file_sha256(args.mask_results)
        and v4_eval.get("results_sha256") == file_sha256(args.v4_results),
        "fresh_audit_bound_to_eval_manifest": eval_manifest.get(
            "source_summary_sha256"
        )
        == file_sha256(args.eval_audit)
        and eval_manifest.get("source_gate_sha256")
        == file_sha256(args.eval_audit_gate)
        and eval_manifest.get("source_signal_sha256")
        == eval_audit.get("event_file_sha256"),
        "base_model_bound": mask_run.get("base_model_sha256")
        == v4_run.get("base_model_sha256")
        == mask_eval.get("base_model_sha256")
        == v4_eval.get("base_model_sha256")
        == protocol.get("base_model_sha256"),
    }
    gate = build_gate(
        mask_eval=mask_eval,
        v4_eval=v4_eval,
        mask_run=mask_run,
        v4_run=v4_run,
        mask_rows=read_jsonl(args.mask_results),
        v4_rows=read_jsonl(args.v4_results),
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
