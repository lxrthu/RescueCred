#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import summarize_evaluation_rows
from scripts.freeze_toolsandbox_v45_learner_protocol import CONFIRMATION_THRESHOLDS, DEVELOPMENT_THRESHOLDS
from scripts.train_toolsandbox_v43_preference import V45_PROTOCOL_STATUS

TOLERANCE = 1e-12


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_exact(summary: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    expected = summarize_evaluation_rows(rows)
    for key, value in expected.items():
        actual = summary.get(key)
        if isinstance(value, float):
            if abs(float(actual) - value) > TOLERANCE:
                return False
        elif actual != value:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("development", "confirmation"), required=True)
    parser.add_argument("--mask-eval", type=Path, required=True)
    parser.add_argument("--v45-eval", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--v45-run", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v45-results", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--candidate-protocol", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--eval-data-gate", type=Path, required=True)
    parser.add_argument("--eval-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mask_eval, v45_eval = _load(args.mask_eval), _load(args.v45_eval)
    mask_run, v45_run = _load(args.mask_run), _load(args.v45_run)
    protocol, candidate_protocol = _load(args.protocol_lock), _load(args.candidate_protocol)
    manifest, data_gate, audit = _load(args.eval_manifest), _load(args.eval_data_gate), _load(args.eval_audit)
    mask_rows, v45_rows = read_jsonl(args.mask_results), read_jsonl(args.v45_results)
    mask_by_id = {str(row["event_id"]): row for row in mask_rows}
    v45_by_id = {str(row["event_id"]): row for row in v45_rows}
    same_ids = len(mask_by_id) == len(mask_rows) and len(v45_by_id) == len(v45_rows) and set(mask_by_id) == set(v45_by_id)
    disagreements = [event_id for event_id in sorted(mask_by_id) if same_ids and mask_by_id[event_id]["selected"] != v45_by_id[event_id]["selected"]]
    wins = sum(v45_by_id[event_id]["causal_correct"] and not mask_by_id[event_id]["causal_correct"] for event_id in disagreements)
    losses = sum(mask_by_id[event_id]["causal_correct"] and not v45_by_id[event_id]["causal_correct"] for event_id in disagreements)
    shifts = {"rescue_preference": [], "reverse_preference": []}
    if same_ids:
        for event_id in sorted(mask_by_id):
            decision = str(mask_by_id[event_id]["decision"])
            if decision in shifts:
                shifts[decision].append(float(v45_by_id[event_id]["margin_b_over_a"]) - float(mask_by_id[event_id]["margin_b_over_a"]))
    def mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))
    rescue_shift, reverse_shift = mean(shifts["rescue_preference"]), mean(shifts["reverse_preference"])
    shift_gap = rescue_shift - reverse_shift
    accuracy_gain = float(v45_eval["causal_accuracy"]) - float(mask_eval["causal_accuracy"])
    terminal_gain = float(v45_eval["mean_selected_terminal_similarity"]) - float(mask_eval["mean_selected_terminal_similarity"])
    progress_gain = float(v45_eval["mean_selected_progress_auc"]) - float(mask_eval["mean_selected_progress_auc"])
    thresholds = DEVELOPMENT_THRESHOLDS if args.role == "development" else CONFIRMATION_THRESHOLDS
    role_lock = protocol[args.role]
    source_identity = bool(protocol.get("source_sha256")) and all(Path(path).is_file() and file_sha256(Path(path)) == digest for path, digest in protocol.get("source_sha256", {}).items())
    expected_total = int(protocol["config"]["epochs"]) * int(protocol["config"]["presentations_per_epoch"])
    identity = {
        "protocol_status": protocol.get("status") == V45_PROTOCOL_STATUS,
        "protocol_source_identity": source_identity,
        "candidate_protocol_preoutcome_bound": candidate_protocol.get("evaluation_role") == args.role and role_lock.get("protocol_sha256") == file_sha256(args.candidate_protocol) and audit.get("protocol_lock_sha256") == file_sha256(args.candidate_protocol),
        "scenario_identity_bound": audit.get("selected_scenario_hashes") == role_lock.get("scenario_identity", {}).get("fresh_hashes"),
        "protocol_bound_to_runs": mask_run.get("protocol_lock_sha256") == v45_run.get("protocol_lock_sha256") == file_sha256(args.protocol_lock),
        "run_and_adapter_identity": mask_eval.get("run_summary_sha256") == file_sha256(args.mask_run) and v45_eval.get("run_summary_sha256") == file_sha256(args.v45_run) and mask_eval.get("adapter_sha256") == mask_run.get("adapter_sha256") and v45_eval.get("adapter_sha256") == v45_run.get("adapter_sha256"),
        "evaluation_files_bound": mask_eval.get("public_events_sha256") == v45_eval.get("public_events_sha256") == manifest.get("public_sha256") and mask_eval.get("private_outcomes_sha256") == v45_eval.get("private_outcomes_sha256") == manifest.get("private_sha256"),
        "result_files_bound": mask_eval.get("results_sha256") == file_sha256(args.mask_results) and v45_eval.get("results_sha256") == file_sha256(args.v45_results),
        "audit_and_data_bound": manifest.get("source_summary_sha256") == file_sha256(args.eval_audit) and manifest.get("source_protocol_sha256") == file_sha256(args.candidate_protocol) and data_gate.get("passed") is True and manifest.get("role") == "evaluation",
        "same_eval_event_set": same_ids and mask_eval.get("event_set_hash") == v45_eval.get("event_set_hash") == manifest.get("event_set_hash"),
        "raw_metrics_recomputed": _summary_exact(mask_eval, mask_rows) and _summary_exact(v45_eval, v45_rows),
        "same_training_data_budget_sequence": mask_run.get("train_file_sha256") == v45_run.get("train_file_sha256") == protocol.get("train_sha256") and mask_run.get("active_event_presentations") == v45_run.get("active_event_presentations") == expected_total and mask_run.get("presented_event_sequence_sha256") == v45_run.get("presented_event_sequence_sha256") == protocol.get("expected_presented_event_sequence_sha256"),
        "expected_method_labels": mask_run.get("presented_decisions") == protocol.get("expected_presented_decisions", {}).get("mask") and v45_run.get("presented_decisions") == protocol.get("expected_presented_decisions", {}).get("v45"),
        "matched_anchor_objective": mask_run.get("loss_definition") == v45_run.get("loss_definition") == "unit_weight*(dpo_shift+absolute_margin)+reference_anchor" and mask_run.get("reference_anchor_coef") == v45_run.get("reference_anchor_coef") == protocol["config"]["reference_anchor_coef"],
        "public_only_model_scoring": mask_eval.get("worker_receives_public_prompt_and_candidates_only") is True and v45_eval.get("worker_receives_public_prompt_and_candidates_only") is True and mask_eval.get("offline_outcomes_joined_after_scoring") is True and v45_eval.get("offline_outcomes_joined_after_scoring") is True,
        "thresholds_frozen": protocol.get("gate_thresholds", {}).get(args.role) == thresholds,
    }
    decisions = v45_eval.get("decisions", {})
    outcomes = {
        "enough_eval_events": int(v45_eval["valid_events"]) >= thresholds["min_eval_events"],
        "enough_eval_rescue_events": int(decisions.get("rescue_preference", 0)) >= thresholds["min_eval_rescue_events"],
        "enough_eval_reverse_events": int(decisions.get("reverse_preference", 0)) >= thresholds["min_eval_reverse_events"],
        "enough_selection_disagreements": len(disagreements) >= thresholds["min_selection_disagreements"],
        "v45_improves_causal_accuracy": accuracy_gain > TOLERANCE if args.role == "development" else accuracy_gain >= thresholds["min_causal_accuracy_improvement"] - TOLERANCE,
        "v45_wins_over_losses": wins > losses,
        "terminal_noninferiority": terminal_gain >= -TOLERANCE,
        "progress_noninferiority": progress_gain >= -TOLERANCE,
        "class_conditional_shift_separation": shift_gap >= thresholds["min_class_conditional_shift_gap"],
    }
    passed = all(identity.values()) and all(outcomes.values())
    gate = {
        "passed": passed,
        "stage": f"toolsandbox_v45_{args.role}_gate_seed42",
        "evaluation_role": args.role,
        "integrity_checks": identity,
        "outcome_checks": outcomes,
        "thresholds": thresholds,
        "eval_events": v45_eval["valid_events"],
        "selection_disagreements": len(disagreements),
        "v45_wins": wins,
        "v45_losses": losses,
        "ties": int(v45_eval["valid_events"]) - len(disagreements),
        "mask_causal_accuracy": mask_eval["causal_accuracy"],
        "v45_causal_accuracy": v45_eval["causal_accuracy"],
        "causal_accuracy_improvement": accuracy_gain,
        "terminal_similarity_improvement": terminal_gain,
        "progress_auc_improvement": progress_gain,
        "mean_rescue_margin_shift": rescue_shift,
        "mean_reverse_margin_shift": reverse_shift,
        "class_conditional_shift_gap": shift_gap,
        "scope": protocol["scope"],
        "next_step": "run frozen confirmation" if passed and args.role == "development" else "freeze multi-seed replication" if passed else "stop and inspect the frozen comparison",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
