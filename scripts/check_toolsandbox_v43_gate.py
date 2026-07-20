#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from scripts.check_toolsandbox_v42_gate import build_gate as build_v42_gate
from scripts.freeze_toolsandbox_v42_protocol import (
    CONFIRMATION_THRESHOLDS as V42_CONFIRMATION_THRESHOLDS,
    DEVELOPMENT_THRESHOLDS as V42_DEVELOPMENT_THRESHOLDS,
)
from scripts.freeze_toolsandbox_v43_protocol import (
    CONFIRMATION_THRESHOLDS,
    DEVELOPMENT_THRESHOLDS,
)
from scripts.train_toolsandbox_v43_preference import PROTOCOL_STATUS


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def build_gate(
    *,
    role: str,
    mask_eval: dict[str, Any],
    v43_eval: dict[str, Any],
    mask_run: dict[str, Any],
    v43_run: dict[str, Any],
    mask_rows: list[dict[str, Any]],
    v43_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    eval_manifest: dict[str, Any],
    eval_audit: dict[str, Any],
    eval_audit_gate: dict[str, Any],
    identity: dict[str, bool],
) -> dict[str, Any]:
    thresholds = (
        DEVELOPMENT_THRESHOLDS if role == "development" else CONFIRMATION_THRESHOLDS
    )
    base_thresholds = (
        V42_DEVELOPMENT_THRESHOLDS
        if role == "development"
        else V42_CONFIRMATION_THRESHOLDS
    )
    mask_by_id = {str(row["event_id"]): row for row in mask_rows}
    v43_by_id = {str(row["event_id"]): row for row in v43_rows}
    same_ids = set(mask_by_id) == set(v43_by_id)
    same_labels = same_ids and all(
        mask_by_id[event_id].get("decision")
        == v43_by_id[event_id].get("decision")
        for event_id in mask_by_id
    )
    shifts = {decision: [] for decision in ("rescue_preference", "reverse_preference")}
    if same_labels:
        for event_id in sorted(mask_by_id):
            decision = str(mask_by_id[event_id]["decision"])
            if decision in shifts:
                shifts[decision].append(
                    float(v43_by_id[event_id]["margin_b_over_a"])
                    - float(mask_by_id[event_id]["margin_b_over_a"])
                )
    rescue_shift = _mean(shifts["rescue_preference"])
    reverse_shift = _mean(shifts["reverse_preference"])
    class_shift_gap = rescue_shift - reverse_shift

    # Reuse the independently recomputed V4.2 gate after a transparent method
    # name normalization. V4.3-specific objective and thresholds are checked
    # separately below; no raw metric is altered.
    normalized_protocol = dict(protocol)
    normalized_protocol["gate_thresholds"] = {
        **protocol["gate_thresholds"],
        role: base_thresholds,
    }
    normalized_protocol["expected_presented_decisions"] = {
        "mask": protocol["expected_presented_decisions"]["mask"],
        "v42": protocol["expected_presented_decisions"]["v43"],
    }
    normalized_v43_eval = {**v43_eval, "method": "v42"}
    normalized_mask_run = {
        **mask_run,
        "loss_definition": "unit_weight*(dpo_shift+absolute_margin)",
    }
    normalized_v43_run = {
        **v43_run,
        "method": "v42",
        "loss_definition": "unit_weight*(dpo_shift+absolute_margin)",
    }
    base = build_v42_gate(
        role=role,
        mask_eval=mask_eval,
        v42_eval=normalized_v43_eval,
        mask_run=normalized_mask_run,
        v42_run=normalized_v43_run,
        mask_rows=mask_rows,
        v42_rows=v43_rows,
        protocol=normalized_protocol,
        eval_manifest=eval_manifest,
        eval_audit=eval_audit,
        eval_audit_gate=eval_audit_gate,
        identity=identity,
    )
    base["integrity_checks"]["v43_anchor_objective"] = (
        mask_run.get("reference_anchor_coef")
        == v43_run.get("reference_anchor_coef")
        == protocol["config"]["reference_anchor_coef"]
        and mask_run.get("loss_definition")
        == v43_run.get("loss_definition")
        == "unit_weight*(dpo_shift+absolute_margin)+reference_anchor"
    )
    base["integrity_checks"]["v43_thresholds_frozen"] = (
        protocol.get("gate_thresholds", {}).get(role) == thresholds
    )
    base["integrity_checks"]["class_shift_rows_matched"] = same_labels and bool(
        shifts["rescue_preference"]
    ) and bool(shifts["reverse_preference"])
    base["outcome_checks"]["class_conditional_shift_separation"] = (
        class_shift_gap >= thresholds["min_class_conditional_shift_gap"]
    )
    base["outcome_checks"]["v43_improves_causal_accuracy"] = base[
        "outcome_checks"
    ].pop("v42_improves_causal_accuracy")
    base["outcome_checks"]["v43_wins_over_losses"] = base["outcome_checks"].pop(
        "v42_wins_over_losses"
    )
    base["passed"] = all(base["integrity_checks"].values()) and all(
        base["outcome_checks"].values()
    )
    base["stage"] = f"toolsandbox_v43_{role}_gate_seed42"
    base["thresholds"] = thresholds
    base["v43_wins"] = base.pop("v42_wins")
    base["v43_losses"] = base.pop("v42_losses")
    base["v43_causal_accuracy"] = base.pop("v42_causal_accuracy")
    base["v43_mean_terminal_similarity"] = base.pop(
        "v42_mean_terminal_similarity"
    )
    base["v43_mean_progress_auc"] = base.pop("v42_mean_progress_auc")
    base["mean_rescue_margin_shift"] = rescue_shift
    base["mean_reverse_margin_shift"] = reverse_shift
    base["class_conditional_shift_gap"] = class_shift_gap
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("development", "confirmation"), required=True)
    parser.add_argument("--mask-eval", type=Path, required=True)
    parser.add_argument("--v43-eval", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--v43-run", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v43-results", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--eval-audit", type=Path, required=True)
    parser.add_argument("--eval-audit-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    mask_eval = load(args.mask_eval)
    v43_eval = load(args.v43_eval)
    mask_run = load(args.mask_run)
    v43_run = load(args.v43_run)
    protocol = load(args.protocol_lock)
    eval_manifest = load(args.eval_manifest)
    eval_audit = load(args.eval_audit)
    eval_audit_gate = load(args.eval_audit_gate)
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
    source_identity = bool(protocol.get("source_sha256")) and all(
        Path(path).is_file() and file_sha256(Path(path)) == expected
        for path, expected in protocol.get("source_sha256", {}).items()
    )
    identity = {
        "protocol_status": protocol.get("status") == PROTOCOL_STATUS,
        "protocol_source_identity": source_identity,
        "protocol_bound_to_runs": mask_run.get("protocol_lock_sha256")
        == v43_run.get("protocol_lock_sha256")
        == file_sha256(args.protocol_lock),
        "run_summaries_bound_to_evaluations": mask_eval.get("run_summary_sha256")
        == file_sha256(args.mask_run)
        and v43_eval.get("run_summary_sha256") == file_sha256(args.v43_run),
        "adapters_bound": mask_eval.get("adapter_sha256")
        == mask_run.get("adapter_sha256")
        and v43_eval.get("adapter_sha256") == v43_run.get("adapter_sha256"),
        "evaluation_files_bound": mask_eval.get("public_events_sha256")
        == v43_eval.get("public_events_sha256")
        == eval_manifest.get("public_sha256")
        and mask_eval.get("private_outcomes_sha256")
        == v43_eval.get("private_outcomes_sha256")
        == eval_manifest.get("private_sha256"),
        "result_files_bound": mask_eval.get("results_sha256")
        == file_sha256(args.mask_results)
        and v43_eval.get("results_sha256") == file_sha256(args.v43_results),
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
            or role_lock.get("known_before_v43_training") is True
            and role_lock.get("manifest_sha256") == file_sha256(args.eval_manifest)
            and role_lock.get("public_sha256")
            == file_sha256(args.eval_manifest.parent / "events.public.jsonl")
            and role_lock.get("private_sha256")
            == file_sha256(args.eval_manifest.parent / "outcomes.private.jsonl")
        ),
        "base_model_bound": mask_run.get("base_model_sha256")
        == v43_run.get("base_model_sha256")
        == mask_eval.get("base_model_sha256")
        == v43_eval.get("base_model_sha256")
        == protocol.get("base_model_sha256"),
    }
    gate = build_gate(
        role=args.role,
        mask_eval=mask_eval,
        v43_eval=v43_eval,
        mask_run=mask_run,
        v43_run=v43_run,
        mask_rows=read_jsonl(args.mask_results),
        v43_rows=read_jsonl(args.v43_results),
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
