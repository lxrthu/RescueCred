from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping


DEFAULT_THRESHOLDS = {
    "min_scenarios": 30,
    "min_controlled_valid_events": 20,
    "min_controlled_nonzero_rate": 0.20,
    "min_natural_eligible_events": 3,
    "min_natural_nonzero_events": 3,
    "min_natural_rescue_events": 2,
    "min_natural_outcome_or_progress_rescue_events": 1,
    "max_natural_harm_rate": 0.10,
    "max_worker_failure_rate": 0.10,
}


def _mode_rows(rows: Iterable[Mapping[str, Any]], mode: str) -> List[Mapping[str, Any]]:
    return [row for row in rows if row.get("mode") == mode]


def _mode_metrics(rows: Iterable[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    selected = _mode_rows(rows, mode)
    valid = [row for row in selected if row.get("replay_valid") is True]
    decisions = Counter(str(row.get("decision")) for row in valid)
    decision_bases = Counter(str(row.get("decision_basis")) for row in valid)
    nonzero = sum(decision != "zero_delta" for decision in decisions.elements())
    terminal_nonzero = sum(
        abs(float(row.get("terminal_delta", row.get("delta", 0.0)) or 0.0)) > 1e-12
        for row in valid
    )
    outcome_or_progress = sum(
        row.get("decision_basis")
        in {"final_official_similarity", "bounded_progress_auc"}
        and row.get("decision") != "zero_delta"
        for row in valid
    )
    efficiency_only = sum(
        row.get("decision_basis")
        in {
            "visible_tool_error_advantage",
            "official_turn_advantage",
            "branch_step_advantage",
        }
        and row.get("decision") != "zero_delta"
        for row in valid
    )
    rescue_events = int(decisions.get("rescue_preference", 0))
    reverse_events = int(decisions.get("reverse_preference", 0))
    outcome_or_progress_rescue = sum(
        row.get("decision_basis")
        in {"final_official_similarity", "bounded_progress_auc"}
        and row.get("decision") == "rescue_preference"
        for row in valid
    )
    return {
        "events": len(selected),
        "valid_events": len(valid),
        "nonzero_events": nonzero,
        "nonzero_rate": nonzero / len(valid) if valid else 0.0,
        "terminal_nonzero_events": terminal_nonzero,
        "terminal_nonzero_rate": terminal_nonzero / len(valid) if valid else 0.0,
        "outcome_or_progress_events": outcome_or_progress,
        "efficiency_only_events": efficiency_only,
        "rescue_events": rescue_events,
        "reverse_events": reverse_events,
        "harm_rate": reverse_events / len(valid) if valid else 0.0,
        "outcome_or_progress_rescue_events": outcome_or_progress_rescue,
        "decisions": dict(sorted(decisions.items())),
        "decision_bases": dict(sorted(decision_bases.items())),
        "mean_terminal_delta": (
            sum(float(row.get("delta", 0.0)) for row in valid) / len(valid)
            if valid
            else 0.0
        ),
    }


def build_summary_and_gate(
    rows: List[Mapping[str, Any]],
    scenarios_requested: int,
    scenarios_selected: int,
    worker_failures: int,
    snapshot_restore_exact: bool,
    official_evaluator_used: bool,
    protocol_required: bool = False,
    protocol_validated: bool = True,
    thresholds: Mapping[str, Any] = DEFAULT_THRESHOLDS,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    controlled = _mode_metrics(rows, "controlled_missing_argument")
    natural = _mode_metrics(rows, "natural_visible_error_repair")
    requests = max(1, scenarios_selected)
    mechanism_checks = {
        "enough_scenarios": scenarios_selected >= int(thresholds["min_scenarios"]),
        "snapshot_restore_exact": bool(snapshot_restore_exact),
        "official_evaluator_used": bool(official_evaluator_used),
        "enough_controlled_valid_events": controlled["valid_events"]
        >= int(thresholds["min_controlled_valid_events"]),
        "controlled_signal_density": controlled["nonzero_rate"]
        >= float(thresholds["min_controlled_nonzero_rate"]),
        "worker_failure_rate": worker_failures / requests
        <= float(thresholds["max_worker_failure_rate"]),
        "protocol_validated": (not protocol_required) or bool(protocol_validated),
    }
    deployable_checks = {
        **mechanism_checks,
        "natural_harness_has_coverage": natural["valid_events"]
        >= int(thresholds["min_natural_eligible_events"]),
        "natural_harness_has_nonzero_credit": natural["nonzero_events"]
        >= int(thresholds["min_natural_nonzero_events"]),
        "natural_harness_has_rescues": natural["rescue_events"]
        >= int(thresholds["min_natural_rescue_events"]),
        "natural_harness_improves_official_outcome_or_progress": natural[
            "outcome_or_progress_rescue_events"
        ]
        >= int(thresholds["min_natural_outcome_or_progress_rescue_events"]),
        "natural_harness_wins_over_losses": natural["rescue_events"]
        > natural["reverse_events"],
        "natural_harness_harm_rate": natural["harm_rate"]
        <= float(thresholds["max_natural_harm_rate"]),
    }
    summary = {
        "status": "completed",
        "stage": "toolsandbox_harness_shadow_signal_audit",
        "scenarios_requested": scenarios_requested,
        "scenarios_selected": scenarios_selected,
        "controlled": controlled,
        "natural": natural,
        "worker_failures": worker_failures,
        "worker_failure_rate": worker_failures / requests,
        "reward_source": "official ToolSandbox EvaluationResult.similarity",
        "reference_boundary": {
            "worker_inputs": [
                "visible task messages",
                "public tool schemas",
                "visible tool receipts",
                "proposal A",
            ],
            "treatment_search_prefix": (
                "reference-free worker actions and visible receipts only"
            ),
            "milestones_and_minefields": "offline branch scoring only",
            "reference_actions": "never read or exported",
        },
        "controlled_scope": (
            "mechanism diagnostic: A removes one public-schema required argument "
            "from the first eligible reference-free model proposal B along a "
            "common visible prefix"
        ),
        "natural_scope": (
            "deployable diagnostic: B is generated only after A yields a visible error"
        ),
    }
    gate = {
        "passed": all(deployable_checks.values()),
        "mechanism_passed": all(mechanism_checks.values()),
        "deployable_harness_passed": all(deployable_checks.values()),
        "stage": "toolsandbox_harness_shadow_signal_gate",
        "checks": deployable_checks,
        "mechanism_checks": mechanism_checks,
        "deployable_harness_checks": deployable_checks,
        "thresholds": dict(thresholds),
        "controlled_nonzero_rate": controlled["nonzero_rate"],
        "controlled_terminal_nonzero_rate": controlled["terminal_nonzero_rate"],
        "natural_nonzero_rate": natural["nonzero_rate"],
        "next_step": (
            "freeze deployable ToolSandbox V4 comparison protocol"
            if all(deployable_checks.values())
            else (
                "controlled mechanism pilot only; no deployable Harness claim"
                if all(mechanism_checks.values())
                else "do not train; inspect ToolSandbox audit failures"
            )
        ),
    }
    return summary, gate
