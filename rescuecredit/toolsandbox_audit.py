from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping


DEFAULT_THRESHOLDS = {
    "min_scenarios": 30,
    "min_controlled_valid_events": 20,
    "min_controlled_nonzero_rate": 0.20,
    "min_natural_eligible_events": 3,
    "max_worker_failure_rate": 0.10,
}


def _mode_rows(rows: Iterable[Mapping[str, Any]], mode: str) -> List[Mapping[str, Any]]:
    return [row for row in rows if row.get("mode") == mode]


def _mode_metrics(rows: Iterable[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    selected = _mode_rows(rows, mode)
    valid = [row for row in selected if row.get("replay_valid") is True]
    decisions = Counter(str(row.get("decision")) for row in valid)
    nonzero = sum(decision != "zero_delta" for decision in decisions.elements())
    return {
        "events": len(selected),
        "valid_events": len(valid),
        "nonzero_events": nonzero,
        "nonzero_rate": nonzero / len(valid) if valid else 0.0,
        "decisions": dict(sorted(decisions.items())),
        "mean_delta": (
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
    thresholds: Mapping[str, Any] = DEFAULT_THRESHOLDS,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    controlled = _mode_metrics(rows, "controlled_missing_argument")
    natural = _mode_metrics(rows, "natural_visible_error_repair")
    requests = max(1, scenarios_selected)
    checks = {
        "enough_scenarios": scenarios_selected >= int(thresholds["min_scenarios"]),
        "snapshot_restore_exact": bool(snapshot_restore_exact),
        "official_evaluator_used": bool(official_evaluator_used),
        "enough_controlled_valid_events": controlled["valid_events"]
        >= int(thresholds["min_controlled_valid_events"]),
        "controlled_signal_density": controlled["nonzero_rate"]
        >= float(thresholds["min_controlled_nonzero_rate"]),
        "natural_harness_has_coverage": natural["valid_events"]
        >= int(thresholds["min_natural_eligible_events"]),
        "worker_failure_rate": worker_failures / requests
        <= float(thresholds["max_worker_failure_rate"]),
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
            "milestones_and_minefields": "offline branch scoring only",
            "reference_actions": "never read or exported",
        },
        "controlled_scope": (
            "mechanism diagnostic: A removes one public-schema required argument "
            "from a reference-free model proposal B"
        ),
        "natural_scope": (
            "deployable diagnostic: B is generated only after A yields a visible error"
        ),
    }
    gate = {
        "passed": all(checks.values()),
        "stage": "toolsandbox_harness_shadow_signal_gate",
        "checks": checks,
        "thresholds": dict(thresholds),
        "controlled_nonzero_rate": controlled["nonzero_rate"],
        "natural_nonzero_rate": natural["nonzero_rate"],
        "next_step": (
            "freeze ToolSandbox V3.1 comparison protocol"
            if all(checks.values())
            else "do not train; inspect ToolSandbox audit failures"
        ),
    }
    return summary, gate
