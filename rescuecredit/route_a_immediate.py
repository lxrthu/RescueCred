from __future__ import annotations

from typing import Any


TOLERANCE = 1e-12


def causal_decision(score_a: float, score_b: float) -> str:
    delta = float(score_b) - float(score_a)
    if delta > TOLERANCE:
        return "rescue_preference"
    if delta < -TOLERANCE:
        return "reverse_preference"
    return "zero_delta"


def preferred_action(score_a: float, score_b: float) -> str | None:
    decision = causal_decision(score_a, score_b)
    if decision == "rescue_preference":
        return "b"
    if decision == "reverse_preference":
        return "a"
    return None


def summarize_immediate_results(
    rows: list[dict[str, Any]], *, event_set_hash: str
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("evaluation_valid")]
    causal = [
        row
        for row in valid
        if abs(float(row["score_b"]) - float(row["score_a"])) > TOLERANCE
    ]
    disagreements = [
        row for row in valid if row.get("mask_selected") != row.get("v2_selected")
    ]

    def selected_score(row: dict[str, Any], method: str) -> float:
        selected = str(row[f"{method}_selected"])
        return float(row["score_b"] if selected == "b" else row["score_a"])

    def correct(row: dict[str, Any], method: str) -> bool:
        return str(row[f"{method}_selected"]) == preferred_action(
            float(row["score_a"]), float(row["score_b"])
        )

    mask_scores = [selected_score(row, "mask") for row in valid]
    v2_scores = [selected_score(row, "v2") for row in valid]
    paired_deltas = [v2 - mask for mask, v2 in zip(mask_scores, v2_scores)]
    mask_mean = sum(mask_scores) / max(1, len(mask_scores))
    v2_mean = sum(v2_scores) / max(1, len(v2_scores))
    mask_causal_accuracy = sum(correct(row, "mask") for row in causal) / max(
        1, len(causal)
    )
    v2_causal_accuracy = sum(correct(row, "v2") for row in causal) / max(
        1, len(causal)
    )
    decisions: dict[str, int] = {}
    for row in valid:
        decision = causal_decision(float(row["score_a"]), float(row["score_b"]))
        decisions[decision] = decisions.get(decision, 0) + 1

    return {
        "status": "completed",
        "stage": "route_a_appworld_immediate_effect_seed42",
        "event_set_hash": event_set_hash,
        "events": len(rows),
        "valid_paired_events": len(valid),
        "invalid_events": len(rows) - len(valid),
        "nonzero_immediate_events": len(causal),
        "immediate_decisions": decisions,
        "selection_disagreements": len(disagreements),
        "mask_mean_immediate_official_score": mask_mean,
        "v2_mean_immediate_official_score": v2_mean,
        "immediate_score_improvement": v2_mean - mask_mean,
        "mask_causal_selection_accuracy": mask_causal_accuracy,
        "v2_causal_selection_accuracy": v2_causal_accuracy,
        "causal_accuracy_improvement": v2_causal_accuracy - mask_causal_accuracy,
        "v2_better_events": sum(delta > TOLERANCE for delta in paired_deltas),
        "v2_worse_events": sum(delta < -TOLERANCE for delta in paired_deltas),
        "ties": sum(abs(delta) <= TOLERANCE for delta in paired_deltas),
        "continuation_used": False,
        "reference_suffix_used": False,
        "azure_used": False,
        "test_split_access": False,
        "reference_role": (
            "offline prefix reconstruction and official scoring only; protected values "
            "are not exported"
        ),
        "scope": (
            "deterministic immediate-effect diagnostic; not end-to-end AppWorld task success"
        ),
    }


def immediate_gate(
    summary: dict[str, Any], *, min_valid: int = 40, min_nonzero: int = 5
) -> dict[str, Any]:
    checks = {
        "enough_valid_paired_events": int(summary["valid_paired_events"])
        >= min_valid,
        "enough_nonzero_immediate_events": int(summary["nonzero_immediate_events"])
        >= min_nonzero,
        "methods_make_different_selections": int(summary["selection_disagreements"])
        >= 3,
        "v2_improves_immediate_official_score": float(
            summary["immediate_score_improvement"]
        )
        > TOLERANCE,
        "v2_has_more_wins_than_losses": int(summary["v2_better_events"])
        > int(summary["v2_worse_events"]),
        "v2_improves_causal_selection_accuracy": float(
            summary["causal_accuracy_improvement"]
        )
        > TOLERANCE,
        "no_continuation_or_reference_suffix": not bool(
            summary.get("continuation_used")
        )
        and not bool(summary.get("reference_suffix_used")),
        "no_azure_or_test_access": not bool(summary.get("azure_used"))
        and not bool(summary.get("test_split_access")),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "stage": "route_a_appworld_immediate_effect_seed42",
        "checks": checks,
        "minimum_valid_events": min_valid,
        "minimum_nonzero_events": min_nonzero,
        "valid_paired_events": summary["valid_paired_events"],
        "nonzero_immediate_events": summary["nonzero_immediate_events"],
        "selection_disagreements": summary["selection_disagreements"],
        "mask_mean_immediate_official_score": summary[
            "mask_mean_immediate_official_score"
        ],
        "v2_mean_immediate_official_score": summary[
            "v2_mean_immediate_official_score"
        ],
        "immediate_score_improvement": summary["immediate_score_improvement"],
        "mask_causal_selection_accuracy": summary[
            "mask_causal_selection_accuracy"
        ],
        "v2_causal_selection_accuracy": summary["v2_causal_selection_accuracy"],
        "v2_better_events": summary["v2_better_events"],
        "v2_worse_events": summary["v2_worse_events"],
        "scope": summary["scope"],
        "next_step": (
            "freeze this diagnostic and run confirmatory seeds"
            if passed
            else "do not expand seeds; immediate causal evidence is insufficient"
        ),
    }
