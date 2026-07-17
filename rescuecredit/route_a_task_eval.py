from __future__ import annotations

import hashlib
import json
from typing import Any


def event_set_hash(rows: list[dict[str, Any]]) -> str:
    identifiers = sorted(str(row["event_id"]) for row in rows)
    payload = json.dumps(identifiers, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validated_action(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    tool = value.get("tool")
    arguments = value.get("arguments")
    if not isinstance(tool, str) or not tool or not isinstance(arguments, dict):
        return None
    if ":/" not in tool and "__" not in tool and "." not in tool:
        return None
    return {"tool": tool, "arguments": arguments}


def summarize_task_results(
    rows: list[dict[str, Any]], *, method: str, events_hash: str
) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("evaluation_valid")]
    scores = [float(row["official_score"]) for row in evaluated]
    return {
        "status": "completed",
        "method": method,
        "event_set_hash": events_hash,
        "events": len(rows),
        "evaluated_events": len(evaluated),
        "mean_official_requirement_score": sum(scores) / max(1, len(scores)),
        "task_success_rate": sum(score >= 1.0 - 1e-12 for score in scores)
        / max(1, len(scores)),
        "exact_correction_rate": sum(
            bool(row.get("correction_matches_reference")) for row in evaluated
        )
        / max(1, len(evaluated)),
        "generation_failure_rate": sum(
            bool(row.get("generation_failed")) for row in rows
        )
        / max(1, len(rows)),
        "candidate_execution_failure_rate": sum(
            bool(row.get("candidate_execution_failed")) for row in evaluated
        )
        / max(1, len(evaluated)),
        "reference_free_model_inputs": True,
        "reference_role": "offline dev fixture and official scoring only",
    }


def paired_gate(
    mask_summary: dict[str, Any],
    v2_summary: dict[str, Any],
    mask_rows: list[dict[str, Any]],
    v2_rows: list[dict[str, Any]],
    *,
    min_events: int = 20,
) -> dict[str, Any]:
    mask_by_id = {row["event_id"]: row for row in mask_rows if row.get("evaluation_valid")}
    v2_by_id = {row["event_id"]: row for row in v2_rows if row.get("evaluation_valid")}
    shared = sorted(set(mask_by_id) & set(v2_by_id))
    mask_scores = [float(mask_by_id[event_id]["official_score"]) for event_id in shared]
    v2_scores = [float(v2_by_id[event_id]["official_score"]) for event_id in shared]
    paired_deltas = [v2 - mask for mask, v2 in zip(mask_scores, v2_scores)]
    mask_mean = sum(mask_scores) / max(1, len(mask_scores))
    v2_mean = sum(v2_scores) / max(1, len(v2_scores))
    mask_success = sum(score >= 1.0 - 1e-12 for score in mask_scores) / max(
        1, len(mask_scores)
    )
    v2_success = sum(score >= 1.0 - 1e-12 for score in v2_scores) / max(
        1, len(v2_scores)
    )
    score_delta = v2_mean - mask_mean
    success_delta = v2_success - mask_success
    checks = {
        "same_frozen_dev_event_set": mask_summary["event_set_hash"]
        == v2_summary["event_set_hash"],
        "enough_paired_events": len(shared) >= min_events,
        "reference_free_model_inputs": bool(
            mask_summary.get("reference_free_model_inputs")
            and v2_summary.get("reference_free_model_inputs")
        ),
        "no_reference_suffix": mask_summary.get("reference_suffix_used") is False
        and v2_summary.get("reference_suffix_used") is False,
        "adapter_scoring_operational": float(
            mask_summary.get("adapter_scoring_failure_rate", 1.0)
        )
        <= 0.05
        and float(v2_summary.get("adapter_scoring_failure_rate", 1.0)) <= 0.05,
        "v2_noninferior_task_success": success_delta >= -1e-12,
        "v2_improves_official_score": score_delta > 1e-12,
    }
    return {
        "passed": all(checks.values()),
        "stage": "route_a_appworld_dev_task_score_seed42",
        "checks": checks,
        "paired_events": len(shared),
        "mask_mean_official_requirement_score": mask_mean,
        "v2_mean_official_requirement_score": v2_mean,
        "official_score_improvement": score_delta,
        "mask_task_success_rate": mask_success,
        "v2_task_success_rate": v2_success,
        "task_success_improvement": success_delta,
        "v2_better_events": sum(delta > 1e-12 for delta in paired_deltas),
        "v2_worse_events": sum(delta < -1e-12 for delta in paired_deltas),
        "ties": sum(abs(delta) <= 1e-12 for delta in paired_deltas),
        "scope": (
            "controlled-state AppWorld dev evaluation; reference trajectories reconstruct "
            "evaluation state but never enter model inputs"
        ),
        "next_step": (
            "freeze the method and run confirmatory seeds"
            if all(checks.values())
            else "do not expand seeds; inspect paired dev failures"
        ),
    }
