from __future__ import annotations

import math
from typing import Any, Dict, Mapping


LEXICOGRAPHIC_COMPONENT_ORDER = (
    "final_official_similarity",
    "bounded_progress_auc",
    "visible_tool_error_advantage",
    "official_turn_advantage",
    "branch_step_advantage",
)
OFFICIAL_SCORE_SOURCE = "official ToolSandbox EvaluationResult.similarity"


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _branch_metrics(branch: Mapping[str, Any]) -> Dict[str, float]:
    score = branch.get("score")
    if not isinstance(score, Mapping):
        raise ValueError("replay-valid branch must contain an official score")
    return {
        "final_similarity": _finite_float(score.get("similarity"), "similarity"),
        "progress_auc": _finite_float(branch.get("progress_auc"), "progress_auc"),
        "tool_errors": _finite_float(branch.get("tool_errors"), "tool_errors"),
        "official_turn_count": _finite_float(
            score.get("turn_count"), "official_turn_count"
        ),
        "branch_steps": _finite_float(branch.get("steps"), "branch_steps"),
    }


def validate_branch_credit_evidence(
    branch: Mapping[str, Any], *, horizon: int, atol: float = 1e-12
) -> Dict[str, float]:
    if branch.get("valid") is not True:
        raise ValueError("cannot validate credit evidence for invalid replay")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    score = branch.get("score")
    trace = branch.get("score_trace")
    padded = branch.get("padded_similarity_trace")
    if not isinstance(score, Mapping) or score.get("source") != OFFICIAL_SCORE_SOURCE:
        raise ValueError("final score lacks official ToolSandbox provenance")
    if not isinstance(trace, list) or not 1 <= len(trace) <= horizon:
        raise ValueError("official score trace length is invalid")
    similarities = []
    for item in trace:
        if not isinstance(item, Mapping) or item.get("source") != OFFICIAL_SCORE_SOURCE:
            raise ValueError("score trace lacks official ToolSandbox provenance")
        similarities.append(_finite_float(item.get("similarity"), "trace similarity"))
    if abs(similarities[-1] - _finite_float(score.get("similarity"), "similarity")) > atol:
        raise ValueError("final score does not equal final trace state")
    expected_padding = similarities + [similarities[-1]] * (horizon - len(similarities))
    if not isinstance(padded, list) or len(padded) != horizon:
        raise ValueError("padded similarity trace length is invalid")
    padded_values = [_finite_float(value, "padded similarity") for value in padded]
    if any(abs(left - right) > atol for left, right in zip(padded_values, expected_padding)):
        raise ValueError("padded similarity trace was not derived from official trace")
    recomputed_auc = sum(expected_padding) / float(horizon)
    if abs(recomputed_auc - _finite_float(branch.get("progress_auc"), "progress_auc")) > atol:
        raise ValueError("progress AUC does not match official score trace")
    return {
        "final_similarity": similarities[-1],
        "progress_auc": recomputed_auc,
    }


def lexicographic_counterfactual_regret(
    branch_a: Mapping[str, Any],
    branch_b: Mapping[str, Any],
    *,
    horizon: int,
    atol: float = 1e-12,
) -> Dict[str, Any]:
    """Return outcome-first counterfactual preference without scalar mixing.

    Positive component values prefer corrected branch B. Error and cost terms
    are written as A minus B so they can never override an earlier official
    outcome or progress difference.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if atol < 0:
        raise ValueError("atol must be nonnegative")
    if branch_a.get("valid") is not True or branch_b.get("valid") is not True:
        return {
            "decision": "invalid",
            "decision_basis": "invalid_replay",
            "decision_value": None,
            "causal_weight": 0.0,
            "components": {},
            "component_order": list(LEXICOGRAPHIC_COMPONENT_ORDER),
        }

    a = _branch_metrics(branch_a)
    b = _branch_metrics(branch_b)
    components = {
        "final_official_similarity": b["final_similarity"] - a["final_similarity"],
        "bounded_progress_auc": b["progress_auc"] - a["progress_auc"],
        "visible_tool_error_advantage": a["tool_errors"] - b["tool_errors"],
        "official_turn_advantage": (
            a["official_turn_count"] - b["official_turn_count"]
        ),
        "branch_step_advantage": a["branch_steps"] - b["branch_steps"],
    }
    for name, value in components.items():
        if not math.isfinite(value):
            raise ValueError(f"non-finite lexicographic component: {name}")

    basis = "all_components_tied"
    decision_value = 0.0
    for name in LEXICOGRAPHIC_COMPONENT_ORDER:
        value = components[name]
        if abs(value) > atol:
            basis = name
            decision_value = value
            break

    if decision_value > 0:
        decision = "rescue_preference"
    elif decision_value < 0:
        decision = "reverse_preference"
    else:
        decision = "zero_delta"

    if basis in {"final_official_similarity", "bounded_progress_auc"}:
        causal_weight = min(1.0, abs(decision_value))
    elif basis in {
        "visible_tool_error_advantage",
        "official_turn_advantage",
        "branch_step_advantage",
    }:
        causal_weight = min(1.0, abs(decision_value) / float(horizon))
    else:
        causal_weight = 0.0

    return {
        "decision": decision,
        "decision_basis": basis,
        "decision_value": decision_value,
        "causal_weight": causal_weight,
        "components": components,
        "component_order": list(LEXICOGRAPHIC_COMPONENT_ORDER),
        "atol": atol,
    }
