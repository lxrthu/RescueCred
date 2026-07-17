from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import RescueEvent

Direction = Literal["a_over_b", "b_over_a"]


@dataclass(frozen=True)
class V2PreferenceDecision:
    ordinary_direction: Direction | None
    causal_direction: Direction | None
    causal_decision: str
    causal_weight: float
    delta: float | None
    audit_probability: float | None


def decide_v2_preference(event: RescueEvent, max_causal_weight: float = 2.5) -> V2PreferenceDecision:
    """Route validity and audited trajectory effects into separate losses.

    ``A`` is the policy proposal and ``B`` is the Harness correction.  Only an
    observed, replay-valid shadow outcome may create a causal preference; the
    residual ``g0_hat`` is deliberately excluded from this decision.
    """

    if max_causal_weight <= 0:
        raise ValueError("max_causal_weight must be positive")
    a_valid = event.metadata.get("a_semantic_valid", "unknown")
    b_valid = event.metadata.get("b_semantic_valid", "unknown")
    if a_valid not in {"true", "false", "unknown"} or b_valid not in {"true", "false", "unknown"}:
        raise ValueError("semantic validity must be true, false, or unknown")

    ordinary: Direction | None = None
    if a_valid == "false" and b_valid == "true":
        ordinary = "b_over_a"
    elif a_valid == "true" and b_valid == "false":
        ordinary = "a_over_b"

    probability = float(event.audit_probability) if event.audit_probability is not None else None
    if event.audit_draw != 1 or event.shadow_return is None or probability is None:
        return V2PreferenceDecision(ordinary, None, "not_audited", 0.0, None, probability)
    if not 0.0 < probability <= 1.0:
        raise ValueError("audited event requires probability in (0, 1]")
    assisted = float(event.assisted_return or 0.0)
    delta = assisted - float(event.shadow_return)
    if abs(delta) <= 1e-12:
        return V2PreferenceDecision(ordinary, None, "zero_delta", 0.0, delta, probability)
    weight = min(abs(delta) / probability, float(max_causal_weight))

    if "unknown" in {a_valid, b_valid}:
        return V2PreferenceDecision(None, None, "validity_unknown", 0.0, delta, probability)

    if delta > 0 and b_valid == "true":
        return V2PreferenceDecision(ordinary, "b_over_a", "rescue_preference", weight, delta, probability)
    if delta < 0 and a_valid == "true" and b_valid == "false":
        return V2PreferenceDecision(ordinary, "a_over_b", "harness_error_reversal", weight, delta, probability)
    if delta < 0 and a_valid == "true" and b_valid == "true":
        return V2PreferenceDecision(ordinary, "a_over_b", "trajectory_preference", weight, delta, probability)
    if delta < 0 and a_valid == "false" and b_valid == "true":
        return V2PreferenceDecision(ordinary, None, "trajectory_conflict", 0.0, delta, probability)
    return V2PreferenceDecision(ordinary, None, "validity_unknown", 0.0, delta, probability)
