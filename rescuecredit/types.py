from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Producer = Literal["policy", "harness", "environment", "tool"]
Stage = Literal["prefix", "intervention", "suffix"]
EventType = Literal["feedback", "reject", "retry", "repair", "replace", "rollback"]


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    producer: Producer
    stage: Stage

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("TokenSpan requires 0 <= start < end")


@dataclass
class RescueEvent:
    run_id: str
    episode_id: str
    group_id: str
    candidate_id: str
    step_id: int
    state_ref: str
    state_hash: str
    proposal_text: str
    proposal_action: dict[str, Any]
    executed_action: dict[str, Any]
    correction_text: str | None
    event_type: EventType
    patch_id: str
    patch_version: str
    verifier_label: float | None
    verifier_confidence: float | None
    verifier_reason: str | None
    deterministic_outcome: bool
    shadow_safe: bool
    teachable_patch: bool
    permanent_safety_patch: bool
    intervention_step: int
    token_spans: list[TokenSpan]
    assisted_return: float | None = None
    audit_probability: float | None = None
    audit_draw: int | None = None
    mu_prediction: float | None = None
    shadow_return: float | None = None
    g0_hat: float | None = None
    rescue_gain_hat: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowResult:
    return_value: float
    task_success: bool
    steps: int
    terminal_reason: str
    state_hash_before: str
    state_hash_after_restore: str
    replay_valid: bool


@dataclass(frozen=True)
class HarnessDecision:
    triggered: bool
    event_type: EventType
    patch_id: str
    corrected_action: dict[str, Any] | None
    feedback_text: str | None
    teachable_patch: bool
    permanent_safety_patch: bool
    changes_execution: bool
    deterministic_outcome: bool = False


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    score: float
    confidence: float
    deterministic_outcome: bool
    reason: str

