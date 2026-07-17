from __future__ import annotations

from dataclasses import dataclass

from .accounting import BudgetCounter
from .audit import AuditLedger, UniformAuditScheduler
from .estimators import PatchEMA, residual_estimate
from .types import RescueEvent, ShadowResult


@dataclass
class EstimateOutcome:
    event: RescueEvent
    audited: bool
    identifiable: bool


class RescueCreditEngine:
    """Implements estimate-before-EMA-update and charges all shadow work."""

    def __init__(
        self,
        ema: PatchEMA,
        scheduler: UniformAuditScheduler,
        ledger: AuditLedger,
        budget: BudgetCounter,
        exact_confidence_threshold: float = 0.99,
    ) -> None:
        self.ema = ema
        self.scheduler = scheduler
        self.ledger = ledger
        self.budget = budget
        self.exact_confidence_threshold = exact_confidence_threshold
        self.eligible_events = 0
        self.audited_events = 0
        self.valid_audits = 0

    def estimate(self, event: RescueEvent, audit_seed: int, shadow: ShadowResult | None = None, shadow_factory=None) -> EstimateOutcome:
        if event.deterministic_outcome and (event.verifier_confidence or 0.0) >= self.exact_confidence_threshold:
            if event.verifier_label is None:
                raise ValueError("deterministic event requires verifier_label")
            event.g0_hat = float(event.verifier_label)
            event.rescue_gain_hat = (event.assisted_return or 0.0) - event.g0_hat
            return EstimateOutcome(event, audited=False, identifiable=True)
        if not event.shadow_safe:
            return EstimateOutcome(event, audited=False, identifiable=False)
        self.eligible_events += 1

        event_id = ":".join(
            [event.run_id, event.episode_id, event.group_id, event.candidate_id, str(event.step_id), event.patch_id]
        )
        mu = self.ema.predict(event.patch_id)
        probability = self.scheduler.probability_for(event, mu)
        commit = self.ledger.commit(event_id, probability, mu)
        draw = self.ledger.draw(event_id, audit_seed)
        event.mu_prediction = mu
        event.audit_probability = commit.probability
        event.audit_draw = draw.draw

        if draw.draw:
            self.audited_events += 1
            if shadow is None and shadow_factory is not None:
                shadow = shadow_factory()
            if shadow is None:
                raise ValueError("selected audit requires a ShadowResult")
            if not shadow.replay_valid:
                self.budget.charge_failed_replay(shadow.steps)
                return EstimateOutcome(event, audited=True, identifiable=False)
            self.budget.charge_shadow(shadow.steps)
            self.valid_audits += 1
            event.shadow_return = shadow.return_value
            event.g0_hat = residual_estimate(mu, 1, probability, shadow.return_value)
            self.ema.update(event.patch_id, shadow.return_value)
        else:
            if shadow is not None:
                raise ValueError("shadow result must not be observed when draw is zero")
            event.g0_hat = residual_estimate(mu, 0, probability, None)
        event.rescue_gain_hat = (event.assisted_return or 0.0) - event.g0_hat
        return EstimateOutcome(event, audited=bool(draw.draw), identifiable=True)
