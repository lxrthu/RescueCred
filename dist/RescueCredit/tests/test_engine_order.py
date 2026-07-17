from rescuecredit.accounting import BudgetCounter
from rescuecredit.audit import AuditLedger, UniformAuditScheduler
from rescuecredit.engine import RescueCreditEngine
from rescuecredit.estimators import PatchEMA
from rescuecredit.types import RescueEvent, ShadowResult, TokenSpan


def make_event():
    return RescueEvent(
        run_id="r", episode_id="e", group_id="g", candidate_id="c", step_id=0,
        state_ref="s", state_hash="h", proposal_text="bad", proposal_action={}, executed_action={},
        correction_text="good", event_type="replace", patch_id="p", patch_version="1",
        verifier_label=None, verifier_confidence=0.5, verifier_reason="semantic",
        deterministic_outcome=False, shadow_safe=True, teachable_patch=True,
        permanent_safety_patch=False, intervention_step=1,
        token_spans=[TokenSpan(0, 1, "policy", "prefix")], assisted_return=1.0,
    )


def test_current_shadow_updates_ema_only_after_estimate(tmp_path):
    ema = PatchEMA(beta=0.5, cold_start=0.0)
    budget = BudgetCounter()
    engine = RescueCreditEngine(ema, UniformAuditScheduler(1.0), AuditLedger(tmp_path / "audit.jsonl"), budget)
    shadow = ShadowResult(1.0, True, 3, "success", "h", "h", True)
    outcome = engine.estimate(make_event(), audit_seed=1, shadow=shadow)
    assert outcome.event.mu_prediction == 0.0
    assert outcome.event.g0_hat == 1.0
    assert ema.predict("p") == 0.5
    assert budget.shadow_steps == 3

