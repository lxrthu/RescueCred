from rescuecredit.types import RescueEvent
from rescuecredit.v2_preference import decide_v2_preference


def event(a="false", b="true", assisted=1.0, shadow=0.0, probability=0.4, draw=1):
    return RescueEvent(
        run_id="r",
        episode_id="e",
        group_id="g",
        candidate_id="c",
        step_id=0,
        state_ref="s",
        state_hash="h",
        proposal_text='{"tool":"A","arguments":{}}',
        proposal_action={"tool": "A", "arguments": {}},
        executed_action={"tool": "B", "arguments": {}},
        correction_text='{"tool":"B","arguments":{}}',
        event_type="replace",
        patch_id="p",
        patch_version="v2",
        verifier_label=None,
        verifier_confidence=None,
        verifier_reason=None,
        deterministic_outcome=False,
        shadow_safe=True,
        teachable_patch=True,
        permanent_safety_patch=False,
        intervention_step=0,
        token_spans=[],
        assisted_return=assisted,
        audit_probability=probability,
        audit_draw=draw,
        shadow_return=shadow,
        metadata={"a_semantic_valid": a, "b_semantic_valid": b},
    )


def test_positive_rescue_prefers_b_with_actual_probability_and_clipping():
    decision = decide_v2_preference(event())
    assert decision.ordinary_direction == "b_over_a"
    assert decision.causal_direction == "b_over_a"
    assert decision.causal_decision == "rescue_preference"
    assert decision.causal_weight == 2.5


def test_harness_error_reverses_only_when_a_true_b_false():
    decision = decide_v2_preference(event(a="true", b="false", assisted=0.0, shadow=1.0))
    assert decision.ordinary_direction == "a_over_b"
    assert decision.causal_direction == "a_over_b"
    assert decision.causal_decision == "harness_error_reversal"


def test_verified_b_negative_delta_is_conflict_not_invalid_action_training():
    decision = decide_v2_preference(event(a="false", b="true", assisted=0.0, shadow=1.0))
    assert decision.ordinary_direction == "b_over_a"
    assert decision.causal_direction is None
    assert decision.causal_decision == "trajectory_conflict"


def test_both_valid_uses_shadow_only_for_trajectory_preference():
    decision = decide_v2_preference(event(a="true", b="true", assisted=0.0, shadow=1.0))
    assert decision.ordinary_direction is None
    assert decision.causal_direction == "a_over_b"
    assert decision.causal_decision == "trajectory_preference"


def test_unaudited_event_never_uses_g0_estimate_for_causal_loss():
    candidate = event(draw=0, shadow=None)
    candidate.g0_hat = 100.0
    decision = decide_v2_preference(candidate)
    assert decision.ordinary_direction == "b_over_a"
    assert decision.causal_direction is None
    assert decision.causal_weight == 0.0


def test_unknown_validity_produces_no_preference():
    decision = decide_v2_preference(event(a="unknown", b="true"))
    assert decision.ordinary_direction is None
    assert decision.causal_direction is None
