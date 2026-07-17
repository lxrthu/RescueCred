from rescuecredit.provenance import build_token_advantages, masked_policy_objective
from rescuecredit.types import TokenSpan


def test_harness_text_never_changes_policy_objective():
    policy_prefix = TokenSpan(0, 2, "policy", "prefix")
    policy_suffix = TokenSpan(4, 6, "policy", "suffix")
    harness_a = TokenSpan(2, 4, "harness", "intervention")
    harness_b = TokenSpan(2, 4, "harness", "intervention")
    advantages_a = build_token_advantages(6, [policy_prefix, harness_a, policy_suffix], -1.0, 0.5)
    advantages_b = build_token_advantages(6, [policy_prefix, harness_b, policy_suffix], -1.0, 0.5)
    log_probs_a = [-0.1, -0.2, -99.0, -88.0, -0.3, -0.4]
    log_probs_b = [-0.1, -0.2, -1.0, -2.0, -0.3, -0.4]
    assert advantages_a[2:4] == [0.0, 0.0]
    assert masked_policy_objective(log_probs_a, advantages_a) == masked_policy_objective(log_probs_b, advantages_b)

