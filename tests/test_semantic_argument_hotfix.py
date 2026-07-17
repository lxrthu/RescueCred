from environments.api_bank import APIBankControlledEnv, APIBankHarness
from rescuecredit.training import CreditAssigner


TASK = {
    "available_tools": [{"name": "SendEmail", "required": ["to", "body"], "optional": []}],
    "reference_actions": [{"tool": "SendEmail", "arguments": {"to": "a@b.com", "body": "hello"}}],
    "max_steps": 4,
}


def test_semantic_argument_mismatch_is_repaired_and_auditable():
    env = APIBankControlledEnv()
    env.reset(TASK, 9)
    bad = {"tool": "SendEmail", "arguments": {"to": "wrong", "body": "hello"}}
    executed, decision = APIBankHarness("H3").execute(env.observation(), bad, env.expected_action())
    assert decision.patch_id == "semantic_argument_mismatch"
    assert decision.triggered and decision.changes_execution
    assert executed == TASK["reference_actions"][0]
    _, _, _, info = env.step(executed)
    assert info["ground_truth_match"] is True

    record = CreditAssigner("rescuecredit", audit_probability=1.0).assign(bad, TASK["reference_actions"][0])
    assert record.intervened and record.teachable
    assert record.audit_draw == 1 and record.shadow_steps == 1


def test_missing_argument_repair_drops_other_wrong_values():
    env = APIBankControlledEnv()
    env.reset(TASK, 10)
    bad = {"tool": "SendEmail", "arguments": {"to": "wrong"}}
    executed, decision = APIBankHarness("H3").execute(env.observation(), bad, env.expected_action())
    assert decision.patch_id == "missing_required_argument"
    assert executed == TASK["reference_actions"][0]
    _, _, _, info = env.step(executed)
    assert info["ground_truth_match"] is True
