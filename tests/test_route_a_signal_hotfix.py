from scripts.appworld_azure_continuation_worker import _next_action
from scripts.train_route_a_preference import (
    balanced_causal_epoch_order,
    preference_loss_components,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, messages, max_tokens=500, temperature=0.0):
        self.calls += 1
        return next(self.responses)


def request():
    return {
        "instruction": "Do the visible task",
        "event_context": {},
        "tool_schemas": [],
        "history": [],
        "remaining_steps": 3,
    }


def test_continuation_repairs_invalid_json_once():
    client = FakeClient(
        [
            "I would call the tool, but this is not JSON",
            '```json\n{"tool":"notes__create","arguments":{"text":"x"}}\n```',
        ]
    )
    result = _next_action(client, request())
    assert result["action"] == {
        "tool": "notes__create",
        "arguments": {"text": "x"},
    }
    assert result["format_repair_attempted"] is True
    assert client.calls == 2


def test_continuation_never_fabricates_default_after_failed_repair():
    client = FakeClient(["not json", '{"tool":"bad","arguments":[]}'])
    result = _next_action(client, request())
    assert result == {
        "action": None,
        "error_type": "invalid_action_shape",
        "format_repair_attempted": True,
    }


def test_v2_epoch_is_balanced_and_budget_matched():
    rows = []
    for index in range(9):
        rows.append({"event_id": f"r{index}", "decision": "rescue_preference"})
    for index in range(15):
        rows.append({"event_id": f"v{index}", "decision": "reverse_preference"})
    for index in range(62):
        rows.append({"event_id": f"z{index}", "decision": "zero_delta"})

    ordered = balanced_causal_epoch_order(rows, 42, 0, len(rows))
    assert len(ordered) == 86
    assert sum(row["decision"] == "rescue_preference" for row in ordered) == 43
    assert sum(row["decision"] == "reverse_preference" for row in ordered) == 43
    assert not any(row["decision"] == "zero_delta" for row in ordered)
    assert ordered == balanced_causal_epoch_order(rows, 42, 0, len(rows))


def test_zero_absolute_coef_exactly_recovers_the_old_dpo_loss():
    import pytest

    torch = pytest.importorskip("torch")

    policy = torch.tensor(-0.2)
    reference = torch.tensor(-0.5)
    total, dpo, _ = preference_loss_components(
        policy,
        reference,
        weight=0.7,
        beta=1.0,
        absolute_margin_coef=0.0,
        target_margin=0.05,
    )
    expected = 0.7 * -torch.nn.functional.logsigmoid(policy - reference)
    assert torch.allclose(total, expected)
    assert torch.allclose(dpo, -torch.nn.functional.logsigmoid(policy - reference))


def test_absolute_margin_term_pushes_the_desired_margin_above_zero():
    import pytest

    torch = pytest.importorskip("torch")

    policy = torch.tensor(-0.2, requires_grad=True)
    reference = torch.tensor(-0.5)
    total, _, absolute = preference_loss_components(
        policy,
        reference,
        weight=1.0,
        beta=1.0,
        absolute_margin_coef=1.0,
        target_margin=0.05,
    )
    total.backward()
    assert float(absolute) > 0
    assert float(policy.grad) < 0  # gradient descent therefore increases the margin
