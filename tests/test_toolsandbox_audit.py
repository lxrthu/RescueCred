import json

from environments.toolsandbox.adapter import (
    action_schema_complete,
    canonical_action,
    controlled_missing_argument,
    score_decision,
)
from rescuecredit.toolsandbox_audit import build_summary_and_gate
from scripts.toolsandbox_azure_worker import _validate


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["phone_number", "content"],
            },
        },
    }
]


def test_controlled_corruption_uses_only_public_required_field():
    action_b = {
        "tool": "send_message",
        "arguments": {"content": "hello", "phone_number": "+123"},
    }
    result = controlled_missing_argument(action_b, SCHEMAS)
    assert result is not None
    action_a, removed = result
    assert removed == "content"
    assert action_schema_complete(action_b, SCHEMAS)
    assert not action_schema_complete(action_a, SCHEMAS)
    assert action_b["arguments"]["content"] == "hello"


def test_controlled_corruption_abstains_without_complete_public_action():
    incomplete = {"tool": "send_message", "arguments": {"phone_number": "+123"}}
    assert controlled_missing_argument(incomplete, SCHEMAS) is None
    assert controlled_missing_argument(
        {"tool": "unknown", "arguments": {}}, SCHEMAS
    ) is None


def test_action_is_canonical_and_decision_is_three_way():
    action = canonical_action(
        {"tool": "send_message", "arguments": {"z": 1, "a": "x"}}
    )
    assert list(action["arguments"]) == ["a", "z"]
    assert score_decision(1.0) == "rescue_preference"
    assert score_decision(-1.0) == "reverse_preference"
    assert score_decision(1e-14) == "zero_delta"


def test_worker_validation_rejects_unknown_tools_and_allows_repair_abstention():
    result, error = _validate(
        {"tool": "send_message", "arguments": {"content": "x"}},
        SCHEMAS,
        False,
    )
    assert error is None
    assert result["action"]["tool"] == "send_message"
    assert _validate({"tool": "hidden", "arguments": {}}, SCHEMAS, False)[1] == "unknown_tool"
    assert _validate({"abstain": True}, SCHEMAS, True)[0]["abstained"] is True


def _rows(controlled_nonzero=8, natural=3):
    rows = []
    for index in range(20):
        delta = 0.2 if index < controlled_nonzero else 0.0
        rows.append(
            {
                "mode": "controlled_missing_argument",
                "replay_valid": True,
                "decision": score_decision(delta),
                "delta": delta,
            }
        )
    for index in range(natural):
        rows.append(
            {
                "mode": "natural_visible_error_repair",
                "replay_valid": True,
                "decision": "rescue_preference",
                "delta": 0.1,
            }
        )
    return rows


def test_signal_gate_passes_dense_controlled_and_natural_audit():
    summary, gate = build_summary_and_gate(
        _rows(),
        scenarios_requested=40,
        scenarios_selected=40,
        worker_failures=0,
        snapshot_restore_exact=True,
        official_evaluator_used=True,
    )
    assert gate["passed"] is True
    assert summary["controlled"]["nonzero_rate"] == 0.4
    assert summary["reference_boundary"]["reference_actions"] == "never read or exported"


def test_signal_gate_rejects_sparse_or_reference_invalid_run():
    _, gate = build_summary_and_gate(
        _rows(controlled_nonzero=2, natural=0),
        scenarios_requested=40,
        scenarios_selected=40,
        worker_failures=0,
        snapshot_restore_exact=False,
        official_evaluator_used=True,
    )
    assert gate["passed"] is False
    assert gate["checks"]["controlled_signal_density"] is False
    assert gate["checks"]["snapshot_restore_exact"] is False
    assert gate["checks"]["natural_harness_has_coverage"] is False


def test_natural_coverage_requires_replay_valid_pairs():
    rows = _rows(controlled_nonzero=8, natural=0)
    rows.extend(
        {
            "mode": "natural_visible_error_repair",
            "replay_valid": False,
            "decision": "invalid",
            "delta": None,
        }
        for _ in range(3)
    )
    _, gate = build_summary_and_gate(
        rows,
        scenarios_requested=40,
        scenarios_selected=40,
        worker_failures=0,
        snapshot_restore_exact=True,
        official_evaluator_used=True,
    )
    assert gate["checks"]["natural_harness_has_coverage"] is False


def test_audit_rows_are_json_serializable():
    json.dumps(_rows(), ensure_ascii=False)
