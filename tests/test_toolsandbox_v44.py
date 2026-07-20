from pathlib import Path

from scripts.audit_toolsandbox_v44_candidates import candidate_values_are_visible
from scripts.freeze_toolsandbox_v44_candidate_protocol import (
    FULL_THRESHOLDS,
    SOURCE_PATHS,
)
from scripts.toolsandbox_azure_worker import _request, _validate_candidate_set


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


def _candidate(content: str) -> dict:
    return {
        "tool_id": "T0000",
        "arguments": {"phone_number": "+123", "content": content},
    }


def test_candidate_set_accepts_distinct_schema_valid_alternatives():
    result, error = _validate_candidate_set(
        {"candidates": [_candidate("y"), _candidate("z")]},
        SCHEMAS,
        "tool_id_v2",
        {
            "tool": "send_message",
            "arguments": {"phone_number": "+123", "content": "x"},
        },
        3,
    )
    assert error is None
    assert [action["arguments"]["content"] for action in result["actions"]] == [
        "y",
        "z",
    ]


def test_diversify_request_exposes_public_catalog_and_returns_unranked_actions():
    class Client:
        def __init__(self):
            self.messages = None

        def complete(self, messages, **kwargs):
            self.messages = messages
            return '{"candidates":[' + (
                '{"tool_id":"T0000","arguments":'
                '{"phone_number":"+123","content":"y"}}]}'
            )

    client = Client()
    result = _request(
        client,
        {
            "mode": "diversify",
            "history": [{"role": "user", "content": "send x or y to +123"}],
            "tool_schemas": SCHEMAS,
            "remaining_steps": 4,
            "candidate_count": 3,
            "proposal_a": {
                "tool": "send_message",
                "arguments": {"phone_number": "+123", "content": "x"},
            },
        },
        "tool_id_v2",
    )
    assert result["actions"][0]["arguments"]["content"] == "y"
    assert "public_tool_catalog" in client.messages[-1]["content"]
    assert "milestone" not in client.messages[-1]["content"].lower()


def test_candidate_set_rejects_copy_duplicate_and_invalid_schema():
    proposal = {
        "tool": "send_message",
        "arguments": {"phone_number": "+123", "content": "x"},
    }
    assert _validate_candidate_set(
        {"candidates": [_candidate("x")]},
        SCHEMAS,
        "tool_id_v2",
        proposal,
        3,
    )[1] == "candidate_matches_proposal"
    assert _validate_candidate_set(
        {"candidates": [_candidate("y"), _candidate("y")]},
        SCHEMAS,
        "tool_id_v2",
        proposal,
        3,
    )[1] == "duplicate_candidate"
    invalid = {"tool_id": "T0000", "arguments": {"content": "y"}}
    assert _validate_candidate_set(
        {"candidates": [invalid]}, SCHEMAS, "tool_id_v2", proposal, 3
    )[1] == "candidate_missing_required_arguments"


def test_candidate_argument_values_require_visible_or_public_provenance():
    proposal = {
        "tool": "send_message",
        "arguments": {"phone_number": "+123", "content": "original"},
    }
    visible = [{"role": "user", "content": "Send either alpha or beta to +123"}]
    supported = {
        "tool": "send_message",
        "arguments": {"phone_number": "+123", "content": "beta"},
    }
    invented = {
        "tool": "send_message",
        "arguments": {"phone_number": "+999", "content": "hallucinated"},
    }
    empty = {
        "tool": "send_message",
        "arguments": {"phone_number": "+123", "content": ""},
    }
    assert candidate_values_are_visible(supported, visible, proposal, SCHEMAS)
    assert not candidate_values_are_visible(invented, visible, proposal, SCHEMAS)
    assert not candidate_values_are_visible(empty, visible, proposal, SCHEMAS)


def test_v44_thresholds_preserve_the_failed_v43_reverse_bar():
    assert FULL_THRESHOLDS["min_nonzero_pairs"] == 60
    assert FULL_THRESHOLDS["min_reverse_pairs"] == 8
    assert FULL_THRESHOLDS["min_reverse_tasks"] == 5
    assert FULL_THRESHOLDS["min_rescue_pairs"] == 8
    assert FULL_THRESHOLDS["max_pairs_per_task"] == 4
    assert "scripts/audit_toolsandbox_signal.py" in SOURCE_PATHS
    assert "scripts/toolsandbox_azure_worker.py" in SOURCE_PATHS


def test_v44_runner_is_sanity_first_and_has_no_training_or_holdout_access():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cloud"
        / "run_toolsandbox_v44_candidate_audit.sh"
    ).read_text(encoding="utf-8")
    freeze_sanity = source.index("freeze_protocol sanity")
    freeze_full = source.index("freeze_protocol full")
    run_sanity = source.index("run_audit sanity")
    sanity_pass = source.index("TOOLSANDBOX_V44_SANITY_PASS")
    run_full = source.index("run_audit full")
    prepare = source.index("prepare_toolsandbox_v44_candidate_data.py", run_full)
    assert freeze_sanity < freeze_full < run_sanity < sanity_pass < run_full < prepare
    assert "--scenario-offset 85" in source
    assert "--candidate-count 3" not in source  # Passed through frozen arguments.
    assert 'freeze_protocol full "$FULL_LOCK" 40 3 4' in source
    assert 'run_audit full "$FULL_LOCK" 40 3 4' in source
    assert "scenario-offset 125" not in source
    assert "scenario-offset 165" not in source
    assert "scenario-offset 205" not in source
    assert "train_toolsandbox" not in source
