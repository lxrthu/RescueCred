import json

import pytest

from rescuecredit.frozen_bank import SCHEMA_VERSION, event_id, validate_public_record


def _record():
    proposal = {"tool": "mail__send", "arguments": {"to": "x@example.com"}}
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id(
            task_id="task-1", call_index=2, parameter="body", proposal=proposal
        ),
        "split": "train",
        "task_id": "task-1",
        "prompt": "visible context only",
        "action_a": proposal,
        "action_b": {
            "tool": "mail__send",
            "arguments": {"to": "x@example.com", "body": "hello"},
        },
        "patch_id": "visible_candidate_repair",
        "provenance": {"reference_free_candidate_selection": True},
    }


def test_public_bank_record_accepts_train_only_visible_fields():
    validate_public_record(_record())


@pytest.mark.parametrize("key", ["expected_action", "correct_after", "ground_truth"])
def test_public_bank_rejects_offline_label_leakage(key):
    record = _record()
    record["provenance"][key] = True
    with pytest.raises(ValueError, match="leaked"):
        validate_public_record(record)


def test_public_bank_rejects_non_train_split():
    record = _record()
    record["split"] = "dev"
    with pytest.raises(ValueError, match="train events only"):
        validate_public_record(record)


def test_event_id_is_deterministic():
    first = _record()["event_id"]
    second = _record()["event_id"]
    assert first == second and len(first) == 64
