import copy

from rescuecredit.deltaguard_protocol import freeze_source_stream

from tests.test_deltaguard import SCHEMAS, _actions


def _row(event_id, decision):
    action_a, action_b = _actions()
    return {
        "event_id": event_id,
        "task_id_hash": "task-" + event_id,
        "scenario_name": "scenario-" + event_id,
        "action_a": action_a,
        "action_b": action_b,
        "treatment_public_tool_schemas": SCHEMAS,
        "treatment_visible_history": [
            {"sender": "RoleType.USER", "content": "Send hello to +15550000000"}
        ],
        "decision": decision,
    }


def test_source_freeze_is_label_blind():
    rows = [_row(f"e{index}", "rescue_preference") for index in range(6)]
    selected_a, audit_a = freeze_source_stream(
        rows,
        families=["messaging"],
        source_events_per_family=4,
        attempt_cap_per_family=6,
        acquisition_rate=0.25,
    )
    changed = copy.deepcopy(rows)
    for row in changed:
        row["decision"] = "reverse_preference"
    selected_b, audit_b = freeze_source_stream(
        changed,
        families=["messaging"],
        source_events_per_family=4,
        attempt_cap_per_family=6,
        acquisition_rate=0.25,
    )
    assert selected_a == selected_b
    assert audit_a == audit_b
    assert audit_a["labels_inspected"] is False
