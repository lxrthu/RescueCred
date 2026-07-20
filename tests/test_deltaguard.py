import copy
import hashlib
import json
from dataclasses import dataclass

from rescuecredit.deltaguard_certificate import build_delta_certificate
from rescuecredit.deltaguard_contract import apply_contract_abstention
from rescuecredit.deltaguard_observers import (
    action_pair_family,
    build_observer_plan,
    public_role_map,
)
from rescuecredit.deltaguard_probe import acquisition_selected, run_paired_probe
from rescuecredit.deltaguard_toolsandbox import public_structure_digest


def _schema(name, description, properties=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


SCHEMAS = [
    _schema("set_cellular_service_status", "Enable / Disable cellular service", {"on": {"type": "boolean"}}),
    _schema("get_cellular_service_status", "Request cellular service status"),
    _schema("send_message_with_phone_number", "Send a message to a recipient using phone number"),
    _schema("search_messages", "Search for a message based on provided arguments"),
]


@dataclass
class Receipt:
    action: dict
    content: str
    exception: str | None
    context: dict


class FakeRuntime:
    def snapshot(self, context):
        return copy.deepcopy(context)

    def context_digest(self, context):
        return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()

    def tool_schemas(self, context):
        return SCHEMAS

    def visible_history(self, context):
        return []

    def execute(self, context, action):
        current = copy.deepcopy(context)
        tool = action["tool"]
        arguments = action["arguments"]
        exception = None
        content = "None"
        if tool == "get_cellular_service_status":
            content = repr(current["cellular"])
        elif tool == "set_cellular_service_status":
            if current["cellular"] == arguments["on"]:
                exception = "ValueError: already in requested state"
            else:
                current["cellular"] = arguments["on"]
        elif tool == "send_message_with_phone_number":
            if not current["cellular"]:
                exception = "ConnectionError: Cellular service is not enabled"
            else:
                current["messages"].append(
                    {
                        "recipient_phone_number": arguments["phone_number"],
                        "content": arguments["content"],
                    }
                )
                content = repr("00000000-0000-4000-8000-000000000001")
        elif tool == "search_messages":
            content = repr(
                [
                    row
                    for row in current["messages"]
                    if row["recipient_phone_number"] == arguments["recipient_phone_number"]
                    and row["content"] == arguments["content"]
                ]
            )
        else:
            raise AssertionError(tool)
        return Receipt(action=dict(action), content=content, exception=exception, context=current)


def _actions():
    action_a = {
        "tool": "send_message_with_phone_number",
        "arguments": {"phone_number": "+15550000000", "content": "hello"},
    }
    action_b = {
        "tool": "set_cellular_service_status",
        "arguments": {"on": True},
    }
    return action_a, action_b


def test_scrambled_schema_roles_use_public_descriptions():
    schemas = [
        _schema("settings_0", "Enable / Disable cellular service"),
        _schema("settings_1", "Request cellular service status"),
        _schema("messages_0", "Search for a message based on provided arguments"),
    ]
    roles = public_role_map(schemas)
    assert roles["set_cellular"] == "settings_0"
    assert roles["get_cellular"] == "settings_1"
    assert roles["search_messages"] == "messages_0"


def test_public_delta_identifies_rescue_and_reverse_without_labels():
    runtime = FakeRuntime()
    action_a, action_b = _actions()
    plan = build_observer_plan(
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
        instruction="Send hello to +15550000000",
    )
    rescue_evidence = run_paired_probe(
        runtime=runtime,
        prefix={"cellular": False, "messages": []},
        action_a=action_a,
        action_b=action_b,
        plan=plan,
    )
    rescue = build_delta_certificate(rescue_evidence)
    assert rescue["relation"] == "b_dominates_a"
    assert rescue["reverse_score"] == 0.0
    assert rescue["route_to_a"] is False

    reverse_evidence = run_paired_probe(
        runtime=runtime,
        prefix={"cellular": True, "messages": []},
        action_a=action_a,
        action_b=action_b,
        plan=plan,
    )
    reverse = build_delta_certificate(reverse_evidence)
    assert reverse["relation"] == "a_dominates_b"
    assert reverse["reverse_score"] == 1.0
    assert reverse["route_to_a"] is True


def test_contract_can_only_preserve_or_abstain_from_a_route():
    runtime = FakeRuntime()
    action_a, action_b = _actions()
    plan = build_observer_plan(
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
        instruction="Send hello to +15550000000",
    )
    certificate = build_delta_certificate(
        run_paired_probe(
            runtime=runtime,
            prefix={"cellular": True, "messages": []},
            action_a=action_a,
            action_b=action_b,
            plan=plan,
        )
    )
    valid = {
        "version": "pic-v2",
        "action_hashes": {"A": certificate["action_hash_a"], "B": certificate["action_hash_b"]},
        "claims": [
            {
                "predicate_id": predicate_id,
                "favours": "A",
                "evidence": [{"source": "observer", "schema_path": "$.parsed"}],
            }
            for predicate_id in certificate["witness_predicates"]
        ],
    }
    assert apply_contract_abstention(certificate, valid)["route_to_a"] is True
    invalid = copy.deepcopy(valid)
    invalid["action_hashes"]["B"] = "drift"
    result = apply_contract_abstention(certificate, invalid)
    assert result["route_to_a"] is False
    assert result["reverse_score"] == 0.5


def test_hmac_acquisition_is_stable_and_pre_outcome():
    first = acquisition_selected(event_id="event-7", eligible=True, key="public", rate=0.25)
    second = acquisition_selected(event_id="event-7", eligible=True, key="public", rate=0.25)
    assert first == second
    assert acquisition_selected(event_id="event-7", eligible=False, key="public", rate=1.0) is False


def test_duplicate_message_is_a_regression_not_progress():
    runtime = FakeRuntime()
    action_a, action_b = _actions()
    plan = build_observer_plan(
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
        instruction="Send hello to +15550000000",
    )
    existing = {
        "cellular": True,
        "messages": [{"recipient_phone_number": "+15550000000", "content": "hello"}],
    }
    certificate = build_delta_certificate(
        run_paired_probe(
            runtime=runtime,
            prefix=existing,
            action_a=action_a,
            action_b=action_b,
            plan=plan,
        )
    )
    message_rows = [row for row in certificate["predicates"] if row["family"] == "messaging"]
    assert any(row["delta_a"] == -1 for row in message_rows)


def test_public_structure_digest_normalizes_replayed_ids_and_timestamps():
    left = {
        "person_id": "11111111-1111-4111-8111-111111111111",
        "creation_timestamp": 1_720_000_000.0,
    }
    right = {
        "person_id": "22222222-2222-4222-8222-222222222222",
        "creation_timestamp": 1_730_000_000.0,
    }
    assert public_structure_digest(left) == public_structure_digest(right)


def test_receipt_only_pair_is_family_eligible_and_routes_only_on_explicit_error():
    action_a = {
        "tool": "search_messages",
        "arguments": {"content": "hello"},
    }
    action_b = {
        "tool": "search_messages",
        "arguments": {"content": "goodbye"},
    }
    assert action_pair_family(action_a, action_b, SCHEMAS) == "messaging"
    evidence = {
        "action_hash_a": "a",
        "action_hash_b": "b",
        "observer_plan": [],
        "pre_observations": [],
        "receipt_family": "messaging",
        "branch_a": {
            "action_receipt": {"exception": None, "value_hash": "success"},
            "post_observations": [],
        },
        "branch_b": {
            "action_receipt": {
                "exception": "ValueError: missing required argument",
                "value_hash": "failure",
            },
            "post_observations": [],
        },
        "prefix_unchanged": True,
    }
    certificate = build_delta_certificate(evidence)
    assert certificate["version"] == "public-paired-delta-v2"
    assert certificate["relation"] == "a_dominates_b"
    assert certificate["route_to_a"] is True


def test_benign_receipt_exception_forces_abstention_without_observers():
    evidence = {
        "observer_plan": [],
        "pre_observations": [],
        "receipt_family": "settings",
        "branch_a": {
            "action_receipt": {"exception": None, "value_hash": "a"},
            "post_observations": [],
        },
        "branch_b": {
            "action_receipt": {
                "exception": "ValueError: already in requested state",
                "value_hash": "b",
            },
            "post_observations": [],
        },
        "prefix_unchanged": True,
    }
    certificate = build_delta_certificate(evidence)
    assert certificate["required_unknown"] is True
    assert certificate["route_to_a"] is False
