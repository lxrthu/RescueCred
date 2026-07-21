from rescuecredit.deltaguard_certificate import build_delta_certificate
from rescuecredit.deltaguard_goal_contract import (
    build_goal_contract,
    goal_predicate_rows,
    verify_goal_contract,
)
from tests.test_deltaguard import SCHEMAS


def _receipt(parsed=None, exception=None, value_hash="hash"):
    return {
        "parsed": parsed,
        "exception": exception,
        "value_hash": value_hash,
    }


def _actions():
    return (
        {
            "tool": "search_messages",
            "arguments": {"content": "project alpha"},
        },
        {
            "tool": "get_current_timestamp",
            "arguments": {},
        },
    )


def test_goal_contract_grounds_only_visible_instruction_values():
    action_a = {
        "tool": "search_messages",
        "arguments": {"content": "project alpha", "sender": "Alice"},
    }
    action_b = {
        "tool": "search_messages",
        "arguments": {"content": "unrelated private value"},
    }
    contract = build_goal_contract(
        instruction="Find Alice's message about project alpha",
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
    )
    verify_goal_contract(contract)
    assert contract["goal_family"] == "messaging"
    assert len(contract["action_grounded_term_ids"]["A"]) == 2
    assert contract["action_grounded_term_ids"]["B"] == []
    assert contract["generated_before_receipts"] is True
    assert contract["outcome_fields_read"] == []
    assert set(contract["action_hashes"]) == {"A", "B"}
    assert set(contract["action_structure_hashes"]) == {"A", "B"}
    assert len(contract["input_sha256"]) == 64


def test_goal_predicates_compare_role_arguments_and_public_receipts():
    action_a, action_b = _actions()
    contract = build_goal_contract(
        instruction="Find the message about project alpha",
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
    )
    evidence = {
        "branch_a": {
            "action_receipt": _receipt(
                [
                    {"content": "project alpha"},
                    {"content": "project alpha"},
                ],
                value_hash="a",
            )
        },
        "branch_b": {"action_receipt": _receipt(1720000000, value_hash="b")},
    }
    rows = {row["predicate_id"]: row for row in goal_predicate_rows(contract, evidence)}
    assert rows["goal:role_alignment"]["delta_a"] == 1
    assert rows["goal:role_alignment"]["delta_b"] == 0
    assert rows["goal:grounded_argument_coverage"]["delta_a"] == 1
    assert rows["goal:grounded_argument_coverage"]["delta_b"] == 0
    assert rows["goal:receipt_term_coverage"]["delta_a"] == 1
    assert rows["goal:receipt_term_coverage"]["delta_b"] == 0


def test_goal_contract_certificate_routes_only_complete_pareto_winner():
    action_a, action_b = _actions()
    contract = build_goal_contract(
        instruction="Find the message about project alpha",
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
    )
    evidence = {
        "action_hash_a": "a",
        "action_hash_b": "b",
        "observer_plan": [],
        "pre_observations": [],
        "receipt_family": "messaging",
        "goal_contract": contract,
        "branch_a": {
            "action_receipt": _receipt([{"content": "project alpha"}], value_hash="a"),
            "post_observations": [],
        },
        "branch_b": {
            "action_receipt": _receipt(1720000000, value_hash="b"),
            "post_observations": [],
        },
        "prefix_unchanged": True,
    }
    certificate = build_delta_certificate(evidence)
    assert certificate["version"] == "public-paired-delta-v3-goal-contract"
    assert certificate["relation"] == "a_dominates_b"
    assert certificate["route_to_a"] is True
    assert "goal:receipt_term_coverage" in certificate["witness_predicates"]

    evidence["branch_b"]["action_receipt"] = _receipt(
        None, exception="ValueError: already complete", value_hash="benign"
    )
    abstained = build_delta_certificate(evidence)
    assert abstained["route_to_a"] is False
    assert abstained["required_unknown"] is True


def test_receipt_outcome_is_not_overridden_by_static_goal_diagnostics():
    action_a = {
        "tool": "search_messages",
        "arguments": {"content": "project alpha"},
    }
    action_b = {
        "tool": "search_messages",
        "arguments": {"content": "other"},
    }
    contract = build_goal_contract(
        instruction="Find the message about project alpha",
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
    )
    evidence = {
        "observer_plan": [],
        "pre_observations": [],
        "receipt_family": "messaging",
        "goal_contract": contract,
        "branch_a": {
            "action_receipt": _receipt([], value_hash="a"),
            "post_observations": [],
        },
        "branch_b": {
            "action_receipt": _receipt(
                [{"content": "project alpha"}], value_hash="b"
            ),
            "post_observations": [],
        },
        "prefix_unchanged": True,
    }
    certificate = build_delta_certificate(evidence)
    assert certificate["relation"] == "b_dominates_a"
    assert certificate["route_to_a"] is False


def test_static_goal_heuristics_cannot_route_without_outcome_witness():
    action_a, action_b = _actions()
    contract = build_goal_contract(
        instruction="Find the message about project alpha",
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
    )
    evidence = {
        "observer_plan": [],
        "pre_observations": [],
        "receipt_family": "messaging",
        "goal_contract": contract,
        "branch_a": {
            "action_receipt": _receipt([], value_hash="a"),
            "post_observations": [],
        },
        "branch_b": {
            "action_receipt": _receipt(1720000000, value_hash="b"),
            "post_observations": [],
        },
        "prefix_unchanged": True,
    }
    certificate = build_delta_certificate(evidence)
    assert certificate["route_to_a"] is False
    assert certificate["witness_predicates"] == []
    diagnostics = {
        row["predicate_id"]: row for row in certificate["predicates"]
    }
    assert diagnostics["goal:role_alignment"]["routing_admissible"] is False
    assert (
        diagnostics["goal:grounded_argument_coverage"]["routing_admissible"]
        is False
    )
