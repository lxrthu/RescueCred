import inspect
import json

from environments.api_bank import (
    DeployableAPIBankHarness,
    OracleAPIBankHarness,
    VisibleContextSemanticValidator,
    merge_visible_tool_context,
    public_harness_observation,
)
from rescuecredit.training import history_patch_id


OBSERVATION = {
    "task_id": "hidden-task-id",
    "user_goal": 'My username is "admin" and the password is adminpass.',
    "available_tools": [
        {"name": "GetUserToken", "required": ["username", "password"], "optional": []},
    ],
    "call_index": 0,
    "num_reference_actions": 99,
    "success_predicate_satisfied": False,
    "state_hash": "visible-state",
    "reference_actions": [
        {"tool": "SECRET_REFERENCE_TOOL", "arguments": {"secret": "DO_NOT_LEAK"}},
    ],
    "reference_tool_receipts": [{"token": "SECRET_RUNTIME_VALUE"}],
}


def test_deployable_api_cannot_receive_expected_action():
    inspect_parameters = inspect.signature(DeployableAPIBankHarness.inspect).parameters
    execute_parameters = inspect.signature(DeployableAPIBankHarness.execute).parameters
    assert "expected" not in inspect_parameters and "reference_actions" not in inspect_parameters
    assert "expected" not in execute_parameters and "reference_actions" not in execute_parameters


def test_public_observation_removes_reference_fields_and_values():
    public = public_harness_observation(OBSERVATION)
    payload = json.dumps(public, sort_keys=True)
    assert "reference" not in payload.lower()
    assert "SECRET_REFERENCE_TOOL" not in payload
    assert "DO_NOT_LEAK" not in payload
    assert "SECRET_RUNTIME_VALUE" not in payload
    assert "num_reference_actions" not in public


def test_visible_context_validator_is_tri_state_not_schema_equals_semantics():
    validator = VisibleContextSemanticValidator()
    supported = {
        "tool": "GetUserToken",
        "arguments": {"username": "admin", "password": "adminpass"},
    }
    contradicted = {
        "tool": "GetUserToken",
        "arguments": {"username": "mallory", "password": "adminpass"},
    }
    unknown = {
        "tool": "GetUserToken",
        "arguments": {"username": "admin", "password": "runtime-secret"},
    }
    assert validator.validate(OBSERVATION, supported).semantic_valid == "true"
    assert validator.validate(OBSERVATION, contradicted).semantic_valid == "false"
    assert validator.validate(OBSERVATION, unknown).semantic_valid in {"false", "unknown"}


def test_deployable_harness_repairs_only_from_visible_goal():
    proposal = {"tool": "GetUserToken", "arguments": {"username": "admin"}}
    executed, decision = DeployableAPIBankHarness("H3").execute(OBSERVATION, proposal)
    assert decision.changes_execution
    assert decision.patch_id == "visible_schema_repair"
    assert executed == {
        "tool": "GetUserToken",
        "arguments": {"username": "admin", "password": "adminpass"},
    }


def test_deployable_harness_skips_when_visible_context_is_insufficient():
    observation = {
        "user_goal": "Please authenticate me.",
        "available_tools": [
            {"name": "GetUserToken", "required": ["username", "password"], "optional": []},
        ],
        "success_predicate_satisfied": False,
    }
    proposal = {"tool": "GetUserToken", "arguments": {"username": "admin"}}
    executed, decision = DeployableAPIBankHarness("H3").execute(observation, proposal)
    assert executed == proposal
    assert decision.corrected_action is None
    assert not decision.changes_execution


def test_oracle_harness_remains_explicit_upper_bound():
    expected = {"tool": "GetUserToken", "arguments": {"username": "admin", "password": "adminpass"}}
    proposal = {"tool": "Other", "arguments": {}}
    executed, decision = OracleAPIBankHarness("H3").execute(OBSERVATION, proposal, expected)
    assert decision.changes_execution and executed == expected


def test_frozen_generator_can_only_supply_visible_missing_values():
    observation = {
        "user_goal": 'Authenticate using "admin" and "p@ssw0rd".',
        "available_tools": [
            {"name": "GetUserToken", "required": ["username", "password"], "optional": []},
        ],
        "success_predicate_satisfied": False,
    }
    seen = {}

    def generator(public, proposal, reason, receipt):
        seen.update(public)
        return {
            "tool": "GetUserToken",
            "arguments": {"username": "admin", "password": "p@ssw0rd"},
        }

    proposal = {"tool": "GetUserToken", "arguments": {"username": "admin"}}
    executed, decision = DeployableAPIBankHarness("H3", correction_generator=generator).execute(
        observation, proposal
    )
    assert decision.patch_id == "generated_visible_schema_repair"
    assert decision.changes_execution
    assert executed["arguments"]["password"] == "p@ssw0rd"
    assert "reference_actions" not in seen


def test_generated_value_not_visible_in_context_is_rejected():
    observation = {
        "user_goal": 'Authenticate using "admin".',
        "available_tools": [
            {"name": "GetUserToken", "required": ["username", "password"], "optional": []},
        ],
        "success_predicate_satisfied": False,
    }

    def generator(_public, _proposal, _reason, _receipt):
        return {
            "tool": "GetUserToken",
            "arguments": {"username": "admin", "password": "invented-secret"},
        }

    proposal = {"tool": "GetUserToken", "arguments": {"username": "admin"}}
    executed, decision = DeployableAPIBankHarness("H3", correction_generator=generator).execute(
        observation, proposal
    )
    assert executed == proposal
    assert decision.corrected_action is None
    assert not decision.changes_execution


def test_frozen_generator_can_replace_existing_unsupported_value_from_visible_goal():
    observation = {
        "user_goal": "My username is admin.",
        "available_tools": [
            {"name": "LookupUser", "required": ["username"], "optional": []},
        ],
        "success_predicate_satisfied": False,
    }

    def generator(_public, _proposal, _reason, _receipt):
        return {
            "tool": "LookupUser",
            "arguments": {"username": "admin"},
        }

    proposal = {
        "tool": "LookupUser",
        "arguments": {"username": "mallory"},
    }
    executed, decision = DeployableAPIBankHarness("H3", correction_generator=generator).execute(
        observation, proposal
    )
    assert decision.patch_id == "generated_visible_argument_repair"
    assert decision.changes_execution
    assert executed["arguments"]["username"] == "admin"


def test_existing_argument_repair_still_rejects_invented_value():
    observation = {
        "user_goal": "My username is admin.",
        "available_tools": [
            {"name": "LookupUser", "required": ["username"], "optional": []},
        ],
        "success_predicate_satisfied": False,
    }

    def generator(_public, _proposal, _reason, _receipt):
        return {
            "tool": "LookupUser",
            "arguments": {"username": "invented-user"},
        }

    proposal = {
        "tool": "LookupUser",
        "arguments": {"username": "mallory"},
    }
    executed, decision = DeployableAPIBankHarness("H3", correction_generator=generator).execute(
        observation, proposal
    )
    assert executed == proposal
    assert decision.corrected_action is None
    assert not decision.changes_execution


def test_feedback_only_patch_is_never_exposed_to_policy_history():
    observation = {
        "user_goal": "Please authenticate me.",
        "available_tools": [
            {"name": "GetUserToken", "required": ["username", "password"], "optional": []},
        ],
        "success_predicate_satisfied": False,
    }
    proposal = {"tool": "GetUserToken", "arguments": {"username": "admin"}}
    _, decision = DeployableAPIBankHarness("H3").execute(observation, proposal)
    assert decision.triggered
    assert not decision.changes_execution
    assert history_patch_id(decision) is None


def test_unique_visible_identifier_repair_needs_no_reference_action():
    observation = {
        "user_goal": "Cancel my appointment.\n56789012",
        "available_tools": [
            {"name": "CancelRegistration", "required": ["appointment_id"], "optional": []},
        ],
        "success_predicate_satisfied": False,
    }
    proposal = {
        "tool": "CancelRegistration",
        "arguments": {"appointment_id": "1234567890"},
    }
    executed, decision = DeployableAPIBankHarness("H3").execute(observation, proposal)
    assert decision.patch_id == "visible_argument_repair"
    assert decision.changes_execution
    assert executed["arguments"]["appointment_id"] == "56789012"


def test_unique_visible_output_producer_is_called_before_token_consumer():
    observation = {
        "user_goal": (
            "Modify my reminder. My username is user3 and my password is user3pass. "
            'The content is "Submit proposal" and the time is "2023-03-25 14:00:00".'
        ),
        "available_tools": [
            {
                "name": "GetUserToken",
                "required": ["password", "username"],
                "optional": [],
                "output_parameters": {"token": {"type": "str"}},
            },
            {
                "name": "ModifyReminder",
                "required": ["content", "time", "token"],
                "optional": [],
                "output_parameters": {"status": {"type": "str"}},
            },
        ],
        "success_predicate_satisfied": False,
    }
    proposal = {
        "tool": "ModifyReminder",
        "arguments": {
            "content": "Submit proposal",
            "time": "2023-03-25 14:00:00",
            "token": "generated_token",
        },
    }
    executed, decision = DeployableAPIBankHarness("H3").execute(observation, proposal)
    assert decision.patch_id == "visible_prerequisite_repair"
    assert decision.changes_execution
    assert executed == {
        "tool": "GetUserToken",
        "arguments": {"password": "user3pass", "username": "user3"},
    }


def test_password_change_prerequisite_uses_old_not_new_password():
    observation = {
        "user_goal": (
            "Change my password. My username is user1 and my old password is user1pass. "
            "My new password is newpass123."
        ),
        "available_tools": [
            {
                "name": "GetUserToken",
                "required": ["password", "username"],
                "optional": [],
                "output_parameters": {"token": {"type": "str"}},
            },
            {
                "name": "ModifyPassword",
                "required": ["new_password", "old_password", "token"],
                "optional": [],
                "output_parameters": {"status": {"type": "str"}},
            },
        ],
        "success_predicate_satisfied": False,
    }
    proposal = {
        "tool": "ModifyPassword",
        "arguments": {
            "new_password": "newpass123",
            "old_password": "user1pass",
            "token": "made_up_token",
        },
    }
    executed, decision = DeployableAPIBankHarness("H3").execute(observation, proposal)
    assert decision.patch_id == "visible_prerequisite_repair"
    assert decision.changes_execution
    assert executed == {
        "tool": "GetUserToken",
        "arguments": {"password": "user1pass", "username": "user1"},
    }


def test_prerequisite_routing_skips_ambiguous_multi_tool_workflow():
    observation = {
        "user_goal": (
            "Open an account first, then check stock, then show my balance. "
            "My username is JohnDoe and my password is pass123."
        ),
        "available_tools": [
            {
                "name": "GetUserToken",
                "required": ["password", "username"],
                "optional": [],
                "output_parameters": {"token": {"type": "str"}},
            },
            {
                "name": "QueryBalance",
                "required": ["token"],
                "optional": [],
                "output_parameters": {"balance": {"type": "float"}},
            },
            {"name": "OpenBankAccount", "required": ["account"], "optional": []},
            {"name": "QueryStock", "required": ["stock_code"], "optional": []},
        ],
        "success_predicate_satisfied": False,
    }
    proposal = {"tool": "QueryBalance", "arguments": {"token": "generated_token"}}
    executed, decision = DeployableAPIBankHarness("H3").execute(observation, proposal)
    assert executed == proposal
    assert not decision.changes_execution


def test_successful_token_receipt_survives_later_noop_feedback():
    context = merge_visible_tool_context(
        None,
        {"status": "ok", "tool": "GetUserToken", "token": "abc123"},
    )
    context = merge_visible_tool_context(
        context,
        {"status": "no_effect", "tool": "OtherTool", "reason": "off path"},
    )
    assert context is not None
    assert context["token"] == "abc123"
    assert context["latest_feedback"]["status"] == "no_effect"
