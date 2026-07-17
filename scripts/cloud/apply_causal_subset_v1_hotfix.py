#!/usr/bin/env python3
"""Install the reference-free causal-subset diagnostic hotfix in place."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "environments/api_bank/deployable.py"
text = path.read_text(encoding="utf-8")

if "def prerequisite_input_value(" not in text:
    anchor = "        return values\n\n    def validate("
    helper = '''        return values

    def prerequisite_input_value(
        self,
        observation: dict[str, Any],
        parameter: str,
        previous_tool_result: dict[str, Any] | None = None,
    ) -> Any | None:
        """Resolve producer credentials without mixing old/new passwords."""

        value = self.explicit_value(observation, parameter, previous_tool_result)
        if value is not None:
            return value
        aliases = {"password": ("old_password", "current_password")}
        candidates: list[Any] = []
        for alias in aliases.get(parameter.lower(), ()):
            for candidate in self.explicit_values(observation, alias, previous_tool_result):
                if _normalize(candidate) not in {_normalize(item) for item in candidates}:
                    candidates.append(candidate)
        return candidates[0] if len(candidates) == 1 else None

    def validate('''
    if text.count(anchor) != 1:
        raise RuntimeError("prerequisite helper anchor missing")
    text = text.replace(anchor, helper, 1)

old_call = '''            for parameter in producer.get("required", []):
                visible = self.validator.explicit_value(
                    observation, str(parameter), previous_tool_result
                )'''
new_call = '''            for parameter in producer.get("required", []):
                visible = self.validator.prerequisite_input_value(
                    observation, str(parameter), previous_tool_result
                )'''
if old_call in text:
    text = text.replace(old_call, new_call, 1)
elif new_call not in text:
    raise RuntimeError("prerequisite call anchor missing")
path.write_text(text, encoding="utf-8")


def self_check() -> None:
    from environments.api_bank import DeployableAPIBankHarness

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
    assert executed == {
        "tool": "GetUserToken",
        "arguments": {"password": "user1pass", "username": "user1"},
    }


self_check()
print("CAUSAL_SUBSET_V1_HOTFIX_OK")
