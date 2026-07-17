#!/usr/bin/env python3
"""Add private tool-receipt replay and reference-free prerequisite routing."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"ALREADY_PATCHED {relative}")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor in {relative}, found {count}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"PATCHED {relative}")


def replace_unless_marker(relative: str, marker: str, old: str, new: str) -> None:
    path = ROOT / relative
    if marker in path.read_text(encoding="utf-8"):
        print(f"ALREADY_PATCHED {relative}")
        return
    replace_once(relative, old, new)


replace_once(
    "environments/api_bank/data.py",
    '''            input_parameters: dict[str, Any] = {}
            description = ""''',
    '''            input_parameters: dict[str, Any] = {}
            output_parameters: dict[str, Any] = {}
            description = ""''',
)
replace_once(
    "environments/api_bank/data.py",
    '''                if isinstance(target, ast.Name) and target.id in {"input_parameters", "description"}:''',
    '''                if isinstance(target, ast.Name) and target.id in {
                    "input_parameters",
                    "output_parameters",
                    "description",
                }:''',
)
replace_once(
    "environments/api_bank/data.py",
    '''                    if target.id == "input_parameters" and isinstance(literal, dict):
                        input_parameters = literal
                    elif target.id == "description" and isinstance(literal, str):''',
    '''                    if target.id == "input_parameters" and isinstance(literal, dict):
                        input_parameters = literal
                    elif target.id == "output_parameters" and isinstance(literal, dict):
                        output_parameters = literal
                    elif target.id == "description" and isinstance(literal, str):''',
)
replace_once(
    "environments/api_bank/data.py",
    '''                    "parameters": input_parameters,
                    "source_file": path.name,''',
    '''                    "parameters": input_parameters,
                    "output_parameters": output_parameters,
                    "source_file": path.name,''',
)
replace_once(
    "environments/api_bank/data.py",
    '''    if len(reference_actions) != len(calls):
        return None
    tools = sorted({action["tool"] for action in reference_actions})''',
    '''    if len(reference_actions) != len(calls):
        return None
    reference_tool_receipts: list[dict[str, Any]] = []
    for call in calls:
        output = call.get("result", {}).get("output")
        receipt: dict[str, Any] = {"status": "ok", "tool": call["api_name"]}
        if isinstance(output, dict):
            receipt.update(output)
        elif output is not None:
            receipt["output"] = output
        reference_tool_receipts.append(receipt)
    tools = sorted({action["tool"] for action in reference_actions})''',
)
replace_once(
    "environments/api_bank/data.py",
    '''        "reference_actions": reference_actions,
        "success_predicate":''',
    '''        "reference_actions": reference_actions,
        # Private environment replay data. It is exposed only after the agent
        # executes the corresponding correct tool call, never in observation().
        "reference_tool_receipts": reference_tool_receipts,
        "success_predicate":''',
)

replace_once(
    "environments/api_bank/adapter.py",
    '''            if expected is not None and normalized == canonical_action(expected):
                ground_truth_match = True
                self.calls.append(normalized)
                self.call_index += 1
                tool_result = {"status": "ok", "call_index": self.call_index, "tool": normalized["tool"]}''',
    '''            if expected is not None and normalized == canonical_action(expected):
                receipt_index = self.call_index
                ground_truth_match = True
                self.calls.append(normalized)
                self.call_index += 1
                receipts = self.task.get("reference_tool_receipts", [])
                if receipt_index < len(receipts):
                    tool_result = copy.deepcopy(receipts[receipt_index])
                    tool_result.update(
                        {"status": "ok", "call_index": self.call_index, "tool": normalized["tool"]}
                    )
                else:
                    tool_result = {"status": "ok", "call_index": self.call_index, "tool": normalized["tool"]}''',
)

replace_unless_marker(
    "environments/api_bank/deployable.py",
    'normalized_parameter = parameter.lower().replace("_", " ").strip()',
    '''        return values

    def validate(''',
    '''        normalized_parameter = parameter.lower().replace("_", " ").strip()
        if not values and normalized_parameter.endswith(" id"):
            identifiers = []
            for candidate in re.findall(r"(?<!\\d)\\d{6,}(?!\\d)", goal):
                if candidate not in identifiers:
                    identifiers.append(candidate)
            if len(identifiers) == 1:
                values.append(identifiers[0])
        return values

    def validate(''',
)

replace_unless_marker(
    "environments/api_bank/deployable.py",
    "def _visible_prerequisite_action(",
    '''        self.validator = validator or VisibleContextSemanticValidator()
        self.correction_generator = correction_generator

    def inspect(''',
    '''        self.validator = validator or VisibleContextSemanticValidator()
        self.correction_generator = correction_generator

    def _visible_prerequisite_action(
        self,
        observation: dict[str, Any],
        proposal: dict[str, Any],
        previous_tool_result: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        catalog = {str(entry.get("name", "")): entry for entry in observation.get("available_tools", [])}
        current_schema = catalog.get(str(proposal.get("tool", "")))
        if current_schema is None:
            return None
        arguments = dict(proposal.get("arguments", {}))
        unsupported: list[str] = []
        for parameter in current_schema.get("required", []):
            value = arguments.get(str(parameter))
            visible_values = self.validator.explicit_values(observation, str(parameter), previous_tool_result)
            if value is None or not any(_normalize(value) == _normalize(item) for item in visible_values):
                unsupported.append(str(parameter))
        candidates: list[dict[str, Any]] = []
        for missing_value in unsupported:
            producers = [
                schema
                for name, schema in catalog.items()
                if name != proposal.get("tool")
                and missing_value in dict(schema.get("output_parameters", {}))
            ]
            if len(producers) != 1:
                continue
            producer = producers[0]
            producer_arguments: dict[str, Any] = {}
            for parameter in producer.get("required", []):
                visible = self.validator.explicit_value(observation, str(parameter), previous_tool_result)
                if visible is None:
                    producer_arguments = {}
                    break
                producer_arguments[str(parameter)] = visible
            if len(producer_arguments) != len(producer.get("required", [])):
                continue
            candidate = {"tool": producer["name"], "arguments": producer_arguments}
            if self.validator.validate(observation, candidate, previous_tool_result).semantic_valid == "true":
                candidates.append(candidate)
        unique = {
            json.dumps(canonical_action(candidate), ensure_ascii=False, sort_keys=True): candidate
            for candidate in candidates
        }
        return next(iter(unique.values())) if len(unique) == 1 else None

    def inspect(''',
)

replace_unless_marker(
    "environments/api_bank/deployable.py",
    "prerequisite = self._visible_prerequisite_action(",
    '''        if schema is None:
            return self._feedback_only("unknown_tool", a_validity.reason)

        corrected = canonical_action(proposal)''',
    '''        if schema is None:
            return self._feedback_only("unknown_tool", a_validity.reason)

        prerequisite = self._visible_prerequisite_action(observation, proposal, previous_tool_result)
        if prerequisite is not None:
            reason = "unique visible producer supplies a required runtime value"
            if self.condition in {"H1", "Hplacebo"}:
                return HarnessDecision(
                    True, "feedback", "visible_prerequisite_repair", None, reason, True, False, False, False
                )
            return HarnessDecision(
                True, "replace", "visible_prerequisite_repair", canonical_action(prerequisite),
                reason, True, False, self.condition == "H3", False,
            )

        corrected = canonical_action(proposal)''',
)

replace_once(
    "environments/api_bank/deployable.py",
    '''                arguments[str(parameter)] = visible
                changed = True
            # Do not rewrite an existing argument''',
    '''                arguments[str(parameter)] = visible
                changed = True
            elif (
                parameter in arguments
                and str(parameter).lower().endswith("_id")
                and visible is not None
                and _normalize(arguments[parameter]) != _normalize(visible)
            ):
                arguments[str(parameter)] = visible
                changed = True
            # Do not rewrite an existing argument''',
)

replace_once(
    "scripts/run_train.py",
    '''                    "visible_schema_repair",
                    "generated_visible_schema_repair",''',
    '''                    "visible_schema_repair",
                    "visible_argument_repair",
                    "visible_prerequisite_repair",
                    "generated_visible_schema_repair",''',
)


def self_check() -> None:
    from environments.api_bank import APIBankControlledEnv, DeployableAPIBankHarness

    observation = {
        "user_goal": (
            "Modify my reminder. My username is user3 and my password is user3pass. "
            'The content is "Submit proposal" and the time is "2023-03-25 14:00:00".'
        ),
        "available_tools": [
            {
                "name": "GetUserToken", "required": ["password", "username"], "optional": [],
                "output_parameters": {"token": {"type": "str"}},
            },
            {
                "name": "ModifyReminder", "required": ["content", "time", "token"], "optional": [],
                "output_parameters": {"status": {"type": "str"}},
            },
        ],
        "success_predicate_satisfied": False,
    }
    proposal = {
        "tool": "ModifyReminder",
        "arguments": {"content": "Submit proposal", "time": "2023-03-25 14:00:00", "token": "bad"},
    }
    executed, decision = DeployableAPIBankHarness("H3").execute(observation, proposal)
    assert decision.patch_id == "visible_prerequisite_repair"
    assert executed["tool"] == "GetUserToken"

    task = {
        "task_id": "t", "user_goal": "g", "available_tools": observation["available_tools"],
        "reference_actions": [executed],
        "reference_tool_receipts": [{"status": "ok", "tool": "GetUserToken", "token": "runtime-token"}],
    }
    env = APIBankControlledEnv()
    public = env.reset(task, 1)
    assert "runtime-token" not in str(public)
    _, _, _, info = env.step(executed)
    assert info["tool_result"]["token"] == "runtime-token"


self_check()
print("TOOL_RECEIPT_PREREQUISITE_HOTFIX_OK")
